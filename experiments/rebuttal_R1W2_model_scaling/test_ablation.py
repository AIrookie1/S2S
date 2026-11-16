"""Verify new architecture hooks, answer scoring and leakage-free splits."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from experiments.rebuttal_R1W2_model_scaling.common import METHODS, aggregate, decoder_layers, execution_device, model_identity, parameter_grid
from experiments.rebuttal_R1W2_model_scaling.rebuttal_R1W2_run import make_hook, prepare_fold, score_answer


class ToyTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=torch.tensor([[3 + ord(c) % 25 for c in text]]))


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_aggregate_uses_question_counts_and_rejects_overlap(self):
        reports = [dict(method=m, fold=f, question_ids=([0, 1] if f == 0 else [2]),
                        metrics={k: (1.0 if f == 0 else 0.0) for k in ('MC1', 'MC2', 'MC3')})
                   for m in METHODS for f in (0, 1)]
        self.assertAlmostEqual(aggregate(reports)[0]['MC1'], 2 / 3)
        reports[1]['question_ids'] = [1]
        with self.assertRaises(ValueError):
            aggregate(reports)

    def test_train_split_excludes_dev_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            q = np.repeat(np.arange(12), 3)
            labels = np.tile([1, 0, 0], 12)
            h = rng.normal(size=(36, 32)).astype('float32')
            h[labels == 1, 0] += 2
            np.savez_compressed(root / 'features.npz', activations=h, labels=labels, q_indices=q,
                                choice_indices=np.tile([0, 1, 2], 12), choices=np.tile(['yes', 'no', 'maybe'], 12))
            config = dict(seed=42, dev_fraction=0.2, epochs=1, mlp_hidden=8, batch_size=8, learning_rate=0.001)
            for fold in (0, 1):
                folder = root / f'fold{fold}'
                split = prepare_fold(root / 'features.npz', config, fold, folder)
                groups = [set(split[key]) for key in ('geometry', 'predictor', 'dev', 'test')]
                self.assertEqual(set.union(*groups), set(range(12)))
                self.assertEqual(sum(map(len, groups)), 12)
                for method in ('l2s', 's2s'):
                    with np.load(folder / f'{method}.npz') as ckpt:
                        self.assertEqual(set(ckpt['train_q_indices']), groups[0] | groups[1])
                        predictor_key = 'predictor_q_indices' if method == 'l2s' else 'mlp_train_q_indices'
                        self.assertEqual(set(ckpt[predictor_key]), groups[1])
                block = torch.nn.Linear(32, 32)
                for method in METHODS:
                    hook = make_hook(method, parameter_grid(method, False)[0], folder, block)
                    original = torch.randn(1, 5, 32)
                    if hook:
                        edited = hook(original.clone(), layer_name=None, start_idx=2)
                        self.assertTrue(torch.equal(edited[:, :2], original[:, :2]))
                        self.assertTrue(torch.isfinite(edited).all())
                        tuple_edited = hook((original.clone(), 'cache'), layer_name=None, start_idx=2)
                        self.assertEqual(tuple_edited[1], 'cache')

    def test_download_receipt_rewrite_does_not_invalidate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"model_type":"qwen2"}')
            receipt = root / 's2s_download.json'
            receipt.write_text(json.dumps(dict(repo_id='Qwen/test', revision='abc')))
            before = model_identity(directory, 'Qwen/test', 'abc')
            receipt.write_text(json.dumps(dict(repo_id='Qwen/test', revision='abc'), indent=2))
            self.assertEqual(before, model_identity(directory, 'Qwen/test', 'abc'))
            with self.assertRaises(ValueError):
                model_identity(directory, 'Qwen/test', 'different')

    def test_offloaded_device_resolution(self):
        from accelerate.hooks import AlignDevicesHook, SequentialHook
        block = torch.nn.Linear(4, 4, device='meta')
        block._hf_hook = SequentialHook(AlignDevicesHook(), AlignDevicesHook(execution_device='cpu', offload=True))
        self.assertEqual(execution_device(block), torch.device('cpu'))
        container = torch.nn.Sequential(block)
        self.assertEqual(execution_device(container), torch.device('cpu'))

    def test_qwen35_and_qwen2_real_decoder_hooks(self):
        qwen2 = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1))
        text = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                    layer_types=['linear_attention', 'full_attention'],
                    linear_num_key_heads=2, linear_num_value_heads=2,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0,
                                     'mrope_section': [2, 3, 3]})
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
                      num_position_embeddings=16, spatial_merge_size=1)
        qwen35 = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision,
                                                             image_token_id=29, video_token_id=30))
        for model in (qwen2, qwen35):
            model.eval()
            name, layers = decoder_layers(model)
            self.assertEqual(len(layers), 2)
            calls = []
            handle = layers[0].register_forward_hook(lambda *args: calls.append(True))
            score = score_answer(model, ToyTokenizer(), layers[0], 'Q?', 'A', None, 'Qwen')
            handle.remove()
            self.assertEqual(calls, [True])
            self.assertTrue(np.isfinite(score))
            # Confirm the score matches an independent shifted-logit calculation.
            from method.S2S_R.eval_utils import get_instruction
            tokenizer = ToyTokenizer()
            prompt = tokenizer(get_instruction('Qwen') + '\n\nQ: Q? A:').input_ids
            answer = tokenizer(' A').input_ids
            with torch.inference_mode():
                logits = model(torch.cat([prompt, answer], 1), use_cache=False).logits
                lp = logits[0, prompt.shape[1] - 1:-1].float().log_softmax(-1)
                expected = lp.gather(1, answer[0, :, None]).sum().item()
            self.assertAlmostEqual(score, expected, places=5)
            def edit(output, start_idx, layer_name=None):
                h = output[0] if isinstance(output, tuple) else output
                out = h.clone()
                out[:, start_idx:, 0] += 10
                return (out,) + output[1:] if isinstance(output, tuple) else out
            changed = score_answer(model, tokenizer, layers[0], 'Q?', 'A', edit, 'Qwen')
            self.assertNotAlmostEqual(score, changed, places=5)
            self.assertFalse(layers[0]._forward_hooks)


if __name__ == '__main__':
    unittest.main()
