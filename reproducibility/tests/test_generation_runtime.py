"""CPU-only EOS and native hook checks; run with the main ML environment."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "ML environment is optional for lightweight CI")
class GenerationRuntimeTests(unittest.TestCase):
    def test_eos_at_last_permitted_token_is_not_a_length_failure(self):
        import torch
        from types import SimpleNamespace
        from reproducibility.generation import generate
        class Tokenizer:
            def __call__(self, prompt, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
            def decode(self, ids, **kwargs):
                return " ".join(map(str, ids))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(6, 2)
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, input_ids.shape[1], 6)
                logits[0, -1, 3 if input_ids.shape[1] == 2 else 5] = 1
                return SimpleNamespace(logits=logits)
        result = generate(Model(), Tokenizer(), "prompt", 2, eos_ids={5})
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertFalse(result["hit_token_limit"])
        result = generate(Model(), Tokenizer(), "prompt", 1, eos_ids={5})
        self.assertTrue(result["hit_token_limit"])

    def test_additive_vector_does_not_use_test_answer_activations(self):
        import numpy as np
        import tempfile
        from pathlib import Path
        from reproducibility.prepare_additive import prepare
        from method.dyn3d_sga.eval_utils import load_steering_vector
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The held-out half has a deliberately enormous, different direction.
            h = np.array([[1,0],[0,0],[3,0],[0,0],[0,999],[0,0],[0,777],[0,0]], dtype=np.float32)
            np.savez(root/"features.npz", activations=h, labels=np.tile([1,0],4), q_indices=np.repeat(np.arange(4),2))
            prepare(root/"features.npz",root/"vectors")
            with np.load(root/"vectors/fold0.npz") as data:
                np.testing.assert_allclose(data["v"], [2,0])
                self.assertEqual(data["test_q_indices"].tolist(), [2,3])
                vector = load_steering_vector(data, vector_source="v", normalize_v=False, device="cpu")
                np.testing.assert_allclose(vector.numpy(), [2,0])


if __name__ == "__main__":
    unittest.main()
