"""Extract raw last-token block outputs, including the final block before norm."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from method.S2S_R.data import get_truthfulqa_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    config = json.loads(Path(args.config).read_text())
    device = config["device"]
    dtype = torch.float32 if device == "cpu" else getattr(torch, config["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"], torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError(f"Invalid layer: {args.layer}")
    dataset = get_truthfulqa_data(style="standard")
    activations = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    handle = model.model.layers[args.layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for prompt in tqdm(dataset["prompts"], desc=f"Extract block {args.layer}"):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                model(**inputs, use_cache=False)
    finally:
        handle.remove()
    values = np.stack(activations)
    if len(values) != len(dataset["prompts"]) or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite block activations")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    # The shared trainer constructs raw negative/mean-positive pairs itself.
    np.savez_compressed(temporary, activations=values,
                        **{key: dataset[key] for key in ("labels", "q_indices", "choice_indices", "choices")},
                        model_name=np.asarray(config["model_name"]),
                        layer=np.asarray(args.layer), extraction=np.asarray("block_output_before_final_norm"))
    temporary.replace(path)


if __name__ == "__main__":
    main()
