# Configuration

- `reproduction/llama.json`, `reproduction/qwen.json`: model identity, main training protocol, and fold-specific inference parameters.
- `experiments.json`: every public CLI experiment ID, native entry, dependency profile, and prerequisite.
- `S2S_L/`, `S2S_R/`, and baseline directories: existing method/task-specific shell presets, kept with their scientific settings.

Use `python -m reproducibility list` and preview the resolved plan before an expensive run. Per-task settings such as Qwen COPA's 400-epoch budget remain in the native task presets.
