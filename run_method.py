
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_module(module_path: Path) -> ModuleType:
    """Load a Python module directly from a file path."""
    module_name = f"_dynamic_entry_{module_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="调用 methods/<method>/<entry> 中的 main() 函数"
    )
    parser.add_argument(
        "--method-dir",
        default="S2S_R",
        help="方法名：S2S_R（S2S_R）",
    )
    parser.add_argument(
        "--entry-file",
        default="evaluate_S2S_R.py",
        help="要运行的 Python 文件名",
    )

    # 未被 run_method.py 识别的参数，全部转交给目标脚本。
    launcher_args, target_args = parser.parse_known_args()

    # 允许使用 `--` 清晰分隔启动器参数和目标脚本参数。
    if target_args and target_args[0] == "--":
        target_args = target_args[1:]

    return launcher_args, target_args


def main() -> None:
    launcher_args, target_args = parse_args()

    project_root = Path(__file__).resolve().parent
    entry_file = launcher_args.entry_file

    if not entry_file.endswith(".py"):
        entry_file += ".py"

    method_names = {"s2s_l": "S2S_L", "s2s_r": "S2S_R"}
    method = method_names.get(launcher_args.method_dir.casefold(), launcher_args.method_dir)
    target_path = (project_root / "methods" / method / entry_file).resolve()

    if not target_path.is_file():
        raise FileNotFoundError(
            "找不到目标脚本：\n"
            f"  {target_path}\n"
            "请检查 --method-dir 和 --entry-file。"
        )

    # 保证目标脚本中的相对路径都以 project 根目录为基准。
    os.chdir(project_root)

    # 让目标模块可以导入 project 根目录和自身目录中的模块。
    for path in (project_root, target_path.parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    module = load_module(target_path)

    target_main = getattr(module, "main", None)
    if not callable(target_main):
        raise AttributeError(
            f"{target_path} 中没有找到可调用的 main() 函数。"
        )

    # 让目标 main() 内部的 argparse 像直接运行脚本时一样读取参数。
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(target_path), *target_args]
        target_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
