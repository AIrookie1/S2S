"""Regression checks for the public S2S_L/S2S_R package layout."""
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / "method/S2S_L").is_dir(), "Canonical layout is checked in the generated release")
class ReleaseLayoutTests(unittest.TestCase):
    def test_canonical_method_packages_and_scripts_exist(self):
        import experiments
        self.assertEqual(Path(experiments.__file__).resolve(), ROOT / "experiments/__init__.py")
        for name in ("S2S_L", "S2S_R"):
            self.assertTrue((ROOT / "method" / name / "__init__.py").is_file())
            self.assertTrue((ROOT / "hyperparams" / name / "LLAMA/truthfulqa.sh").is_file())
            self.assertTrue((ROOT / "script" / name / "LLAMA" / f"run_{name}.sh").is_file())

    def test_cli_preserves_case_for_both_canonical_packages(self):
        spec = importlib.util.spec_from_file_location("s2s_release_launcher", ROOT / "test.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        for name in ("S2S_L", "S2S_R"):
            entry = f"evaluate_{name}.py"
            for spelling in (name, name.lower()):
                called = []
                def load(path):
                    called.append(path)
                    return SimpleNamespace(main=lambda: None)
                with patch.object(sys, "argv", ["test.py", "--method-dir", spelling, "--entry-file", entry]), patch.object(launcher, "load_module", load), patch.object(launcher.os, "chdir"):
                    launcher.main()
                self.assertEqual(called[0], ROOT / "method" / name / entry)

    def test_retired_method_labels_are_absent(self):
        retired = re.compile(r"dyn[23]d|[23]d[_-]dyn", re.I)
        for path in ROOT.rglob("*"):
            if any(x in (".git", "__pycache__", ".pytest_cache") for x in path.parts):
                continue
            self.assertIsNone(retired.search(str(path.relative_to(ROOT))), str(path.relative_to(ROOT)))
            if path.is_file() and path.suffix in (".py", ".sh", ".md", ".json", ".csv", ".yml", ".ini"):
                self.assertIsNone(retired.search(path.read_text()), str(path.relative_to(ROOT)))

    def test_planned_python_targets_exist(self):
        from reproducibility.run import build_plan
        from reproducibility.tests.test_workflow import args
        for suite in ("main", "component", "gate", "margin", "generation", "bipo", "scaling"):
            for job in build_plan(args(suite)):
                if job["command"][0] == "bash":
                    self.assertTrue((ROOT / job["command"][1]).is_file())
                    continue
                command = job["command"][1:]
                if command[0] == "-m":
                    target = ROOT / (command[1].replace(".", "/") + ".py")
                elif command[0] == "test.py":
                    target = ROOT / "method" / command[command.index("--method-dir")+1] / command[command.index("--entry-file")+1]
                else:
                    target = ROOT / command[0]
                self.assertTrue(target.is_file(), str(target.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
