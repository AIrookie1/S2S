"""Keep modern-Transformers scaling tests in their dedicated environment."""
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):
    if collection_path.name != "rebuttal_R1W2_model_scaling":
        return False
    return not any("rebuttal_R1W2_model_scaling" in str(argument) for argument in config.args)
