import json
from functools import lru_cache
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.json")


@lru_cache(maxsize=None)
def load_config(path=CONFIG_PATH):
    with Path(path).open() as handle:
        return json.load(handle)
