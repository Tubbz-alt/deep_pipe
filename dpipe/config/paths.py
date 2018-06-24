""""File with utilities to get library paths."""
import os
import json


def get_paths():
    current_path = os.path.realpath(os.path.dirname(__file__))
    repository_path = os.path.join(current_path, os.pardir, os.pardir)
    with open(os.path.join(repository_path, 'paths.json')) as f:
        paths = json.load(f)

    return {key: os.path.realpath(os.path.join(repository_path, path)) for key, path in paths.items()}
