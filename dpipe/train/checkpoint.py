import shutil
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from dpipe.io import PathLike

__all__ = 'CheckpointManager',


def save_pickle(o, path):
    with open(path, 'wb') as file:
        pickle.dump(o.__dict__, file)


def load_pickle(o, path):
    with open(path, 'rb') as file:
        state = pickle.load(file)
        for key, value in state.items():
            setattr(o, key, value)


def save_torch(o, path):
    torch.save(o.state_dict(), path)


def load_torch(o, path):
    o.load_state_dict(torch.load(path))


class CheckpointManager:
    """
    Saves the most recent iteration to ``base_path`` and removes the previous one.

    Parameters
    ----------
    base_path: str
        path to save/restore checkpoint object in/from.
    objects: Dict[PathLike, Any]
        objects to save. Each key-value pair represents
        the path relative to ``base_path`` and the corresponding object.
    frequency: int
        the frequency with which the objects are stored.
        By default only the latest checkpoint is saved.
    """

    def __init__(self, base_path: PathLike, objects: Dict[PathLike, Any], frequency: int = np.inf):
        self.base_path: Path = Path(base_path)
        self._checkpoint_prefix = 'checkpoint_'
        self.objects = objects or {}
        self.frequency = frequency

    def _get_checkpoint_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration}'

    def _clear_checkpoint(self, iteration):
        if (iteration + 1) % self.frequency != 0:
            shutil.rmtree(self._get_checkpoint_folder(iteration))

    @staticmethod
    def _dispatch_saver(o):
        if isinstance(o, (torch.nn.Module, torch.optim.Optimizer)):
            return save_torch
        return save_pickle

    @staticmethod
    def _dispatch_loader(o):
        if isinstance(o, (torch.nn.Module, torch.optim.Optimizer)):
            return load_torch
        return load_pickle

    def save(self, iteration: int):
        """Save the states of all tracked objects."""
        current_folder = self._get_checkpoint_folder(iteration)
        current_folder.mkdir(parents=True)

        for path, o in self.objects.items():
            save = self._dispatch_saver(o)
            save(o, current_folder / path)

        if iteration:
            self._clear_checkpoint(iteration - 1)

    def restore(self) -> int:
        """Restore the most recent states of all tracked objects and return next iteration's index."""
        if not self.base_path.exists():
            return 0

        max_iteration = -1
        for file in self.base_path.iterdir():
            file = file.name
            if file.startswith(self._checkpoint_prefix):
                max_iteration = max(max_iteration, int(file[len(self._checkpoint_prefix):]))

        # no backups found
        if max_iteration < 0:
            return 0

        iteration = max_iteration + 1
        last_folder = self._get_checkpoint_folder(iteration - 1)

        for path, o in self.objects.items():
            load = self._dispatch_loader(o)
            load(o, last_folder / path)

        return iteration
