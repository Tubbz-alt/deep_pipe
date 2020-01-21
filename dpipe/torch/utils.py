from pathlib import Path
from typing import Callable, Union, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from dpipe.medim.io import PathLike
from dpipe.medim.itertools import squeeze_first, collect

__all__ = [
    'load_model_state', 'save_model_state',
    'get_device', 'to_device', 'is_on_cuda', 'to_cuda',
    'to_var', 'sequence_to_var', 'to_np', 'sequence_to_np',
    'set_params', 'set_lr',
]

Device = Union[torch.device, nn.Module, torch.Tensor, str]
ArrayLike = Union[np.ndarray, Iterable, int, float]


def load_model_state(module: nn.Module, path: PathLike, modify_state_fn: Callable = None) -> nn.Module:
    """
    Updates the ``module``'s state dict by the one located at ``path``.

    Parameters
    ----------
    module
    path
    modify_state_fn: Callable(current_state, loaded_state)
        if not ``None``, two arguments will be passed to the function:
        current state of the model and the state loaded from the path.
        This function should modify states as needed and return the final state to load.
        For example, it could help you to transfer weights from similar but not completely equal architecture.
    """
    state_to_load = torch.load(path, map_location=get_device(module))
    if modify_state_fn is not None:
        current_state = module.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    module.load_state_dict(state_to_load)
    return module


def save_model_state(module: nn.Module, path: PathLike):
    """Saves the ``module``'s state dict to ``path``."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(module.state_dict(), path)


def get_device(x: Device = None) -> torch.device:
    """
    Determines the correct device based on the input.

    Parameters
    ----------
    x: torch.device, torch.nn.Module, torch.Tensor, str, None
        | if ``torch.Tensor`` - returns the device on which it is located
        | if ``torch.nn.Module`` - returns the device on which its parameters are located
        | if ``str`` or ``torch.device`` - returns `torch.device(x)`
        | if ``None`` - same as 'cuda' if CUDA is available, 'cpu' otherwise.
    """
    if isinstance(x, nn.Module):
        try:
            return next(x.parameters()).device
        except StopIteration:
            raise ValueError('The device could not be determined as the passed model has no parameters.') from None
    if isinstance(x, torch.Tensor):
        return x.device

    if x is None:
        x = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(x)


def is_on_cuda(x: Union[nn.Module, torch.Tensor]):
    if isinstance(x, nn.Module):
        x = next(x.parameters())

    return x.is_cuda


def to_var(*arrays: ArrayLike, device: Device = None, requires_grad: bool = False):
    """
    Convert numpy arrays to torch Tensors.

    Parameters
    ----------
    arrays: array-like
        objects, that will be converted to torch Tensors.
    device
        the device on which to move ``x``. See `get_device` for details.
    requires_grad
        whether the tensors require grad.

    Notes
    -----
    If ``arrays`` contains a single argument the result will not be contained in a tuple:
    >>> x = to_var(x)
    >>> x, y = to_var(x, y)

    If this is not the desired behaviour, use `sequence_to_var`, which always returns a tuple of tensors.
    """
    return squeeze_first(tuple(sequence_to_var(*arrays, device=device, requires_grad=requires_grad)))


def to_np(*tensors: torch.Tensor):
    """
    Convert torch Tensors to numpy arrays.

    Notes
    -----
    If ``tensors`` contains a single argument the result will not be contained in a tuple:
    >>> x = to_np(x)
    >>> x, y = to_np(x, y)

    If this is not the desired behaviour, use `sequence_to_np`, which always returns a tuple of arrays.
    """
    return squeeze_first(tuple(sequence_to_np(*tensors)))


@collect
def sequence_to_var(*arrays: ArrayLike, device: Device = None, requires_grad: bool = False):
    for x in arrays:
        x = torch.from_numpy(np.asarray(x))
        if requires_grad:
            x.requires_grad_()
        yield to_device(x, device)


@collect
def sequence_to_np(*tensors: torch.Tensor):
    for x in tensors:
        # TODO: raise if not tensor?
        if isinstance(x, torch.Tensor):
            x = x.data.cpu().numpy()
        yield x


def to_device(x: Union[nn.Module, torch.Tensor], device: Device = None):
    """
    Move ``x`` to ``device``.

    Parameters
    ----------
    x
    device
        the device on which to move ``x``. See `get_device` for details.
    """
    return x.to(device=get_device(device))


def to_cuda(x, cuda: Union[nn.Module, torch.Tensor, bool] = None):
    """
    Move ``x`` to cuda if specified.

    Parameters
    ----------
    x
    cuda
        whether to move to cuda. If None, torch.cuda.is_available() is used to determine that.
    """
    if isinstance(cuda, (nn.Module, torch.Tensor)):
        cuda = is_on_cuda(cuda)
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def set_lr(optimizer: Optimizer, lr: float) -> Optimizer:
    """Change an ``optimizer``'s learning rate to ``lr``."""
    return set_params(optimizer, lr=lr)


def set_params(optimizer: Optimizer, **params) -> Optimizer:
    """Change an ``optimizer``'s parameters by the ones passed in ``params``."""
    for name, value in params.items():
        for param_group in optimizer.param_groups:
            param_group[name] = value
    return optimizer
