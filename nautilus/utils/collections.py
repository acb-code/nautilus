from collections.abc import Sequence
from typing import Any

import numpy as np


def concatenate_dict_of_arrays(
    base: dict[str, np.ndarray], new: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Concatenate values in a dict of numpy arrays along axis 0.
    Mutates and returns `base`.
    """
    for k, v in new.items():
        if k in base:
            base[k] = np.concatenate((base[k], v), axis=0)
        else:
            base[k] = v
    return base


def concatenate_dict_of_lists(
    base: dict[str, list[Any]], new: dict[str, Any]
) -> dict[str, list[Any]]:
    """
    Append values in `new` to lists in `base`.
    If a key isn't present in `base`, it is created with a one-element list.
    Mutates and returns `base`.
    """
    for k, v in new.items():
        if k in base:
            base[k].append(v)
        else:
            base[k] = [v]
    return base


def find_common(list_of_lists: Sequence[Sequence[Any]]) -> list[Any]:
    """
    Return the sorted list of elements common to every list in `list_of_lists`.
    """
    if not list_of_lists:
        return []
    common = set(list_of_lists[0])
    for items in list_of_lists[1:]:
        common &= set(items)
    return sorted(common)


def flatten_list(list_of_lists: Sequence[Sequence[Any]]) -> list[Any]:
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]
