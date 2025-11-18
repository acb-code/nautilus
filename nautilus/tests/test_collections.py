import numpy as np

from nautilus.utils.collections import (
    concatenate_dict_of_arrays,
    concatenate_dict_of_lists,
    find_common,
    flatten_list,
)


def test_concatenate_dict_of_arrays_appends_along_axis_zero():
    base = {"a": np.array([[1], [2]])}
    new = {"a": np.array([[3]]), "b": np.array([[4]])}

    result = concatenate_dict_of_arrays(base, new)

    np.testing.assert_array_equal(result["a"], np.array([[1], [2], [3]]))
    np.testing.assert_array_equal(result["b"], np.array([[4]]))


def test_concatenate_dict_of_lists_appends_and_creates_keys():
    base = {"a": [1]}
    new = {"a": 2, "b": 3}

    result = concatenate_dict_of_lists(base, new)

    assert result["a"] == [1, 2]
    assert result["b"] == [3]


def test_find_common_handles_empty_and_multiple_lists():
    assert find_common([]) == []
    assert find_common([[1, 2, 3], [3, 2], [2, 3, 4]]) == [2, 3]


def test_flatten_list():
    nested = [[1, 2], [3], [], [4, 5]]
    assert flatten_list(nested) == [1, 2, 3, 4, 5]
