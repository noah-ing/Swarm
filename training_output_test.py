import pytest
import sys
import os

# Add the directory containing training_output.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_output import flatten, flatten_safe

class TestFlatten:
    def test_basic_nested_list(self):
        """Test basic list nesting"""
        assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]

    def test_empty_list(self):
        """Test empty list handling"""
        assert flatten([]) == []
        assert flatten([[], []]) == []
        assert flatten([1, [], [2, []], 3]) == [1, 2, 3]

    def test_mixed_data_types(self):
        """Test lists with mixed data types"""
        result = flatten([1, "hello", [2.5, [True, None]], {"key": "value"}])
        assert result == [1, "hello", 2.5, True, None, {"key": "value"}]

    def test_deep_nesting(self):
        """Test deeply nested lists"""
        deeply_nested = [[[[[1]]]]]
        assert flatten(deeply_nested) == [1]

    def test_single_elements(self):
        """Test single element scenarios"""
        assert flatten([42]) == [42]
        assert flatten(42) == [42]

    def test_tuples_and_mixed_iterables(self):
        """Test flattening of tuples and mixed iterables"""
        assert flatten((1, (2, 3), [4, (5, 6)])) == [1, 2, 3, 4, 5, 6]
        
        # Test sets and generators (as lists)
        assert flatten([1, 2, [3, 4]]) == [1, 2, 3, 4]

    def test_none_and_falsy_values(self):
        """Test handling of None and falsy values"""
        assert flatten([None, [None], []]) == [None, None]
        assert flatten([0, [False], [""]]) == [0, False, ""]

    def test_complex_nesting(self):
        """Test complex nested structures"""
        assert flatten([1, [2, [3, [4]], 5], 6, [7, [8, 9]]]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_circular_reference(self):
        """Test circular reference detection"""
        circular_list = [1, 2]
        circular_list.append(circular_list)
        
        with pytest.raises(ValueError, match="Circular reference detected"):
            flatten(circular_list)

    def test_flatten_safe_depth_limit(self):
        """Test depth limit in flatten_safe"""
        # Create a very deeply nested list
        deep_list = [1]
        for _ in range(150):
            deep_list = [deep_list]
        
        # Test that it raises RecursionError when exceeding max depth
        with pytest.raises(RecursionError, match="Maximum recursion depth"):
            flatten_safe(deep_list, max_depth=100)

    def test_non_iterable_inputs(self):
        """Test handling of non-iterable inputs"""
        assert flatten("string") == ["string"]
        assert flatten(123) == [123]
        assert flatten(None) == [None]

    def test_nested_dicts(self):
        """Test handling of nested dictionaries"""
        assert flatten([1, {"a": 2, "b": [3, 4]}]) == [1, {"a": 2, "b": [3, 4]}]

def test_performance_large_list():
    """Test performance and correctness with large nested list"""
    # Create a large nested list
    large_list = list(range(1000))
    for _ in range(10):
        large_list = [large_list]
    
    result = flatten(large_list)
    assert len(result) == 1000
    assert result == list(range(1000))

def test_performance_nested_elements():
    """Performance test with many nested elements"""
    elements = list(range(1000))
    nested_list = elements.copy()
    for _ in range(50):
        nested_list = [nested_list, elements]
    
    result = flatten(nested_list)
    assert len(result) == 1000 * 51  # 1000 elements * 51 repetitions