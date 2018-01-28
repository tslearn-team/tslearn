import numpy as np
from numpy.testing import assert_array_equal
import pytest
from tslearn.utils import _arraylike_copy
from tslearn.utils import bit_length

@pytest.mark.parametrize(
	"test_array, expected_output",
	[([5], np.array([5])),
	 (np.array([6]), np.array([6]))
	])
def test_arraylike_copy(test_array, expected_output):
	#check array contents
	assert_array_equal(_arraylike_copy(test_array), expected_output)
	#check array type
	assert isinstance(_arraylike_copy(test_array), np.ndarray)
	#check deep copy for numpy array input
	assert not np.shares_memory(test_array, _arraylike_copy(test_array))

def test_bit_length():
	test_input = 10
	expected_bit_length = 4
	test_bit_length = bit_length(test_input)
	assert test_bit_length == expected_bit_length

def  


