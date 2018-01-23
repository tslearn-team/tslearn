import numpy as np
from numpy.testing import assert_array_equal
import pytest
from tslearn.utils import _arraylike_copy

@pytest.mark.parametrize(
	"test_array, expected_output",
	[([5], np.array([5])),
	 (np.array([6]), np.array([6]))
	])
def test_arraylike_copy(test_array, expected_output):
	assert_array_equal(test_array, expected_output)
	#check deep copy for numpy array input
	assert not np.may_share_memory(test_array, expected_output)



