import pytest
from pdcleaner.utils.utils import raise_if_not_in

def test_raise_if_not_in():
    legal_values = [0, 1]
    message = 'error'
    with pytest.raises(ValueError, match="error"):
        raise_if_not_in(-1, legal_values, message)