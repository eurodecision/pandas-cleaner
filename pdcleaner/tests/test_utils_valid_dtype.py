from pdcleaner.utils.utils import is_a_valid_dtype


def test_is_a_valid_dtype():
    assert is_a_valid_dtype('int')
    assert is_a_valid_dtype('float64')
    assert is_a_valid_dtype('object')
    assert is_a_valid_dtype('category')
    assert is_a_valid_dtype('datetime64[ns]')


def test_is_a_valid_dtype_fail():
    assert not is_a_valid_dtype('toto')
