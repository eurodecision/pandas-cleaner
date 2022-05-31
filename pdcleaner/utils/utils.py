"""
Utilities
"""
import re

from inspect import signature
from typing import Callable

import pandas as pd


def add_method(cls, name):
    """
    This decorator adds the function as a method of the class cls
    under the name name

    Inspired by :
    https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
    """
    def decorator(func):
        setattr(cls, name, func)
        return func
    return decorator


def all_subclasses(cls):
    """retuns all children classes of a given class"""
    lst_subclasses = cls.__subclasses__()
    if len(lst_subclasses) > 0:
        return set(lst_subclasses).union(
            [subsubclass for subclass in lst_subclasses
                for subsubclass in all_subclasses(subclass)])
    return set()


def is_valid_detection_method_name(name):
    """check if a string is valid name for a detection method"""
    name_re = r"[a-z][a-z0-9_]*"
    return re.match(name_re, name)


def raise_if_not_in(value, legal_values, message):
    """raises a ValueError if the value is illegal

    Parameters:
    -----------
    value : tested value
    legal_values : list of allowed values
    message : message to raise in the ValueError
    """
    if value not in legal_values:
        raise ValueError(message)


def is_a_valid_dtype(type_: str) -> bool:
    """checks if a string is a valid python type"""
    try:
        pd.Series([], dtype='int').astype(type_)
        return True
    except TypeError:
        return False


def nb_of_args(function: Callable) -> int:
    """Get a function's number of arguments via its signature"""
    return len(signature(function).parameters)
