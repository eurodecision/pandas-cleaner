"""Utilities for report() display/print"""

import textwrap


def print_line(cpt: int = 0, symbol: str = '=', nb: int = 78) -> None:
    """Prints a new line of nb characters symbol. If cpt is odd, start a new line."""

    if (cpt % 2) != 0:
        print("\n" + symbol * nb)
    else:
        print(symbol * nb)
    return None


def print_name_value(name: str = '', value: str = '', cpt: int = 0) -> None:
    """Print name, value pairs in a 18/18 2-columns format"""

    if (cpt % 2) == 0:
        print(f"{name: <18}{value: >18}"+" "*6, end='')
    else:
        print(f"{name: <18}{value: >18}")

    return None


def print_fixed_width(text: str = '', width: int = 78) -> None:
    """Prints a string on multiple lines width a max width"""

    print('\n'.join(textwrap.wrap(text, width)))
    return None
