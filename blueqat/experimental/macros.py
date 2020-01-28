import re

from blueqat import BlueqatGlobalSetting
from blueqat.pauli import term_from_chars

def is_symbol(s):
    return re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', s)


def def_macro(func, name: str = None, allow_overwrite: bool = False):
    if name is None:
        if isinstance(func, str):
            name = func
            def _wrapper(func):
                BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
                return func
            return _wrapper
        if callable(func):
            # Direct call or decorator
            name = func.__name__
            if not is_symbol(name):
                raise ValueError('Invalid function name')
            BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
            return func
    if isinstance(name, str) and callable(func):
        BlueqatGlobalSetting(name, func)
        return func
    raise ValueError('Invalid argument')


@def_macro
def evo(c, pauli, t):
    if isinstance(pauli, str):
        pauli = term_from_chars(pauli)
    else:
        pauli = pauli.to_term()
    pauli.get_time_evolution()(c, t)
    return c
