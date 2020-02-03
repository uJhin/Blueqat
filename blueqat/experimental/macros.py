from blueqat import BlueqatGlobalSetting
from blueqat.pauli import term_from_chars

from .utils import def_macro, targetable
from .operations import GLOBAL_MACROS


@def_macro
def evo(c, pauli, t):
    if isinstance(pauli, str):
        pauli = term_from_chars(pauli)
    else:
        pauli = pauli.to_term()
    pauli.get_time_evolution()(c, t)
    return c


@def_macro
@targetable
def iqft(c, target):
    from . import Ops
    from math import pi
    dummy = Ops().i[target].ops[0]
    n_qubits = c.n_qubits
    if hasattr(target, '__index__'):
        n_qubits = max(n_qubits, target.__index__())
    if isinstance(target, slice) and target.stop is not None:
        n_qubits = max(n_qubits, target.stop)
    if isinstance(target, tuple):
        for t in target:
            if hasattr(t, '__index__'):
                n_qubits = max(n_qubits, t.__index__())
            if isinstance(t, slice) and t.stop is not None:
                n_qubits = max(n_qubits, t.stop)
    target = tuple(dummy.target_iter(n_qubits))
    n_target = len(target)
    for i in range(n_target):
        angle = -0.5 * pi
        for j in range(i + 1, n_target):
            c.cphase(angle)[target[j], target[i]]
            angle *= 0.5
        c.h[target[i]]
    return c

@def_macro
def macros(_):
    print(GLOBAL_MACROS)
