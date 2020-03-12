from functools import partial

from blueqat import Circuit, BlueqatGlobalSetting
from blueqat.pauli import term_from_chars
from blueqat.circuit import DEFAULT_BACKEND_NAME

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


def _expand_ops(ops, n_qubits):
    # TODO: Implement
    return ops

def _extend_circuit_macro():
    def wrap_init(f):
        def wrapper(self, *args, **kwargs):
            if 'macros' in kwargs:
                self.macros = kwargs['macros']
                del kwargs['macros']
            else:
                self.macros = {}
            f(self, *args, **kwargs)
        return wrapper
    Circuit.__init__ = wrap_init(Circuit.__init__)
    def wrap_getattr(f):
        def wrapper(self, name):
            if name in self.macros:
                return partial(self.macros[name], self)
            return f(self, name)
        return wrapper
    Circuit.__getattr__ = wrap_getattr(Circuit.__getattr__)
    def new_run(self, *args, backend=None, **kwargs):
        if backend is None:
            if self._default_backend is None:
                backend = self._Circuit__get_backend(DEFAULT_BACKEND_NAME)
            else:
                backend = self._Circuit__get_backend(self._default_backend)
        elif isinstance(backend, str):
            backend = self._Circuit_get_backend(backend)
        return backend.run(_expand_ops(self.ops, self.n_qubits), self.n_qubits, *args, **kwargs)
    Circuit.run = new_run

_extend_circuit_macro()
