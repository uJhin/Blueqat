# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keyword import iskeyword
from functools import wraps, update_wrapper
from typing import Tuple, Dict, Any, Callable, Union, Optional
from dataclasses import dataclass, field

import numpy as np

from blueqat import Circuit, BlueqatGlobalSetting
from . import Ops
from .operations import MacroWrapper


def circuit_to_unitary(circ: Circuit, *runargs, **runkwargs):
    runkwargs.setdefault('returns', 'statevector')
    runkwargs.setdefault('ignore_global', False)
    n_qubits = circ.n_qubits
    vecs = []
    if n_qubits == 0:
        return np.array([[1]])
    for i in range(1 << n_qubits):
        bitmask = tuple(k for k in range(n_qubits) if (1 << k) & i)
        c = Circuit()
        if bitmask:
            c.x[bitmask]
        c += circ
        vecs.append(c.run(*runargs, **runkwargs))
    return np.array(vecs).T


def def_macro(func: Optional[Union[Callable[[Any], Any], str]] = None, *, allow_overwrite: bool = False):
    """@def_macro decorator.

    Typical usage:

    Case 1: no arguments

        @def_macro
        def ham(c):
            ...

    equivalent to this:

        def ham(c):
            ...
        BlueqatGlobalSetting.register_macro('ham', ham)


    Case 2: with name:

        @def_macro('egg')
        def ham(c):
            ...

    is equivalent with

        def ham(c):
            ...
        BlueqatGlobalSetting.register_macro('egg', ham)

    Case 3: with allow_overwrite keyword argument

        @def_macro(allow_overwrite=True)
        def ham(c):
            ...

    or

        @def_macro('egg', allow_overwrite=True)
        def ham(c):
            ...

    call BlueqatGlobalSetting.register_macro with allow_overwrite=True.
    """
    if callable(func):
        # @def_macro pattern.
        name = func.__name__
        if not name.isidentifier() or iskeyword(name):
            raise ValueError(f'Function name {name} is not a valid macro name. ')
        BlueqatGlobalSetting.register_macro(name, func)
        return func
    if isinstance(func, str):
        # @def_macro(name) or @def_macro(name, allow_overwrite) pattern.
        name = func
        def _wrapper(func):
            BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
            return func
        return _wrapper
    if func is None:
        # @def_macro(allow_overwrite) pattern.
        def _wrapper(func):
            name = func.__name__
            if not name.isidentifier() or iskeyword(name):
                raise ValueError(f'Function name {name} is not a valid macro name. ')
            BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
            return func
        return _wrapper
    raise TypeError('Invalid type for first argument.')



def targetable(func, call='never'):
    """Decorator to create gate-like class

    call: 'never', 'must', 'optional'
    """
    @dataclass
    class Inner:
        circuit: Union[Circuit, Ops]
        args: Tuple[Any] = ()
        kwargs: Dict[str, Any] = field(default_factory=dict)
        called: bool = False

        def __call__(self, *args, **kwargs):
            if call == 'never':
                raise AttributeError(func.__name__, 'is not callable.')
            if self.called:
                raise ValueError('Already called.')
            self.args = args
            self.kwargs = kwargs
            self.called = True
            return self

        def __getitem__(self, item):
            if self.circuit is None:
                raise ValueError('No circuit.')
            if call == 'must' and not self.called:
                raise ValueError('Not called.')
            return func(self.circuit, item, *self.args, **self.kwargs)

    class Wrapper(MacroWrapper):
        def __call__(self, *args, **kwargs):
            return Inner(*args, **kwargs)

    w = Wrapper()
    update_wrapper(w, func)
    return w
