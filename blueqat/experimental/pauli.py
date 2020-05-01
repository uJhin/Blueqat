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

"""The module for calculate Pauli matrices."""

from typing import NamedTuple, Tuple
from blueqat.pauli import Term, _PauliImpl
from math import pi

half_pi = pi * 0.5


def get_time_evolution(self):
        """Get the function to append the time evolution of this term.

        Returns:
            function(circuit: Circuit, t: float):
                Add gates for time evolution to `circuit` with time `t`
        """
        term = self.simplify()
        coeff = term.coeff
        if coeff.imag:
            raise ValueError("Not a real coefficient.")
        ops = term.ops
        return TimeEvolution(ops, coeff)


class TimeEvolution(NamedTuple):
    ops: Tuple[_PauliImpl]
    coeff: float

    def __call__(self, circuit, t):
        ops = self.ops
        coeff = self.coeff
        if not ops:
            return
        for op in ops:
            n = op.n
            if op.op == "X":
                circuit.h[n]
            elif op.op == "Y":
                circuit.rx(-half_pi)[n]
        for i in range(1, len(ops)):
            circuit.cx[ops[i-1].n, ops[i].n]
        circuit.rz(-2 * coeff * t)[ops[-1].n]
        for i in range(len(ops)-1, 0, -1):
            circuit.cx[ops[i-1].n, ops[i].n]
        for op in ops:
            n = op.n
            if op.op == "X":
                circuit.h[n]
            elif op.op == "Y":
                circuit.rx(half_pi)[n]
        return circuit

Term.get_time_evolution = get_time_evolution
