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

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, Tuple, Iterable
import itertools
import random
import warnings

import numpy as np
from scipy.optimize import minimize as scipy_minimizer
from ..circuit import Circuit
from ..utils import to_inttuple

@dataclass
class Objective:
    ansatz: "AnsatzBase"
    sampler: Callable[[Circuit, Iterable], Dict[Tuple[int], float]]
    def __call__(self, params):
        circuit = self.ansatz.get_circuit(params)
        circuit.make_cache()
        return self.ansatz.get_energy(circuit, self.sampler)


@dataclass
class VerboseObjective:
    objective: Objective
    def __call__(self, params):
        val = self.objective(params)
        print("params:", params, "val:", val)
        return val


class AnsatzBase:
    def __init__(self, hamiltonian, n_params):
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        self.n_qubits = self.hamiltonian.max_n() + 1

    def get_circuit(self, params):
        """Make a circuit from parameters."""
        raise NotImplementedError

    def get_energy(self, circuit, sampler):
        """Calculate energy from circuit and sampler."""
        val = 0.0
        for meas in self.hamiltonian:
            c = circuit.copy()
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(-np.pi / 2)[op.n]
            measured = sampler(c, meas.n_iter())
            for bits, prob in measured.items():
                if sum(bits) % 2:
                    val -= prob * meas.coeff
                else:
                    val += prob * meas.coeff
        return val.real

    def get_objective(self, sampler):
        """Get an objective function to be optimized."""
        return Objective(self, sampler)


class HamiltonianGroupingAnsatz(AnsatzBase):
    def __init__(self, hamiltonian, n_params, grouping_strategy=None):
        super().__init__(hamiltonian, n_params)
        if grouping_strategy is None:
            grouping_strategy = grouping_hamiltonian
        self.grouped_hamiltonian = grouping_strategy(hamiltonian)
        self.x_list = []
        self.y_list = []
        self.m_list = []
        n_qubits = self.n_qubits
        for grp in self.grouped_hamiltonian:
            meas = ['.'] * n_qubits
            for term in grp:
                for op in term.ops:
                    if op.op == 'I':
                        continue
                    assert meas[op.n] == '.' or meas[op.n] == op.op
                    meas[op.n] = op.op
            self.x_list.append(tuple(n for n, m in enumerate(meas) if m == 'X'))
            self.y_list.append(tuple(n for n, m in enumerate(meas) if m == 'Y'))
            self.m_list.append(tuple(n for n, m in enumerate(meas) if m != '.'))

        def term_mask(m_list, n_iter):
            n = tuple(n_iter)
            return tuple(int(m in n) for m in m_list)

        self.grps = [[(term.coeff, term_mask(m_list, term.n_iter())) for term in grp]
                     for m_list, grp in zip(self.m_list, self.grouped_hamiltonian)]
#        print(f'''{hamiltonian=}
#{self.grouped_hamiltonian=}
#{self.x_list=}
#{self.y_list=}
#{self.m_list=}
#{self.grps=}''')


    def get_energy(self, circuit, sampler):
        """Calculate energy from circuit and sampler."""
        tuplemask = lambda t1, t2: tuple(e1 & e2 for e1, e2 in zip(t1, t2))
        val = 0.0
        for xs, ys, ms, grp in zip(self.x_list, self.y_list, self.m_list, self.grps):
            c = circuit.copy()
            c.h[xs].rx(-np.pi * 0.5)[ys]
            measured = sampler(c, ms)
            #print('measured:', measured)
            for coeff, mask in grp:
                #print('mask:', mask)
                for bits, prob in measured.items():
                    #print('bits:', bits)
                    #print('---->', tuplemask(mask, bits))
                    if sum(tuplemask(mask, bits)) % 2:
                        val -= prob * coeff
                    else:
                        val += prob * coeff
        return val.real


class QaoaAnsatz(HamiltonianGroupingAnsatz):
    def __init__(self, hamiltonian, step=1, init_circuit=None):
        super().__init__(hamiltonian, step * 2)
        self.hamiltonian = hamiltonian.to_expr().simplify()
        if not self.check_hamiltonian():
            raise ValueError("Hamiltonian terms are not commutable")

        self.step = step
        self.n_qubits = self.hamiltonian.max_n() + 1
        if init_circuit:
            self.init_circuit = init_circuit
            if init_circuit.n_qubits > self.n_qubits:
                self.n_qubits = init_circuit.n_qubits
        else:
            self.init_circuit = Circuit(self.n_qubits).h[:]
        self.init_circuit.make_cache()
        self.time_evolutions = [term.get_time_evolution() for term in self.hamiltonian]

    def check_hamiltonian(self):
        """Check hamiltonian is commutable. This condition is required for QaoaAnsatz"""
        return self.hamiltonian.is_all_terms_commutable()

    def get_circuit(self, params):
        c = self.init_circuit.copy()
        betas = params[:self.step]
        gammas = params[self.step:]
        for beta, gamma in zip(betas, gammas):
            beta *= np.pi
            gamma *= 2 * np.pi
            for evo in self.time_evolutions:
                evo(c, gamma)
            c.rx(beta)[:]
        return c

class VqeResult:
    def __init__(self, vqe=None, params=None, circuit=None):
        self.vqe = vqe
        self.params = params
        self.circuit = circuit
        self._probs = None

    def most_common(self, n=1):
        return tuple(sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    @property
    def probs(self):
        """Get probabilities. This property is obsoleted. Use get_probs()."""
        warnings.warn("VqeResult.probs is obsoleted. " +
                      "Use VqeResult.get_probs().", DeprecationWarning)
        return self.get_probs()

    def get_probs(self, sampler=None, rerun=None, store=True):
        """Get probabilities."""
        if rerun is None:
            rerun = sampler is not None
        if self._probs is not None and not rerun:
            return self._probs
        if sampler is None:
            sampler = self.vqe.sampler
        probs = sampler(self.circuit, range(self.circuit.n_qubits))
        if store:
            self._probs = probs
        return probs


class Vqe:
    def __init__(self, ansatz, minimizer=None, sampler=None):
        self.ansatz = ansatz
        self.minimizer = minimizer or get_scipy_minimizer(
            method="Powell",
            options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000}
        )
        self.sampler = sampler or non_sampling_sampler
        self._result = None

    def run(self, verbose=False):
        objective = self.ansatz.get_objective(self.sampler)
        if verbose:
            objective = VerboseObjective(objective)
        params = self.minimizer(objective, self.ansatz.n_params)
        c = self.ansatz.get_circuit(params)
        return VqeResult(self, params, c)

    @property
    def result(self):
        """Vqe.result is deprecated. Use `result = Vqe.run()`."""
        warnings.warn("Vqe.result is deprecated. Use `result = Vqe.run()`",
                      DeprecationWarning)
        return self._result if self._result is not None else VqeResult()

def get_scipy_minimizer(**kwargs):
    """Get minimizer which uses `scipy.optimize.minimize`"""
    def minimizer(objective, n_params):
        params = [random.random() for _ in range(n_params)]
        result = scipy_minimizer(objective, params, **kwargs)
        return result.x
    return minimizer

def expect(qubits, meas):
    "For the VQE simulation without sampling."
    result = {}
    i = np.arange(len(qubits))
    meas = tuple(meas)

    def to_mask(n):
        return reduce(lambda acc, im: acc | (n & (1 << im[0])) << (im[1] - im[0]), enumerate(meas), 0)

    def to_key(k):
        return tuple(1 if k & (1 << i) else 0 for i in meas)

    mask = reduce(lambda acc, v: acc | (1 << v), meas, 0)

    cnt = defaultdict(float)
    probs = qubits.real ** 2
    probs += qubits.imag ** 2
    for i, p in enumerate(probs):
        if p != 0.0:
            cnt[i & mask] += p
    return {to_key(k): v for k, v in cnt.items()}


# TODO: Shall be change sampler specifications for grouping Hamiltonian.

def statevector_sampler(circuit, meas, *args, **kwargs):
    """Calculate the expectations without sampling."""
    kwargs.setdefault('returns', 'statevector')
    meas = tuple(meas)
    n_qubits = circuit.n_qubits
    return expect(circuit.run(*args, **kwargs), meas)

non_sampling_sampler = statevector_sampler


def get_measurement_sampler(n_sample):
    """Returns a function which get the expectations by sampling the measured circuit"""
    def sampling_by_measurement(circuit, meas, *args, **kwargs):
        def reduce_bits(bits, meas):
            bits = [int(x) for x in bits[::-1]]
            return tuple(bits[m] for m in meas)

        kwargs.setdefault('returns', 'shots')
        kwargs.setdefault('shots', n_sample)
        meas = tuple(meas)
        circuit.measure[meas]
        counter = circuit.run(*args, **kwargs)
        counts = Counter({reduce_bits(bits, meas): val for bits, val in counter.items()})
        return {k: v / n_sample for k, v in counts.items()}

    return sampling_by_measurement


def get_statevector_counting_sampler(n_sample):
    """Returns a function which get the expectations by sampling the state vector"""
    def sampling_by_measurement(circuit, meas, *args, **kwargs):
        val = 0.0
        kwargs.setdefault('returns', 'statevector')
        e = expect(circuit.run(*args, **kwargs), meas)
        bits, probs = zip(*e.items())
        dists = np.random.multinomial(n_sample, probs) / n_sample
        return dict(zip(tuple(bits), dists))
    return sampling_by_measurement

get_state_vector_sampler = get_statevector_counting_sampler

# TODO: This function shall be moved to pauli lib. and shall be renamed.
def trim_small_term(hamiltonian, small=1e-6):
    cls = hamiltonian.__class__
    return cls(tuple(term for term in hamiltonian if abs(term.coeff) > small))

def grouping_hamiltonian(hamiltonian):
    hamiltonian = trim_small_term(hamiltonian.to_expr())
    groups = []
    for i_term, term in enumerate(hamiltonian):
        for i_grp, g in enumerate(reversed(groups)):
            if all(map(term.is_commutable_with, g)):
                #print(f'{i_term}\t{len(groups) - 1 - i_grp} appended')
                g.append(term)
                break
        else:
            groups.append([term])
            #print(f'{i_term}\tnew group')
    return groups
