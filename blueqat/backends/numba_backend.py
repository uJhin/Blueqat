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

from collections import Counter
import cmath
import math
import random
import warnings

import numpy as np
from numba import jit, njit, prange
import numba

from ..gate import *
from .backendbase import Backend

# TODO: Use this
DEFAULT_DTYPE = np.complex128

# Typedef
# Index of Qubit
_QBIdx = numba.uint32
# Mask of Quantum State
_QSMask = numba.uint64
# Index of Quantum State
_QSIdx = _QSMask

class _NumbaBackendContext:
    """This class is internally used in NumbaBackend"""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        self.save_cache = True
        self.shots_result = Counter()
        self.cregs = None

    def prepare(self, cache):
        """Prepare to run next shot."""
        if cache is not None:
            np.copyto(self.qubits, cache)
        else:
            self.qubits.fill(0.0)
            self.qubits[0] = 1.0
        self.cregs = [0] * self.n_qubits

    def store_shot(self):
        """Store current cregs to shots_result"""
        def to_str(cregs):
            return ''.join(str(b) for b in cregs)
        key = to_str(self.cregs)
        self.shots_result[key] = self.shots_result.get(key, 0) + 1


@njit(_QSIdx(_QSMask, _QBIdx),
      locals={'lower': _QSMask, 'higher': _QSMask},
      cache=True)
def _shifted(lower_mask, idx):
    lower = idx & lower_mask
    higher = (idx & ~lower_mask) << 1
    return higher + lower


# Thanks to Morino-san!
@njit(_QBIdx[:](_QBIdx[:], _QSIdx),
        locals={'in_pos': _QBIdx, 'i': _QBIdx, 'shift_map': _QBIdx[:]},
      cache=True)
def _create_bit_shift_map(poslist, n_qubits):
    shift_map = np.zeros(n_qubits, dtype=np.uint32)
    for i, b in enumerate(sorted(poslist)):
        in_pos = b - i
        if in_pos < n_qubits:
            shift_map[in_pos] += 1
    for i in range(len(poslist) - 1):
        shift_map[i + 1] += shift_map[i]
    for i in range(n_qubits):
        shift_map[i] += i
    return shift_map


@njit(_QSMask(_QBIdx[:]),
      cache=True)
def _create_mask_from_shift_map(shift_map):
    mask = 0
    for m in shift_map:
        mask |= 1 << m
    return mask


@njit(_QBIdx(_QSIdx, _QBIdx[:]),
      cache=True)
def _create_idx_from_shift_map(noshiftidx, shift_map):
    idx = 0
    for m in shift_map:
        if (1 << m) & noshiftidx:
            idx |= 1 << m
    return idx


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _zgate(qubits, n_qubits, target):
    lower_mask = (1 << target) - 1
    for i in prange(1 << (n_qubits - 1)):
        qubits[_shifted(lower_mask, i)] *= -1


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _xgate(qubits, n_qubits, target):
    lower_mask = (1 << target) - 1
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        qubits[i0] = qubits[i0 + (1 << target)]
        qubits[i0 + (1 << target)] = t


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _ygate(qubits, n_qubits, target):
    lower_mask = (1 << target) - 1
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        # Global phase is ignored.
        qubits[i0] = -qubits[i0 + (1 << target)]
        qubits[i0 + (1 << target)] = t


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _hgate(qubits, n_qubits, target):
    sqrt2_inv = 0.7071067811865475
    lower_mask = (1 << target) - 1
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = (t + u) * sqrt2_inv
        qubits[i0 + (1 << target)] = (t - u) * sqrt2_inv


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx, numba.complex128),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _diaggate(qubits, n_qubits, target, factor):
    lower_mask = (1 << target) - 1
    for i in prange(1 << (n_qubits - 1)):
        i1 = _shifted(lower_mask, i) + (1 << target)
        # Global phase is ignored.
        qubits[i1] *= factor


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx, numba.float64),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _rygate(qubits, n_qubits, target, ang):
    lower_mask = (1 << target) - 1
    ang *= 0.5
    cos = math.cos(ang)
    sin = math.sin(ang)
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = cos * t - sin * u
        qubits[i0 + (1 << target)] = sin * t + cos * u


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx, numba.float64),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _rxgate(qubits, n_qubits, target, ang):
    lower_mask = (1 << target) - 1
    ang *= 0.5
    cos = math.cos(ang)
    nisin = math.sin(ang) * -1.j
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = cos * t + nisin * u
        qubits[i0 + (1 << target)] = nisin * t + cos * u


@njit(numba.void(numba.complex128[:], _QBIdx, _QBIdx,
                 numba.float64, numba.float64, numba.float64),
      locals={'lower_mask': _QSMask},
      parallel=True)
def _u3gate(qubits, n_qubits, target, theta, phi, lambd):
    lower_mask = (1 << target) - 1
    theta *= 0.5
    cos = math.cos(theta)
    sin = math.sin(theta)
    expadd = cmath.exp((phi + lambd) * 0.5j)
    expsub = cmath.exp((phi - lambd) * 0.5j)
    a = expadd.conjugate() * cos
    b = -expsub.conjugate() * sin
    c = expsub * sin
    d = expadd * cos
    for i in prange(1 << (n_qubits - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = a * t + c * u
        qubits[i0 + (1 << target)] = b * t + d * u


@njit(numba.void(numba.complex128[:], _QBIdx[:], _QBIdx),
      parallel=True)
def _czgate(qubits, controls_target, n_qubits):
    target = controls_target[-1]
    shift_map = _create_bit_shift_map(controls_target, n_qubits)
    print(shift_map)
    all1 = 0
    for i in controls_target:
        all1 |= 1 << i
    for i in range(1 << (n_qubits - len(controls_target))):
        i11 = _create_idx_from_shift_map(i, shift_map) | all1
        qubits[i11] *= -1


@njit(numba.void(numba.complex128[:], _QBIdx[:], _QBIdx),
      parallel=True)
def _cxgate(qubits, controls_target, n_qubits):
    control = controls_target[0]
    target = controls_target[-1]
    shift_map = _create_bit_shift_map(controls_target, n_qubits)
    for i in range(1 << (n_qubits - len(controls_target))):
        i10 = _create_idx_from_shift_map(i, shift_map) | (1 << control)
        i11 = i10 | (1 << target)
        t = qubits[i10]
        qubits[i10] = qubits[i11]
        qubits[i11] = t


class NumbaBackend(Backend):
    """Simulator backend which uses numba."""
    __return_type = {
        "statevector": lambda ctx: _ignore_globals(ctx.qubits),
        "shots": lambda ctx: ctx.shots_result,
        "statevector_and_shots": lambda ctx: (_ignore_globals(ctx.qubits), ctx.shots_result),
        "_inner_ctx": lambda ctx: ctx,
    }
    DEFAULT_SHOTS = 1024

    def __init__(self):
        self.cache = None
        self.cache_idx = -1

    def __clear_cache(self):
        self.cache = None
        self.cache_idx = -1

    def __clear_cache_if_invalid(self, n_qubits, dtype):
        if self.cache is None:
            self.__clear_cache()
            return
        if len(self.cache) != 2**n_qubits:
            self.__clear_cache()
            return
        if self.cache.dtype != dtype:
            self.__clear_cache()
            return

    def run(self, gates, n_qubits, *args, **kwargs):
        def __parse_run_args(shots=None, returns=None, **_kwargs):
            if returns is None:
                if shots is None:
                    returns = "statevector"
                else:
                    returns = "shots"
            if returns not in self.__return_type.keys():
                raise ValueError(f"Unknown returns type '{returns}'")
            if shots is None:
                if returns in ("statevector", "_inner_ctx"):
                    shots = 1
                else:
                    shots = self.DEFAULT_SHOTS
            if returns == "statevector" and shots > 1:
                warnings.warn("When `returns` = 'statevector', `shots` = 1 is enough.")
            return shots, returns

        shots, returns = __parse_run_args(*args, **kwargs)

        self.__clear_cache_if_invalid(n_qubits, DEFAULT_DTYPE)
        ctx = _NumbaBackendContext(n_qubits)

        def run_single_gate(gate):
            nonlocal ctx
            action = self._get_action(gate)
            if action is not None:
                ctx = action(gate, ctx)
            else:
                for g in gate.fallback(n_qubits):
                    run_single_gate(g)

        for _ in range(shots):
            ctx.prepare(self.cache)
            for gate in gates[self.cache_idx + 1:]:
                run_single_gate(gate)
                if ctx.save_cache:
                    self.cache = ctx.qubits.copy()
                    self.cache_idx += 1
            if ctx.cregs:
                ctx.store_shot()

        return self.__return_type[returns](ctx)

    def make_cache(self, gates, n_qubits):
        self.run(gates, n_qubits)

    def gate_x(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _xgate(qubits, n_qubits, target)
        return ctx

    def gate_y(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _ygate(qubits, n_qubits, target)
        return ctx

    def gate_z(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _zgate(qubits, n_qubits, target)
        return ctx

    def gate_h(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _hgate(qubits, n_qubits, target)
        return ctx

    def gate_cz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _czgate(qubits, np.array([control, target], dtype=np.uint32), n_qubits)
        return ctx

    def gate_cx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _cxgate(qubits, np.array([control, target], dtype=np.uint32), n_qubits)
        return ctx

    def gate_rx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rxgate(qubits, n_qubits, target, theta)
        return ctx

    def gate_ry(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rygate(qubits, n_qubits, target, theta)
        return ctx

    def gate_rz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(1.j * gate.theta)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_t(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_tdg(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(-0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_s(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = 1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_sdg(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = -1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_ccz(self, gate, ctx):
        c1, c2, t = gate.targets
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        indices = (i & (1 << c1)) != 0
        indices &= (i & (1 << c2)) != 0
        indices &= (i & (1 << t)) != 0
        qubits[indices] *= -1
        return ctx

    def gate_ccx(self, gate, ctx):
        c1, c2, t = gate.targets
        ctx = self.gate_h(HGate(t), ctx)
        ctx = self.gate_ccz(CCZGate(gate.targets), ctx)
        ctx = self.gate_h(HGate(t), ctx)
        return ctx

    def gate_u1(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(1.j * gate.lambd)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_u3(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        phi = gate.phi
        lambd = gate.lambd
        for target in gate.target_iter(n_qubits):
            _u3gate(qubits, n_qubits, target, theta, phi, lambd)
        return ctx

    def gate_measure(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            p_zero = np.linalg.norm(qubits[(i & (1 << target)) == 0]) ** 2
            rand = random.random()
            if rand < p_zero:
                qubits[(i & (1 << target)) != 0] = 0.0
                qubits /= np.sqrt(p_zero)
                ctx.cregs[target] = 0
            else:
                qubits[(i & (1 << target)) == 0] = 0.0
                qubits /= np.sqrt(1.0 - p_zero)
                ctx.cregs[target] = 1
        ctx.save_cache = False
        return ctx


# TODO: Parallel
@njit
def _ignore_globals(qubits):
    for q in qubits:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            break
    return qubits
