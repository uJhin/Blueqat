'''QUBO maker, or QUBO mapper a.k.a. Quma.
'''
from typing import List, Mapping, Tuple
from numbers import Real
from functools import reduce
import operator
import blueqat.pauli

class Quma:
    def __init__(self, terms):
        self.terms = terms

    @staticmethod
    def num(val):
        return Quma({frozenset(): val})

    @staticmethod
    def var(n, coeff=1):
        return Quma({frozenset((n,)): coeff})

    @staticmethod
    def __add_terms_inplace(lhs, rhs):
        for k in rhs:
            if k in lhs:
                lhs[k] += rhs[k]
                if lhs[k] == 0:
                    del lhs[k]
            else:
                lhs[k] = rhs[k]
        return lhs

    @staticmethod
    def __neg_terms(terms):
        return {k: -v for k, v in terms.items()}

    @staticmethod
    def __mul_terms(lhs, rhs):
        d = {}
        for k1, v1 in lhs.items():
            for k2, v2 in rhs.items():
                coeff = v1 * v2
                syms = k1 | k2
                if syms in d:
                    d[syms] += coeff
                else:
                    d[syms] = coeff
        return {k: v for k, v in d.items() if v}

    def is_term(self) -> bool:
        return len(self.terms) == 1

    def is_const(self) -> bool:
        return len(self.terms) == 1 and frozenset() in self.terms

    def subs(self, key, value=None):
        if isinstance(value, (int, float, Quma)):
            if isinstance(key, int):
                key = frozenset((key,))
            elif isinstance(key, Quma):
                if not key.is_term():
                    raise ValueError("key shall be a term. (for example, q0 or q1*q2)")
                key, = key.terms
            if not isinstance(key, frozenset):
                raise ValueError("key shall be single term or integer.")

            d = {}
            for k, v in self.terms.items():
                if key <= k:
                    newkey = k - key
                    newval = v * value
                    if newkey in d:
                        d[newkey] += newval
                    else:
                        d[newkey] = newval
                else:
                    if k in d:
                        d[k] += v
                    else:
                        d[k] = v
            return Quma({k: v for k, v in d.items() if v})

        if hasattr(key, '__iter__') and value is None:
            expr = self
            if isinstance(key, dict):
                key = key.items()
            for k, v in key:
                expr = expr.subs(k, v)
            return expr

        raise ValueError("Quma.subs(Quma, val) or Quma.subs([(Quma, val)]) is supported")

    def __neg__(self):
        return Quma(Quma.__neg_terms(self.terms))

    def __add__(self, other):
        if isinstance(other, Quma):
            return Quma(Quma.__add_terms_inplace(self.terms.copy(), other.terms))
        if isinstance(other, (int, float)):
            return Quma(Quma.__add_terms_inplace(self.terms.copy(), Quma.num(other).terms))
        raise ValueError(f"`Quma` + `{other.__class__.__name__}` is not supported.")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Quma):
            return Quma(Quma.__mul_terms(self.terms, other.terms))
        if isinstance(other, (int, float)):
            if other:
                return Quma({k: v * other for k, v in self.terms.items()})
            return Quma.num(0)
        raise ValueError(f"`Quma` * `{other.__class__.__name__}` is not supported.")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1 / other)
        raise ValueError(f"`Quma` / `{other.__class__.__name__}` is not supported.")

    def __pow__(self, other):
        if isinstance(other, int):
            if not other:
                return Quma.num(1)
            ret = Quma(self.terms.copy())
            for i in range(other - 1):
                ret *= self
            return ret
        raise ValueError(f"`Quma` ** `{other.__class__.__name__}` is not supported.")

    def __repr__(self):
        return 'Quma(' + repr(self.terms) + ')'

    def __str__(self):
        def str_v(v, has_var):
            if v >= 0:
                if v == 1 and has_var:
                    return '+ '
                return f'+ {v} '
            if v == -1 and has_var:
                return '- '
            return f'- {-v} '

        if not self.terms:
            return '0'
        sterms = [str_v(v, k) + ' '.join(f'q{i}' for i in sorted(k))
                  for k, v in self.terms.items()]
        if sterms[0][0] == '+':
            sterms[0] = sterms[0][2:]
        else:
            sterms[0] = '-' + sterms[0][2:]
        return ' '.join(sterms)

    def pauli(self, simplify=True) -> blueqat.pauli.Expr:
        expr = sum(reduce(operator.mul, map(blueqat.pauli.qubo_bit, ind)) * val if ind else val
                   for ind, val in self.terms.items())
        if simplify:
            return expr.to_expr().simplify()
        return expr.to_expr()

    def is_quadratic(self) -> bool:
        return all(map(lambda k: len(k) <= 2, self.terms))

    def qubo_matrix(self) -> List[List[Real]]:
        n = self.max_n() + 1
        mat = [[0.] * n for _ in range(n)]
        for k, v in self.terms.items():
            if not k:
                continue
            elif len(k) == 1:
                i, = k
                mat[i][i] += v
            elif len(k) == 2:
                i, j = k
                if i > j:
                    i, j = j, i
                mat[i][j] += v
            else:
                raise ValueError('Not a quadratic matrix.')
        return mat

    def max_n(self):
        return max(max(term) for term in self.terms)

    def qubits(self):
        return reduce(operator.or_, self.terms, set())

    def reduce_unused_vars(self) -> Tuple['Quma', Mapping[int, int]]:
        mapping = {q: i  for i, q in enumerate(sorted(self.qubits())) if i != q}
        expr = self.subs(mapping)
        return expr, mapping

def parse(s: str) -> Quma:
    pass

for i in range(500):
    exec(''.join(f'q{i}=Quma.var({i});'))

class _Qubits:
    def __getitem__(self, item):
        if isinstance(item, slice):
            return (Quma.var(i) for i in _slice_generator(item))
        if isinstance(item, tuple):
            return [Quma.var(i) for i in item]
        return Quma.var(item.__index__())

    def __call__(self, item):
        return self[item]

qubits = _Qubits()

def _slice_generator(item):
    def get_index(v):
        try:
            return v.__index__()
        except AttributeError:
            raise TypeError('Indices must be integers or slice, not', v.__class__.__name__)

    if item.step is None:
        step = 1
    else:
        step = get_index(item.step)
        if step == 0:
            raise ValueError('Step must not be 0')

    if step > 0:
        i = 0 if item.start is None else max(0, get_index(item.start))
        if item.stop is not None:
            last = get_index(item.stop)
            while i < last:
                yield i
                i += step
        else:
            while 1:
                yield i
                i += step
    else:
        if item.start is None:
            raise ValueError('Start index is required when step < 0')
        last = -1 if item.stop is None else max(-1, get_index(item.stop))
        while i > last:
            yield i
            i += step
