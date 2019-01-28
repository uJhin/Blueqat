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

from abc import abstractmethod
from enum import Enum, auto
import re
import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Set, TextIO, Tuple, Union, NoReturn, Match
from itertools import takewhile


_restr_symbol = r'[a-zA-Z_][a-zA-Z0-9_]*'
_restr_quoted_str = r'"[^"]*"'
_restr_uint = r'[1-9][0-9]*|0'
_restr_float = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
_restr_funcs = r'sin|cos|tan|exp|ln|sqrt'

_regex_tokens = re.compile(r'OPENQASM 2.0|->|==|' +
        '|'.join((
            _restr_symbol,
            _restr_quoted_str,
            _restr_float,
        )) +
        r'|//|\S')


def split_tokens(qasmstr: str) -> Iterable[Tuple[int, str]]:
    for i, line in enumerate(qasmstr.split('\n')):
        toks = _regex_tokens.findall(line)
        yield from takewhile(
            lambda x: not x[1].startswith('//'),
            ((i, tok) for tok in toks))


def _err_with_lineno(lineno: int, msg: str) -> NoReturn:
    raise ValueError(f"Line {lineno}: {msg}")


class TokenGetter:
    def __init__(self, it: Iterable[Tuple[int, str]]) -> None:
        self.it = it
        self.buf = []
        self.lineno = 1

    def get(self) -> Tuple[int, str]:
        if self.buf:
            tok = self.buf.pop()
        else:
            try:
                tok = next(self.it)
            except StopIteration:
                return self.lineno, None
        self.lineno = tok[0]
        return tok

    def unget(self, tok: Tuple[int, str]) -> None:
        self.buf.append(tok)

    def _fail(self, action: Any) -> Any:
        if action is None:
            return None
        elif isinstance(action, str):
            _err_with_lineno(self.lineno, action)

    def get_if(self, cond: Any, or_else: Any = None) -> Union[Tuple[int, str], None]:
        tok = self.get()
        if tok is None:
            return self._fail(or_else)
        if isinstance(cond, str):
            if tok[1] == cond:
                return tok
            self.unget(tok)
            return self._fail(or_else)
        if hasattr(cond, '__call__'):
            if cond(tok[1]):
                return tok
            self.unget(tok)
            return self._fail(or_else)
        raise ValueError('Unknown conditions')

    def assert_semicolon(self, msg: str = '";" is expected.') -> None:
        self.get_if(';', msg)


def parse_qasm(qasmstr: str) -> 'QasmProgram':
    tokens = TokenGetter(split_tokens(qasmstr))
    errmsg = 'Program shall be start with "OPENQASM 2.0;".'
    tokens.get_if('OPENQASM 2.0', errmsg)
    tokens.assert_semicolon()
    stmts = []
    gates = {}
    qregs = {}
    cregs = {}
    included = set()
    _parse_statements(tokens, stmts, gates, qregs, cregs, included)
    return QasmProgram(stmts, gates, qregs, cregs, included)


def parse_qasmf(qasmfile: Union[str, TextIO], *args, **kwargs) -> 'QasmProgram':
    if isinstance(qasmfile, str):
        with open(qasmfile) as f:
            return parse_qasmf(f, *args, **kwargs)
    return parse_qasm(qasmfile.read(), *args, **kwargs)


class QasmNode:
    pass


class QasmProgram(QasmNode):
    def __init__(self,
                 statements: List[Any],
                 qregs: Dict[str, int],
                 cregs: Dict[str, int],
                 gates: Dict[str, Any],
                 included: Set[str]) -> None:
        self.statements = statements
        self.qregs = qregs
        self.cregs = cregs
        self.gates = gates
        self.included = included


    def __repr__(self) -> str:
        return f'QasmProgram({repr(self.statements)}, ' + \
               f'{repr(self.qregs)}, {repr(self.cregs)}, ' + \
               f'{repr(self.gates)}, {repr(self.included)})'


class QasmExpr(QasmNode):
    @abstractmethod
    def eval(self):
        pass

    def __repr__(self):
        if hasattr(self, 'value'):
            return f'{self.__class__.__name__}({repr(self.value)})'
        return f'{self.__class__.__name__}({repr(self.lhs)}, {repr(self.rhs)})'


class QasmRealExpr(QasmExpr):
    @abstractmethod
    def eval(self) -> float:
        pass


class QasmRealAdd(QasmRealExpr):
    def __init__(self, lhs: QasmExpr, rhs: QasmExpr):
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        return self.lhs.eval() + self.rhs.eval()


class QasmRealSub(QasmRealExpr):
    def __init__(self, lhs: QasmExpr, rhs: QasmExpr):
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        return self.lhs.eval() - self.rhs.eval()


class QasmRealMul(QasmRealExpr):
    def __init__(self, lhs: QasmExpr, rhs: QasmExpr):
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        return self.lhs.eval() * self.rhs.eval()


class QasmRealDiv(QasmRealExpr):
    def __init__(self, lhs: QasmExpr, rhs: QasmExpr):
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        return self.lhs.eval() / self.rhs.eval()


class QasmRealValue(QasmRealExpr):
    def __init__(self, value: float):
        self.value = value

    def eval(self):
        return self.value


class QasmRealUnaryFunctions(Enum):
    Sin = auto()
    Cos = auto()
    Tan = auto()
    Exp = auto()
    Ln = auto()
    Sqrt = auto()

    @staticmethod
    def from_str(s: str) -> 'QasmRealUnaryFunctions':
        if s == 'sin':
            return QasmRealUnaryFunctions.Sin
        if s == 'cos':
            return QasmRealUnaryFunctions.Cos
        if s == 'tan':
            return QasmRealUnaryFunctions.Tan
        if s == 'exp':
            return QasmRealUnaryFunctions.Exp
        if s == 'ln':
            return QasmRealUnaryFunctions.Ln
        if s == 'sqrt':
            return QasmRealUnaryFunctions.Sqrt
        raise ValueError('Unexpected value')

class QasmRealCall(QasmRealExpr):
    def __init__(self, func: QasmRealUnaryFunctions, arg: QasmRealExpr):
        self.func = func
        self.arg = arg

    def eval(self):
        func = None
        if self.func == QasmRealUnaryFunctions.Sin:
            func = math.sin
        elif self.func == QasmRealUnaryFunctions.Cos:
            func = math.cos
        elif self.func == QasmRealUnaryFunctions.Tan:
            func = math.tan
        elif self.func == QasmRealUnaryFunctions.Exp:
            func = math.exp
        elif self.func == QasmRealUnaryFunctions.Ln:
            func = math.log
        elif self.func == QasmRealUnaryFunctions.Sqrt:
            func = math.sqrt
        else:
            raise ValueError('Unexpected Enum value.')
        return func(self.arg.eval())

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.func)}, {repr(self.arg)})'


QasmRealConstValues = Enum('QasmRealConstValues', 'Pi')

class QasmRealConst(QasmRealExpr):
    def __init__(self, value: QasmRealConstValues):
        self.value = value

    def eval(self):
        if self.value == QasmRealConstValues.Pi:
            return math.pi
        raise ValueError('Unexpected Enum value.')



class QasmGateDef(QasmNode):
    pass


class QasmApplyGate(QasmNode):
    def __init__(self, gate, params, qregs):
        self.gate = gate
        self.params = params
        self.qregs = qregs

    def __repr__(self):
        return f'QasmApplyGate({self.gate}, {self.params}, {self.qregs})'


class QasmBarrier(QasmNode):
    def __init__(self, qregs):
        self.qregs = qregs

    def __repr__(self):
        return f'QasmBarrier({self.qregs})'


class QasmIf(QasmNode):
    def __init__(self, creg: str, num: int, node: QasmNode):
        self.creg = creg
        self.num = num
        self.node = node

    def __repr__(self):
        return f'QasmIf({self.creg}, {self.num}, {self.node})'


class QasmReset(QasmNode):
    def __init__(self, qregs):
        self.qregs = qregs

    def __repr__(self):
        return f'QasmReset({self.qregs})'


class QasmMeasure(QasmNode):
    def __init__(self, qregs, cregs):
        self.qregs = qregs
        self.cregs = cregs

    def __repr__(self):
        return f'QasmMeasure({self.qregs}, {self.cregs})'


class QasmGateApply(QasmNode):
    def __init__(self,
                 gate: 'QasmAbstractGate',
                 params: List[QasmRealExpr],
                 qregs: List[Tuple[str, int]]) -> None:
        self.gate = gate
        self.params = params
        self.qregs = qregs

    def __repr__(self) -> str:
        return f'QasmGateApply({repr(self.gate)}, {self.params}, {self.qregs})'


QasmGateType = Enum('QasmGateType', 'Gate Opaque Builtin')


class QasmAbstractGate:
    def __init__(self, name: str, params: List[str], qargs: List[str]):
        self.name = name
        self.params = params
        self.n_params = len(params)
        self.qargs = qargs
        self.n_qargs = len(qargs)

    @classmethod
    @abstractmethod
    def gatetype(cls) -> QasmGateType:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}', {self.params}, {self.qargs})"


class QasmGate(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Gate

    def __init__(self, gatedef: QasmGateDef):
        self.gatedef = gatedef
        name = ''; params = []; qargs = [] # TODO: Impl.
        super().__init__(name, params, qargs)


    def __repr__(self) -> str:
        return f'QasmGate({repr(self.gatedef)})'


class QasmOpaque(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Opaque

    def __repr__(self) -> str:
        return f"QasmOpaque('{self.name}')"


class QasmBuiltinGate(QasmAbstractGate):
    @classmethod
    def gatetype(cls):
        return QasmGateType.Builtin

    def __repr__(self) -> str:
        return f"QasmBuiltinGate('{self.name}')"


def _get_matcher(regex: str) -> Callable[[str], Match]:
    _re = re.compile('^' + regex + '$')
    def matcher(s: str) -> Match:
        return _re.match(s)
    return matcher


_is_symbol = _get_matcher(_restr_symbol)
_is_quoted_str = _get_matcher(_restr_quoted_str)
_is_uint = _get_matcher(_restr_uint)
_is_float = _get_matcher(_restr_float)
_is_func = _get_matcher(_restr_funcs)


def _parse_statements(tokens,
                      stmts: List[QasmNode],
                      qregs: Dict[str, int],
                      cregs: Dict[str, int],
                      gates: Dict[str, QasmAbstractGate],
                      included: Set[str]):
    lineno, tok = tokens.get()
    while tok:
        if tok == 'qreg':
            sym, num = _parse_def_reg(tokens)
            if sym in qregs or sym in cregs:
                _err_with_lineno(lineno, f'Register "{sym}" is already defined.')
            qregs[sym] = num
        elif tok == 'creg':
            sym, num = _parse_def_reg(tokens)
            if sym in qregs or sym in cregs:
                _err_with_lineno(lineno, f'Register "{sym}" is already defined.')
            cregs[sym] = num
        elif tok == 'include':
            incfile = _parse_include_stmt(tokens)
            if incfile in included:
                _err_with_lineno(lineno, f'File "{incfile}" is already included.')
            included.add(incfile)
            if incfile == "qelib1.inc":
                load_qelib1(gates)
                #print('qelib1 loaded.')
                #print(gates)
            else:
                try:
                    with open(incfile) as f:
                        _parse_statements(f.read(), stmts, qregs, cregs, gates, included)
                except FileNotFoundError:
                    _err_with_lineno(lineno, f'Included file "{incfile}" is not exists.')
                except IsADirectoryError:
                    _err_with_lineno(lineno, f'Included file "{incfile}" is a directory.')
                except PermissionError:
                    _err_with_lineno(lineno, f'Cannot access to "{incfile}". Permission denied.')
                except OSError as e:
                    _err_with_lineno(lineno, f'During reading file {incfile}, Error occured. {e}')
        elif tok in ('gate', 'opaque'):
            if tok == 'gate':
                gate = _parse_def_gate(tokens)
            else:
                gate = _parse_opaque(tokens)
            if gate in gates:
                _err_with_lineno(lineno, f'Gate {gate} is already defined.')
            gates[gate] = gate
        elif tok == 'barrier':
            stmts.append(_parse_barrier_stmt(tokens, qregs))
        elif tok == 'if':
            stmts.append(_parse_if_stmt(tokens, gates, qregs, cregs))
        elif tok == 'reset':
            stmts.append(_parse_reset_stmt(tokens, qregs))
        elif tok == 'measure':
            stmts.append(_parse_measure_stmt(tokens, qregs, cregs))
        elif tok in gates:
            stmts.append(_parse_apply_gate(tokens, gates[tok], qregs))
        else:
            print(f"?{lineno}: {tok}")
        #print('stmts:', stmts)
        lineno, tok = tokens.get()


def _parse_idlist(tokens, endtoken: str) -> List[Any]:
    params = []
    while 1:
        param = tokens.get()
        if param is None:
            _err_with_lineno(param[0], 'Unexpected end of file.')
        params.append(param[1])
        delim = tokens.get()
        if delim is None:
            _err_with_lineno(param[0], 'Unexpected end of file.')
        if delim[1] == endtoken:
            return params
        if delim[1] != ',':
            _err_with_lineno(param[0], f'Unexpected token "{delim[1]}".')


def _parse_exprlist(tokens, endtoken: str) -> List[Any]:
    params = []
    while 1:
        param = _parse_expr(tokens)
        if param is None:
            _err_with_lineno(tokens.lineno, 'Unexpected end of file.')
        params.append(param)
        delim = tokens.get()
        if delim is None:
            _err_with_lineno(delim[0], 'Unexpected end of file.')
        if delim[1] == endtoken:
            return params
        if delim[1] != ',':
            _err_with_lineno(delim[0], f'Unexpected token "{delim[1]}".')


def _parse_params(tokens,
                  allow_no_params: bool = False,
                  allow_empty: bool = False) -> List[Any]:
    if allow_no_params:
        paren = tokens.get_if('(')
        if not paren:
            return []
    else:
        paren = tokens.get_if('(', 'No parameter found.')
    params = _parse_idlist(tokens, ')')
    if not params and not allow_empty:
        _err_with_lineno(paren[0], 'Empty parameter is not allowed.')
    return params


def _parse_qparams(tokens):
    params = _parse_idlist(tokens, ';')
    if not params:
        _err_with_lineno(tokens.lineno, 'Empty parameter is not allowed.')
    return params


def _parse_def_reg(tokens):
    sym = tokens.get_if(_is_symbol, 'After "qreg", symbol is expected.')
    tokens.get_if('[', f'Unexpected token after "qreg {sym}".')
    num = tokens.get_if(_is_uint, f'After "qreg {sym}[", unsigned integer is expected.')
    tokens.get_if(']', 'Unclosed bracket "[".')
    tokens.assert_semicolon()
    return sym[1], int(num[1])


def _parse_include_stmt(tokens):
    incfile = tokens.get_if(_is_quoted_str, 'After "include", file path is expected.')
    tokens.assert_semicolon()
    return incfile[1][1:-1]


def _parse_if_stmt(tokens, gates, qregs, cregs):
    tokens.get_if('(', '"(" is expected for if statements.')
    line, c = tokens.get()
    if c not in cregs:
        _err_with_lineno(line, f'creg is expected, found "{c}".')
    tokens.get_if('==', f'"==" is expected.')
    line, num = tokens.get_if(_is_uint, f'unsigned integer value is expected.')
    line, tok = tokens.get()
    if tok is None:
        _err_with_lineno(line, 'Unexpected enf of file.')
    if tok == 'measure':
        node = _parse_measure_stmt(tokens, qregs, cregs)
    elif tok == 'reset':
        node = _parse_reset_stmt(tokens, qregs)
    elif tok in gates:
        node = _parse_apply_gate(tokens, gates[tok], qregs)
    else:
        _err_with_lineno(line, f'Unexpected token "{tok}" is found.')
    return QasmIf(c, num, node)


def _parse_reset_stmt(tokens, qregs):
    q = _parse_qregs(tokens, qregs)
    tokens.assert_semicolon()
    return QasmReset(q)


def _parse_measure_stmt(tokens, qregs, cregs):
    q = _parse_qregs(tokens, qregs)
    tokens.get_if('->', '"->" is expected.')
    c = _parse_qregs(tokens, cregs)
    tokens.assert_semicolon()
    return QasmMeasure(q, c)


def _parse_def_gate(tokens):
    name = tokens.get_if(_is_symbol, 'After "gate", name is expected.')
    params = _parse_params(tokens, allow_no_params=True, allow_empty=False)
    qparams = _parse_qparams(tokens)
    tokens.get_if('{', '`{` is expected for gate definition.')
    # TODO: Impl.
    tokens.get_if('}', 'Corresponed paren `}` is required.')
    return QasmGateDef()


def _parse_opaque(tokens):
    name = tokens.get_if(_is_symbol, 'After "opaque", name is expected.')
    params = _parse_params(tokens, allow_no_params=True, allow_empty=False)
    qparams = _parse_qparams(tokens)
    tokens.assert_semicolon()
    return QasmOpaque(name, params, qparams)


def _parse_args(tokens) -> List[Any]:
    paren = tokens.get_if('(', 'No parens found')
    params = _parse_exprlist(tokens, ')')
    return params


def _parse_qregs(tokens, qregs, n_qregs=-1):
    def parse_qreg(must_found):
        if must_found:
            lineno, reg = tokens.get_if(_is_symbol, 'qreg is expected.')
        else:
            tok = tokens.get_if(_is_symbol)
            if tok is None:
                return None
            lineno, reg = tok
        if reg not in qregs:
            _err_with_lineno(lineno, 'Undefined qreg: "{reg}".')
        if tokens.get_if('['):
            lineno, num = tokens.get_if(_is_uint, 'Index is expected.')
            num = int(num)
            if num >= qregs[reg]:
                _err_with_lineno(lineno, f'Size of qreg "{reg}" is {qregs[reg].size} but {num} specified.')
            tokens.get_if(']', '"]" is expected.')
            return reg, num
        return reg, None

    regs = []
    while 1:
        tok = parse_qreg(False)
        if tok is None:
            return regs
        regs.append(tok)
        delim = tokens.get_if(',')
        if not delim:
            if n_qregs != -1 and n_qregs != len(regs):
                _err_with_lineno(tokens.lineno,
                                 f'{n_qregs} parameters are expected. {len(regs)} parameters found.')
            return regs


def _parse_apply_gate(tokens, gate, qregs):
    params = ()
    if gate.n_params:
        params = _parse_args(tokens)
    qregs = _parse_qregs(tokens, qregs, gate.n_qargs)
    tokens.assert_semicolon()
    return QasmApplyGate(gate, params, qregs)


def _parse_barrier_stmt(tokens, qregs):
    qregs = _parse_qregs(tokens, qregs)
    tokens.assert_semicolon()
    return QasmBarrier(qregs)


def _parse_expr(tokens):
    # expr := term ('+'|'-' term)*
    # term := factor ('*'|'/' factor)*
    # factor := '('expr')' | const | funccall | float
    # const := pi
    # funccall := function '(' args ')'
    # float := floating point number
    def _parse_number(tokens):
        line, tok = tokens.get()
        tokens.unget((line, tok))
        line, numstr = tokens.get_if(_is_float, 'Floating point number is expected.')
        return QasmRealValue(float(numstr))

    def _parse_factor(tokens):
        if tokens.get_if('('):
            expr = _parse_expr(tokens)
            tokens.get_if(')', 'Corresponded `)` not found.')
            return expr
        if tokens.get_if('pi'):
            return QasmRealConst(QasmRealConstValues.Pi)
        line_tok = tokens.get_if(_is_func)
        if line_tok:
            func = QasmRealUnaryFunctions.from_str(line_tok[1])
            tokens.get_if('(', '`(` is required after function')
            arg = _parse_expr(tokens)
            tokens.get_if(')', 'Corresponded `)` not found.')
            return QasmRealCall(func, arg)
        return _parse_number(tokens)

    def _parse_term(tokens):
        lhs = _parse_factor(tokens)
        while 1:
            if tokens.get_if('*'):
                rhs = _parse_factor(tokens)
                lhs = QasmRealMul(lhs, rhs)
                continue
            if tokens.get_if('/'):
                rhs = _parse_factor(tokens)
                lhs = QasmRealDiv(lhs, rhs)
                continue
            break
        return lhs

    lhs = _parse_term(tokens)
    while 1:
        if tokens.get_if('+'):
            rhs = _parse_term(tokens)
            lhs = QasmRealAdd(lhs, rhs)
            continue
        if tokens.get_if('-'):
            rhs = _parse_term(tokens)
            lhs = QasmRealSub(lhs, rhs)
            continue
        break
    return lhs


def load_qelib1(gates: Dict[str, QasmAbstractGate]) -> None:
    # TODO: Use circuit.GATE_SET.
    gates.update({
        'x': QasmBuiltinGate('x', [], ['t']),
        'y': QasmBuiltinGate('y', [], ['t']),
        'z': QasmBuiltinGate('z', [], ['t']),
        'h': QasmBuiltinGate('h', [], ['t']),
        't': QasmBuiltinGate('t', [], ['t']),
        'tdg': QasmBuiltinGate('tdg', [], ['t']),
        's': QasmBuiltinGate('s', [], ['t']),
        'sdg': QasmBuiltinGate('sdg', [], ['t']),
        'cx': QasmBuiltinGate('cx', [], ['c', 't']),
        'cz': QasmBuiltinGate('cz', [], ['c', 't']),
        'cnot': QasmBuiltinGate('cnot', [], ['c', 't']),
        'rx': QasmBuiltinGate('rx', ['theta'], ['t']),
        'ry': QasmBuiltinGate('ry', ['theta'], ['t']),
        'rz': QasmBuiltinGate('rz', ['theta'], ['t']),
        'phase': QasmBuiltinGate('rz', ['theta'], ['t']),
        'u1': QasmBuiltinGate('u1', ['lambda'], ['t']),
        'u2': QasmBuiltinGate('u2', ['phi', 'lambda'], ['t']),
        'u3': QasmBuiltinGate('u3', ['theta', 'phi', 'lambda'], ['t']),
        'cu1': QasmBuiltinGate('cu1', ['lambda'], ['c', 't']),
        'cu2': QasmBuiltinGate('cu2', ['phi', 'lambda'], ['c', 't']),
        'cu3': QasmBuiltinGate('cu3', ['theta', 'phi', 'lambda'], ['c', 't']),
        'swap': QasmBuiltinGate('swap', [], ['c', 't']),
        'ccx': QasmBuiltinGate('ccx', [], ['c1', 'c2', 't']),
        'toffoli': QasmBuiltinGate('ccx', [], ['c1', 'c2', 't']),
        # TODO: These gates are not defined in qelib1, but defined in language specifications.
        'U': QasmBuiltinGate('u3', ['theta', 'phi', 'lambda'], ['t']),
        'CX': QasmBuiltinGate('cx', [], ['c', 't'])
    })


def output_qasm(ast: QasmProgram) -> str:
    indent_nspace = 4
    lines = ['OPENQASM 2.0;']
    def add_line(line, indentlevel=0):
        lines.append(' ' * (indent_nspace * indentlevel) + line)

    def output_stmts(stmts, indentlevel=0):
        for stmt in stmts:
            if isinstance(stmt, QasmApplyGate):
                line = stmt.gate.name
                if stmt.params:
                    line += '(' + ', '.join(str(x.eval()) for x in stmt.params) + ')'
                line += ' '
                line += ', '.join(q if n is None else f'{q}[{n}]' for q, n in stmt.qregs)
                line += ';'
                add_line(line, indentlevel)
            elif isinstance(stmt, QasmBarrier):
                line = 'barrier '
                line += ', '.join(q if n is None else f'{q}[{n}]' for q, n in stmt.qregs)
                line += ';'
                add_line(line, indentlevel)
            elif isinstance(stmt, QasmMeasure):
                line = 'measure '
                line += ', '.join(q if n is None else f'{q}[{n}]' for q, n in stmt.qregs)
                line += ' -> '
                line += ', '.join(c if n is None else f'{c}[{n}]' for c, n in stmt.cregs)
                line += ';'
                add_line(line, indentlevel)
            else:
                raise ValueError('Unknown node ' + stmt.__class__.__name__)

    for inc in ast.included:
        add_line(f'include "{inc}";')
    for q, n in ast.qregs.items():
        add_line(f'qreg {q}[{n}];')
    for c, n in ast.cregs.items():
        add_line(f'creg {c}[{n}];')
    # TODO: Output gates: Dict[str, Any]
    output_stmts(ast.statements)
    return '\n'.join(lines)


def output_circuit(ast: QasmProgram) -> 'Circuit':
    from blueqat import Circuit
    c = Circuit()
    qmaps = {}
    qidx = 0
    for q, n in ast.qregs.items():
        qmaps[q] = (qidx, qidx + n)
        qidx += n

    def get_qubits(q, n):
        if n is None:
            return slice(qmaps[q][0], qmaps[q][1])
        return qmaps[q][0] + n

    def output_stmts(stmts):
        for stmt in stmts:
            if isinstance(stmt, QasmApplyGate):
                g = getattr(c, stmt.gate.name)
                bits = tuple(get_qubits(q, n) for q, n in stmt.qregs)
                if stmt.params:
                    g(*[x.eval() for x in stmt.params])[bits]
                else:
                    g[bits]
            elif isinstance(stmt, QasmBarrier):
                pass
            elif isinstance(stmt, QasmMeasure):
                for q, n in stmt.qregs:
                    bits = get_qubits(q, n)
                    c.m[bits]
            else:
                raise ValueError('Unknown node ' + stmt.__class__.__name__)

    output_stmts(ast.statements)
    return c


if __name__ == '__main__':
    # This QFT code is copied from IBM, OpenQASM project.
    # https://github.com/Qiskit/openqasm/blob/master/examples/generic/qft.qasm
    qftstr = '''
// quantum Fourier transform
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[0];
x q[2];
barrier q;
h q[0];
cu1(pi/2) q[1],q[0];
h q[1];
cu1(pi/4) q[2],q[0];
cu1(pi/2) q[2],q[1];
h q[2];
cu1(pi/8) q[3],q[0];
cu1(pi/4) q[3],q[1];
cu1(pi/2) q[3],q[2];
h q[3];
measure q -> c;'''

    #print(list(split_tokens(qftstr)))
    qp = parse_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nu1(sin(0 + cos(1 * pi - pi)) + 0.5 * pi / 2 - 0.5 * (pi - 0)) q[0];')
    print(qp)
    print(output_qasm(qp))
    exit()
    qp = parse_qasm(qftstr)
    print(qp)
    qasm = output_qasm(qp)
    print(qasm)
    qp2 = parse_qasm(qasm)
    print(qp2)
    qasm2 = output_qasm(qp2)
    assert qasm == qasm2
    from blueqat import Circuit
    print(output_circuit(parse_qasm(Circuit().h[0].m[:].to_qasm())).run(shots=100))
    print(output_circuit(parse_qasm(Circuit().h[0].cx[0, 1].m[:].to_qasm())).run(shots=100))
    print('end.')
