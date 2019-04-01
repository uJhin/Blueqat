from itertools import product
from .backendbase import Backend

class _BinaryBackendContext:
    def __init__(self, n_bits, ibits, ilabels, obits, olabels, disp):
        self.n_bits = n_bits
        self.ops = []
        self.ibits = ibits
        self.ilabels = ilabels
        self.obits = obits
        self.olabels = olabels
        self.disp = disp


class BinaryBackend(Backend):
    def __parse_runargs(self, n_bits, inputs=None, outputs=None, display=True):
        # Parse inputs. {'label': n, ...} or [n, m, ...] or None.
        if inputs is None:
            inputs = list(range(n_bits))
            inputs_label = [f'{i}' for i in range(n_bits)]
        if isinstance(inputs, dict):
            inputs_label = list(inputs.keys())
            inputs = list(inputs.values())
            if len(set(inputs)) != len(inputs):
                raise ValueError('Duplicate input bits!')
        else:
            inputs = list(inputs)
            inputs_label = [f'{i}' for i in inputs]
            if len(set(inputs)) != len(inputs):
                raise ValueError('Duplicate input bits!')
        # Parse outputs. {'label': n, ...} or [n, m, ...] or None.
        if outputs is None:
            outputs = list(range(n_bits))
            outputs_label = [f'{i}' for i in range(n_bits)]
        if isinstance(outputs, dict):
            outputs_label = list(outputs.keys())
            outputs = list(outputs.values())
            if len(set(outputs)) != len(outputs):
                raise ValueError('Duplicate output bits!')
        else:
            outputs = list(outputs)
            outputs_label = [f'{i}' for i in outputs]
            if len(set(outputs)) != len(outputs):
                raise ValueError('Duplicate output bits!')
        ctx = _BinaryBackendContext(n_bits, inputs, inputs_label, outputs, outputs_label, display)
        return ctx

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        return gates, self.__parse_runargs(n_qubits, *args, **kwargs)

    def _postprocess_run(self, ctx):
        n_bits = ctx.n_bits
        inputs = ctx.ibits
        outputs = ctx.obits
        ops = ctx.ops
        disp = ctx.disp
        if disp:
            inlabel = [s or ' ' for s in ctx.ilabels]
            outlabel = [s or ' ' for s in ctx.olabels]
            label = ' '.join(inlabel) + ' | '  + ' '.join(outlabel)
            print(label)
            print('-' * len(label))
            fmt = ''
            for s in inlabel:
                fmt += ' ' * (len(s) - 1) + '{} '
            fmt += '|'
            for s in outlabel:
                fmt += ' ' * (len(s) - 1) + ' {}'
        results = []
        for bits in product((0, 1), repeat=len(inputs)):
            mem = [0] * n_bits
            for i, b in zip(inputs, bits):
                mem[i] = b
            for t, cs in ops:
                if all(mem[c] for c in cs):
                    mem[t] ^= 1
            outs = [mem[i] for i in outputs]
            results.append((bits, outs))
            if disp:
                print(fmt.format(*bits, *outs))
        return results

    def gate_x(self, gate, ctx):
        for idx in gate.target_iter(ctx.n_bits):
            ctx.ops.append((idx, ()))
        return ctx

    def gate_cx(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx.n_bits):
            ctx.ops.append((target, (control,)))
        return ctx

    def gate_ccx(self, gate, ctx):
        c1, c2, t = gate.targets
        ctx.ops.append((t, (c1, c2)))
        return ctx
