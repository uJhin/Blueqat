from ..vqe import *

def statevector_sampler(circuit, meas, *args, **kwargs):
    """Calculate the expectations without sampling."""
    kwargs.setdefault('returns', 'statevector')
    meas = tuple(meas)
    n_qubits = circuit.n_qubits
    return expect(circuit.run(*args, **kwargs), meas)

# Overwrite
non_sampling_sampler = statevector_sampler
