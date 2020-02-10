from ..vqe import *

def statevector_sampler(circuit, meas, *args, **kwargs):
    """Calculate the expectations without sampling."""
    kwargs.setdefault('returns', 'statevector')
    meas = tuple(meas)
    n_qubits = circuit.n_qubits
    return expect(circuit.run(*args, **kwargs), meas)

# Overwrite
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

# Overwrite
get_state_vector_sampler = get_statevector_counting_sampler
