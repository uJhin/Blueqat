from blueqat import Circuit
from blueqat.circuit import _GateWrapper
from blueqat import circuitmacro

@circuitmacro
def gphase(c: Circuit, phase: float) -> _GateWrapper:
    """Add global phase."""
    return c.u(0.0, 0.0, 0.0, phase)

@circuitmacro
def su(c: Circuit, theta: float, phi: float, lam: float) -> _GateWrapper:
    """SU(2) matrix gate."""
    return c.u(theta, phi, lam, -0.5 * (phi * lam))
