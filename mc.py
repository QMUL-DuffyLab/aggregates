from numba import jit, int32, float64
from numba.experimental import jitclass
import numpy as np
from kmc import Pulse, Rates, Iteration

@jitclass([('start', int32), ('end', int32), ('float', float64)])
class Process(start, end, base_rate, emissive):
    name: str
    start: int
    end: int
    base_rate: float
    emissive: bool

def annihilation(ni, i, base_rate):
    n = ni[0, i] + ni[1, i]
    return (n * (n - 1) * base_rate) / 2.

def stimulated_emission(ni, i, pulse, t, dt):
    return pulse.ft[int(t / dt)] * (ni[0, i] + ni[1, i]) / 24.0

def absorption(ni, i, pulse, t, dt):
    n = ni[0, i] + ni[1, i]
    return (pulse.ft[int(t / dt)] / 24.0) * (24.0 - n)

@njit
def rate_calc(ind, t, state, rates):
    r = rates.copy()
    if (state == "A"):
        n = ni[0, ind]
    elif (state == "P"):
        n = ni[1, ind]
    elif (state == "Q"):
        n = ni[2, ind]
    if (t < np.size(pulse) * dt):
        if ((state == "A" and n < n_max) 
                or (state == "P" and n < 1)):
            ft = pulse.ft[int(t/dt)]
            r[0] = (ft / 24.0) * (24.0 - n)
            r[1] = (ft / 24.0) * (n)
    # the first two rates will be zero by default
    # so outside of the pulse time we can ignore them
    if (is_q[ind]):
        if (state != "P" and ni[1, ind] > 0):
            r[-3] = 0.0
        elif (state == "P" and ni[2, ind] > 0):
            r[-3] = 0.0
    for i in range(2, np.size(rates) - 2):
        r[i] = r[i] * n
        if (i < np.size(rates) - 4):
            nn = neighbours[ind, i - 2]
            # total population on this neighbour is A + P
            r[i] = r[i] * ds[n, (ni[0, nn] + ni[1, nn])]
    if (is_q[ind] and state != "Q"):
        n = ni[0, i] + ni[1, i]
    af = (n * (n - 1)) / 2.0
    r[-1] = r[-1] * af
    return r

 

@njit
def hop_entropy(n_max, k_hop):
    ds = np.empty((n_max + 1, n_max + 1))
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            if (i < j):
                ds[i, j] = k_hop * ((i * (n_max - j)) /
                        ((j + 1) * (n_max - i + 1)))
            else:
                ds[i, j] = k_hop
    return ds

def allocate_quenchers(rng, rho_q, n):
    n_q = int(rho_q * n)
    is_q = np.full(n_q, False)
    quenchers = np.full(n_q, -1)
    for i in range(n_q):
        c = rng.randint(n_q)
        while is_q[c]:
            c = rng.randint(n_q)
        quenchers[i] = c
    return quenchers

# processes

def add(ni, i):
    ni[i] = ni[i] + 1

def remove(ni, i):
    ni[i] = ni[i] - 1

def hop(ni, i, j):
    ni[i] = ni[i] - 1
    ni[j] = ni[j] + 1

def hop_pq(ni, npq, i):

