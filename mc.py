from numba import jit, njit, prange, vectorize, int32, float64, boolean, void
from numba.experimental import jitclass
import numpy as np
from kmc import Pulse, Rates, Iteration

n_sites = 1
n_states = 3
n_current = 0
loss = [False, False, False, False, False]
ni = np.zeros((n_sites, n_states))
n_max = [14, 1, 1]
dt = 1.0

# in principle i think we could parallelise the inner loop here but
# we only need to calculate it once per simulation so i haven't bothered
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

@njit
def allocate_quenchers(rng, rho_q, n):
    n_q = int(rho_q * n)
    is_q = np.full(n_q, False)
    quenchers = np.full(n_q, -1)
    for i in range(n_q):
        c = rng.integers(n_q)
        while is_q[c]:
            c = rng.integers(n_q)
        quenchers[i] = c
    for i in range(n):
        if not is_q[i]:
            # zero the A->P rate
            pass
    return (is_q, quenchers)

@njit
def rate_calc(ind, t, state, rates):
    # should use base_rates here - will have
    # correct rates for neighbours for each site
    r = rates.copy()
    if (state == "A"):
        n = ni[ind, 0]
    elif (state == "P"):
        n = ni[ind, 1]
    elif (state == "Q"):
        n = ni[ind, 2]
    if (t < np.size(pulse) * dt):
        if ((state == "A" and n < n_max) 
                or (state == "P" and n < 1)):
            ft = pulse.ft[int(t/dt)]
            r[0] = (ft / 24.0) * (24.0 - n)
            r[1] = (ft / 24.0) * (n)
    # the first two rates will be zero by default
    # so outside of the pulse time we can ignore them
    if (is_q[ind]):
        if (state != "P" and ni[ind, 1] > 0):
            r[-3] = 0.0
        elif (state == "P" and ni[ind, 2] > 0):
            r[-3] = 0.0
    for i in range(2, np.size(rates) - 2):
        r[i] = r[i] * n
        if (i < np.size(rates) - 4):
            nn = neighbours[ind, i - 2]
            # total population on this neighbour is A + P
            r[i] = r[i] * ds[n, (ni[nn, 0] + ni[nn, 1])]
    if (is_q[ind] and state != "Q"):
        n = ni[i, 0] + ni[i, 1]
    af = (n * (n - 1)) / 2.0
    r[-1] = r[-1] * af
    return r

@njit(float64[:](float64[:], float64), parallel=True)
def prob_calc(rates, dt):
    p = np.zeros_like(rates)
    for i in range(np.size(rates)):
        p[i] = rates[i] * np.exp(-1.0 * rates[i] * dt)
    return p


@njit
def move(ind, process, state, loss):
    if (state == "A"):
        if (process == 0):
            # absorption
            add(ind, 0)
        elif (process == 1):
            # simulated emission
            remove(ind, 0)
            # NB: add a loss column for SE?
        elif (1 < process < (rate_size - 4)):
            # hop to neighbouring A
            transfer(ind, 0, neighbours[ind, i - 2], 0)
        elif (process == rate_size - 4):
            pass
        elif (process == rate_size - 3):
            # hop to P
            transfer(ind, 0, ind, 1)
        elif (process == rate_size - 2):
            # decay
            remove(ind, 0)
            loss[1] = True
        elif (process == rate_size - 1):
            # annihilation
            # A and P can annihilate with each other: if
            # both are populated, choose one to remove at random
            if (is_q[ind]):
                c = rng.integers(ni[ind, 0] + ni[ind, 1])
                if (c == 0): # check rng.integers behaviour
                    s = 1
                else:
                    s = 0
            else:
                s = 0
            remove(ind, s)
            loss[0] = True
    elif (state == "P"):
        if (process == 0):
            # absorption
            add(ind, 0)
        elif (process == 1):
            # simulated emission
            remove(ind, 0)
        elif (process == rate_size - 4):
            # hop back to A
            transfer(ind, 1, ind, 0)
        elif (process == rate_size - 3):
            # hop to Q
            transfer(ind, 1, ind, 2)
        elif (process == rate_size - 2):
            # decay
            remove(ind, 0, loss=2)
            loss[2] = True
        elif (process == rate_size - 1):
            # annihilation
            # if P is occupied this is a quencher, and for
            # this rate to be nonzero A must also be populated
            c = rng.integers(ni[ind, 0] + ni[ind, 1])
            if (c == 0): # check rng.integers behaviour
                s = 1
            else:
                s = 0
            remove(ind, s)
            loss[0] = True
        else:
            raise ValueError("Incorrect process on pre-quencher.")
    elif (state == "Q"):
        if (process == rate_size - 3):
            # hop back to P
            transfer(ind, 2, ind, 1)
        elif (process == rate_size - 2):
            # decay
            remove(ind, 0, loss=3)
            loss[3] = True
        elif (process == rate_size - 1):
            # annihilation
            # this should not be possible
            raise ValueError("Annihilation on quencher.")
        else:
            raise ValueError("Incorrect process on quencher.")

@njit(void(int32, int32))
def add(i, s):
    global n_current
    ni[i, s]  += 1
    n_current += 1

@njit(void(int32, int32))
def remove(i, s):
    global n_current
    ni[i, s]  -= 1
    n_current -= 1

@njit(void(int32, int32, int32, int32))
def transfer(i, s_i, f, s_f):
    remove(i, s_i)
    add(f, s_f)

# this is what it should eventually look like!
@njit(float64[:](int32, int32, float64, float64[:]))
def rate_calc(i, s, t, rates):
    # should use base_rates here - will have
    # correct rates for neighbours for each site
    r = rates.copy()
    n = ni[i, s]
    if (t < np.size(pulse) * dt):
        if (s != 2 and n < n_max[s]):
            ft = pulse.ft[int(t/dt)]
            r[0] = (ft / 24.0) * (24.0 - n)
            r[1] = (ft / 24.0) * (n)
    if (s < 2):
        na = ni[i, 0] + ni[i, 1]
    else:
        na = 0.0
    af = (na * (na - 1)) / 2.0
    r[2] = r[2] * af

    r[3:] = r[3:] * n
    if (s != n_states - 1):
        if (ni[i, s + 1] == n_max[s + 1]):
            r[4] = 0.0
    if (s != 0):
        if (ni[i, s - 1] == n_max[s - 1]):
            r[5] = 0.0
    if (s == 0):
        for i in range(6, np.size(rates)):
            nn = neighbours[i, i - 6]
            # total population on this neighbour is A + P
            r[i] = r[i] * ds[n, (ni[nn, 0] + ni[nn, 1])]
    return r

@njit(void(int32, int32, int32, boolean[:]))
def move(p, i, s, loss):
    if (p == 0):
        add(i, s)
    elif (p == 1):
        remove(i, s) # stimulated emission
        loss[0] = True
    elif (p == 2):
        remove(i, s) # annihilation
        loss[1] = True
    elif (p == 3):
        remove(i, s) # decay from state s
        loss[s + 2] = True
    elif (p == 4):
        transfer(i, s, i, s + 1) # forward transfer
    elif (p == 5):
        transfer(i, s, i, s - 1) # backward transfer
    else:
        if (s != 0):
            raise ValueError("process %d called for state %d" % p, s)
        transfer(i, 0, neighbours[i, x - 6], 0) # hop

@njit
def mc_step(rng, dt, n_q):
    n_attempts = n_sites + (2 * n_q)
    for i in range(n_attempts):
        loss = [False, False, False, False, False]
        ri = rng.integers(n_attempts)
        s = ri // n_sites + ri // (n_sites + n_q)
        if (ri < n_sites):
            ind = ri
            s = 0
        if (n_sites <= ri < n_sites + n_q):
            ind = ri - n_sites
            s = 1
        else:
            ind = ri - (n_sites + n_q)
            s = 2

        rates = rate_calc(ind, s, t, base_rates)
        if not np.any(rates):
            continue
        nz = np.nonzero(rates)
        for j in range(nz):
            if not np.any(rates):
                break # break out of trying to do stuff on this site
            c = rng.integers(nz)
            ri = 0
            for k in range(np.size(rates)):
                if (rates[k] > 0.0):
                    ri += 1
                if ri == c:
                    c = k
                    break

        r = rng.random()
        if (r < rates[c] * np.exp(-1.0 * rates[c] * dt)):
            move(c, ind, s, loss)
            rates = rate_calc(ind, s, t, base_rates)
        if np.any(loss):
            for j in range(np.size(loss)):
                if (loss[j]):
                    counts[j, np.floor(t / binwidth)] += 1
