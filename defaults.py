import numpy as np
from rates import Rates

protein_r = 5. # radius of a protein - only relevant for packing of real aggregates
n_trimers = 200 # (approximate) number of trimers to place in lattice
max_count = 10000 # maximum count to reach in histogram
binwidth = 25.0 # histogram bin width (picoseconds)

# pulse stuff
"""
  xsec_485nm = 2.74E-15 cm^2 for an LHCII trimer at 485nm
  (taken from digitising Leonas's figure and using his reported
  cross section at 534 nm). The fortran divides by 24 to get a cross
  section per chlorophyll a, so don't do that here.
  Note that this wasn't important before because we
  were converting to excitation densities, but if you're reproducing
  experimental fluences, this will matter!
"""
xsec = 2.74E-15
# example experimental fluences. give in photons cm^{-2} pulse^{-1}
fluences = [1.0e12, 5.0e12, 1.0e13, 5.0e13, 1.0e14]
# example of using excitation densities instead of real fluences - if
# you want to do this you can just change the variable name from
# exc_dens to fluences and the code will use these instead
exc_dens = [x / xsec for x in [0.05, 0.10, 0.25, 0.50,
            0.75, 1.0, 1.25, 1.5, 2., 3., 4., 5.]]

pulse_fwhm = 50. # fwhm of pulse in ps
pulse_mu = 200. # peak time of pulse in ps

# rate stuff - all in picoseconds
hop = 25.0              # hopping rate between trimers
chl_decay = 2200.0      # decay of a chlorophyll
pq_decay = chl_decay    # decay of pre-quencher
q_decay = 10.0          # decay of a quencher (carotenoid)
ann = 16.0              # annihilation rate for excitons on same trimer
omega = 5.0             # entropy ratio (n_pool / n_pq)

# the fortran code actually bins four decay processes:
# [annihilation, pool decay, PQ decay, Q decay].
# emissive here tells the code which of these to include when
# keeping track of the max count
emissive = [False, True, True, False]
# for argument ordering see rates.py
rates_dict = {
 'detergent': Rates(np.inf, chl_decay, pq_decay, q_decay, np.inf, np.inf,
     np.inf, np.inf, ann,
     emissive,
     True, True),
 'hop_only': Rates(hop, chl_decay, pq_decay, q_decay, np.inf, np.inf,
     np.inf, np.inf, ann, emissive, True, True),
 # entropic here refers to omega: applies an entropic penalty for
 # transfer to the pre-quencher. roughly speaking, assumes that
 # the chlorophyll which transfers energy to the carotenoid isn't
 # strongly coupled to the rest of the Chls.
 # slow, medium, fast denote the transfer rate from PQ to Q:
 # slow 100ps, medium 10ps, fast 1ps
 'slow_entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     omega, 1.0, 100., np.inf, ann, emissive, True, True),
 'medium_entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     omega, 1.0, hop, np.inf, ann, emissive, True, True),
 'fast_entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     omega, 1.0, 1.0, np.inf, ann, emissive, True, True),
 # non-entropic doesn't apply the entropic penalty: basically, assumes
 # that all the chlorophylls are connected. slow, medium, fast as before
 'slow_non-entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     1.0, 1.0, 100., np.inf, ann, emissive, True, True),
 'medium_non-entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     1.0, 1.0, hop, np.inf, ann, emissive, True, True),
 'fast_non-entropic': Rates(hop, chl_decay, pq_decay, q_decay,
     1.0, 1.0, 1.0, np.inf, ann, emissive, True, True),
 }

# maximum number of exponentials to fit
n_max = 3
# initial guesses for fit components - these are in nanoseconds!
tau_init = 0.001 * np.array([chl_decay, ann, 500])
