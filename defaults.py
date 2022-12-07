import numpy as np
from kmc import Rates

protein_r = 5. # radius of a protein - only relevant for packing of real aggregates
n_trimers = 200 # (approximate) number of trimers to place in lattice
max_count = 10000 # maximum count to reach in histogram
binwidth = 25. # histogram bin width

# pulse stuff
"""
  xsec_485nm = 2.74E-15 cm^2 for an LHCII trimer at 485nm
  (taken from digitising Leonas's figure and using his reported
  cross section at 534 nm). our effective cross section is smaller
  than that though - we only consider excitations that end up on Chl a
  # xsec_485nm = 2.74E-15
"""
xsec = 2.74E-16
# fluences from Sophie's data
# fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
#         1.9E14, 3.22E14, 6.12E14, 9.48E14]
# fluences from Lekshmi's data
fluences_1Mhz = [1.79E14, 4.48E14, 8.96E14, 1.34E15,
        1.79E15, 4.48E15, 8.96E15, 1.34E16, 1.52E16, 1.61E16]
fluences_10Mhz = [1.79E13, 4.48E13, 8.96E13, 1.34E14,
        1.79E14, 4.48E14, 8.96E14, 1.34E15, 1.52E15, 1.61E15,
        1.79E15, 2.24E15, 2.6E15]
# these aren't really fluences, i'm abusing notation because the rest
# of the code looks for fluences. they're arbitrary excitation densities
# - by doing this we get average excitations per trimer of 0.05, 0.10, etc
fluences = [x / xsec for x in [0.05, 0.10, 0.25, 0.50,
            0.75, 1.0, 1.25, 1.5, 2., 3., 4., 5.]]
fluences = [x / xsec for x in [0.05, 5.]]
pulse_fwhm = 50. # fwhm of pulse in ps
pulse_mu = 200. # peak time of pulse in ps

# rate stuff
hop = 25. # hopping rate between trimers
chl_decay = 2200. # decay of a chlorophyll
q_decay = 10. # decay of a quencher (carotenoid)
ann = 16. # annihilation rate for excitons on same trimer
omega = 5. # entropy ratio - \tau_{pool->pq} / \tau_{pq->pool}
rates_dict = {
 'detergent': Rates(np.inf, chl_decay, chl_decay, q_decay, np.inf, np.inf,
     np.inf, np.inf, ann, [False, True, True, False], True, True),
 'hop_only': Rates(hop, chl_decay, chl_decay, q_decay, np.inf, np.inf,
     np.inf, np.inf, ann, [False, True, True, False], True, True),
 'slow_entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     omega, 1.0, 100., np.inf, ann, [False, True, True, False], True, True),
 'medium_entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     omega, 1.0, hop, np.inf, ann, [False, True, True, False], True, True),
 'fast_entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     omega, 1.0, 1.0, np.inf, ann, [False, True, True, False], True, True),
 'slow_non-entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     1.0, 1.0, 100., np.inf, ann, [False, True, True, False], True, True),
 'medium_non-entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     1.0, 1.0, hop, np.inf, ann, [False, True, True, False], True, True),
 'fast_non-entropic': Rates(hop, chl_decay, chl_decay, q_decay,
     1.0, 1.0, 1.0, np.inf, ann, [False, True, True, False], True, True),
 }
