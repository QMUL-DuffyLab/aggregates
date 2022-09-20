import numpy as np
from kmc import Rates

protein_r = 5. # radius of a protein - only relevant for packing of real aggregates
n_trimers = 200 # (approximate) number of trimers to place in lattice
max_count = 10000 # maximum count to reach in histogram
binwidth = 50. # histogram bin width

# pulse stuff
# fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
#         1.9E14, 3.22E14, 6.12E14, 9.48E14]
xsec_485nm = 2.74E-15 # xsec of LHCII at 485nm - could change
# fluences = [9.48E14, 6.12E14, 3.22E14, 1.9E14, 1.31E14,
#         6.24E13, 3.03E13, 6.07E12]
fluences_1Mhz = [1.79E14, 4.48E14, 8.96E14, 1.34E15,
        1.79E15, 4.48E15, 8.96E15, 1.34E16, 1.52E16, 1.61E16]
fluences_10Mhz = [1.79E13, 4.48E13, 8.96E13, 1.34E14,
        1.79E14, 4.48E14, 8.96E14, 1.34E15, 1.52E15, 1.61E15,
        1.79E15, 2.24E15, 2.6E15]
fluences = fluences_1Mhz
pulse_fwhm = 200. # fwhm of pulse in ps
pulse_mu = 350. # peak time of pulse in ps

# rate stuff
hop = 25. # hopping rate between trimers
chl_decay = 3600. # decay of a chlorophyll
car_decay = 10. # decay of a quencher (carotenoid)
ann = 50. # annihilation rate for excitons on same trimer
pool_to_pq = 5.
pq_to_pool = 1.
rates_dict = {
 'detergent': Rates(np.inf, chl_decay, chl_decay, car_decay, np.inf, np.inf,
     np.inf, np.inf, ann, [False, True, True, False], True, True),
 'hop_only': Rates(hop, chl_decay, chl_decay, car_decay, np.inf, np.inf,
     np.inf, np.inf, ann, [False, True, True, False], True, True),
 'irrev': Rates(hop, chl_decay, chl_decay, car_decay, # car-eet
     pool_to_pq, pq_to_pool, 20., np.inf, ann, [False, True, True, False], True, True),
 'rev': Rates(hop, chl_decay, chl_decay, car_decay, # mennucci
     pool_to_pq, pq_to_pool, 20., 20., ann, [False, True, True, False], True, True),
 'fast_irrev': Rates(hop, chl_decay, chl_decay, car_decay, # schlau-cohen
     pool_to_pq, pq_to_pool, 1.1, np.inf, ann, [False, True, True, False], True, True),
 'fast_rev': Rates(hop, chl_decay, chl_decay, car_decay, # schlau-cohen
     pool_to_pq, pq_to_pool, 1.1, 1.1, ann, [False, True, True, False], True, True),
 'slow': Rates(hop, chl_decay, chl_decay, 833., # holzwarth
     180., 550., 260., 3300., ann, [False, True, False, False], True, True),
 'exciton': Rates(hop, chl_decay, 40., 40.,
     pool_to_pq, pq_to_pool, 1000., 1000., ann, [False, True, False, False], True, True),
 }
