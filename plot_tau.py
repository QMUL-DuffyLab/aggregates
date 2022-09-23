import sys
import os
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import defaults

file_path = sys.argv[1]
rho_q = sys.argv[2]
fwhm = sys.argv[3]
# if os.path.isfile("out/lhcii_agg_mica.txt".format(file_path)):
# exp = np.loadtxt("out/exp_lhcii_solution.dat")

fig, ax = plt.subplots()
npt = defaults.xsec * np.array(defaults.fluences)
files = ["{}/{}_{:3.2f}_{}_fit_info.csv".format(file_path, rho_q,
    n, fwhm) for n in npt]
# list the parameters we need (plus their errors)
k = ["a1", "a2", "tau1", "tau2", "tau_amp"]
k = k + [x + "_err" for x in k]
arr = np.zeros((len(k), len(files)))
for i, f in enumerate(files):
    if os.path.isfile(f):
        d = pd.read_csv(f)
        # find the number of exponentials with the smallest tau_amp error
        d = d.set_index('n_exp')
        minloc = d.idxmin(axis=0)['tau_amp_err']
        # loop over the keys and if they're there, put the values in
        # missing keys will -> 0, missing values -> nan (I think?)
        for j, key in enumerate(k):
            if key in d.keys():
                arr[j, i] = d[key][minloc]
# put back in a dict to make the following more readable
d = dict(zip(k, arr))

# plot tau_amp and the actual time constants with errors
plt.errorbar(npt, d["tau_amp"], yerr=d["tau_amp_err"],
    label=r'$ \left< \tau_{\text{amp.}} \right> $',
    elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
plt.errorbar(npt, d["tau1"], yerr=d["tau1_err"],
    label=r'$ \tau_{1} $',
    elinewidth=0.5, capsize=2.0, marker='s', ms=6.0, lw=1.0)
plt.errorbar(npt, d["tau2"], yerr=d["tau2_err"],
    label=r'$ \tau_{2} $',
    elinewidth=0.5, capsize=2.0, marker='^', ms=6.0, lw=1.0)
# add markers to show the relative size of the amplitudes
plt.scatter(npt, d["tau1"], s=d["a1"] * 500., c='#888888')
plt.scatter(npt, d["tau2"], s=d["a2"] * 500., c='#888888')
# if float(rho_q) > 0.0:
#     exp_data = np.loadtxt("out/lhcii_lipid_mica.dat")
#     plt.errorbar(exp_data[:, 0], exp_data[:, 1], exp_data[:, 2],
#             label='LHCII w/ lipid bilayer', color='k', lw=4.0,
#             elinewidth=0.5, capsize=2.0, marker='^', ms=10.0)

plt.grid()
plt.legend()
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Excitations per trimer $ \rho_{\text{exc.}} $')
ax.set_ylabel(r'$ \left< \tau \right> (\text{ns}) $')
ax.minorticks_off()
plt.minorticks_off()
fig.savefig("{}/{}_{}_tau_scatter.pdf".format(file_path, rho_q, fwhm))
