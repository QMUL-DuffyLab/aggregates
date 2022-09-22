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
npt = defaults.xsec * defaults.fluences
files = ["{}/{}_{:3.2f}_{}_fit_info.csv".format(file_path, rho_q,
    n, fwhm) for n in npt]
for i, f in enumerate(files):
    if os.path.isfile(f):
        d = pd.read_csv(f)
        l = ma.masked_invalid(l)
        # the mask here repeats for the whole row of the lifetime matrix
        # for any row where the error is greater than the lifetime
        lm = ma.array(l, mask=np.repeat(l[:, 2] > l[:, 1], l.shape[1]))
        # this throws a warning about converting masked element to nan
        # cannot for the life of me figure out why, should work fine?
        plt.errorbar(fluences, lm[:, 1]/1000., lm[:, 2]/1000.,
            label=r'$ \left< \tau_{\text{%s.}} \right> $' % d["n_exp"],
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)

if float(rho_q) > 0.0:
    exp_data = np.loadtxt("out/lhcii_lipid_mica.dat")
    plt.errorbar(exp_data[:, 0], exp_data[:, 1], exp_data[:, 2],
            label='LHCII w/ lipid bilayer', color='k', lw=4.0,
            elinewidth=0.5, capsize=2.0, marker='^', ms=10.0)

plt.grid()
plt.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Excitations per trimer $ \rho_{\text{exc.}} $')
f_fmt = ["{:4.2f}".format(n) for n in npt]
ax.set_ylabel(r'$ \left< \tau \right> (\text{ns}) $')
ax.set_yticks([4., 3., 2.5, 2., 1.5, 1., 0.5, 0.2])
ax.set_yticklabels(["4", "3", "2.5", "2", "1.5", "1", "0.5", "0.2"])
ax.set_xticks(npt)
ax.set_xticklabels(f_fmt)
ax.minorticks_off()
plt.minorticks_off()
fig.savefig("{}/{}_{}_tau_loglog.pdf".format(file_path, rho_q, po_pq_ent))
