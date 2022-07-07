import sys
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

file_path = sys.argv[1]
rho_q = sys.argv[2]
po_pq_ent = sys.argv[3]
# if os.path.isfile("out/lhcii_agg_mica.txt".format(file_path)):
# exp = np.loadtxt("out/exp_lhcii_solution.dat")

fig, ax1 = plt.subplots()
n_exp = ["mono", "bi", "tri"]
files = ["{}/{}_{}_{}_tau.dat".format(file_path, rho_q, po_pq_ent, ex) for ex in n_exp]
fluences = []
for i, f in enumerate(files):
    if os.path.isfile(f):
        l = np.loadtxt(f)
        fluences = l[:, 0]
        l = ma.masked_invalid(l)
        # the mask here repeats for the whole row of the lifetime matrix
        # for any row where the error is greater than the lifetime
        lm = ma.array(l, mask=np.repeat(l[:, 2] > l[:, 1], l.shape[1]))
        # this throws a warning about converting masked element to nan
        # cannot for the life of me figure out why, should work fine?
        plt.errorbar(fluences, lm[:, 1]/1000., lm[:, 2]/1000.,
            label=r'$ \left< \tau_{\text{%s.}} \right> $' % n_exp[i],
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)

if float(rho_q) > 0.0:
    exp_data = np.loadtxt("out/lhcii_lipid_mica.dat")
    plt.errorbar(exp_data[:, 0], exp_data[:, 1], exp_data[:, 2],
            label='LHCII w/ lipid bilayer', color='k', lw=4.0,
            elinewidth=0.5, capsize=2.0, marker='^', ms=10.0)

plt.grid()
plt.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Fluence $ (h \nu \times 10^{14} \; \text{pulse}^{-1} \text{cm}^{-2}) $')
ax1.set_ylabel(r'$ \left< \tau \right> (\text{ns}) $')
ax1.set_yticks([4., 3., 2.5, 2., 1.5, 1., 0.6])
ax1.set_yticklabels(["4", "3", "2.5", "2", "1.5", "1", "0.6"])
ax1.set_xticks(fluences)
ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticks(fluences)
f_fmt = ["{:4.2f}".format(x * 1E-14) for x in fluences]
ax1.set_xticklabels(f_fmt)
rho_fmt = ["{:4.2f}".format(x * 1.1E-14) for x in fluences]
ax2.set_xticklabels(rho_fmt)
ax2.set_xlabel(r'Excitations per trimer $ \rho_{\text{exc.}} $', labelpad=9.0)
ax2.set_yticks([4., 3., 2.5, 2., 1.5, 1., 0.6])
ax2.set_yticklabels(["4", "3", "2.5", "2", "1.5", "1", "0.6"])
plt.minorticks_off()
ax1.minorticks_off()
fig.savefig("{}/{}_tau_loglog.pdf".format(file_path, rho_q))
