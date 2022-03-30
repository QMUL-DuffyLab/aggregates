import sys
import os
import numpy as np
import matplotlib.pyplot as plt

file_path = sys.argv[1]
print(file_path)
exp = np.loadtxt("out/exp_lhcii_solution.dat")
fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
        1.9E14, 3.22E14, 6.12E14, 9.48E14]
rho = 1.1E-14 * np.array(fluences)
def forward(x):
    '''convert fluence to average excitons per trimer'''
    return x * 1.1E-14

fig, ax1 = plt.subplots()
if os.path.isfile("{}/lifetimes.dat".format(file_path)):
    lifetimes = np.loadtxt("{}/lifetimes.dat".format(file_path))
    plt.errorbar(fluences, lifetimes[:, 0]/1000., yerr=lifetimes[:, 1]/1000.,
            label=r'$ \left< \tau_{\text{avg.}} \right> $', 
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
if os.path.isfile("{}/mono_tau.dat".format(file_path)):
    lifetimes = np.loadtxt("{}/mono_tau.dat".format(file_path))
    plt.errorbar(fluences, lifetimes[:, 0]/1000., yerr=lifetimes[:, 1]/1000.,
            label=r'$ \left< \tau_{\text{mono.}} \right> $', 
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
if os.path.isfile("{}/bi_tau.dat".format(file_path)):
    lifetimes = np.loadtxt("{}/bi_tau.dat".format(file_path))
    plt.errorbar(fluences, lifetimes[:, 0]/1000., yerr=lifetimes[:, 1]/1000.,
            label=r'$ \left< \tau_{\text{bi.}} \right> $', 
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
if os.path.isfile("{}/tri_tau.dat".format(file_path)):
    lifetimes = np.loadtxt("{}/tri_tau.dat".format(file_path))
    plt.errorbar(fluences, lifetimes[:, 0]/1000., yerr=lifetimes[:, 1]/1000.,
            label=r'$ \left< \tau_{\text{tri.}} \right> $', 
            elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)

plt.errorbar(exp[:, 0], exp[:, 1], yerr=exp[:, 2], label=r'Exp.', 
        elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
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
rhos = [forward(x) for x in fluences]
rho_fmt = ["{:4.2f}".format(x) for x in rhos]
ax2.set_xticklabels(rho_fmt)
ax2.set_xlabel(r'Excitations per trimer $ \rho_{\text{exc.}} $', labelpad=9.0)
ax2.set_yticks([4., 3., 2.5, 2., 1.5, 1., 0.6])
ax2.set_yticklabels(["4", "3", "2.5", "2", "1.5", "1", "0.6"])
plt.minorticks_off()
ax1.minorticks_off()
fig.savefig("{}/tau_loglog.pdf".format(file_path))
