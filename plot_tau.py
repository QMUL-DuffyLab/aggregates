import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists

# path = "/home/callum/code/aggregates/out"
# n_iterations = "{:d}".format(1000)
# n_quenchers = "{:3.2f}".format(0.0)
print(str(sys.argv))
file_path = sys.argv[1]
print(file_path)

exp = np.loadtxt("out/exp_lhcii_solution.dat")
fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
        1.9E14, 3.22E14, 6.12E14, 9.48E14]
rho = 1.1E-14 * np.array(fluences)
def forward(x):
    return x * 1.1E-14
def backward(x):
    return x / 1.1E-14

taus = []
for f in fluences:
    d = np.loadtxt("{}_{:4.2E}_decays.dat".format(file_path, f))
    m = np.mean(d[:, 0])
    err = np.std(d[:, 0]) / np.sqrt(1000)
    print("Fluence = {:4.2e}".format(f))
    print("Total mean = {:6.2f}, standard error = {:6.2f}".format(m, err))
    taus.append([f, m, err])

taus = np.array(taus)
np.savetxt("{}/taus.dat".format(os.path.dirname(file_path)), taus)
plot_exp_fits = True
if plot_exp_fits:
    mono_tau = np.loadtxt("{}/mono_tau.dat".format(os.path.dirname(file_path)))
    bi_tau = np.loadtxt("{}/bi_tau.dat".format(os.path.dirname(file_path)))
# taus_pr_01 = np.loadtxt("{}/lut_eet/hex/taus_pr_0.1.dat".format(path))

fig, ax1 = plt.subplots()
plt.errorbar(taus[:, 0], taus[:, 1]/1000., yerr=np.sqrt(taus[:, 2]/1000.), 
        label=r'$ \left< \tau \right> $', elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
if plot_exp_fits:
    plt.plot(mono_tau[:, 0], mono_tau[:, 1]/1000., label=r'$ \left< \tau_{\text{mono}} \right> $', marker='o', ms=6.0, lw=3.0)
    plt.plot(bi_tau[:, 0][np.where(bi_tau[:, 1]>500.)], bi_tau[:, 1][np.where(bi_tau[:, 1]>500.)]/1000., label=r'$ \left< \tau_{\text{bi}} \right> $', marker='o', ms=6.0, lw=3.0)
# plt.errorbar(taus_pr_01[:, 0], taus_pr_01[:, 1]/1000., yerr=np.sqrt(taus_pr_01[:, 2]/1000.), 
#         label='Const. 0.1', elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
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
ax2.set_xticks(taus[:, 0])
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
fig.savefig("{}/tau_loglog.pdf".format(os.path.dirname(file_path)))
