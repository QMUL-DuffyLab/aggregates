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
# a1 = []
# a1_err = []
# a2 = []
# a2_err = []
# tau1 = []
# tau1_err = []
# tau2 = []
# tau2_err = []
# tau_amp = []
# tau_amp_err = []
k = ["a1", "a2", "tau1", "tau2", "tau_amp"]
k = k + [x + "_err" for x in k]
print(k)
arr = np.zeros((len(k), len(files)))
for i, f in enumerate(files):
    if os.path.isfile(f):
        d = pd.read_csv(f)
        d = d.set_index('n_exp')
        # d = d.fillna(value=0.0)
        minloc = d.idxmin(axis=0)['tau_amp_err']
        print(d.keys())
        print('a2' in d.keys())
        for j, key in enumerate(k):
            if key in d.keys():
                arr[j, i] = d[key][minloc]
        # a1.append(d["a1"][minloc])
        # a1_err.append(d["a1"][minloc])
        # a2.append(d["a1"][minloc])
        # a2_err.append(d["a1"][minloc])
        # tau1.append(d["tau1"][minloc])
        # tau1_err.append(d["tau1_err"][minloc])
        # tau2.append(d["tau2"][minloc])
        # tau2_err.append(d["tau2_err"][minloc])
        # tau_amp.append(d["tau_amp"][minloc])
        # tau_amp_err.append(d["tau_amp_err"][minloc])
new_d = dict(zip(k, arr))
print(new_d)

plt.errorbar(npt, arr[4, :], yerr=arr[9, :],
    label=r'$ \left< \tau_{\text{amp.}} \right> $',
    elinewidth=0.5, capsize=2.0, marker='o', ms=6.0, lw=3.0)
plt.errorbar(npt, arr[2, :], yerr=arr[7, :],
    label=r'$ \tau_{1} $',
    elinewidth=0.5, capsize=2.0, marker='s', ms=6.0, lw=1.0)
plt.errorbar(npt, arr[3, :], yerr=arr[8, :],
    label=r'$ \tau_{2} $',
    elinewidth=0.5, capsize=2.0, marker='^', ms=6.0, lw=1.0)
plt.scatter(npt, arr[2, :], s=arr[0, :] * 500., c='#888888')
plt.scatter(npt, arr[3, :], s=arr[1, :] * 500., c='#888888')
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
# ax.set_xticks([0.0, 0])
# f_fmt = ["{:4.2f}".format(n) for n in npt]
ax.set_ylabel(r'$ \left< \tau \right> (\text{ns}) $')
ax.set_yticks([4., 3., 2.5, 2., 1.5, 1., 0.5, 0.2])
ax.set_yticklabels(["4", "3", "2.5", "2", "1.5", "1", "0.5", "0.2"])
# ax.set_xticks(npt)
# ax.set_xticklabels(f_fmt)
ax.minorticks_off()
plt.minorticks_off()
fig.savefig("{}/{}_{}_tau_scatter.pdf".format(file_path, rho_q, fwhm))
