#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trimer import Aggregate, theoretical_aggregate

class Rates():
    '''
    take a set of rates and normalise them to the hopping rate:
    e.g. a fast pre-quencher to quencher rate of 400fs^{-1}
    compared to a hopping rate of 20ps^{-1} gives a rate of 20/0.4 = 50.
    Give as equivalent times in ps!
    '''
    def __init__(self, tau_hop, tau_pool, tau_pq, tau_q,
            t_po_pq, t_pq_po, t_pq_q, t_q_pq, t_annihilation):
        self.tau_hop = tau_hop
        self.hop     = 1. / tau_hop
        self.g_pool  = 1. / tau_pool
        self.g_pq    = 1. / tau_pq
        self.g_q     = 1. / tau_q
        self.k_po_pq = 1. / t_po_pq
        self.k_pq_po = 1. / t_pq_po
        self.k_pq_q  = 1. / t_pq_q
        self.k_q_pq  = 1. / t_q_pq
        self.k_ann   = 1. / t_annihilation

class Iteration():
    def __init__(self, aggregate, rates, seed, rho_quenchers,
            n_steps, n_excitations, verbose=False, draw_frames=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.rates = rates
        self.n_st = n_steps
        self.n_e = n_excitations
        self.n_current = self.n_e
        self.rng = np.random.default_rng(seed=seed)
        self.currently_occupied = np.zeros(self.n_e, dtype=int)
        # + 2 for the pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.previous_pool = np.zeros(self.n_e, dtype=int)
        self.quenchers = np.full(len(self.aggregate.trimers), False, dtype=bool)
        self.t = 0. # probably need to write a proper init function
        self.ti = [0.]
        self.emissions = np.full((self.n_e), np.nan)
        # keep track of time spent on pre-quencher and quencher?
        self.time_on_pq = np.zeros(self.n_e, dtype=float)
        self.time_on_q = np.zeros(self.n_e, dtype=float)
        # four ways to lose population: annihilation, decay from a
        # chl pool (trimer), decay from pre-quencher, decay from quencher
        self.loss_times = np.full((4, self.n_e), np.nan)
        self.kmc_setup(rho_quenchers)
        self.transitions = np.array(self.transition_calc())
        if seed == 0:
            self.draw("frames/init_{:03d}.jpg".format(seed))
        if self.n_st > 0:
            for i in range(self.n_st):
                print("Step {}, time = {:8.3e}".format(i, self.t))
                if draw_frames:
                    if i % 100 == 0:
                        self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
                self.kmc_step()
        else:
            i = 0
            while self.n_current > 0:
                self.kmc_step()
                i += 1
                if draw_frames:
                    if i % 100 == 0:
                        self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
        self.kmc_cleanup()
            
    def transition_calc(self):
        print(self.n_sites)
        self.transitions = [[] for _ in range(self.n_sites)]
        for i in range(self.n_sites):
            t = []
            if i < self.n_sites - 2:
                # trimer (pool)
                for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                    t.append(rates.hop)
                if self.quenchers[i]:
                    t.append(rates.k_po_pq)
                else:
                    t.append(0.)
                t.append(rates.g_pool)
                t.append(rates.k_ann)
            elif i == self.n_sites - 2:
                # pre-quencher
                t.append(rates.k_pq_po)
                t.append(rates.k_pq_q)
                t.append(rates.g_pq)
            elif i == self.n_sites - 1:
                # quencher
                t.append(rates.k_q_pq)
                t.append(rates.g_q)
            self.transitions[i] = np.array(t)
            print(i, self.transitions[i], file=self.output)
        return self.transitions
    
    def kmc_setup(self, rho):
        self.n_q = int(len(self.aggregate.trimers) * rho)
        for i in range(self.n_q):
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            while self.quenchers[choice]:
                choice = self.rng.integers(low=0,
                    high=len(self.aggregate.trimers))
            self.quenchers[choice] = True

        for i in range(self.n_e):
            # only generate excitations on the pools, not quenchers
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            self.n_i[choice] += 1
            self.currently_occupied[i] = choice

    def kmc_cleanup(self):
        for i in range(self.n_e):
            # only generate excitations on the pools, not quenchers
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            self.n_i[choice] += 1
            self.currently_occupied[i] = choice

    def kmc_step(self):
        '''
        first draft of one kMC step
        extremely inefficient
        '''
        if self.n_current == 0:
            print("no excitations left!")
            return -1
        on_pq = [False for _ in range(self.n_e)]
        on_q = [False for _ in range(self.n_e)]
        for i in range(self.n_current):
            # annihilation, pool decay, pq decay, q decay
            pop_loss = [False for _ in range(4)]
            print("Currently occupied = {}".format(self.currently_occupied),
                    file=self.output)
            i = self.rng.integers(low=0, high=self.n_current)
            rand1 = self.rng.random()
            rand2 = self.rng.random()
            ind = self.currently_occupied[np.where(self.currently_occupied >= 0)][i]
            i = np.nonzero(self.currently_occupied == ind)[0][0]
            print(self.n_current, i, ind, file=self.output)
            if ind < self.n_sites - 2:
                # pool
                n = self.n_i[ind]
                if n >= 2:
                    (q, k_tot) = select_process(self.transitions[ind], rand1)
                else:
                    (q, k_tot) = select_process(self.transitions[ind][:-1], rand1)
                # uncomment next line to turn off annihilation
                # (q, k_tot) = select_process(self.transitions[ind][:-1], rand1)
                print("q = {}, kp = {}".format(q, self.transitions[ind]),
                        file=self.output)
                if (q == len(self.transitions[ind]) - 1):
                    # annihilation
                    print("po ann from trimer {}".format(ind), file=self.output)
                    self.n_i[ind] -= 1
                    self.n_current -= 1
                    self.currently_occupied[i] = -1
                    pop_loss[0] = True
                elif (q == len(self.transitions[ind]) - 2):
                    # decay
                    print("po decay from trimer {}".format(ind), file=self.output)
                    self.n_i[ind] -= 1
                    self.n_current -= 1
                    self.currently_occupied[i] = -1
                    pop_loss[1] = True
                elif (q == len(self.transitions[ind]) - 3):
                    # hop to pre-quencher
                    print("po->pq from trimer {}".format(ind), file=self.output)
                    self.n_i[ind] -= 1
                    self.n_i[-2]  += 1
                    self.currently_occupied[i] = self.n_sites - 2
                    self.previous_pool[i] = ind
                    print("i = {}, on_pq = {}, previous pool = {}".format(i, on_pq, self.previous_pool),
                            file=self.output)
                    on_pq[i] = True
                else:
                    # hop to neighbour
                    nn = self.aggregate.trimers[ind].get_neighbours()[q].index
                    print(q, ind, i, nn,
                            [self.aggregate.trimers[ind].get_neighbours()[j].index 
                            for j in range(len(
                            self.aggregate.trimers[ind].get_neighbours()))], 
                            file=self.output)
                    print("neighbour: {} to {}".format(ind, nn), file=self.output)
                    self.n_i[ind] -= 1
                    self.n_i[nn] += 1
                    self.currently_occupied[i] = nn
            elif ind == self.n_sites - 2:
                '''
                NB: no annihilation here - in principle every trimer is
                connected to a pre-quencher and through that to a quencher;
                this is just a convenient way of bookkeeping
                '''
                # pre-quencher
                (q, k_tot) = select_process(self.transitions[ind], rand1)
                if (q == 0):
                    # hop back to pool
                    # excitations on the pre-quencher are indistinguishable:
                    # pick one at random from previous_pool and put it there
                    # with previous_pool.pop() it'd be first in first out
                    pp = self.previous_pool[np.where(self.previous_pool >= 0)]
                    choice = self.rng.integers(low=0, high=len(pp))
                    self.n_i[ind] -= 1
                    self.n_i[pp[choice]] += 1
                    self.currently_occupied[i] = pp[choice]
                    self.previous_pool[i] = -1
                    print("pq->po: ind {}, choice {}, previous pool = {}".format(
                        ind, choice, self.previous_pool), file=self.output)
                    print("pq->po after delete: previous pool = {}".format(
                        self.previous_pool), file=self.output)
                elif (q == 1):
                    # hop to quencher
                    print("pq->q", file=self.output)
                    self.n_i[ind] -= 1
                    self.n_i[-1] = 1
                    self.currently_occupied[i] = self.n_sites - 1
                elif (q == 2):
                    # decay
                    print("pq decay", file=self.output)
                    self.n_i[ind] -= 1
                    self.currently_occupied[i] = -1
                    self.n_current -= 1
                    self.previous_pool[i] = -1
                    print("previous pool = {}".format(self.previous_pool)
                            , file=self.output)
                    pop_loss[2] = True
            elif ind == self.n_sites - 1:
                # quencher
                (q, k_tot) = select_process(self.transitions[ind], rand1)
                if (q == 0):
                    # hop back to pre-quencher
                    print("q->pq", file=self.output)
                    self.n_i[ind] -= 1
                    self.n_i[-2] += 1
                elif (q == 1):
                    # decay
                    print("q decay", file=self.output)
                    self.n_i[ind] -= 1
                    self.n_current -= 1
                    self.previous_pool[i] = -1
                    self.currently_occupied[i] = -1
                    print("previous pool = {}".format(self.previous_pool),
                            file=self.output)
                    pop_loss[3] = True
            # i think this is correct???? need to figure out
            self.t -= 1./ (k_tot) * np.log(rand2)
            self.ti.append(self.t)
            if pop_loss[1] or pop_loss[2]:
                # emissive decays
                # print("EMISSION AT T = {:6.3e}".format(self.t))
                self.emissions[i] = self.t
            if any(pop_loss):
                # add this time to the relevant stat
                # we only do one move at a time, so only one of pop_loss
                # can be true at any one time; hence it's safe to do [0][0]
                # self.loss_times[np.nonzero(pop_loss)[0][0]].append(self.t)
                self.loss_times[np.nonzero(pop_loss)[0][0]][self.n_current] = self.t
                # zero the time to get time between decays!
                self.t = 0.
        return

    def draw(self, filename):
        '''
        draw the current state of the system.
        pretty ugly to do this manually in opencv
        imo. but it's quick and it works
        '''
        font = cv2.FONT_HERSHEY_DUPLEX
        xmax = np.max([np.abs(t.x) for t in self.aggregate.trimers])
        scale = 4
        pic_side = (2 * scale * int(xmax + 4 * r))
        # the + 200 is to add space to put pre-quencher,
        # quencher, time and population details in
        img = np.zeros((pic_side + 200,
            pic_side + 200, 3), np.uint8)
        for i, t in enumerate(self.aggregate.trimers):
            if self.quenchers[i]:
                colour = (232, 139, 39)
            else:
                colour = (255, 255, 255)
            cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                int(scale * (t.x + xmax + 2. * r))), 
                int(scale * t.r), colour, -1)
            if (self.n_i[i] != 0):
                # coloured circle to indicate exciton
                cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                    int(scale * (t.x + xmax + 2. * r))), 
                    int(scale * 0.75 * t.r), (26, 0, 153), -1)
                cv2.putText(img, "{:1d}".format(self.n_i[i]),
                        (scale * int(t.y + xmax + 1.75 * r),
                    scale * int(t.x + xmax + 2.25 * r)),
                        font, 0.75, (255, 255, 255), 3)
        # pre-quencher
        pq_colour = (0, 94, 20)
        cv2.rectangle(img, pt1=(pic_side + 5, 100), 
            pt2=(pic_side + 55, 150),
            color=pq_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-2]),
                (pic_side + 5, 135),
                font, 1, (255, 255, 255), 2)
        cv2.putText(img, "PQ",
                (pic_side + 15, 90),
                font, 0.75, pq_colour, 3)
        # quencher
        q_colour = (60, 211, 242)
        cv2.rectangle(img, pt1=(pic_side + 5, 200), 
            pt2=(pic_side + 55, 250),
            color=q_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-1]),
                (pic_side + 5, 235),
                font, 1, (255, 255, 255), 2)
        cv2.putText(img, "Q",
                (pic_side + 20, 190),
                font, 0.75, q_colour, 3)
        cv2.putText(img, "t = {:6.2f}ps".format(self.t),
                (pic_side + 5, 300),
                font, 0.75, (255, 255, 255), 2)
        cv2.putText(img, "n = {:02d}".format(np.sum(self.n_i)),
                (pic_side + 5, 350),
                font, 0.75, (255, 255, 255), 2)
        cv2.imwrite(filename, img)

def select_process(k_p, rand):
    '''
    BKL algorithm for KMC.
    choose which configuration to jump to given a set of transition rates
    to those configurations from the current one, and a random number in [0,1].
    '''
    k_p_s = np.cumsum(k_p)
    k_tot = k_p_s[-1]
    i = 0
    while rand * k_tot > k_p_s[i]:
        i += 1
    return (i, k_tot)

def estimate_posterior_mean(loss_times):
    lambda_i = []
    for k_i in loss_times:
        k_i = k_i[k_i > 0.].flatten()
        if len(k_i) > 0:
            l = 1./len((k_i.flatten())) * np.sum(k_i.flatten())
        else:
            l = np.nan
        lambda_i.append(l)
    return lambda_i

def emission_histogram(emissions, filename):
    '''
    plot a histogram of all the emissive decays via matplotlib;
    return the set of bin values and edges so we can fit them after
    '''
    import matplotlib.pyplot as plt
    num_bins = 100
    (n, bins, patches)= plt.hist(emissions, num_bins,
            histtype="step", color='C0')
    plt.gca().set_ylabel("Counts")
    plt.gca().set_xlabel("Time (ps)")
    plt.savefig(filename)
    plt.close()
    return n, bins

def lm(no_exp, x, y, rates):
    from lmfit.models import ExponentialModel
    ''' use lmfit to a mono or biexponential '''
    exp1 = ExponentialModel(prefix='exp1')
    pars = exp1.make_params(exp1decay=1./rates.g_pool,
                            exp1amplitude=np.max(y))
    mod = exp1
    if no_exp == 2:
        exp2 = ExponentialModel(prefix='exp2')
        pars.update(exp2.make_params(exp2decay=rates.k_ann,
                                     exp2amplitude=1.))
        mod = mod + exp2
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    return out

if __name__ == "__main__":
    r = 5.
    lattice_type = "hex"
    n_iter = 8 # 434 trimers for honeycomb
    n_iterations = 100
    rho_quenchers = 0.0
    n_excitons = 50
    rates_dict = {
     'lut_eet': Rates(20., 4000., 4000., 14., 
         7., 1., 20., np.inf, 24.),
     'schlau_cohen': Rates(20., 4000., 4000., 14., 
         7., 1., 0.4, 0.4, 24.)
     }
    rates_key = 'schlau_cohen'
    rates = rates_dict[rates_key]

    path = "out/{}/{}".format(rates_key, lattice_type)
    os.makedirs(path, exist_ok=True)
    file_prefix = "{:d}_{:3.2f}_{:d}".format(
            n_iterations, rho_quenchers, n_excitons)

    agg = theoretical_aggregate(r, 0., lattice_type, n_iter)
    loss_times = []
    emissions = []
    for i in range(n_iterations):
        print("iteration {}:".format(i))
        it = Iteration(agg, rates, i,
                rho_quenchers, 0, n_excitons, False)
        print("Loss times:")
        print("annihilations: ",it.loss_times[0])
        print("pool decays: ",it.loss_times[1])
        print("pq decays: ",it.loss_times[2])
        print("q decays: ",it.loss_times[3])
        loss_times.append(it.loss_times)
        emissions.append(it.emissions)

    loss_times = np.array(loss_times)
    decays = np.array([np.transpose(loss_times[:, j, :]) for j in range(4)])
    print("Decays:", decays[decays > 0.])
    '''
    NB: i am estimating for each decay mode separately here
    which is probably wrong. tau is just a straight estimation of everything.
    Also need to figure out errors on these!
    '''
    l = np.stack([loss_times[:, i, :].flatten() for i in range(4)])
    lambda_i = estimate_posterior_mean(l)
    print("Posterior means: ", lambda_i)
    tau = np.mean(decays[decays > 0.])
    sigma_tau = np.std(decays[decays > 0.])
    print("Total posterior mean, sigma: ", tau, sigma_tau)
    np.savetxt("{}/{}_tau.dat".format(path, file_prefix), [tau, sigma_tau])
    np.savetxt("{}/{}_means.dat".format(path, file_prefix), lambda_i)
    l = np.column_stack([loss_times[:, i, :].flatten() for i in range(4)])
    np.savetxt("{}/{}_decays.dat".format(path, file_prefix), l)
    np.savetxt("{}/{}_emissions.dat".format(path, file_prefix), l)

    ax = sns.histplot(data=l, element="step", fill=False)
    # otherwise the legend will just be the array index
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax.legend(handles, ["Ann.", "Pool", "PQ", "Q"], title="Decays")
    ax.set_xlabel("Time (ps)")
    plt.axvline(x=tau, ls="--", c='k')
    plt.savefig("{}/{}_plot.pdf".format(path, file_prefix))
    plt.close()

    ax = sns.histplot(data=np.ravel(emissions), element="step",
                      binwidth=50., fill=False, stat="density")
    ax.set_xlabel("Time (ps)")
    plt.savefig("{}/{}_emissions.pdf".format(path, file_prefix))
    plt.close()

    # matplotlib histogram - output bins and vals for lmfit
    histvals, histbins = emission_histogram(np.ravel(emissions),
            "{}/{}_emissions_mpl.pdf".format(path, file_prefix))
    x = histbins[:-1] + (np.diff(histbins) / 2.)
    mono_fit = lm(1, x, histvals, rates)
    print(mono_fit.fit_report())
    bi_fit = lm(2, x, histvals, rates)
    print(bi_fit.fit_report())
    plt.hist(np.ravel(emissions), 200, histtype="step",
             color='C0', label="hist", log=True)
    plt.plot(x, mono_fit.best_fit, color='C1', label="mono fit")
    plt.plot(x, bi_fit.best_fit, color='C2', label="bi fit")
    plt.legend()
    plt.savefig("{}/{}_fit.pdf".format(path, file_prefix))
    plt.close()

    fig = mono_fit.plot(xlabel="Time (ps)", ylabel="Counts")
    axes = fig.gca()
    axes.set_yscale('log')
    plt.savefig("{}/{}_mono.pdf".format(path, file_prefix))
    plt.close()

    fig = bi_fit.plot(xlabel="Time (ps)", ylabel="Counts")
    axes = fig.gca()
    axes.set_yscale('log')
    plt.savefig("{}/{}_bi.pdf".format(path, file_prefix))
    plt.close()
