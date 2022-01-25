#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trimer import Aggregate, theoretical_aggregate

class Model():
    '''
    Simple container for a set of relevant rates - tau_x is the time constant
    of process x. e.g. we assume "pool" chlorophylls decay on a timescale
    of a few nanoseconds, whereas the decay of a carotenoid quencher might
    be 10-15ps. Give these in ps!
    emissive is a bool list I use to track which of the decays are emissive.
    Not currently useful but might be.
    '''
    def __init__(self, tau_hop, tau_pool, tau_pq, tau_q,
            t_po_pq, t_pq_po, t_pq_q, t_q_pq, t_annihilation,
            emissive):
        self.tau_hop  = tau_hop
        self.hop      = 1. / tau_hop
        self.g_pool   = 1. / tau_pool
        self.g_pq     = 1. / tau_pq
        self.g_q      = 1. / tau_q
        self.k_po_pq  = 1. / t_po_pq
        self.k_pq_po  = 1. / t_pq_po
        self.k_pq_q   = 1. / t_pq_q
        self.k_q_pq   = 1. / t_q_pq
        self.k_ann    = 1. / t_annihilation
        self.emissive = emissive

class Iteration():
    '''
    Take an aggregate - either one constructed from an experimental image
    or a generated one - populate it with a fraction of quenchers and a
    set of excitons (the number of excitons is dependent on fluence), and
    run kinetic Monte Carlo until all the excitons are gone.
    Record the intervals between exciton losses and whether these were
    emissive or not.
    '''
    def __init__(self, aggregate, model, seed, rho_quenchers,
            n_steps, fluence, verbose=False, draw_frames=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.model = model
        self.n_st = n_steps
        self.rng = np.random.default_rng(seed=seed)
        # + 2 for the pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.quenchers = np.full(len(self.aggregate.trimers), False, dtype=bool)
        self.kmc_setup(rho_quenchers, fluence)
        # self.n_e and self.currently_occupied are set in kmc_setup
        self.n_current = self.n_e
        self.previous_pool = np.zeros(self.n_e, dtype=int)
        self.t = 0.
        self.ti = [0.]
        self.loss_times = np.full((self.n_e), np.nan)
        # four ways to lose population: annihilation, decay from a
        # chl pool (trimer), decay from pre-quencher, decay from quencher
        self.decay_type = np.full((self.n_e), -1, dtype=int)
        self.transitions = self.transition_calc()
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
            
    def transition_calc(self):
        '''
        generate a list of numpy arrays where each array is the set
        of all possible transition rates for the corresponding trimer.
        kMC then picks a trimer and looks up these rates to determine moves.
        it's a list of np arrays because they're ragged atm; different
        trimers can have different numbers of neighbours. this is
        sufficient for now but see the note below otherwise
        '''
        self.transitions = [[] for _ in range(self.n_sites)]
        max_neighbours = np.max(np.fromiter((len(x.get_neighbours()) 
                     for x in self.aggregate.trimers), int))
        # note: if it was necessary to make all these the same length,
        # e.g. if it was in C++ or needed compiling down or whatever,
        # you could do this by padding the neighbours left with zeroes!
        # t = np.zeroes(max_neighbours + 3, dtype=float)
        # for j in range(n_neigh):
        #     t[max_neighbours - (j + 1)] = model.hop
        # for max_neighbours = 6, n_neigh = 3, this gives
        # [0., 0., 0., hop, hop, hop]
        # for the pre-quencher and quencher you'd have to do t[-3] etc.
        for i in range(self.n_sites):
            t = []
            if i < self.n_sites - 2:
                # trimer (pool)
                n_neigh = len(self.aggregate.trimers[i].get_neighbours())
                for j in range(n_neigh):
                    t.append(model.hop)
                if self.quenchers[i]:
                    t.append(model.k_po_pq)
                else:
                    t.append(0.)
                t.append(model.g_pool)
                t.append(model.k_ann)
            elif i == self.n_sites - 2:
                # pre-quencher
                t.append(model.k_pq_po)
                t.append(model.k_pq_q)
                t.append(model.g_pq)
            elif i == self.n_sites - 1:
                # quencher
                t.append(model.k_q_pq)
                t.append(model.g_q)
            self.transitions[i] = np.array(t)
            print(i, self.transitions[i], file=self.output)
        return self.transitions
    
    def kmc_setup(self, rho, fluence):
        '''
        randomly allocate quenchers based on ρ_q
        '''
        self.n_q = int(len(self.aggregate.trimers) * rho)
        for i in range(self.n_q):
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            while self.quenchers[choice]:
                choice = self.rng.integers(low=0,
                    high=len(self.aggregate.trimers))
            self.quenchers[choice] = True

        '''
        now loop over trimers and excite based on the
        absorption cross section σ and the fluence.
        σ @ 480nm \approx 1.1E-14 - might need changing!
        '''
        l = fluence * 1.1E-14
        penalise = True
        '''
        the first block does not penalise multiple excitation and
        just places a number of excitations on each trimer which is
        drawn from the corresponding Poisson distribution.
        to penalise multiple excitation we have to do a bit more.
        first get a total number of excitations for this iteration.
        then thresh makes it less and less likely multiple excitations
        will be placed on the same trimer; change this to penalise more/less.
        this way probably isn't the most efficient way of placing them,
        but it's python, so that's the least of our worries speed-wise
        '''
        if penalise is False:
            self.n_e = 0
            occupied = []
            for i in range(len(self.aggregate.trimers)):
                p = self.rng.poisson(lam=l)
                self.n_i[i] = p
                self.n_e += p
                for k in range(p):
                    occupied.append(i)
        else:
            self.n_e = 0
            photons = 0
            occupied = []
            for i in range(len(self.aggregate.trimers)):
                photons += self.rng.poisson(lam=l)
            attempts = 0
            while attempts < photons:
                i = self.rng.integers(low=0,
                        high=len(self.aggregate.trimers))
                # no penalty if trimer's unpopulated, obviously
                if self.n_i[i] == 0:
                    self.n_i[i] += 1
                    self.n_e += 1
                    occupied.append(i)
                else:
                    # penalise
                    # note to self: time-based penalty?
                    # i.e. a trimer can only absorb one photon
                    # every 5ps or something?
                    r = self.rng.random()
                    sigma_ratio = 3.
                    thresh = (24. - (1. + sigma_ratio) * self.n_i[i]) / 24.
                    if thresh > 0. and r < thresh:
                        self.n_i[i] += 1
                        occupied.append(i)
                        self.n_e += 1
                attempts += 1
        self.currently_occupied = np.array(occupied)

    def kmc_step(self):
        '''
        first draft of one kMC step
        extremely inefficient
        '''
        if self.n_current == 0:
            print("no excitations left!")
            return -1
        for j in range(self.n_current):
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
                    # second order! need to draw another number
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
                    print("i = {}, on_pq, previous pool = {}".format(i,
                        self.previous_pool), file=self.output)
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
            if any(pop_loss):
                # add this time to the relevant stat
                # we only do one move at a time, so only one of pop_loss
                # can be true at any one time; hence it's safe to do [0][0]
                # self.loss_times[np.nonzero(pop_loss)[0][0]].append(self.t)
                dt = np.nonzero(pop_loss)[0][0]
                print("dt = {}".format(dt), file=self.output)
                print("loss time = {}".format(self.t), file=self.output)
                self.loss_times[self.n_current] = self.t
                self.decay_type[self.n_current] = dt
                # zero the time to get time between decays!
                self.t = 0.
        return

    def draw(self, filename):
        import cv2
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

def emission_histogram(emissions, filename, num_bins=200):
    '''
    plot a histogram of all the emissive decays via matplotlib;
    return the set of bin values and edges so we can fit them after
    '''
    import matplotlib.pyplot as plt
    (n, bins, patches)= plt.hist(emissions, num_bins,
            histtype="step", color='C0')
    plt.gca().set_ylabel("Counts")
    plt.gca().set_xlabel("Time (ps)")
    plt.savefig(filename)
    plt.close()
    return n, bins

def lm(no_exp, x, y, model):
    from lmfit.models import ExponentialModel
    ''' use lmfit to a mono or biexponential '''
    exp1 = ExponentialModel(prefix='exp1')
    pars = exp1.make_params(exp1decay=1./model.g_pool,
                            exp1amplitude=np.max(y))
    mod = exp1
    if no_exp == 2:
        exp2 = ExponentialModel(prefix='exp2')
        pars.update(exp2.make_params(exp2decay=1./model.k_ann,
                                     exp2amplitude=np.max(y)))
        mod = mod + exp2
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    return out
    
if __name__ == "__main__":
    r = 5.
    lattice_type = "hex"
    n_iter = 8 # 434 trimers for honeycomb
    n_iterations = 1000
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # lazy - make a different main file to loop over models and fluences
    # annihilation, pool decay, pq decay, q decay
    model_dict = {
     'lut_eet': Model(20., 3800., 3800., 14., 
         7., 1., 20., np.inf, 48., [False, True, True, False]),
     'schlau_cohen': Model(20., 3800., 3800., 14., 
         7., 1., 0.4, 0.4, 48., [False, True, True, False])
     }
    model_key = 'lut_eet'
    model = model_dict[model_key]
    mono_tau = []
    bi_tau = []

    for fluence in fluences:
        print("Fluence = {:4.2e}, n_iterations = {:d}".format(
            fluence, n_iterations))
        path = "out/{}/{}".format(model_key, lattice_type)
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:d}_{:3.2f}_{:4.2e}".format(
                n_iterations, rho_quenchers, fluence)
        decay_filename = "{}/{}_decays.dat".format(path, file_prefix)
        emission_filename = "{}/{}_emissions.dat".format(path, file_prefix)

        # note - second parameter here is the nn cutoff. set to 0 to
        # disable excitation hopping between trimers
        agg = theoretical_aggregate(r, 0., lattice_type, n_iter)
        n_es = []
        means = []
        stddevs = []
        emission_means = []
        emission_stddevs = []
        yields = []
        decay_file     = open(decay_filename, mode='w')
        emissions_file = open(emission_filename, mode='w')
        for i in range(n_iterations):
            verbose = False
            emissions = []
            it = Iteration(agg, model, i,
                    rho_quenchers, 0, fluence, verbose=verbose)
            n_es.append(it.n_e)
            for k in range(it.n_e):
                print("{:1.5e} {:1d}".format(it.loss_times[k], it.decay_type[k]), 
                        file=decay_file)
                if model.emissive[it.decay_type[k]] is True:
                    print("{:1.5e}".format(it.loss_times[k]), file=emissions_file)
                    emissions.append(it.loss_times[k])
            means.append(np.mean(it.loss_times))
            stddevs.append(np.std(it.loss_times))
            emission_means.append(np.mean(emissions))
            emission_stddevs.append(np.std(emissions))
            yields.append(emission_means[-1]/means[-1])
            if verbose is True:
                print("Iteration {:d}".format(i))
                print("=== μ, σ ===")
                print(means[-1], stddevs[-1])
                print("=== EMISSION μ, σ ===")
                print(emission_means[-1], emission_stddevs[-1])
            else:
                width = os.get_terminal_size().columns - 20
                print("\rProgress: [{0}{1}] {2}%".format(
                    '█'*int((i + 1) * width/n_iterations),
                    ' '*int(width - ((i + 1) * width/n_iterations)),
                    int((i + 1) * 100 / n_iterations)), end='')

        print() # newline after progress bar
        decay_file.close()
        emissions_file.close()
        '''
        tau is just a straight estimation of everything
        mean of means and mean of emissive means reported separately
        statistics of these????? are they valid things to report???
        '''
        decays = np.loadtxt(decay_filename)
        emissions = np.loadtxt(emission_filename)
        tau = np.mean(decays[:, 0])
        sigma_tau = np.std(decays[:, 0])
        print("Total μ, σ: ", tau, sigma_tau)
        print("μ, σ of means: ", np.mean(means),
                np.std(means))
        print("μ, σ of emission means: ", np.mean(emission_means),
                np.std(emission_means))
        print("μ, σ of excitation numbers: ", np.mean(n_es),
                np.std(n_es))
        print("Average fraction of excited trimers ρ_exc: ",
                np.mean(n_es) / len(it.aggregate.trimers))

        np.savetxt("{}/{}_total_mean_std.dat".format(path, file_prefix),
                [tau, sigma_tau])
        np.savetxt("{}/{}_total_emission_mean_std.dat".format(path, file_prefix),
                [np.mean(emissions), np.std(emissions)])
        np.savetxt("{}/{}_n_es.dat".format(path, file_prefix), n_es)
        np.savetxt("{}/{}_means.dat".format(path, file_prefix), means)
        np.savetxt("{}/{}_stddevs.dat".format(path, file_prefix), stddevs)
        np.savetxt("{}/{}_emission_means.dat".format(path, 
            file_prefix), emission_means)
        np.savetxt("{}/{}_emission_stddevs.dat".format(path, 
            file_prefix), emission_stddevs)

        '''
        NB: latex will work in column names and captions (e.g.
        in typedict below, if it were needed). just have e.g. 0: r'$ \sigma $',
        '''
        decay_pd = pd.DataFrame(decays, columns=["Time (ps)", "Decay type"])
        # typedict = {"Ann": 0., "Pool": 1., "PQ":, 2., "Q": 3.}
        typedict = {0.: "Ann.", 1.: "Pool", 2.: "PQ", 3: "Q"}
        decay_pd = decay_pd.replace({"Decay type": typedict})
        ax = sns.histplot(data=decay_pd, x="Time (ps)", hue="Decay type",
                element="step", fill=False)
        plt.axvline(x=tau, ls="--", c='k')
        plt.savefig("{}/{}_plot.pdf".format(path, file_prefix))
        plt.close()

        ax = sns.histplot(data=decays[:, 0], element="step",
                          binwidth=25., fill=False)
        ax.set_xlabel("Time (ps)")
        plt.savefig("{}/{}_emissions.pdf".format(path, file_prefix))
        plt.close()

        # matplotlib histogram - output bins and vals for lmfit
        histvals, histbins = emission_histogram(decays[:, 0],
                "{}/{}_emissions_mpl.pdf".format(path, file_prefix))
        x = histbins[:-1] + (np.diff(histbins) / 2.)
        try:
            mono_fit = lm(1, x, histvals, model)
            print(mono_fit.fit_report())
            fig = mono_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            axes.set_yscale('log')
            plt.savefig("{}/{}_mono.pdf".format(path, file_prefix))
            plt.close()
            comps = mono_fit.best_values
            print(comps)
            mono_tau.append([fluence, comps['exp1decay']])
            print("Mono-exponential <tau> = {:8.3f}".format(comps['exp1decay']))
        except ValueError:
            print("Mono-exponential fit failed!")
            pass
        try:
            bi_fit = lm(2, x, histvals, model)
            print(bi_fit.fit_report())
            fig = bi_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            axes.set_yscale('log')
            plt.savefig("{}/{}_bi.pdf".format(path, file_prefix))
            plt.close()
            comps = bi_fit.best_values
            avg_tau = (comps['exp1decay'] * comps['exp1amplitude'] \
                + comps['exp2decay'] * comps['exp2amplitude']) \
                / (comps['exp1amplitude'] + comps['exp2amplitude'])
            bi_tau.append([fluence, avg_tau])
            print("Bi-exponential <tau> = {:8.3f}".format(avg_tau))
        except ValueError:
            print("Bi-exponential fit failed!")
            pass
        
    np.savetxt("{}/mono_tau.dat".format(path), np.array(mono_tau))
    np.savetxt("{}/bi_tau.dat".format(path), np.array(bi_tau))
