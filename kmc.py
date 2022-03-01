#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from trimer import Aggregate

class Pulse():
    def __init__(self, fwhm, mu):
        self.fwhm = fwhm # ps
        self.mu = mu
        self.sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
        self.t = np.arange(0., 2. * mu, 1.)
        self.ft = 1. / (self.sigma * np.sqrt(2. * np.pi)) * \
    np.exp(- (self.t - self.mu)**2 / (np.sqrt(2.) * self.sigma)**2)

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
            emissive, decay_on_pq, ann_on_pq):
        self.tau_hop        = tau_hop
        self.hop            = 1. / tau_hop
        self.g_pool         = 1. / tau_pool
        self.g_pq           = 1. / tau_pq
        self.g_q            = 1. / tau_q
        self.k_po_pq        = 1. / t_po_pq
        self.k_pq_po        = 1. / t_pq_po
        self.k_pq_q         = 1. / t_pq_q
        self.k_q_pq         = 1. / t_q_pq
        self.k_ann          = 1. / t_annihilation
        self.emissive       = emissive
        self.decay_on_pq    = decay_on_pq
        self.ann_on_pq      = ann_on_pq

class Iteration():
    import numpy as np
    import cv2
    '''
    Take an aggregate - either one constructed from an experimental image
    or a generated one - populate it with a fraction of quenchers and a
    set of excitons (the number of excitons is dependent on fluence), and
    run kinetic Monte Carlo until all the excitons are gone.
    Record the intervals between exciton losses and whether these were
    emissive or not.
    '''
    def __init__(self, aggregate, model, pulse, rho_quenchers,
            n_iter, path, fluence, verbose=False, draw_frames=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.model = model
        self.pulse = pulse
        self.fluence = fluence
        self.n_iterations = n_iter
        # self.rng = np.random.default_rng(seed=seed)
        self.rho_quenchers = rho_quenchers
        # pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.quenchers = np.full(len(self.aggregate.trimers), False, dtype=bool)
        self.quencher_setup(self.rho_quenchers)
        self.base_rates = self.transition_calc()
        self.rates = self.base_rates.copy()
        self.pq = []
        self.q = []
        self.t = 0.
        self.t_tot = 0.
        self.n_current = 0
        self.loss_times = []
        # four ways to lose population: annihilation, decay from a
        # chl pool (trimer), decay from pre-quencher, decay from quencher
        self.decay_type = []
        self.write_arrays(path)
        # initialise the rates!
        # for i in range(self.n_sites):
        #     self.update_rates(i, self.n_i[i], self.t_tot)
        # if seed == 0:
        #     self.draw("frames/init_{:03d}.jpg".format(seed))
        # if self.n_st > 0:
        #     for i in range(self.n_st):
        #         print("Step {}, time = {:8.3e}, t_tot = {:8.3e}".format(i, self.t, self.t_tot))
        #         if draw_frames:
        #             if i % 100 == 0:
        #                 self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
        #         self.kmc_step()
        # else:
        #     i = 0
        #     res = 0
        #     while self.t_tot < 2. * self.pulse.mu:
        #         self.mc_step(1.)
        #     for i in range(self.n_sites):
        #         self.update_rates(i, self.n_i[i], self.t_tot)
        #     while res == 0:
        #         res = self.kmc_step()
        #         i += 1
        #         if draw_frames:
        #             if i % 100 == 0:
        #                 self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
        # self.loss_times = np.array(self.loss_times)
        # self.decay_type = np.array(self.decay_type)
            
    def quencher_setup(self, rho_quench):
        '''
        randomly allocate quenchers based on ρ_q
        '''
        self.n_q = int(len(self.aggregate.trimers) * rho_quench)
        for i in range(self.n_q):
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            while self.quenchers[choice]:
                choice = self.rng.integers(low=0,
                    high=len(self.aggregate.trimers))
            self.quenchers[choice] = True

    def transition_calc(self):
        '''
        generate a list of numpy arrays where each array is the set
        of all possible transition rates for the corresponding trimer.
        the order's arbitrary, but has to be followed in the Metropolis
        and kinetic Monte Carlo simulations below; the function `move`
        carries out the process based on which of these rates we pick.
        note that the final two sets of rates are for the pre-quenching
        and quenching states - we assume these states are connected to
        certain trimers at random, but that they are identical for each
        '''
        self.max_neighbours = np.max(np.fromiter((len(x.get_neighbours()) 
                     for x in self.aggregate.trimers), int))
        self.base_rates = np.zeros((self.n_sites, self.max_neighbours + 4),\
                dtype=float)
        for i in range(self.n_sites):
            t = self.base_rates[i].copy()
            # first element is generation
            if i < self.n_sites - 2:
                # trimer (pool)
                n_neigh = len(self.aggregate.trimers[i].get_neighbours())
                for j in range(n_neigh):
                    t[self.max_neighbours - (j)] = self.model.hop
                # if self.quenchers[i]:
                t[self.max_neighbours + 1] = self.model.k_po_pq
                # else:
                #     t[self.max_neighbours + 1] = 0.
                t[self.max_neighbours + 2] = self.model.g_pool
                t[self.max_neighbours + 3] = self.model.k_ann
            elif i == self.n_sites - 2:
                if (self.rho_quenchers != 0):
                # pre-quencher
                    t[self.max_neighbours] = self.model.k_pq_po
                    t[self.max_neighbours + 1] = self.model.k_pq_q
                    t[self.max_neighbours + 2] = self.model.g_pq
                    t[self.max_neighbours + 3] = self.model.k_ann
            elif i == self.n_sites - 1:
                # quencher
                if (self.rho_quenchers != 0):
                    t[self.max_neighbours + 1] = self.model.k_q_pq
                    t[self.max_neighbours + 2] = self.model.g_q
                    t[self.max_neighbours + 3] = self.model.k_ann
            self.base_rates[i] = t
            # print(i, self.base_rates[i], file=self.output)
        return self.base_rates

    def update_rates(self, index, n, t):
        '''
        the base set of rates calculated in transition_calc are actually
        population-dependent for the most part; photon absorption is also
        time-dependent. this function takes the time and the population
        and updates the rates for a given trimer accordingly
        NB: for speed, it'd be good to update the cumulative set of rates
        here as well, then we don't have to do np.cumsum() every time
        '''
        # if n = 0 the rates will go to zero and stay there - prevent this
        self.rates[index] = self.base_rates[index].copy()
        if t < 2 * self.pulse.mu:
            # generation term non-zero
            t_index = int(t)
            ft = self.pulse.ft[t_index]
            # σ @ 480nm \approx 1.1E-14
            # sigma_ratio = σ_{se} / σ
            xsec = 1.1E-14
            sigma_ratio = 1.
            n_pigments = 24.
            if ((1 + sigma_ratio) * n) <= n_pigments:
                '''
                \int ft dt = 1 so xsec * fluence * ft
                over the whole pulse equals the average number
                of absorbed photons, per trimer.
                '''
                self.rates[index][0] = xsec * self.fluence * ft * \
                ((n_pigments - (1 + sigma_ratio) * n) / n_pigments)
        for k in range(1, len(self.rates[index]) - 1):
            self.rates[index][k] *= n # account for number of excitations
        '''
        annihilation can happen on the pre-quencher or quencher,
        in principle. but only if they're on the same trimer!
        check this here. this should actually sum over every count
        larger than 1 and multiply by all the factors, i think
        '''
        if index == self.n_sites - 2:
            if len(self.pq) > 0:
                (uniques, counts) = np.unique(self.pq, return_counts=True)
                n_max = np.max(counts)
            else:
                n_max = 0
            ann_fac = n_max * (n_max - 1) / 2.
        elif index == self.n_sites - 1:
            if len(self.q) > 0:
                (uniques, counts) = np.unique(self.q, return_counts=True)
                n_max = np.max(counts)
            else:
                n_max = 0
            ann_fac = n_max * (n_max - 1) / 2.
        else:
            ann_fac = n * (n - 1) / 2.
        self.rates[index][-1] *= ann_fac

    def move(self, i, q, rates, pop_loss):
        '''
        this function is the guts of both the Metropolis and kinetic
        Monte Carlo code below - take a site (whichever trimer we picked
        at random), an index into the rate array (the index determines
        which move we're doing, based on the order in transition_calc above),
        and carry out the corresponding process. Note that the possible
        processes are different for the pre-quencher and quencher as above.
        pop_loss is updated to tell me if any loss of population occurred
        as a result, and if so what type.
        '''
        if i < self.n_sites - 2:
            # pool
            # if (q == 0):
                # nothing
                # print("nothing")
            if (q == 0):
                # generation
                self.n_i[i] += 1
                self.n_current += 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                print("generation on {}".format(i), file=self.output)
            if (q == len(rates) - 1):
                # annihilation
                print("po ann from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_current -= 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                pop_loss[0] = True
            if (q == len(rates) - 2):
                # decay
                print("po decay from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_current -= 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                pop_loss[1] = True
            if (q == len(rates) - 3):
                # hop to pre-quencher
                print("po->pq from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_i[-2]  += 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.update_rates(-2, self.n_i[-2], self.t_tot)
                self.pq.append(i) # keep track of which trimer it came from
                print("i = {} -> pq".format(i), file=self.output)
            if (0 < q < self.max_neighbours):
                # hop to neighbour
                nn = self.aggregate.trimers[i].get_neighbours()[q - 1].index
                print(q, i, nn,
                        [self.aggregate.trimers[i].get_neighbours()[p].index 
                        for p in range(len(
                        self.aggregate.trimers[i].get_neighbours()))], 
                        file=self.output)
                print("neighbour: {} to {}".format(i, nn), file=self.output)
                self.n_i[i] -= 1
                self.n_i[nn] += 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.update_rates(nn, self.n_i[nn], self.t_tot)
        elif i == self.n_sites - 2:
            '''
            pre-quencher
            note that excitations can be created here, too
            need to fix the annihilation behaviour on this and the quencher,
            since atm excitations can annihilate on these even if they didn't
            come from the same trimer!!!
            '''
            # if (q == 0):
                # print("nothing")
            if (q == 0):
                self.n_i[i] += 1
                self.n_current += 1
                print("generation on pq", file=self.output)
                # choose a random trimer for it to hop to
                choice = self.rng.integers(low=0, high=self.n_sites - 2)
                self.pq.append(choice)
                self.update_rates(i, self.n_i[i], self.t_tot)
            if (q == len(rates) - 4):
                # hop back to pool
                # excitations on the pre-quencher are indistinguishable:
                # pick one at random from pq and put it back
                # with pq.pop() it'd be first in first out
                choice = self.rng.integers(low=0, high=len(self.pq))
                print("pq->po: choice {} of {}, previous = {}".format(
                    choice, len(self.pq), self.pq[choice]),
                    file=self.output)
                self.n_i[i] -= 1
                self.n_i[self.pq[choice]] += 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.update_rates(self.pq[choice], self.n_i[self.pq[choice]], 
                        self.t_tot)
                self.pq.remove(choice)
                print("pq->po after delete: pq = {}".format(
                    self.pq), file=self.output)
            elif (q == len(rates) - 3):
                # hop to quencher
                print("pq->q", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.pq))
                self.n_i[i] -= 1
                self.n_i[-1] += 1
                self.q.append(self.pq[choice])
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.update_rates(-1, self.n_i[-1], self.t_tot)
                self.pq.remove(choice)
            elif (q == len(rates) - 2):
                # decay
                print("pq decay", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.pq))
                self.n_i[i] -= 1
                print("previous = {}".format(self.pq[choice]), 
                        file=self.output)
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.pq.remove(choice)
                pop_loss[2] = True
            elif (q == len(rates) - 1):
                # annihilation
                print("pq ann", file=self.output)
                '''
                annihilation can only occur here if two excitons are on
                the same quenching trimer. because i only consider one
                pre-quencher since they're all identical, this means we
                have to be careful about annihilation! first, find which
                trimer(s) have multiple excitons currently on the pre-quencher.
                then pick one of those multiples and remove the first exciton
                on that trimer.
                '''
                (uniques, counts) = np.unique(self.pq, return_counts=True)
                multiples = np.nonzero(counts > 1)[0]
                choice = self.rng.integers(low=0, high=len(multiples))
                self.n_i[i] -= 1
                # uniques[multiples][choice] gives us which of the trimers
                # with multiple excitons on the pre-quencher is annihilating.
                # np.where()[0][0] gives us the first index of that trimer
                # on self.pq
                index = np.where(self.pq == uniques[multiples][choice])[0][0]
                print("previous = {}".format(self.pq[index]), file=self.output)
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.pq.remove(index)
                pop_loss[0] = True
        elif i == self.n_sites - 1:
            '''
            quencher
            we can generate excitations here too in principle
            '''
            # if (q == 0):
            #     print("nothing")
            if (q == 0):
                self.n_i[i] += 1
                self.n_current += 1
                print("generation on q", file=self.output)
                choice = self.rng.integers(low=0, high=self.n_sites - 2)
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.q.append(choice)
            if (q == len(rates) - 3):
                # hop back to pre-quencher
                print("q->pq", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.q))
                self.n_i[i] -= 1
                self.n_i[-2] += 1
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.update_rates(-2, self.n_i[-2], self.t_tot)
                self.pq.append(self.q[choice])
                self.q.remove(choice)
            elif (q == len(rates) - 2):
                # decay
                print("quencher decay, n = {}".format(self.n_i[i]),
                        file=self.output)
                choice = self.rng.integers(low=0, high=len(self.q))
                self.n_i[i] -= 1
                print("previous chl was = {}".format(self.q[choice]),
                        file=self.output)
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.q.remove(choice)
                pop_loss[3] = True
            elif (q == len(rates) - 1):
                # annihilation
                print("quencher annihilation", file=self.output)
                (uniques, counts) = np.unique(self.q, return_counts=True)
                multiples = np.nonzero(counts > 1)[0]
                choice = self.rng.integers(low=0, high=len(multiples))
                self.n_i[i] -= 1
                index = np.where(self.q == uniques[multiples][choice])[0][0]
                print("previous chl was = {}".format(self.q[index]),
                        file=self.output)
                self.update_rates(i, self.n_i[i], self.t_tot)
                self.q.remove(index)
                pop_loss[0] = True

    def mc_step(self, dt):
        '''
        for a given time step, we can calculate the probability of a Poisson
        process with the associated rate happening within that time step.
        (this is just an exponential distribution).
        these can be set as the acceptance probabilities for Metropolis,
        since we don't have to worry about detailed balance (this is not an
        equilibrium process!). we loop over trimers, propose a move at random
        (absorption, hop, decay, annihilation if n > 1),
        then run it through Metropolis.
        '''
        if self.rho_quenchers != 0.:
            n_attempts = self.n_sites
        else:
            n_attempts = self.n_sites - 2
        for i in range(n_attempts):
            # annihilation, pool decay, pq decay, q decay
            pop_loss = [False for _ in range(4)]
            trimer = self.rng.integers(low=0, high=n_attempts)
            # as the time changes, so does the generation rate - update this
            self.update_rates(trimer, self.n_i[trimer], self.t_tot)
            rates = self.rates[trimer]
            # print(rates)
            probs = np.fromiter((rate * np.exp(-rate * dt) for rate in rates),
                dtype=float) # acceptance probabilities for Metropolis
            # ignore moves with zero rate
            choice = self.rng.integers(low=0, high=np.count_nonzero(probs))
            proposed_move = np.nonzero(probs)[0][choice]
            rand = self.rng.random()
            if (rand < probs[proposed_move]):
                # carry out the move
                # print("before n, rates", self.n_i[trimer], self.rates[trimer])
                self.move(trimer, proposed_move, rates, pop_loss)
                # print("after n, rates", self.n_i[trimer], self.rates[trimer])
                print('Move accepted. index = {:d}, p = {:f}, '\
                        'rand = {:f}, t_tot = {:6.3f}, '\
                        'n_current = {:d}'.format(proposed_move,
                            probs[proposed_move], rand, self.t_tot,
                            self.n_current), file=self.output)
                '''
                not sure that the decay time is treated properly here?
                '''
                if any(pop_loss):
                    # add this time to the relevant stat
                    # we only do one move at a time, so only one of pop_loss
                    # can be true at any one time; hence it's safe to do [0][0]
                    decay_type = np.nonzero(pop_loss)[0][0]
                    print("decay type = {}".format(decay_type),
                            file=self.output)
                    print("loss time = {}".format(self.t), file=self.output)
                    self.loss_times.append(self.t_tot)
                    self.decay_type.append(decay_type)
                    # zero the time to get time between decays!
                    self.t = 0.
        self.t += dt
        self.t_tot += dt
    
    def kmc_step(self):
        '''
        after the pulse we have no exciton generation - switch to
        kinetic Monte Carlo to simulate annihilations and decays
        '''
        if self.n_current == 0 and self.t_tot >= 2. * self.pulse.mu:
            return -1
        pop_loss = [False for _ in range(4)]
        rand1 = self.rng.random()
        rand2 = self.rng.random()
        if np.any(self.rates):
            (i, q, k_tot) = self.bkl(rand1)
            rates = self.rates[i]
            self.move(i, q, rates, pop_loss)
            self.t -= 1./ (k_tot) * np.log(rand2)
            self.t_tot += self.t
            if any(pop_loss):
                # add this time to the relevant stat
                # we only do one move at a time, so only one of pop_loss
                # can be true at any one time; hence it's safe to do [0][0]
                decay_type = np.nonzero(pop_loss)[0][0]
                print("decay type = {}".format(decay_type),
                        file=self.output)
                print("loss time = {}".format(self.t), file=self.output)
                self.loss_times.append(self.t_tot)
                self.decay_type.append(decay_type)
                # zero the time to get time between decays!
                self.t = 0.
        else:
            print("all rates zero. n_current = {}, t_tot = {}".format(self.n_current, self.t_tot),
                    file=self.output)
        return 0

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
        pic_side = (2 * scale * int(xmax + 4 *
            self.aggregate.trimers[0].r))
        # the + 200 is to add space to put pre-quencher,
        # quencher, time and population details in
        img = np.zeros((pic_side + 200,
            pic_side + 200, 3), np.uint8)
        for i, t in enumerate(self.aggregate.trimers):
            if self.quenchers[i]:
                colour = (232, 139, 39)
            else:
                colour = (255, 255, 255)
            cv2.circle(img, (int(scale * (t.y + xmax + 2. * t.r)),
                int(scale * (t.x + xmax + 2. * t.r))),
                int(scale * t.r), colour, -1)
            if (self.n_i[i] != 0):
                # coloured circle to indicate exciton
                cv2.circle(img, (int(scale * (t.y + xmax + 2. * t.r)),
                    int(scale * (t.x + xmax + 2. * t.r))),
                    int(scale * 0.75 * t.r), (26, 0, 153), -1)
                cv2.putText(img, "{:1d}".format(self.n_i[i]),
                        (scale * int(t.y + xmax + 1.75 * t.r),
                    scale * int(t.x + xmax + 2.25 * t.r)),
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

    def bkl(self, rand):
        '''
        BKL algorithm for KMC.
        choose which configuration to jump to given a set of transition rates
        to those configurations from the current one, and a random number in [0,1].
        '''
        k_p_s = np.cumsum(self.rates.flatten())
        k_tot = k_p_s[-1]
        '''
        binary search to find the correct process to execute
        we want the first index where k_p_s[i] >= rand * k_tot
        '''
        l = 0
        r = len(k_p_s) - 1
        while l < r:
            m = (l + r) // 2
            if k_p_s[m] < rand * k_tot:
                l = m + 1
            else:
                r = m
        # each set of rates is the same length
        (n, q) = np.divmod(l, len(self.base_rates[0]))
        print("bkl: l, n, q = ", l, n, q)
        return (n, q, k_tot)

    def write_arrays(self, path, binwidth=50., max_time=10000.):
        self.params_file = "{}/params".format(path)
        neighbours_file = "{}/neighbours.dat".format(path)
        rates_file = "{}/base_rates.dat".format(path)
        pulse_file = "{}/pulse.dat".format(path)
        # np.savetxt(rates_file, self.base_rates.flatten(order='F'))
        np.savetxt(rates_file, self.base_rates.flatten())
        neighbours = np.zeros((self.n_sites - 2, self.max_neighbours))
        for i in range(self.n_sites - 2):
            for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                neighbours[i][self.max_neighbours - j] = self.aggregate.trimers[i].get_neighbours()[j]
        np.savetxt(neighbours_file, neighbours.flatten(order='F'))
        with open(self.params_file, 'w') as f:
            f.write("{:d}\n".format(self.n_iterations))
            f.write("{:d}\n".format(self.n_sites))
            f.write("{:d}\n".format(self.max_neighbours))
            f.write("{:f}\n".format(self.rho_quenchers))
            f.write("{:f}\n".format(self.fluence))
            f.write("{:f}\n".format(self.pulse.mu))
            f.write("{:f}\n".format(self.pulse.fwhm))
            f.write("{:f}\n".format(binwidth))
            f.write("{:f}\n".format(max_time))
            f.write(rates_file)
            f.write("\n")
            f.write(neighbours_file)
