#!/usr/bin/env python3
import numpy as np
from shapefill import ShapeFill
import cv2

class Trimer():
    '''
    Base trimer class - various types of quenched trimers
    will be derived from this. Very few common parameters tbh.
    '''
    def __init__(self, decay_time, neighbours):
        self.decay_time = decay_time
        self.neighbours = neighbours

    def get_decay_time(self):
        ''' Return decay time of this trimer '''
        return self.decay_time

    def get_neighbours(self):
        ''' Return list of neighbours of this trimer '''
        return self.neighbours

class QuenchedTrimer(Trimer):
    '''
    Not sure yet whether this will be the base for different
    kinds of quenched trimer or whether I'll just make one Trimer
    subclass for each type.
    '''
    def __init__(self, decay_time, neighbours, num_levels):
        self.num_levels = num_levels
        super().__init__(decay_time, neighbours)

    def get_num_levels(self):
        '''
        Different types of quenching are represented here by
        different levels in the trimer - return the number of
        levels for this one
        '''
        return self.num_levels

    def get_decay_time(self):
        return self.decay_time

    def get_neighbours(self):
        return self.neighbours

    def _construct_adjacency_matrix(self):
        '''
        Depending on what kind of quencher we have, we need to
        add extra rows to the adjacency matrix representing the
        trap(s) - do this here. idk what this will involve yet
        '''
        return

class Aggregate:
    '''
    connected components bits: stats, image
    (monochrome with only this aggregate) then pass to shapefill,
    add the circles and adjacency info to the aggregate as well.
    '''
    def __init__(self, img, x, y, w, h, area, n, rho, nn_cutoff, max_pulls):
        self.img = img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        self.max_pulls = max_pulls
        self.shapefill = ShapeFill(img, n=n, rho=rho, colours=['#99001A'])
        self.fd = self.fractal_dimension()
        self.A = self._adj(nn_cutoff)
        self.shapefill.guard = 500

    def pack(self, n):
        '''
        pack the aggregate with circles by
        placing and then pulling them around.
        '''
        nplaced = self.shapefill.make_circles()
        nplaced_total = nplaced
        pulls = 0
        print('First run: {}/{} circles placed.'.format(nplaced_total, n))
        while nplaced != 0 and pulls <= self.max_pulls and nplaced_total <= n:
            self.shapefill.pull_circles()
            nplaced = self.shapefill.make_circles()
            nplaced_total += nplaced
            print('{} circles placed.'.format(nplaced))
            pulls += 1
        print('Done. {}/{} total circles placed.'.format(nplaced_total, n))

    def fractal_dimension(self):
        '''
	https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
        '''

        # our image is binary but it's 0 and 255: fix that
        Z = self.img / 255

        # Only for 2d image
        assert(len(self.img.shape) == 2)

        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k*k))[0])

        # Minimal dimension of image
        p = min(Z.shape)

        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))

        # Extract the exponent
        n = int(np.log(n)/np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))

        print(counts)
        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def _adj(self, nn_cutoff):
        '''
        construct an adjacency matrix
        '''
        adj = np.zeros((len(self.shapefill.circles), len(self.shapefill.circles)))
        for i, c in enumerate(self.shapefill.circles):
            neighbours = np.array([c.is_nn(c2.cx, c2.cy, nn_cutoff) for c2 in self.shapefill.circles])
            adj[i] = neighbours.astype(int)

        # the above will make each circle its own neighbour
        for i in range(len(self.shapefill.circles)):
            adj[i][i] = 0

        return adj

    def make_neighbours(self, nn_cutoff, filename):
        '''
        draw all the neighbours onto the image so we can check!
        '''
        col = self.shapefill.colour_img.copy()
        [c1.draw_circle(col, int(c1.r)) for c1 in self.shapefill.circles]
        [c1.draw_neighbour(c2.cx, c2.cy, col) for c1 in self.shapefill.circles for c2 in self.shapefill.circles if c1.is_nn(c2.cx, c2.cy, nn_cutoff)]
        cv2.imwrite(filename, col)
        self.shapefill.colour_img = col
