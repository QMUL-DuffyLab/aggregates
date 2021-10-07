#!/usr/bin/env python3
import numpy as np
from shapefill import ShapeFill
import cv2

class Trimer():
    '''
    Base trimer class - various types of quenched trimers
    will be derived from this. Very few common parameters tbh.
    '''
    def __init__(self, x, y, r, decay_time):
        self.x, self.y, self.r = x, y, r
        self.decay_time = decay_time

    def get_decay_time(self):
        ''' Return decay time of this trimer '''
        return self.decay_time

    def get_neighbours(self):
        ''' Return list of neighbours of this trimer '''
        return self.neighbours

    def add_neighbour(self, trimer):
        self.neighbours.append(trimer)

    def overlap_with(self, x, y, r):
        """Does the circle overlap with another of radius r at (cx, cy)?"""

        d = np.hypot(x-self.x, y-self.y)
        return d < r + self.r

    def is_nn(self, x, y, nn_cutoff):
        """Do we consider this circle as a neighbour to the circle (cx, cy)?"""
        d = np.hypot(x-self.x, y-self.y)
        return d < nn_cutoff

    def draw_neighbour(self, cx, cy, img):
        """ draw a line between two neighbours """
        cv2.line(img, (self.cy, self.cx), (cy, cx), (232, 139, 39))

    def draw_circle(self, img, r):
        """Write the circle's SVG to the output stream, fo."""
        # next line is for when i figure out how to use cv2 everywhere lol
        cv2.circle(img, (self.cy, self.cx), r, (26, 0, 153), -1, cv2.LINE_AA)

    def move(self, cx, cy):
        """ move to (cx, cy). assumes we've checked for collisions! """
        self.cx = cx
        self.cy = cy

class State:
    def __init__(self, type, ):
        self.type = type


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
    def __init__(self, img, x, y, w, h, area, n, r, nn_cutoff, max_pulls):
        self.img = img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        if img is not None:
            self.shapefill = ShapeFill(img, n=n, r=r, max_pulls=max_pulls, colours=['#99001A'])
        # else:
            # self.trimers = generate_lattice()
        self.fd = self.fractal_dimension()

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

        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def adj(self, nn_cutoff):
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

    def make_neighbours(self, filename):
        '''
        draw all the neighbours onto the image so we can check!
        '''
        if len(self.shapefill.circles) < 2: # no neighbours
            return None

        col = self.shapefill.colour_img.copy()
        for i, c1 in enumerate(self.shapefill.circles):
            c1.draw_circle(col, int(c1.r))
            for j, c2 in enumerate(self.shapefill.circles):
                if self.A[i][j]:
                    c1.draw_neighbour(c2.cx, c2.cy, col)
        cv2.imwrite(filename, col)
        self.shapefill.colour_img = col

def generate_lattice(lattice_type, num_iter, r):
    '''
    test function to generate 1d, square, hex and hopefully honeycomb lattices.
    draws packed circles to represent trimers on the lattice.
    gonna use this as a basis for creating theoretical aggregates of trimers
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from shapefill import Circle
    if lattice_type == "line":
        basis = [[0,0]]
        a = 2.00000001 * r
        symmetry = 2
    elif lattice_type == "square":
        basis = [[0,0]]
        a = 2.00000001 * r
        symmetry = 4
    elif lattice_type == "hex":
        basis = [[0,0]]
        a = 2.00000001 * r
        symmetry = 6
    elif lattice_type == "honeycomb":
        basis = [
                2. * r * np.array([0, 0]),
                2. * r * np.array([0, -1]),
                ]
        a = 2.00000001 * np.sqrt(3.) * r
        symmetry = 6

    nn_vectors = []
    for i in range(symmetry):
        phi = i * 2 * np.pi / symmetry
        ri = np.array([
            [np.cos(phi), np.sin(phi)],
            [-np.sin(phi), np.cos(phi)]]) @ np.array(a * np.array([1, 0]))
        nn_vectors.append(ri)

    print(nn_vectors)
    sites = []
    sites.append(Circle(basis[0][0], basis[0][1], r - 0.000001))

    fig, ax = plt.subplots()
    for b in basis:
        ax.add_patch(mpatches.Circle((b[0], b[1]), r - 0.000001, 
            ec='C0', fc='C0', lw=0.25 * r))

    i = 0
    while i < num_iter:
        new_sites = []
        colour = "C{:1d}".format(i + 1)
        for site in sites:
            for n in nn_vectors:
                r0 = np.array([site.cx, site.cy])
                ri = np.array(r0) + np.array(n)
                t = Circle(ri[0], ri[1], r - 0.0000001)
                '''
                nested list comp to check overlap with the existing 
                sites from previous iterations and also the ones we're 
                currently adding
                '''
                if not any(t.overlap_with(t2.cx, t2.cy, r) for t2 in [a for b in [sites, new_sites] for a in b]):
                    new_sites.append(t)
        for site in new_sites:
            sites.append(site)
        i += 1

    trimers = []
    for site in sites:
        for ind, b in enumerate(basis):
            t = Trimer(site.cx + b[0], site.cy + b[1], r - 0.00001, 4.)
            if not any(t.overlap_with(t2.x, t2.y, r) for t2 in trimers):
                trimers.append(t)
                ax.add_patch(mpatches.Circle((site.cx + b[0], site.cy + b[1]),
                    site.r - 0.00001, ec='C{:1d}'.format(ind), fc='w'))
            else:
                print("Trimer collision!!!")

    print("Number of sites placed = {}".format(len(sites)))
    print("Number of trimers = {}".format(len(trimers)))
    xmax = np.max(np.array(np.abs([s.cx for s in sites])))
    ax.set_xlim([-xmax - (2. * r), xmax + (2. * r)])
    ax.set_ylim([-xmax - (2. * r), xmax + (2. * r)])
    fig.savefig("{}_lattice_agg.pdf".format(lattice_type))

    return trimers

if __name__ == "__main__":
    lattice_type = "hex"
    num_iter = 5
    r = 5.
    generate_lattice(lattice_type, num_iter, r)
