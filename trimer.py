#!/usr/bin/env python3
import numpy as np
from shapefill import ShapeFill
import cv2

class Trimer():
    '''
    Base trimer class - various types of quenched trimers
    will be derived from this. Very few common parameters tbh.
    '''
    def __init__(self, x, y, r, decay_time=4.):
        self.x, self.y, self.r = x, y, r
        self.decay_time = decay_time
        self.neighbours = []

    def get_decay_time(self):
        ''' Return decay time of this trimer '''
        return self.decay_time

    def get_neighbours(self):
        ''' Return list of neighbours of this trimer '''
        return self.neighbours

    def add_neighbour(self, trimer):
        self.neighbours.append(trimer)

    def overlap_with(self, x, y, r):
        """Does this trimer overlap with another at (x, y)? """
        d = np.hypot(x-self.x, y-self.y)
        return d < r + self.r

    def is_nn(self, x, y, nn_cutoff):
        """Do we consider this trimer as a neighbour to one at (x, y)?"""
        d = np.hypot(x-self.x, y-self.y)
        return d < nn_cutoff

    def draw_circle(self, img, r):
        """ draw this trimer """
        cv2.circle(img, (self.y, self.x), r, (26, 0, 153), -1, cv2.LINE_AA)

    def draw_neighbour(self, x, y, img):
        """ draw a line between two neighbours """
        cv2.line(img, (self.y, self.x), (y, x), (232, 139, 39))

    def move(self, cx, cy):
        """ move to (cx, cy). assumes we've checked for collisions! """
        self.cx = cx
        self.cy = cy

class State:
    def __init__(self, type, rates, lifetime):
        self.type = type
        self.rates = rates
        self.lifetime = lifetime

def real_aggregate(r, nn_cutoff, img, n, max_pulls, component):
    sf = ShapeFill(img, n=n, r=r, max_pulls=max_pulls, colours=['#99001A'])
    sf.make_image('components/{:03d}.jpg'.format(component))
    sf.pack()
    sf.make_image('components/{:03d}_pulled.jpg'.format(component))
    trimers = []
    for c in sf.circles:
        trimers.append(Trimer(c.cx, c.cy, r))

    '''
    this code is duplicated below: check whether the actual
    trimer objects will be modified if called from a function.
    if so, we can just make it a function and call that
    '''
    A = np.zeros((len(trimers), len(trimers)))
    for i, t1 in enumerate(trimers):
        for j, t2 in enumerate(trimers):
            if i == j:
                continue
            else:
                if (t1.is_nn(t2.x, t2.y, nn_cutoff)):
                    A[i][j] = 1
                    t1.add_neighbour(t2)

    for i in range(len(trimers)):
        if A[i][i] != 0:
            print("adjacency matrix broken")
        
    return Aggregate(trimers, A, img)


def theoretical_aggregate(r, nn_cutoff, lattice_type, n_iter):
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

        print("{} lattice generation.".format(lattice_type.capitalize()))

        nn_vectors = []
        for i in range(symmetry):
            phi = i * 2 * np.pi / symmetry
            ri = np.array([
                [np.cos(phi), np.sin(phi)],
                [-np.sin(phi), np.cos(phi)]]) @ np.array(a * np.array([1, 0]))
            nn_vectors.append(ri)

        sites = []
        sites.append(Circle(basis[0][0], basis[0][1], r - 0.000001))

        i = 0
        while i < n_iter:
            new_sites = []
            for site in sites:
                for n in nn_vectors:
                    r0 = np.array([site.cx, site.cy])
                    ri = np.array(r0) + np.array(n)
                    c = Circle(ri[0], ri[1], r - 0.0000001)
                    '''
                    nested list comp to check overlap with the existing 
                    sites from previous iterations and also the ones we're 
                    currently adding
                    '''
                    if not any(c.overlap_with(c2.cx, c2.cy, r) 
                            for c2 in [a for b in [sites, new_sites] 
                            for a in b]):
                        new_sites.append(c)
            sites.extend(new_sites)
            i += 1

        trimers = []
        xmax = 0.
        for site in sites:
            for ind, b in enumerate(basis):
                if (np.abs(site.cx + b[0]) > xmax):
                    xmax = np.abs(site.cx + b[0])
                t = Trimer(site.cx + b[0], site.cy + b[1], r - 0.00001)
                if not any(t.overlap_with(t2.x, t2.y, r) for t2 in trimers):
                    trimers.append(t)
                else:
                    print("Trimer collision!!!")

        # do this after appending all the trimers to
        # ensure that A_{ij} and nn pairs are symmetric
        A = np.zeros((len(trimers), len(trimers)))
        img = np.zeros((2 * int(xmax + 4. * r),
            2 * int(xmax + 4. * r), 3), np.uint8)
        for i, t1 in enumerate(trimers):
            cv2.circle(img, (int(t1.y + xmax + 2. * r),
                int(t1.x + xmax + 2. * r)), int(t1.r) + 1, (255, 255, 255), -1)
            for j, t2 in enumerate(trimers):
                if i == j:
                    continue
                else:
                    if (t1.is_nn(t2.x, t2.y, nn_cutoff)):
                        A[i][j] = 1
                        t1.add_neighbour(t2)

        colour_img = img.copy()
        for i, t in enumerate(trimers):
            print("Trimer no. {}: x = {:8.3f}, y = {:8.3f}".format(i, t.x, t.y))
            for j, t2 in enumerate(t.get_neighbours()):
                print("Neighbour {}: x = {:8.3f}, y = {:8.3f}".format(j, t2.x, t2.y))
                cv2.line(colour_img, (int(t.y + xmax + 2. * r),
                int(t.x + xmax + 2. * r)), (int(t2.y + xmax + 2. * r),
                int(t2.x + xmax + 2. * r)), (232, 139, 39))

        cv2.imwrite("{}_lattice_cv2_neighbours.jpg".format(lattice_type),
                colour_img)
        print("Number of trimers placed = {}".format(len(trimers)))
        img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return Aggregate(trimers, A, img_binary)

class Aggregate:
    '''
    connected components bits: stats, image
    (monochrome with only this aggregate) then pass to shapefill,
    add the circles and adjacency info to the aggregate as well.
    '''
    def __init__(self, trimers, A, img):
        self.trimers = trimers
        self.A = A
        self.img = img
        self.fd = self.fractal_dimension()
        print("Fractal dimension = {:6.4f}".format(self.fd))

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
        if len(self.trimers) < 2: # no neighbours
            return None

        col = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for t1 in self.trimers:
            t1.draw_circle(col, int(t1.r))
            for t2 in t1.get_neighbours():
                t1.draw_neighbour(t2.x, t2.y, col)
        cv2.imwrite(filename, col)

if __name__ == "__main__":
    lattice_type = "hex"
    n_iter = 7
    r = 5.
    agg = theoretical_aggregate(r, 2.5*r, lattice_type, n_iter)
