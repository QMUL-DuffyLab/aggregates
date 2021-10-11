import numpy as np
import cv2

class Circle:
    """A little class representing an SVG circle."""

    def __init__(self, cx, cy, r, icolour=None):
        """Initialize the circle with its centre, (cx,cy) and radius, r.

        icolour is the index of the circle's colour.

        """
        self.cx, self.cy, self.r = cx, cy, r
        self.icolour = icolour

    def move(self, cx, cy):
        """ move to (cx, cy). assumes we've checked for collisions! """
        self.cx = cx
        self.cy = cy

    def overlap_with(self, cx, cy, r):
        """Does the circle overlap with another of radius r at (cx, cy)?"""

        d = np.hypot(cx-self.cx, cy-self.cy)
        return d < r + self.r

    def is_nn(self, cx, cy, nn_cutoff):
        """Do we consider this circle as a neighbour to the circle (cx, cy)?"""
        d = np.hypot(cx-self.cx, cy-self.cy)
        return d < nn_cutoff

    def draw_circle(self, img, r):
        """Draw a circle on the image using cv2"""
        cv2.circle(img, (self.cy, self.cx), r, (26, 0, 153), -1, cv2.LINE_AA)

    def draw_neighbour(self, cx, cy, img):
        """ draw a line between two neighbours """
        cv2.line(img, (self.cy, self.cx), (cy, cx), (232, 139, 39))

class ShapeFill():
    """A class for filling a shape with circles."""

    def __init__(self, img, r, n, max_pulls=10, colours=None, *args, **kwargs):
        """Initialize the class with an image specified by filename.

        The image should be white on a black background.
        Circle radius is r - in pixels
        The maximum number of circles to pack is given by n
        colours is a list of RGB hex codes (currently unused)
        """

        self.img = img
        self.colour_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.r, self.n = r, n
        self.width, self.height = np.shape(img)[0], np.shape(img)[1]
        self.guard = 500
        self.max_pulls = max_pulls
        self.circles = []
        self.colours = colours or ['#99001A', '#278BE8','#F2D33C','#832591','#F28FEA']

    def _circle_fits(self, icx, icy, r):
        """pretty self explanatory"""

        if icx-r < 0 or icy-r < 0:
            return False
        if icx+r >= self.width or icy+r >= self.height:
            return False

        if not all((self.img[icx-r,icy], self.img[icx+r,icy],
                self.img[icx,icy-r], self.img[icx,icy+r])):
            return False
        return True

    def apply_circle_mask(self, icx, icy, r):
        """Zero all elements of self.img in circle at (icx, icy), radius r."""

        x, y = np.ogrid[0:self.width, 0:self.height]
        r2 = (r+1)**2
        mask = (x-icx)**2 + (y-icy)**2 <= r2
        self.img[mask] = 0

    def remove_circle_mask(self, icx, icy, r):
        """1 all elements of self.img in circle at (icx, icy), radius r."""

        x, y = np.ogrid[0:self.width, 0:self.height]
        r2 = (r+1)**2
        mask = (x-icx)**2 + (y-icy)**2 <= r2
        self.img[mask] = 1

    def _place_circle(self, r, c_idx=None):
        """Attempt to place a circle of radius r within the image figure.
 
        c_idx is a list of indexes into the self.colours list, from which
        the circle's colour will be chosen. If None, use all colours.

        """

        if not c_idx:
            c_idx = range(len(self.colours))

        # Get the coordinates of all non-zero image pixels
        img_coords = np.nonzero(self.img)
        if not img_coords:
            return False

        # The guard number: if we don't place a circle within this number
        # of trials, we give up.
        guard = self.guard
        # For this method, r must be an integer. Ensure that it's at least 1.
        r = max(1, int(r))
        while guard:
            # Pick a random candidate pixel...
            i = np.random.randint(len(img_coords[0]))
            icx, icy = img_coords[0][i], img_coords[1][i]
            # ... and see if the circle fits there
            if self._circle_fits(icx, icy, r):
                self.apply_circle_mask(icx, icy, r)
                circle = Circle(icx, icy, r, icolour=np.random.choice(c_idx))
                self.circles.append(circle)
                return True
            guard -= 1
        # print('guard reached.')
        return False

    def make_circles(self, c_idx=None):
        """Place the little circles inside the big one.

        c_idx is a list of colour indexes (into the self.colours list) from
        which to select random colours for the circles. If None, use all
        the colours in self.colours.

        """

        nplaced = 0
        for i in range(self.n):
            if self._place_circle(self.r, c_idx):
                nplaced += 1
        # print('{}/{} circles placed successfully.'.format(nplaced, self.n))
        return nplaced

    def pull_circles(self):
        '''
        pull all the circles towards a given point to make space.
        '''
        # pick a pixel within the image to act as a centre of gravity
        # note: _place_circle applies a mask which turns the location of
        # each circle black, so this should only pick out unoccupied pixels
        img_coords = np.nonzero(self.img)
        ri = np.random.randint(len(img_coords[0]))
        icx, icy = img_coords[0][ri], img_coords[1][ri]
        print("centre pixel: ({}, {})".format(icx, icy))
        cv2.circle(self.colour_img, (icy, icx), 2, (255, 0, 0), -1, cv2.LINE_AA)
        r = [(i, np.sqrt((circle.cx - icx)**2 + (circle.cy - icy)**2)) for i, circle in enumerate(self.circles)]
        r.sort(key=lambda t: t[1]) # try moving the closest first
        mask = [True] * len(r) # for checking circle overlaps
        for i in [t[0] for t in r]:
            c = self.circles[i]
            # next line is to make sure _circle_fits works correctly:
            # if we leave this circle masked it will trivially not move
            self.remove_circle_mask(c.cx, c.cy, c.r)
            mask[i] = False # remove this circle from the overlap check
            rx, ry = (icx - c.cx), (icy - c.cy)
            dx, dy = 0, 0
            '''
            quick and dirty solution:
            move horizontally, one pixel at a time, as long as the
            moving circle wouldn't leave the grain or overlap any other.
            then do the same vertically. in principle we could
            try doing it the other way round, both at once, etc.
            but that'll be slower, so hopefully this is good enough.
            '''
            can_move = True
            while (np.abs(dx) < np.abs(rx)) and can_move:
                dx = dx + np.sign(rx)
                if self._circle_fits(c.cx + np.sign(rx), c.cy, c.r):
                    if not any(circle.overlap_with(c.cx + np.sign(rx), c.cy, c.r) for circle in [b for a, b in zip(mask, self.circles) if a]):
                        c.move(c.cx + np.sign(rx), c.cy)
                        dx = dx + np.sign(rx)
                    else:
                        can_move = False
                else:
                    can_move = False

            can_move = True
            while (np.abs(dy) < np.abs(ry)) and can_move:
                if self._circle_fits(c.cx, c.cy + np.sign(ry), c.r):
                    if not any(circle.overlap_with(c.cx, c.cy + np.sign(ry), c.r) for circle in [b for a, b in zip(mask, self.circles) if a]):
                        c.move(c.cx, c.cy + np.sign(ry))
                        dy = dy + np.sign(ry)
                    else:
                        can_move = False
                else:
                    can_move = False
            
            self.apply_circle_mask(c.cx, c.cy, c.r)
            mask[i] = True # reset the mask

    def pack(self):
        '''
        pack the aggregate with circles by
        placing and then pulling them around.
        '''
        nplaced = self.make_circles()
        nplaced_total = nplaced
        pulls = 0
        print('First run: {}/{} circles placed.'.format(nplaced_total, self.n))
        while pulls <= self.max_pulls and nplaced_total <= self.n:
            self.pull_circles()
            nplaced = self.make_circles()
            nplaced_total += nplaced
            print('{} circles placed.'.format(nplaced))
            pulls += 1
        print('Done. {}/{} total circles placed.'.format(nplaced_total, self.n))

    def make_image(self, filename, *args, **kwargs):
        """ add the circles to the image and write it out. """
        colour_img = self.colour_img.copy()
        for circle in self.circles:
            circle.draw_circle(colour_img, int(self.r))
        cv2.imwrite(filename, colour_img)

