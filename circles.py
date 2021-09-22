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

    def overlap_with(self, cx, cy, r):
        """Does the circle overlap with another of radius r at (cx, cy)?"""

        d = np.hypot(cx-self.cx, cy-self.cy)
        return d < r + self.r

    def is_nn(self, cx, cy, nn_cutoff):
        """Do we consider this circle as a neighbour to the circle (cx, cy)?"""

        d = np.hypot(cx-self.cx, cy-self.cy)
        return d < nn_cutoff

    def draw_circle(self, img):
        """Write the circle's SVG to the output stream, fo."""
        # next line is for when i figure out how to use cv2 everywhere lol
        cv2.circle(img, (self.cy, self.cx), 4, (26, 0, 153), -1, cv2.LINE_AA)

    def move(self, cx, cy):
        """ move to (cx, cy). assumes we've checked for collisions! """
        self.cx = cx
        self.cy = cy

class Circles:
    """A class for drawing circles-inside-a-circle."""
    
    def __init__(self, width=600, height=600, R=250, n=800, rho_min=0.005,
                 rho_max=0.05, colours=None):
        """Initialize the Circles object.

        width, height are the SVG canvas dimensions
        R is the radius of the large circle within which the small circles are
        to fit.
        n is the maximum number of circles to pack inside the large circle.
        rho_min is rmin/R, giving the minimum packing circle radius.
        rho_max is rmax/R, giving the maximum packing circle radius.
        colours is a list of SVG fill colour specifiers to be referenced by
            the class identifiers c<i>. If None, a default palette is set.

        """

        self.width, self.height = width, height
        self.R, self.n = R, n
        # The centre of the canvas
        self.CX, self.CY = self.width // 2, self.height // 2
        self.rmin, self.rmax = R * rho_min, R * rho_max
        self.colours = colours or ['#993300', '#a5c916', '#00AA66', '#FF9900']
        self.circles = []
        # The "guard number": we try to place any given circle this number of
        # times before giving up.
        self.guard = 500

    def make_image(self, filename, *args, **kwargs):
        """ add the circles to the image and write it out. """
        col = self.colour_img
        colour_img = self.colour_img.copy()
        for circle in self.circles:
            circle.draw_circle(colour_img)
        cv2.imwrite(filename, colour_img)
        self.colour_img = col

    def _place_circle(self, r, c_idx=None):
        """Attempt to place a circle of radius r within the larger circle.
        
        c_idx is a list of indexes into the self.colours list, from which
        the circle's colour will be chosen. If None, use all colours.

        """

        if not c_idx:
            c_idx = range(len(self.colours))

        # The guard number: if we don't place a circle within this number
        # of trials, we give up.
        guard = self.guard
        while guard:
            # Pick a random position, uniformly on the larger circle's interior
            cr, cphi = ( self.R * np.sqrt(np.random.random()),
                         2*np.pi * np.random.random() )
            cx, cy = cr * np.cos(cphi), cr * np.sin(cphi)
            if cr+r < self.R:
            # The circle fits inside the larger circle.
                if not any(circle.overlap_with(self.CX+cx, self.CY+cy, r)
                                    for circle in self.circles):
                    # The circle doesn't overlap any other circle: place it.
                    circle = Circle(cx+self.CX, cy+self.CY, r,
                                    icolour=np.random.choice(c_idx))
                    self.circles.append(circle)
                    return True
            guard -= 1
        # Warn that we reached the guard number of attempts and gave up for
        # for this circle.
        print('guard reached.')
        return False

    def make_circles(self, c_idx=None):
        """Place the little circles inside the big one.

        c_idx is a list of colour indexes (into the self.colours list) from
        which to select random colours for the circles. If None, use all
        the colours in self.colours.

        """

        # First choose a set of n random radii and sort them. We use
        # random.random() * random.random() to favour small circles.
        r = self.rmin + (self.rmax - self.rmin) * np.random.random(
                                self.n) * np.random.random(self.n)
        r[::-1].sort()
        # Do our best to place the circles, larger ones first.
        nplaced = 0
        for i in range(self.n):
            if self._place_circle(r[i], c_idx):
                nplaced += 1
        print('{}/{} circles placed successfully.'.format(nplaced, self.n))
                

if __name__ == '__main__':
    circles = Circles(n=2000)
    circles.make_circles()
    circles.make_svg('circles.svg')

