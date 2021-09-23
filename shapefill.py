from PIL import Image
import numpy as np
from circles import Circle, Circles
import cv2

class ShapeFill(Circles):
    """A class for filling a shape with circles."""

    def __init__(self, img, *args, **kwargs):
        """Initialize the class with an image specified by filename.

        The image should be white on a black background.

        The maximum and minimum circle sizes are given by rho_min and rho_max
        which are proportions of the minimum image dimension.
        The maximum number of circles to pack is given by n
        colours is a list of SVG fill colour specifiers (a default palette is
        used if this argument is not provided).

        """

        self.img = img
        self.masked_img = img
        self.colour_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.width, self.height = np.shape(img)[0], np.shape(img)[1]
        dim = min(self.width, self.height)
        super().__init__(self.width, self.height, dim, *args, **kwargs)

    def _circle_fits(self, icx, icy, r):
        """If I fits, I sits."""

        if icx-r < 0 or icy-r < 0:
            return False
        if icx+r >= self.width or icy+r >= self.height:
            return False

        if not all((self.masked_img[icx-r,icy], self.masked_img[icx+r,icy],
                self.masked_img[icx,icy-r], self.masked_img[icx,icy+r])):
            return False
        return True

    def apply_circle_mask(self, icx, icy, r):
        """Zero all elements of self.img in circle at (icx, icy), radius r."""

        x, y = np.ogrid[0:self.width, 0:self.height]
        r2 = (r+1)**2
        mask = (x-icx)**2 + (y-icy)**2 <= r2
        self.masked_img[mask] = 0

    def remove_circle_mask(self, icx, icy, r):
        """1 all elements of self.img in circle at (icx, icy), radius r."""

        x, y = np.ogrid[0:self.width, 0:self.height]
        r2 = (r+1)**2
        mask = (x-icx)**2 + (y-icy)**2 <= r2
        self.masked_img[mask] = 1

    def _place_circle(self, r, c_idx=None):
        """Attempt to place a circle of radius r within the image figure.
 
        c_idx is a list of indexes into the self.colours list, from which
        the circle's colour will be chosen. If None, use all colours.

        """

        if not c_idx:
            c_idx = range(len(self.colours))

        # Get the coordinates of all non-zero image pixels
        img_coords = np.nonzero(self.masked_img)
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

    def pull_circles(self):
        '''
        pull all the circles towards a given point to make space.
        '''
        # pick a pixel within the image to act as a centre of gravity
        # note: _place_circle applies a mask which turns the location of
        # each circle black, so this should only pick out unoccupied pixels
        img_coords = np.nonzero(self.masked_img)
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
                

if __name__ == '__main__':
    # Land colours, sea colours.
    c1 = ['#99001A']
    c2 = ['#173f5f', '#20639b', '#3caea3']

    # First fill the land silhouette.
    shape = ShapeFill('004_new.bmp', n=3000, rho_min=0.0045, rho_max=0.0045, colours=c1+c2)
    # expects black shape on white bg?
    shape.img = 255 - shape.img
    shape.guard = 1000
    shape.make_circles(c_idx=range(len(c1)))
    # shape.make_svg('uk-1.svg')

    # Now load the image again, invert it and fill the sea with circles.
    # shape.read_image('uk.png')
    # shape.n = 5000
    # shape.make_circles(c_idx=[len(c1)+i for i in range(len(c2))])
    # shape.make_svg('uk-2.svg')
    
