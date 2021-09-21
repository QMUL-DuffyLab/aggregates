#!/usr/bin/env python3
from shapefill import ShapeFill

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
    def __init__(self, image, x, y, w, h, area, n, rho):
        self.image = image
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        self.shapefill = ShapeFill(image, n=n, rho_min=rho, rho_max=rho, colours=['#99001A'])
        self.shapefill.guard = 250
        self.shapefill.make_circles()


    def _adj(self):
        '''
        construct an adjacency matrix
        '''
        nn_cutoff = 1.2 * rho # rho won't work i don't think, need to work out r in pixels
        adj = np.zeros((len(self.shapefill.circles), len(self.shapefill.circles)))
        for c in self.shapefill.circles:
            # this makes no sense yet lol
            neighbours = np.where([circle.is_nn(self.CX + cx, self.CY + cy, nn_cutoff) for c2 in self.shapefill.circles])
            adj[i] = neighbours
