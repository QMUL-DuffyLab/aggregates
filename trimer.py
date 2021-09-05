#!/usr/bin/env python3

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
