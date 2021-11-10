import numpy as np


class Footprint:
    def __init__(self, values, xs=None, ys=None, pointers=None, img=None):
        self.values = values
        self.xs = xs
        self.ys = ys
        self.pointers = pointers
        self.img = img
