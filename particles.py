import numpy as np


class _Particle():

    def __init__(self, r, v, name):
        """_summary_

        Args:
            r (array_like): Positional vector in crtbp units
            v (array_like): Velocity vector in crtbp units
        """
        self.name = name
        self.r0 = r
        self.v0 = v
        self.r = self.r0
        self.v = self.v0
        self.STM = np.identity(6)
        self.time = 0
        self.period = None

    def __repr__(self):  # common special method to represent
        return "Particle(name(id) = {}, r0 = {}, v0 ={}, r = {}, v = {}, time = {}, period = {})".format(
            self.name,
            self.r0,
            self.v0,
            self.r,
            self.v,
            self.time,
            self.period
        )
