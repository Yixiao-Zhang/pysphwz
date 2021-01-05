#!/usr/bin/env python3

import math

import numpy as np
import tinyarray as ta

from sph_core import SPH_APP, Point, domain, mid_domain

dx = 0.8
density0 = dx**(-len(domain))
stiffness = 1.0e3
viscosity = 0.1
nudge = 0.0
k = 7

ca = math.sqrt(stiffness*k/density0)

print(ca)

class Wall(Point):
    def __init__(self, postion):
        velocity = ta.zeros(len(position))
        super().__init__(position, velocity)


class LiquidPoint(Point):
    def force(self, model):
        res = ta.array([0.0, 0.0])
        res += -model.gradient('pressure', self)

        if viscosity:
            res += viscosity*model.laplace('velocity', self)
        if nudge:
            res -= self.velocity*nudge

        # res += ta.array([0.0, 1.0])

        return res

    @property
    def color(self):
        return '#FF0000' if self.velocity[0] > 0 else '#FFFF00'

    @property
    def pressure(self):
        return stiffness*((self.density/density0)**k - 1)


def main():

    xs = np.arange(0.0, domain[0], dx)
    ys = np.arange(0.0, domain[1], dx)
    vmax = 40.0
    # vxs = vmax*(2*(ys > mid_domain[1]) - 1)
    vxs = vmax*(ys > 0.40*domain[1])*(ys < 0.60*domain[1])
    vys = np.random.uniform(-vmax*1e-3, vmax*1e-3, len(xs))

    xs, ys = map(lambda a: np.reshape(a, (-1, )), np.meshgrid(xs, ys))
    vys, vxs = map(lambda a: np.reshape(a, (-1, )), np.meshgrid(vys, vxs))

    positions = map(ta.array, map(list, zip(xs, ys)))
    velocities = map(ta.array, map(list, zip(vxs, vys)))

    points = tuple(map(LiquidPoint, positions, velocities))

    model = SPH_APP(points, dt=0.005, nsnapshot=10)
    model.run()


if __name__ == '__main__':
    main()
