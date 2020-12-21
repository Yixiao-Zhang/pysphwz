#!/usr/bin/env python3

import math

import numpy as np
import tinyarray as ta

from sph_core import SPH_APP, Point, domain, mid_domain, norm

dx = 1.0
density0 = dx**(-len(domain))
stiffness = 1.0e2
viscosity = 1.0
nudge = 0.0
k = 7

ca = math.sqrt(stiffness*k/density0)

print(ca)

class LiquidPoint(Point):
    def force(self, model):
        res = ta.array([0.0, 0.0])
        res += -model.gradient('pressure', self)

        if viscosity:
            res += viscosity*model.laplace('velocity', self)
        if nudge:
            res -= self.velocity*nudge

        if model.t < 1.0:
            res -= self.velocity*1.0
            res = res*ta.array([0.0, 10.0])

        res += ta.array([0.0, 1.0e2])

        return res

    @property
    def color(self):
        return '#FF0000' # if self.velocity[0] > 0 else '#FFFF00'

    @property
    def pressure(self):
        return stiffness*((self.density/density0)**k - 1)


class Wall(LiquidPoint):
    def __init__(self, position):
        velocity = ta.zeros(len(position))
        super().__init__(position, velocity)

    def force(self, model):
        res = ta.array([0.0, 0.0])
        return res

    @property
    def color(self):
        return '#FFFFFF'


def create_wall(*positions):
    positions = list(map(ta.array, positions))
    end = positions[0]
    dist = 0.95*dx
    for i in range(len(positions) - 1):
        start = end
        end = (positions[i+1] - positions[i]) + start
        length = norm(start-end)
        unit = (end - start)/length
        num = math.ceil(length/dist)
        for k in range(num):
            yield Wall(start + unit*(dist*k))
        end = start + unit*dist*k

def main():

    walls = tuple(create_wall(
            (31.5, 0.5),
            (31.5, 15.5),
            (0.5, 15.5),
            (0.5, 0.5),
        ))
    wallxmin = min(wall.position[0] for wall in walls)
    wallymax = max(wall.position[1] for wall in walls)

    xs = np.arange(wallxmin + dx, 8.0, dx)
    ys = np.arange(2.0, wallymax - dx, dx)

    vmax = 8.0
    vxs = np.random.uniform(-vmax*1e-3, vmax*1e-3, len(ys))
    vys = np.random.uniform(-vmax*1e-3, vmax*1e-3, len(xs))

    xs, ys = map(lambda a: np.reshape(a, (-1, )), np.meshgrid(xs, ys))
    vys, vxs = map(lambda a: np.reshape(a, (-1, )), np.meshgrid(vys, vxs))

    positions = map(ta.array, map(list, zip(xs, ys)))
    velocities = map(ta.array, map(list, zip(vxs, vys)))

    points = tuple(map(LiquidPoint, positions, velocities))

    points = points + walls

    model = SPH_APP(points, dt=0.005, nsnapshot=10)
    model.run()


if __name__ == '__main__':
    main()
