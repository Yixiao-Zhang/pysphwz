#!/usr/bin/env python3

import tkinter as tk
import time
import itertools
import logging
import math

import numpy as np
import tinyarray as ta

domain = ta.array([16.0, 9.0])
mid_domain = domain/2
pradius = 1.0
npoint = 576
density0 = npoint/math.exp(sum(map(math.log, domain)))
stiffness = 10.0

def norm(array):
    return math.sqrt(sum(array*array))

def weight(displacement):
    dist = norm(displacement)
    return max((2/math.pi)*(1 - dist**2), 0.0)

def grad_weight(displacement):
    dist = norm(displacement)
    return (4/math.pi)*displacement if dist < 1.0 else 0.0

def laplace(displacement):
    return sum(displacement*grad_weight(displacement))/(norm(displacement)**2 + 1.0e-2)

def pressure(density):
    return stiffness*((density/density0)**7 - 1)

class Point:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

    def move(self, dt:float):
        self.position += dt*self.velocity
        self.position %= domain

    def accelarate(self, dt:float, *args, **kwargs):
        self.velocity += dt*self.force(*args, **kwargs)

    def displacement(self, point):
        return ((self.position - point.position + mid_domain)%domain
                    - mid_domain)

    def force(self, model):
        return 0.0

class SFH:
    def __init__(self, points, t0=0.0):
        self.points = points
        self.t = t0

    def step(self, dt=None):
        vmax = max(norm(point.velocity) for point in self.points)
        if dt is None:
            dt = 0.4*pradius/max(vmax, 1.0)

        print('CFL', vmax*dt/pradius)

        for point in self.points:
            point.accelarate(dt, self)

        for point in self.points:
            point.move(dt)

        self.t += dt

class SFH_APP(SFH, tk.Canvas):
    scale = 50.0
    radius = 4.0
    def __init__(self, points):
        SFH.__init__(self, points)

        self.root = tk.Tk()
        tk.Canvas.__init__(self, self.root,
            width=domain[0]*self.scale, height=domain[1]*self.scale)
        self.ovals = self.plot_points()
        self.timer = self.create_text(20, 20, anchor=tk.NW,
            font="Courier 20 bold",
            text="")
        self.pack()

    def coords_oval(self, point):
        x, y = point.position*self.scale
        return (x - self.radius, y - self.radius,
                x + self.radius, y + self.radius)

    def plot_points(self):
        ovals = list()
        for point in self.points:
            try:
                color = point.color
            except:
                color = '#000000'
            oval = self.create_oval(*self.coords_oval(point),
                fill=color)
            ovals.append(oval)
        return tuple(ovals)

    real_time_sep = 0.2
    def loop_func(self):
        time_next = time.time()
        while True:
            self.step(dt=0.01)
            for point, oval in zip(self.points, self.ovals):
                self.coords(oval, *self.coords_oval(point))
            self.itemconfig(self.timer, text = f't = {self.t:.02f}')
            self.update()
            time_next += self.real_time_sep
            time_sleep = time_next - time.time()
            if time_sleep < 0.0:
                logging.warning('FPS drop')
            time.sleep(max(0.0, time_sleep))

    def run(self):
        self.root.after(0, self.loop_func)
        tk.mainloop()

class LiquidApp(SFH_APP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_subdomains()
        self.density = density0 # dummy value
        self.pressure = 0.0 # dummy value

    class LiquidPoint(Point):
        def force(self, model):
            res = ta.array([0.0, 0.0])
            res += -model.gradient('pressure', self)
            res += 0.2*model.laplace('velocity', self)
            res += (self.position - domain/2)*ta.array([0.0, -1.0])
            return res

        @property
        def color(self):
            return '#FF0000' if self.position[1] > domain[1]/2 else '#FFFF00'

    def step(self, *args, **kwargs):
        self.update_subdomains()
        self.update_density()
        super().step(*args, **kwargs)

    # make sure that the subdomain is large enough to capture
    # partical-partical interactions
    nsubs = (16, 9)
    def update_subdomains(self):
        self.subs = [[set() for _ in range(nsub)] for nsub in self.nsubs]
        for point in self.points:
            for i, subi in enumerate(self.subdomain_index(point)):
                self.subs[i][subi].add(point)
        self._neiborghs = dict()
        for indexes in itertools.product(*map(range, self.nsubs)):
            sets = []
            for i, subi in enumerate(indexes):
                sets.append(set.union(*(self.subs[i][(subi+k)%self.nsubs[i]]
                        for k in (-1, 0, 1))))
            self._neiborghs[indexes] = set.intersection(*sets)

    def update_density(self):
        for point in self.points:
            point.density = sum(weight(point.position - neighbor.position)
                for neighbor in self.neiborghs(point))
            point.pressure = pressure(point.density)

    def subdomain_index(self, point):
        return (int(point.position[i]*self.nsubs[i]/domain[i])
                    for i in range(len(self.nsubs)))

    def neiborghs(self, point):
        return self._neiborghs[tuple(self.subdomain_index(point))]

    def gradient(self, vname, point):
        return sum(
               (getattr(point, vname)/point.density**2
               + getattr(neighbor, vname)/neighbor.density**2)
                * grad_weight(neighbor.position - point.position)
                for neighbor in self.neiborghs(point)
        )

    def laplace(self, vname, point):
        return -2*sum(
               getattr(point, vname)/neighbor.density
                * laplace(neighbor.position - point.position)
                for neighbor in self.neiborghs(point)
        )


def main():
    xs = np.random.uniform(0.0, domain[0], npoint)
    ys = np.random.uniform(0.0, domain[1], npoint)
    positions = map(ta.array, map(list, zip(xs, ys)))

    vxs = np.random.uniform(-1.0, 1.0, npoint)
    vys = np.random.uniform(-1.0, 1.0, npoint)
    velocities = map(ta.array, map(list, zip(vxs, vys)))

    points = tuple(map(LiquidApp.LiquidPoint, positions, velocities))

    model = LiquidApp(points)
    model.run()


if __name__ == '__main__':
    main()
