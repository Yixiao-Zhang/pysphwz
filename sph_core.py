#!/usr/bin/env python3

import tkinter as tk
import itertools
import math
import operator
import functools
from pathlib import Path
import sys

import tinyarray as ta

domain = ta.array([32.0, 16.0])
mid_domain = domain/2
pradius = 1.0

def norm(vec):
    return math.sqrt(sum(vec*vec))

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

def weight_1d(x):
    q = x/pradius
    if abs(q) > 1.0:
        return 0.0
    elif abs(q) > 1/3:
        return (27/16/pradius)*(1 - abs(q))**2
    else:
        return (9/8 - 27/8*q**2)/pradius

def grad_weight_1d(x):
    q = x/pradius
    if abs(q) > 1.0:
        return 0.0
    elif abs(q) > 1/3:
        return math.copysign((27/8/pradius**2)*(1 - abs(q)), q)
    else:
        return 27/4*q/pradius**2

def weight(displacement):
    return prod(map(weight_1d, displacement))

def grad_weight(displacement):
    res = list()
    for i in range(len(displacement)):
        res.append(
            grad_weight_1d(displacement[i])
            * weight(
                displacement[j] for j in range(len(displacement)) if j != i
            )
        )
    return ta.array(res)


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


class NS:
    def __init__(self, points):
        self.points = points
        self.nsubs = tuple(map(math.ceil, domain/pradius))

    def update_ns(self):
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

        for point in self.points:
            point.density = sum(weight(neighbor.displacement(point))
                for neighbor in self.neiborghs(point))

    def subdomain_index(self, point):
        return (int(point.position[i]*self.nsubs[i]/domain[i])%self.nsubs[i]
                    for i in range(len(self.nsubs)))

    def neiborghs(self, point):
        return self._neiborghs[tuple(self.subdomain_index(point))]


class SPH(NS):
    def __init__(self, points, t0=0.0):
        super().__init__(points)
        self.t = t0
        self.nstep = 0

    def step(self, dt:float):

        self.update_ns()

        vmax = max(norm(point.velocity) for point in self.points)
        rho_max = max(point.density for point in self.points)
        rho_min = min(point.density for point in self.points)

        for point in self.points:
            point.accelarate(dt, self)

        for point in self.points:
            point.move(dt)

        self.t += dt
        self.nstep += 1

    def gradient(self, vname, point):
        return sum(
               (getattr(point, vname)/point.density**2
               + getattr(neighbor, vname)/neighbor.density**2)
                * grad_weight(neighbor.displacement(point))
                for neighbor in self.neiborghs(point) if neighbor != point
        )

    @staticmethod
    def laplace_marco(displacement):
        return (
                    sum(displacement*grad_weight(displacement))
                    / (sum(displacement*displacement) + pradius**2*1.0e-2)
                )

    def laplace(self, vname, point):
        return 2*sum(
               (getattr(neighbor, vname)- getattr(point, vname))
                * self.laplace_marco(neighbor.displacement(point))
                /neighbor.density
                for neighbor in self.neiborghs(point) if neighbor != point
        )


class SPH_APP(SPH, tk.Canvas):
    scale = 25.0
    radius = 4.0
    def __init__(self, points, dt, nsnapshot=None):
        SPH.__init__(self, points)
        self.dt = dt
        self.nsnapshot = nsnapshot

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

    def save_eps(self):
        output = '{0:s}-{1:04d}.eps'.format(Path(sys.argv[0]).stem, self.nstep)
        self.postscript(file=output)

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

    def loop_func(self):
        while True:

            for point, oval in zip(self.points, self.ovals):
                self.coords(oval, *self.coords_oval(point))
            self.itemconfig(self.timer, text = f't = {self.t:.02f}')
            self.update()

            if self.nsnapshot is not None:
                if self.nstep%self.nsnapshot == 0:
                    self.save_eps()

            self.step(dt=self.dt)

    def run(self):
        self.root.after(0, self.loop_func)
        tk.mainloop()

