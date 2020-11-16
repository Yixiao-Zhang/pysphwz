#!/usr/bin/env python3

import tkinter as tk
import time
import logging

import numpy as np

domain = np.array([16.0, 16.0])
mid_domain = domain/2
pradius = 2.0

class Point:
    def __init__(self, position:np.ndarray, velocity:np.ndarray):
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
        if dt is None:
            vmax = max(np.linalg.norm(point.velocity) for point in self.points)
            dt = 0.4*pradius/max(vmax, 1.0)

        for point in self.points:
            point.move(dt)

        for point in self.points:
            point.accelarate(dt, self)

        self.t += dt

class SFH_APP(SFH, tk.Canvas):
    scale = 32.0
    radius = 4.0
    def __init__(self, points):
        SFH.__init__(self, points)

        self.root = tk.Tk()
        tk.Canvas.__init__(self, self.root,
            width=domain[0]*self.scale, height=domain[1]*self.scale)
        self.ovals = self.plot_points()
        self.timer = self.create_text(20, 20, anchor=tk.NW,
            font="Courier 10 bold",
            text="")
        self.pack()

    def coords_oval(self, point):
        x, y = point.position*self.scale
        return (x - self.radius, y - self.radius,
                x + self.radius, y + self.radius)

    def plot_points(self):
        ovals = list()
        for point in self.points:
            oval = self.create_oval(*self.coords_oval(point),
                fill='#476042')
            ovals.append(oval)
        return tuple(ovals)

    real_time_sep = 0.2
    def loop_func(self):
        time_next = time.time()
        while True:
            self.step(dt=0.05)
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

    class LiquidPoint(Point):
        def force(self, model):
            res = np.array([0.0, 0.0])
            for point in model.neiborghs(self):
                displacement = self.displacement(point)
                distance = np.linalg.norm(displacement)
                if distance < pradius:
                    res += -1.0*displacement/max(distance, 1.0e-3)
            return res

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self.update_subdomains()

    # make sure that the subdomain is large enough to capture
    # partical-partical interactions
    nsubs = (32, 32)
    def update_subdomains(self):
        self.subs = [[set() for _ in range(nsub)] for nsub in self.nsubs]
        for point in self.points:
            for i, subi in enumerate(self.subdomain_index(point)):
                self.subs[i][subi].add(point)

    def subdomain_index(self, point):
        return (int(point.position[i]*self.nsubs[i]/domain[i])
                    for i in range(len(self.nsubs)))

    def neiborghs(self, point):
        sets = []
        for i, subi in enumerate(self.subdomain_index(point)):
            sets.append(set.union(*(self.subs[i][(subi+k)%self.nsubs[i]]
                    for k in (-1, 0, 1))))
        return set.union(*sets)


def main():
    npoint = 256
    xs = np.random.uniform(0.0, domain[0], npoint)
    ys = np.random.uniform(0.0, domain[1], npoint)
    positions = map(np.array, map(list, zip(xs, ys)))

    vxs = np.random.uniform(-1.0, 1.0, npoint)
    vys = np.random.uniform(-1.0, 1.0, npoint)
    velocities = map(np.array, map(list, zip(vxs, vys)))

    points = tuple(map(LiquidApp.LiquidPoint, positions, velocities))

    model = LiquidApp(points)
    model.run()


if __name__ == '__main__':
    main()
