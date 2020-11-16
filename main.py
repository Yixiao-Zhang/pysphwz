#!/usr/bin/env python3

import tkinter as tk
import time

import numpy as np

domain = np.array([16.0, 16.0])
radias = 1.0

class Point:
    def __init__(self, position:np.ndarray, velocity:np.ndarray):
        self.position = position
        self.velocity = velocity

    def move(self, dt:float):
        self.position += dt*self.velocity
        self.position %= domain

    def accelarate(self, dt:float):
        self.velocity += dt*self.force()

    def force(self):
        return 0.0

class SFH:
    def __init__(self, points, t0=0.0):
        self.points = points
        self.t = t0

    def step(self, dt=None):
        if dt is None:
            vmax = max(np.linalg.norm(point.velocity) for point in self.points)
            dt = 0.4*radias/max(vmax, 1.0)

        for point in self.points:
            point.move(dt)

        for point in self.points:
            point.accelarate(dt)

        self.t += dt

class PointGravity(Point):
    def force(self):
        return np.array([0.0, 5.0 if self.position[1] < domain[1]/2 else -5.0])

class SFH_APP(SFH, tk.Canvas):
    scale = 32.0
    radius = 4.0
    def __init__(self, points, root):
        SFH.__init__(self, points)
        tk.Canvas.__init__(self, root,
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

    def loop_func(self):
        time_next = time.time()
        while True:
            self.step(0.1)
            for point, oval in zip(self.points, self.ovals):
                self.coords(oval, *self.coords_oval(point))
            self.itemconfig(self.timer, text = f't = {self.t:.02f}')
            self.update()
            time_next += 0.05
            time_sleep = time_next - time.time()
            time.sleep(max(0.0, time_sleep))


def main():
    npoint = 64
    xs = np.random.uniform(0.0, domain[1], npoint)
    ys = np.random.uniform(0.0, domain[1], npoint)
    positions = map(np.array, map(list, zip(xs, ys)))

    vxs = np.random.uniform(-1.0, 1.0, npoint)
    vys = np.random.uniform(-1.0, 1.0, npoint)
    velocities = map(np.array, map(list, zip(vxs, vys)))

    points = tuple(map(PointGravity, positions, velocities))

    root = tk.Tk()
    model = SFH_APP(points, root=root)
    root.after(100, model.loop_func)
    tk.mainloop()


if __name__ == '__main__':
    main()
