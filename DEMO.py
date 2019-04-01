# A simple DEM DEMO code for hands-on experiences
# in using Discrete Element Methods
# Version 0.1, by Yixiang Gan
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# Material and Geometry
BoxSizeX = 2.0
BoxSizeY = 2.0
ParticleDensity = 1.0
ParticleSize = 0.3
Stiffness = 100.0
Dissipation = 0.05  # COR
Gravity = 9.8
ParticleN = 10
ParticleMass = np.pi * ParticleDensity * ParticleSize**2

# Time increment
dt = 0.25 * np.sqrt(ParticleMass / Stiffness)

# Class Definitions


class ParticleBox:
    #    init_state is an [N x 4] array, where N is the number of particles:
    #       [[x1, y1, vx1, vy1],
    #        [x2, y2, vx2, vy2],
    #        ...               ]
    #    bounds is the size of the box: [xmin, xmax, ymin, ymax]

    def __init__(self,
                 init_state=[[1, 0, 0, -1],
                             [-0.5, 0.5, 0.5, 0.5],
                             [-0.5, -0.5, -0.5, 0.5]],
                 bounds=[-BoxSizeX, BoxSizeX, -BoxSizeY, BoxSizeY],
                 size=ParticleSize,
                 M=ParticleMass,
                 G=Gravity):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        # Update Particle Positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # Contact Detection
        D = squareform(pdist(self.state[:, :2], 'euclidean'))
        ind1, ind2 = np.where(D < 2.0 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # Contact Pairs, Contact Laws
        # (linear elasticity, no friction, no rotation)
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]
            dist = D[i1, i2]
            overlap = 2.0 * self.size - dist
            force = Stiffness * overlap

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            nr_rel = r_rel / dist  # unit vector
            v_rel = v1 - v2

            # collisions of spheres reflect v_rel over r_rel
            # rr_rel = np.dot(r_rel, r_rel)
            # vr_rel = np.dot(v_rel, r_rel)
            # v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] += force / m1 * nr_rel * dt
            self.state[i2, 2:] -= force / m2 * nr_rel * dt

            # add local dissipation
            self.state[i1:, 2:] -= Dissipation * v_rel * dt
            self.state[i2:, 2:] += Dissipation * v_rel * dt

        # check for boundaries
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -(1.0 - Dissipation)
        self.state[crossed_y1 | crossed_y2, 3] *= -(1.0 - Dissipation)

        # add gravity
        self.state[:, 3] -= self.G * dt


# Initialisation
#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((ParticleN, 4))
init_state[:, :2] *= 1.9 * BoxSizeX

box = ParticleBox(init_state, size=ParticleSize)

# Visualisation
#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1.0 * BoxSizeX, 1.0 * BoxSizeX),
                     ylim=(-1.0 * BoxSizeY, 1.0 * BoxSizeY))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'o', ms=6,)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)


def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect


def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)
    EKine = 0.5 * ParticleMass * np.sum(box.state[:, 2:]**2)
    ms = int(fig.dpi * 0.5 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    time_text.set_text('time = %.2f \ntime step = %f\nkinetic energy = %.2f ' % (box.time_elapsed, dt, EKine))

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect, time_text


ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)

plt.show()

# Outputs and Results Analysis
