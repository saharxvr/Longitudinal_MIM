import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import numpy as np
from constants import PROJECT_FOLDER


def rotate_ax(fig, ax, path):
    def get_rotate(c_ax):

        def rotate(angle):
            if angle < 360:
                c_ax.view_init(azim=angle)
            else:
                c_ax.view_init(elev=angle - 360, azim=359)

        return rotate

    angle_interval = 3

    ani = animation.FuncAnimation(fig, get_rotate(ax), frames=np.arange(0, 720, angle_interval), interval=2000)
    ani.save(path, writer=animation.PillowWriter(fps=20))