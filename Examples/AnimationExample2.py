import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 6), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

def init():
    print("init")
    line.set_data([], [])
    return line,

def animate(i):
    print(i)
    x, y = line.get_data()
    print(len(y))
    new_x = (1/60)*i
    new_y = np.sin(2 * np.pi * new_x)
    x.append(new_x)
    y.append(new_y)
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=250, interval=20, blit=True)
plt.show()