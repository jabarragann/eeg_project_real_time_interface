from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

class Scope(object):
    def __init__(self, ax, maxt=10, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.line.set_linestyle('--')
        self.line.set_marker('*')
        self.line.set_markersize(2)
        self.line.set_color('red')


        self.ax.add_line(self.line)

        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

        self.init_time = time.time()

        self.init_time2 = time.time()
        self.total_time = 0

        self.start_removing = False

    def update(self, y):

        #print(y)

        #Get Last Sample
        lastt = self.tdata[-1]

        # Calculate current time
        self.dt = time.time() - self.init_time
        self.total_time += self.dt
        #print(self.dt)

        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.start_removing=True
            # self.tdata = [self.tdata[-1]]
            # self.ydata = [self.ydata[-1]]
            # self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            # #self.ax.figure.canvas.draw()
        if self.start_removing:
            self.tdata = self.tdata[1:]
            self.ydata = self.ydata[1:]
            self.ax.set_xlim(self.tdata[0], self.tdata[-1] + self.dt*10)


        #Calculate new sample
        new_y = self.sineSignal()

        #Update artists data
        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(new_y)
        self.line.set_data(self.tdata, self.ydata)

        #Reset init time
        self.init_time = time.time()


        return self.line,

    def init_animation(self):
        self.init_time = time.time()
        self.init_time2 = time.time()
        return self.line,

    def sineSignal(self):
        return np.clip(np.sin(0.5*np.pi * self.total_time), -0.5,0.5)



def emitter(p=0.03):
    'return a random value with probability p, else 0'
    while True:
        v = np.random.rand(1)
        if v > p:
            yield [0.,1.,2.]
        else:
            yield np.random.rand(1)


# Fixing random state for reproducibility
np.random.seed(19680801)

initTime= time.time()
fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, func = scope.update, frames = emitter, init_func=scope.init_animation, interval=10, blit=True)

print("animation")
plt.show()
print("Closing")