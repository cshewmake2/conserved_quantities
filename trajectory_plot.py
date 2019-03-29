import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class TrajectoryPlot(object):

    def __init__(self, ax, dt, ax_size = 1, win_size = 10):
        self.ax = ax
        self.axis_size = ax_size
        self.window_size = win_size
        self.dt = dt
        self.x1_frame = []
        self.x2_frame = []

        # Add line to plot
        self.line = Line2D(self.x1_frame, self.x2_frame,color='b')
        self.ax.add_line(self.line)

        # Set initial axis limits
        self.ax.set_ylim(-self.axis_size/2,self.axis_size/2)
        self.ax.set_xlim(-self.axis_size/2,self.axis_size/2)

    def update_limits(self,x1_val,x2_val):
        x_min,x_max = self.ax.get_xlim()
        if x1_val < x_min:
            x_min = x1_val - 0.1
            self.ax.set_xlim(x_min, x_max) # or 0 to 255, or min in history to max in history (keep in object as m_vars)
        if x1_val > x_max:
            x_max = x1_val + 0.1
            self.ax.set_xlim(x_min, x_max) # or 0 to 255, or min in history to max in history (keep in object as m_vars)

        y_min,y_max = self.ax.get_ylim()
        if x2_val < y_min:
            y_min = x2_val - 0.1
            self.ax.set_ylim(y_min, y_max) # or 0 to 255, or min in history to max in history (keep in object as m_vars)
        if x2_val > y_max:
            y_max = x2_val + 0.1
            self.ax.set_ylim(y_min, y_max) # or 0 to 255, or min in history to max in history (keep in object as m_vars)

    def update_buffer(self):
        if len(self.x1_frame)*self.dt >= self.window_size: # after the first time hitting the boundary
            self.x1_frame.pop(0)
            self.x2_frame.pop(0)
            self.line.set_data(self.x1_frame,self.x2_frame)

    def update(self, x1_val, x2_val):
        # append new signal values to frame
        self.x1_frame.append(x1_val)
        self.x2_frame.append(x2_val)
        self.line.set_data(self.x1_frame,self.x2_frame)

        self.update_limits(x1_val,x2_val)
        self.update_buffer()

        return self.line
