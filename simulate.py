import numpy as np
import matplotlib
matplotlib.interactive(True)
import matplotlib.pyplot as plt

from scipy.integrate import odeint

x0 = [1,1]
t = np.linspace(0,5,1000)

def f(x,t):
    return [x[0],-x[1]]

x_t = odeint(f, x0, t)

plt.plot(t,x[:,0])
