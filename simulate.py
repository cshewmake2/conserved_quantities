import numpy as np
import matplotlib
matplotlib.interactive(True)
import matplotlib.pyplot as plt
from scipy.integrate import odeint

x0 = [1,0]
t = np.linspace(0,10,1000)

def f(x,t):
    return [x[1],-x[0]]

x_t = odeint(f, x0, t)

plt.plot(x_t[:,0],x_t[:,1])
