import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import auto_encoder as ae

x0 = [1,0]
t = np.linspace(0,10,1000)

def f(x,t):
    return [x[1],-x[0]]


x_t = odeint(f, x0, t)


x_dot = [f(x,0) for x in x_t]

x_dot = np.array(x_dot)

[autoencoder, encoder, decoder] = ae.train_encoder(x_t, 1 , 1 , 0)

x_hat = autoencoder.predict(x_t)

z_t = encoder.predict(x_t)

plt.plot(x_t[:,0],x_t[:,1])
plt.plot(x_hat[:,0],x_hat[:,1])
#plt.plot(z_t[:,0], z_t[:,1])
plt.show()


plt.plot(t, z_t)
plt.show()