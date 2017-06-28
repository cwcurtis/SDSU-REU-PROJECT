import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import afm_dno_solver_suit as afm_dno
import time

K = 128
Llx = 10
tf = 40.
dt = 1e-3
nsteps = int(np.floor(tf/dt))
KT = 2*K
dx = 2.*Llx/KT
X = np.arange(-Llx, Llx, dx)

ep = .1
mu = np.sqrt(.1)
Mval = 10

usol = pyfftw.empty_aligned(KT,dtype='complex128')

t0 = time.time()
usol[:] = afm_dno.afm_dno_solver(K, ep, mu, Llx, tf, Mval, dt)
t1 = time.time()

print t1-t0

plt.plot(X,usol)
plt.show()