from glob import glob
import numpy as np
from scipy import integrate, interpolate
from sys import argv
if len(argv) > 1:
    import matplotlib.pyplot as plt

nblocks = 10000
lambdas = list()
mean_forces = list()

dirs = glob("0.*") + ["1.000"]
for d in dirs:
    if d == '0.000'  or d == '1.000':
        continue
    lambdas.append(float(d))
    with open(f"{d}/driver.out") as fp:
        dEdl = list()
        for line in fp:
            dEdl.append(float(line.split()[-1]))
        ntot = len(dEdl)
        nchunk = ntot // nblocks
        mean_force = list()
        for i in range(nblocks):
            mean_force.append(np.mean(dEdl[i*nchunk:(i+1)*nchunk]))
        mean_forces.append(mean_force)

lambdas = np.array(lambdas)
mean_forces = np.array(mean_forces)
order = np.argsort(lambdas)
lambdas = lambdas[order]
mean_forces = mean_forces[order]

for iblock in range(nblocks):
    f = interpolate.interp1d(lambdas, mean_forces[:,iblock].ravel(), kind='cubic', fill_value='extrapolate')
    F = 0
    for x in np.arange(0.1, 0.9, 0.1):
        F += integrate.quad(f, x, x + 0.1)[0]
    F += integrate.quad(f, 0.001, 0.1)[0]
    F += integrate.quad(f, 0.9, 0.999)[0]
    F += 0.001 * f(0.001)
    F += 0.001 * f(0.999)
    print(F)
    if len(argv) > 1:
        plt.plot(np.linspace(0,1,100), f(np.linspace(0,1,100)))
        plt.scatter(lambdas, mean_forces[:,iblock].ravel())
if len(argv) > 1:
    plt.show()

