import re
from glob import glob
import numpy as np
from scipy import integrate, interpolate
from sys import argv
if len(argv) > 2:
    import matplotlib.pyplot as plt

nblocks = int(argv[1])
lambdas = list()
mean_forces = list()

dirs = glob("0.*") + ["1.000"]
print("# lambda windows:")
for d in dirs:
    try:
        lambdas.append(float(d))
        print("#", d)
    except:
        continue
    with open(f"{d}/driver.out") as fp:
        dEdl = list()
        for line in fp:
            if line[:4] == 'dhdl':
                dEdl.append(float(line.split()[-1]))
        ntot = len(dEdl)
        nchunk = ntot // nblocks
        mean_force = list()
        for i in range(nblocks):
            mean_force.append(np.mean(dEdl[i*nchunk:(i+1)*nchunk]))
        mean_forces.append(mean_force)
print("# -------------------------------------")

lambdas = np.array(lambdas)
mean_forces = np.array(mean_forces)
order = np.argsort(lambdas)
lambdas = lambdas[order]
mean_forces = mean_forces[order]

for iblock in range(nblocks):
    f = interpolate.interp1d(lambdas, mean_forces[:,iblock].ravel(), kind='cubic', fill_value='extrapolate')
    F = 0
    for x in np.arange(0, 1, 0.1):
        F += integrate.quad(f, x, x + 0.1)[0]
    print(F)
    if len(argv) > 2:
        plt.plot(np.linspace(0,1,100), f(np.linspace(0,1,100)))
        plt.scatter(lambdas, mean_forces[:,iblock].ravel())
if len(argv) > 2:
    plt.show()

np.savetxt('mean_force.dat', np.transpose([lambdas, np.mean(mean_forces, axis=1), np.std(mean_forces, axis=1)]))
