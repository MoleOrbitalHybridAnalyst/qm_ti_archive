import numpy as np
from constants import kb, hartree, kcal

mp2 = np.loadtxt('./mp2_rerun.out')
ene0 = np.loadtxt('./simulation.out', usecols=3)

beta = 1 / (kb * 310.15 * kcal / hartree)
expE = np.exp(-beta * (mp2 - ene0))
dF0 = -1/beta * np.log(np.mean(expE))
print("Correction to F0", dF0)
print("Correction to F1-F0", -dF0)
