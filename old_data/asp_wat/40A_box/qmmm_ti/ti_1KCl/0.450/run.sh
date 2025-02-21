export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2

# latest plumed that supports opes flooding
source ~/packages/plumed-2.9.0/sourceme.sh

# important to use latest ipi to work with latest plumed
export PYTHONPATH=/home/chhli/packages/i-pi-2.6.1:$PYTHONPATH
ipi=/home/chhli/packages/i-pi-2.6.1/bin/i-pi

$ipi nvt.xml > nvt.out &
echo "$!" > ipi.pid
sleep 60

~/packages/lammps-2Aug2023/build/lmp -in lmp.in > lmp.out &

env OMP_NUM_THREADS=8 python -u driver.py 0.450 > driver.out
