#/bin/bash

echo "are you sure?"
read arg
if [ $arg == "yes" ]
then
    rm -rf simulation.* *COLVAR *KERNELS *RESTART_INFO  RESTART *.out log.lammps time_wall \#*
fi
