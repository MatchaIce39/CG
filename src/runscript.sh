#!/bin/bash

#SBATCH --output=fileoutput.out
#SBATCH --error=fileerror.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=debug

cd /users/njohnson77/CG/src

srun -n 4 ./cg
