#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-schuberm
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=1GB
#SBATCH --gres=gpu:a100_3g.20gb:1
              

#SBATCH --time=03:00:00

module load python/3 cuda/12.2 
module load cudacore/.12.2.2


export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/lib64:$LD_LIBRARY_PATH


python CPU_COMB.py