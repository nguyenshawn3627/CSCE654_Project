#!/bin/bash
#SBATCH -N 1                 # number of nodes
#SBATCH -C cpu               # use CPU-only nodes
#SBATCH -q regular           # queue/partition
#SBATCH -t 00:05:00          # time limit HH:MM:SS
#SBATCH -J blockbpe          # job name
#SBATCH -o blockbpe_%j.out   # STDOUT (%j = job ID)
#SBATCH -e blockbpe_%j.err   # STDERR

# Load required programming environment
module load PrgEnv-gnu
module load nlohmann-json

# Set OpenMP parameters
export OMP_NUM_THREADS=64          # Perlmutter CPU nodes have 64 cores
export OMP_PLACES=cores
export OMP_PROC_BIND=spread        # helps improve scaling

echo "Running BlockBPE with ${OMP_NUM_THREADS} threads"
echo "----------------------------------------------"

# Run your executable
./blockbpe
