#!/bin/bash
#SBATCH --job-name=HPC-MPI-Weak-Scaling
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive

module load openMPI/4.1.6/gnu/14.2.1
mpicc -fopenmp mandelbrot.c -o mandelbrot -lm

OUTPUT_FILE="../csvs/mpi_weak_scaling.csv"
echo "cores,threads,width,height,time" > $OUTPUT_FILE

# Using just 1 thread
export OMP_NUM_THREADS=1

# Parameters for the image
X_LEFT=-2.0
Y_LOWER=-1.0
X_RIGHT=1.0
Y_UPPER=1.0
MAX_ITERATIONS=255
C=1000000

for CORES in {1..256}; do

    # Width and height of the image proportional to the number of cores
    n=$(echo "scale=2; sqrt($CORES * $C)" | bc)

    EXEC_TIME=$(mpirun -np $CORES --map-by core --bind-to core ./mandelbrot $n $n $X_LEFT $Y_LOWER $X_RIGHT $Y_UPPER $MAX_ITERATIONS 1)

    # Savings
    echo "$CORES,1,$n,$n,$EXEC_TIME" >> $OUTPUT_FILE

    # Control output
    echo "Eseguito con cores=$CORES, threads=1, width=$n, height=$n, time=$EXEC_TIME"
done

