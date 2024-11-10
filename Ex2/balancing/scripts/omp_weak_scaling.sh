#!/bin/bash
#SBATCH --job-name=HPC-OMP-Scaling
#SBATCH --nodes=1                
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=128       
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive
 
mpicc -fopenmp mandelbrot.c -o mandelbrot -lm
module load openMPI/4.1.6/gnu/14.2.1

# Output file
OUTPUT_FILE="../csvs/omp_weak_scaling.csv"
echo "cores,threads,width,height,time" > $OUTPUT_FILE

# Parameters for the image
X_LEFT=-2.0
Y_LOWER=-1.0
X_RIGHT=1.0
Y_UPPER=1.0
MAX_ITERATIONS=255
C=1000000

for THREADS in {1..128}; do
    
    export OMP_NUM_THREADS=$THREADS
    n=$(echo "scale=2; sqrt($THREADS * $C)" | bc)
    export OMP_PLACES=threads
    export OMP_PROC_BIND=true

    # Execution with just 1 process MPI (bind to/map by socket is necessary)
    EXEC_TIME=$(mpirun -np 1 ./mandelbrot --map-by socket --bind-to socket $n $n $X_LEFT $Y_LOWER $X_RIGHT $Y_UPPER $MAX_ITERATIONS $THREADS)

    # Savings
    echo "1,$THREADS,$n,$n,$EXEC_TIME" >> $OUTPUT_FILE

    # Control output
    echo "Eseguito con cores=1, threads=$THREADS, width=$n, height=$n, time=$EXEC_TIME"
done
