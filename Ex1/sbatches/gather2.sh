#!/bin/bash
#SBATCH --job-name=HPC
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:59:58
#SBATCH --partition EPYC
#SBATCH --exclusive

module load openMPI/4.1.6/gnu/14.2.1

echo "Processes,Size,Latency" > gather2.csv

# #interazions
repetitions=10000

for power in {1..8}
do
	processes=$((2**power))
	for size_power in {1..18}
	do
		size=$((2**size_power))
		result_gather=$(mpirun --map-by core -np $processes --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_gather_algorithm 2 ../../osu_gather -m $size -x $repetitions -i $repetitions | tail -n 1 | awk '{print $2}')
		
		echo "$processes,$size,$result_gather"
		echo "$processes,$size,$result_gather" >> gather2.csv
	done
done	
