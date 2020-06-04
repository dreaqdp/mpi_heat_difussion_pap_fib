#!/bin/bash

if [ $# -ne 1 ]
then
    echo "USAGE: iterate.sh <mode>\n
                mode -> 0 (mpi_only), 1 (hybrid)\n"
    exit
fi
if [ $1 -eq 0 ]
then
    mkdir timing_mpi
    cd timing_mpi
else
    mkdir timing_hybrid
    cd timing_hybrid
fi

for nodes in {1..2}
do
#nodes=2
    ranks=$(( 1 * $nodes ))
    max_ranks=$(( 16 * $nodes ))

    while [ $ranks -le $max_ranks ]
    do
        name='nodes_'${nodes}'_ranks_'${ranks}
        mkdir $name
        cd $name

        for k in {1..5}
        do
            mkdir run_$k
            cd run_$k

            echo "nodes $nodes ranks $ranks "
            qsub -pe mpich $nodes -l execution2 ../../../submit-mpi.sh $nodes $ranks
            cd ..
        done

        cd ..
        ((ranks<<=1))
    done
done

