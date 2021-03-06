/*
 * Iterative solver for heat distribution
 */

#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, "Usage: %s <input file> [result file]\n\n", s);
}

int distribute_rows (int mpi_id, int mpi_ranks, int rows) {
    return (rows/mpi_ranks) + ((rows % mpi_ranks) > mpi_id);
}

int main( int argc, char *argv[] )
{
    int columns, rows;
    int iter, maxiter;
    double residual=0.0;

    int myid, numprocs, len;
    MPI_Status status;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(hostname, &len);

    if (myid == 0) {
        printf("I am the master running on %s, distributing work to %d additional workers ...\n", hostname, numprocs-1);

        // Input and output files
        FILE *infile, *resfile;
        char *resfilename;

        // algorithmic parameters
        algoparam_t param;

        double runtime, flop;

        // check arguments
        if( argc < 2 ) {
            usage( argv[0] );
            return 1;
        }

        // check input file
        if( !(infile=fopen(argv[1], "r")) ) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
            usage(argv[0]);
            return 1;
        }

        // check result file
        resfilename= (argc>=3) ? argv[2]:"heat.ppm";

        if( !(resfile=fopen(resfilename, "w")) ) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);
            usage(argv[0]);
            return 1;
        }

        // check input
        if( !read_input(infile, &param) ) {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        if( !initialize(&param) ) {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        maxiter = param.maxiter;
        // full size (param.resolution are only the inner points)
        columns = param.resolution + 2;
        rows = columns;
        int local_rank = distribute_rows(0, numprocs, param.resolution) + 2;


        // starting time
        runtime = wtime();

        // send to workers the necessary information to perform computation
        for (int i=1; i<numprocs; i++) {
                int sent_rows = distribute_rows(i, numprocs, param.resolution) + 2;
        //fprintf(stdout, "index %d \n", i*(sent_rows-3));
                MPI_Send(&maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // broadcast?
                MPI_Send(&columns, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // broadcast?
                MPI_Send(&sent_rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.u[i * (sent_rows - 2) * columns], sent_rows*columns, MPI_DOUBLE, i, 5, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[i * (sent_rows - 2) *columns], sent_rows*columns, MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
        }
        iter = 0;

        while(1) {
            residual = relax_jacobi(param.u, param.uhelp, local_rank, columns, myid, numprocs);
            // Copy uhelp into u, uhelp contains solver result
            double * tmp = param.u; param.u = param.uhelp; param.uhelp = tmp;

            iter++;
            
            // solution good enough ?
            if (residual < 0.00005) break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (maxiter>0 && iter>=maxiter) break;
        }

        // receive information
        for (int i = 1; i < numprocs; i++) {
            int sent_rows = distribute_rows(i, numprocs, param.resolution);

        fprintf(stdout, "index %d del proc %d\n", i*sent_rows, myid);
            MPI_Recv(&param.u[i * sent_rows * columns], sent_rows * columns, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }



        // stopping time
        runtime = wtime() - runtime;

        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;

        fprintf(stdout, "Time: %04.3f \n", runtime);
        fprintf(stdout, "Flops and Flops per second: (%3.3f GFlop => %6.2f MFlop/s)\n",
                flop/1000000000.0,
                flop/runtime/1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // for plot...
        if (param.resolution < 1024) {
            coarsen( param.u, rows, columns, param.uvis, param.visres+2, param.visres+2 );
            write_image( resfile, param.uvis, param.visres+2, param.visres+2 );
        }

        finalize( &param );

        MPI_Finalize();
        return 0;
    } 
    else {
        printf("I am worker %d on %s and ready to receive work from master ...\n", myid, hostname);

        // receive information from master to perform computation locally
        MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // allocate memory for worker
        double * u     = calloc( sizeof(double),rows*columns );
        double * uhelp = calloc( sizeof(double),rows*columns );

        if( !u || !uhelp ) {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }

        // fill initial values for matrix with values received from master
        MPI_Recv(&u[0], rows*columns, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(&uhelp[0], rows*columns, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);

        //fprintf(stdout, "PROC %d; columns: %d, rows:%d\n", myid, columns, rows);
        iter = 0;
        while(1) {
            residual = relax_jacobi(u, uhelp, rows, columns, myid, numprocs);
            // Copy uhelp into u
            double * tmp = u; u = uhelp; uhelp = tmp;

            iter++;


            // solution good enough ?
            if (residual < 0.00005) break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (maxiter>0 && iter>=maxiter) break;
        }

        fprintf(stdout, "Process %d finished computing after %d iterations with residual value = %f\n", myid, iter, residual);

        fprintf(stdout, "PROC %d; index %d\n", myid, 1*columns);
    if (myid == 1) {
    //    for( int i=1; i<sizex-1; i++ ) {
            for( int j=1; j<columns-1; j++ ) {
                fprintf(stdout, "%f ", u[1*columns + j]);
            }
            fprintf(stdout,"\n");
     //   }
    }
        MPI_Send(&u[1*columns], (rows - 2)*columns, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        MPI_Finalize();
        return 0;
    }
}
