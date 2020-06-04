#include "heat.h"

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey, int mpi_id, int mpi_size)
{
    double diff, sum=0.0;

    for( int i=1; i<sizex-1; i++ ) {
        for( int j=1; j<sizey-1; j++ ) {
            utmp[i*sizey + j] = 0.20 * (u[ i*sizey     + j     ]+  // center
                    u[ i*sizey     + (j-1) ]+  // left
                    u[ i*sizey     + (j+1) ]+  // right
                    u[ (i-1)*sizey + j     ]+  // top
                    u[ (i+1)*sizey + j     ]); // bottom

            diff = utmp[i*sizey + j] - u[i*sizey + j];
            sum += diff * diff;
        }
    }

    /*
    if (mpi_id == 1) {
    //    for( int i=1; i<sizex-1; i++ ) {
            for( int j=1; j<sizey-1; j++ ) {
                fprintf(stdout, "%f ", utmp[1*sizey + j]);
            }
            fprintf(stdout,"\n");
     //   }
    }
    */
    // mpi sharing rows for next iteration

    MPI_Request row_send_req[2];
    MPI_Request row_recv_req[2];

    if (mpi_id < (mpi_size - 1)) {
        MPI_Isend(&utmp[(sizex - 2) * sizey], sizey, MPI_DOUBLE, mpi_id + 1, 3, MPI_COMM_WORLD, &row_send_req[1]);
        /*
        if (mpi_id == 1) {
            fprintf(stdout, "Process %d fila -1 abans:\n", mpi_id);
            for(int i = 0; i < sizey; i++) {
                fprintf(stdout, "%f ", utmp[(sizex-1)*sizey + i]);
            }
            fprintf(stdout, "\n");
        }
        */
        MPI_Irecv(&utmp[(sizex - 1) * sizey], sizey, MPI_DOUBLE, mpi_id + 1, 2, MPI_COMM_WORLD, &row_recv_req[1]);
    }
    if (mpi_id) {
        MPI_Irecv(&utmp[0], sizey, MPI_DOUBLE, mpi_id - 1, 3, MPI_COMM_WORLD, &row_recv_req[0]);
        MPI_Isend(&utmp[1*sizey], sizey, MPI_DOUBLE, mpi_id - 1, 2, MPI_COMM_WORLD, &row_send_req[0]);
    }

    if (mpi_id < mpi_size - 1) {
        MPI_Wait(&row_send_req[1], MPI_STATUS_IGNORE);
        MPI_Wait(&row_recv_req[1], MPI_STATUS_IGNORE);
    }
    if (mpi_id) {
        MPI_Wait(&row_send_req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&row_recv_req[0], MPI_STATUS_IGNORE);
    }
    /*
    if (mpi_id == 1) {
        fprintf(stdout, "Process %d fila -1 despres:\n", mpi_id);
        for(int i = 0; i < sizey; i++) {
            fprintf(stdout, "%f ", utmp[(sizex-1)*sizey + i]);
        }
        fprintf(stdout, "\n");
    }
    */
/*
    if (mpi_id == 1) {
    //    for( int i=1; i<sizex-1; i++ ) {
            for( int j=1; j<sizey-1; j++ ) {
                fprintf(stdout, "%f ", utmp[1*sizey + j]);
            }
            fprintf(stdout,"\n");
     //   }
    }
    */
    // mpi sharing sum result to decide if to continue computing

    double sum_reduced = sum;
    MPI_Allreduce(&sum, &sum_reduced, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum_reduced;
}
