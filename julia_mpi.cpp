/*
*  Julia set (C++)
*
*  = Auto SIMD vectorization + MPI
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#define SAVE_IMAGE 1

using namespace std ;

const int MAX_ITER = 255 ;
const float z_Re_min = -0.8 ;
const float z_Re_max =  0.8 ;
const float z_Im_min = -0.8 ;
const float z_Im_max =  0.8 ;
const float c_Re = -0.622772 ;
const float c_Im =  0.42193  ;

int main(int argc, char *argv[]){

    int i ;

    MPI_Status mpi_status ;
    MPI_Init(&argc, &argv);

    int mpi_id  ; // Rank of thie MPI process
    int mpi_num ; // Total number of MPI process
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_num);

    int N_RE, N_IM ;
    if( argc==3 ){
      N_RE = atoi(argv[1]) ;
      N_IM = atoi(argv[2]) ;
    }else{
      std::printf("Incorrect input argument -- Need N_RE and N_IM.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Divide work between MPI processes
    long npoint = (long)ceilf((float)N_RE*N_IM/mpi_num) ;
    long base = npoint*mpi_id ;
    if( mpi_id == mpi_num-1 ) npoint = (long)N_RE*N_IM - base ;

    int *nIter ;
    if( mpi_id == 0 ){
        nIter = (int*)calloc( (long)N_RE*N_IM, sizeof(int));
    }else{
        nIter = (int*)calloc( npoint, sizeof(int));
    }


    const double start_time = MPI_Wtime();

    // *** Julia set ***
    for(i=0 ; i<npoint ; ++i)
    {
        long xi = (base+i) % N_RE ;
        long yj = (long) (base + i - xi)/N_RE ;

        float z_Re = z_Re_min + ( (float)xi/(float)N_RE ) * (z_Re_max - z_Re_min) ;
        float z_Im = z_Im_max - ( (float)yj/(float)N_IM ) * (z_Im_max - z_Im_min) ; // Reversed
        float re, im ;

        int k=0;
        while( k<MAX_ITER )
        {
            if( z_Re*z_Re + z_Im*z_Im  > 4.0 ) break ;
            re = z_Re ;
            im = z_Im ;
            z_Re = re*re - im*im + c_Re ;
            z_Im = 2*re*im + c_Im ;
            k++ ;
        }

        nIter[i] = k ;
        //printf("Local i=%d of MPI rank %d is assigned to thread %d\n", i, mpi_id, thread_id);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if( mpi_id==0 ){
        std::printf("%d MPI procs :: Image size %d x %d :: Elapsed time = %f seconds.\n", mpi_num, N_RE, N_IM, MPI_Wtime()-start_time);
    }


    // Write out results
    if( SAVE_IMAGE ){
        if( mpi_id == 0 ){

            int worker ;
            for( worker=1 ; worker<mpi_num-1 ; ++worker )
            {
                // Most worker
                base = npoint*worker ;
                MPI_Recv(&nIter[base], npoint, MPI_INT, worker, 0, MPI_COMM_WORLD, &mpi_status);
            }
            // Last worker == mpi_num-1
            base = npoint*(mpi_num-1) ;
            npoint = (long)N_RE*N_IM - base ;
            MPI_Recv(&nIter[base], npoint, MPI_INT, mpi_num-1, 0, MPI_COMM_WORLD, &mpi_status);

            FILE *outfile ;
            outfile = std::fopen("julia.pgm","w");
            std::fprintf(outfile,"P2\n#julia.pgm\n%d %d\n%d\n", N_RE, N_IM, MAX_ITER);

            int j ;
            for(j=0 ; j<N_IM ; ++j)
            {
            for(i=0 ; i<N_RE ; ++i)
            {
                std::fprintf(outfile,"%d ", nIter[ j*N_RE+i ]);
            }
            std::fprintf(outfile,"\n");
            }
            std::fclose(outfile);

        }else{
            MPI_Send(&nIter[0], npoint, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

    }

    std::free(nIter);
    MPI_Finalize();


return 0; }

