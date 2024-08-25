/*
*  Julia set (C++)
*
*  = Auto SIMD vectorization + OpenMP
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

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

    int N_RE, N_IM ;
    if( argc==3 ){
      N_RE = atoi(argv[1]) ;
      N_IM = atoi(argv[2]) ;
    }else{
      std::printf("Incorrect input argument -- Need N_RE and N_IM.\n");
      return 1;
    }

    long npoint = (long) N_RE*N_IM ;
    int *nIter = (int*)calloc( npoint, sizeof(int));
    int num_threads ;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // *** Julia set ***
    #pragma omp parallel firstprivate(nIter)
    {
    num_threads = omp_get_num_threads() ;
    //int thread_id = omp_get_thread_num();
    //printf("Hello from MPI rank %d thread %d\n", mpi_id, thread_id);

    #pragma omp for
    for(i=0 ; i<npoint ; ++i)
    {
        long xi = i % N_RE ;
        long yj = (long) (i - xi)/N_RE ;

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
    }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::printf("%d OMP threads :: Image size %d x %d :: Elapsed time = %f seconds.\n", num_threads, N_RE, N_IM, static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f);

    // Write out results
    if( SAVE_IMAGE ){
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
    }

    std::free(nIter);

return 0; }

