/*
*  Julia set (C++)
*
*  = Auto SIMD vectorization + MPI + Pthread
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <pthread.h>

#define SAVE_IMAGE 1

const int MAX_ITER = 255 ;
const float z_Re_min = -0.8 ;
const float z_Re_max =  0.8 ;
const float z_Im_min = -0.8 ;
const float z_Im_max =  0.8 ;
const float c_Re = -0.622772 ;
const float c_Im =  0.42193  ;

// *** JULIA ***

int num_threads ;
int N_RE, N_IM ;
int *nIter ;
long long int npoint, base ;

void *julia(void *detail)
{
  const int thread_id = *static_cast<int*>(detail);

  long long int num = static_cast<long long int>( ceil((double)npoint/num_threads) ) ;
  const long long int start = num*thread_id ;

  if( thread_id==num_threads-1 )
    num = npoint - num*thread_id ;

  int *ptr = &nIter[0] ;


  // *** Julia set ***
  for(long long int i=start ; i<start+num; ++i)
  {
      long xi = (base+i) % N_RE ;
      long yj = (long) (base+i - xi)/N_RE ;

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

      ptr[i] = k ;
  }


return nullptr; }


// *** PTHREAD ***

// Should implement error checking !!!
void dispatch_pthread(void *(*main_work)(void *args))
{
  bool normal = true ;

  int *worker_id = static_cast<int*>(malloc((num_threads-1)*sizeof(int)));
  pthread_t *worker = static_cast<pthread_t*>(malloc((num_threads-1)*sizeof(pthread_t)));

  int i ;
  for(i=0 ; i != num_threads-1 ; ++i)
  {
    worker_id[i] = i ;
    if( pthread_create(&worker[i], nullptr, main_work, &worker_id[i]) != 0 ){
      std::printf("Error cannot dispatch child thread %d.\n", i);
      normal = false ;
      break;
    }
  }

  if( normal ){
    int master_id = num_threads-1 ;
    main_work(&master_id);

    for(i=0 ; i != num_threads-1 ; ++i){
      pthread_join(worker[i], nullptr);
    }
  }else{
    for(--i; i>=0 ; --i){
      pthread_cancel(worker[i]);
    }
  }


  std::free(worker_id);
  std::free(worker);

return; }


// *** MAIN ***

int main(int argc, char *argv[]){

  MPI_Status mpi_status ;
  MPI_Init(&argc, &argv);

  int mpi_id, mpi_num ;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_num);
  const int root = mpi_num - 1 ;


  if( argc==4 ){
    num_threads = static_cast<int>(atoi(argv[1])) ;
    N_RE = static_cast<int>(atoi(argv[2])) ;
    N_IM = static_cast<int>(atoi(argv[3])) ;
  }else{
    std::printf("Incorrect input argument -- Need NumThread, N_RE and N_IM.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  // Divide work between MPI processes
  const long long int total_point = static_cast<long long int>(N_RE)*static_cast<long long int>(N_IM) ;
  npoint = static_cast<long long int>( ceil((double)total_point/mpi_num) ) ;
  base   = npoint*mpi_id ;

  const long long int max_npoint = npoint ;
  if( mpi_id==root )
    npoint = total_point - base ;

  // *** Julia set ***
  nIter = static_cast<int*>(calloc( npoint, sizeof(int)));
  const double start_time = MPI_Wtime();

  dispatch_pthread(julia);

  MPI_Barrier(MPI_COMM_WORLD);
  if( mpi_id==root ){
    std::printf("%d MPI procs x %d PThreads :: Image size %d x %d :: Elapsed time = %f seconds.\n", mpi_num, num_threads, N_RE, N_IM, MPI_Wtime()-start_time);
  }


  // Write out results
  if( SAVE_IMAGE ){

    if( mpi_id==root ){
      //printf("Start writing 'julia.pgm'.\n");

      FILE *outfile ;
      outfile = std::fopen("julia.pgm","w");
      std::fprintf(outfile,"P2\n%d %d\n%d\n", N_RE, N_IM, MAX_ITER);
      int *temp = (int*)calloc(max_npoint, sizeof(int));

      for(int proc=0 ; proc<mpi_num-1 ; ++proc)
      {
        MPI_Recv(&temp[0], max_npoint, MPI_INT, proc, 0, MPI_COMM_WORLD, &mpi_status);
        //printf("Receive Result from MPI process %d.\n", proc);

        //printf("-- Start writing result received from MPI process %d.\n", proc);
        base = max_npoint*proc ;
        for(int i=0 ; i<max_npoint ; ++i)
        {
          std::fprintf(outfile,"%d ", temp[i]);
          if( (base+i)%N_RE == N_RE-1 )
            std::fprintf(outfile,"\n");
        }
        //printf("-- Done writing result received from MPI process %d.\n", proc);
      }
      std::free(temp);
      //printf("-- Start writing result of MPI process %d.\n", root);
      base = max_npoint*root ;
      for(int i=0 ; i<npoint ; ++i)
      {
        std::fprintf(outfile,"%d ", nIter[i]);
        if( (base+i)%N_RE == N_RE-1 )
          std::fprintf(outfile,"\n");
      }

      //printf("Finish writing 'julia.pgm'.\n");
    }else{

      MPI_Send(&nIter[0], npoint, MPI_INT, root, 0, MPI_COMM_WORLD);

    } // END mpi_id==root

  }

  std::free(nIter);
  MPI_Finalize();

return 0; }
