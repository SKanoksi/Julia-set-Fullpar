/*
*  Julia set (C++)
*
*  = Auto SIMD vectorization + Pthread
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
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
long long int total_point ;

void *julia(void *detail)
{
  const int thread_id = *static_cast<int*>(detail);

  long long int num = static_cast<long long int>( ceil((double)total_point/num_threads) ) ;
  const long long int start = num*thread_id ;

  if( thread_id==num_threads-1 )
    num = total_point - num*thread_id ;

  int *ptr = &nIter[0] ;


  // *** Julia set ***
  for(long long int i=start ; i<start+num; ++i)
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

  if( argc==4 ){
    num_threads = static_cast<int>(atoi(argv[1])) ;
    N_RE = static_cast<int>(atoi(argv[2])) ;
    N_IM = static_cast<int>(atoi(argv[3])) ;
  }else{
    std::printf("Incorrect input argument -- Need NumThread, N_RE and N_IM.\n");
    return 1;
  }

  total_point = static_cast<long long int>(N_RE)*static_cast<long long int>(N_IM) ;
  nIter = static_cast<int*>(calloc( total_point, sizeof(int)));
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // *** Julia set ***
  dispatch_pthread(julia);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::printf("%d Pthreads :: Image size %d x %d :: Elapsed time = %f seconds.\n", num_threads, N_RE, N_IM, static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f);

  // Write out results
  if( SAVE_IMAGE ){
    FILE *outfile ;
    outfile = std::fopen("julia.pgm","w");
    std::fprintf(outfile,"P2\n#julia.pgm\n%d %d\n%d\n", N_RE, N_IM, MAX_ITER);

    int i, j ;
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
