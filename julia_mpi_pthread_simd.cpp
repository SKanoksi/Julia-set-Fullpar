/*
*  Julia set (C++)
*
*  = Explicit AVX2 SIMD + MPI + Pthread
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <mpi.h>
#include <pthread.h>

#define SAVE_IMAGE 1

using namespace std ;


// *** AVX2 = 256 bits = 32 bytes ***

#define VEC_SIZE 256
#define VEC_NUM_E 8

typedef float float_vec __attribute__ ((vector_size (32)));
using float_mem_lv0 = __m256 ;  // level N = 2**N elements
using float_mem_lv1 = __m128 ;
using float_mem_lv3 =  float ;

typedef int int_vec __attribute__ ((vector_size (32)));
using int_mem_lv0 = __m256i ;
using int_mem_lv1 = __m128i ;
using int_mem_lv2 = __m64   ;
using int_mem_lv3 =   int   ;

typedef union alignas(alignof(float_mem_lv0)) {
  float_vec      v    ;
  float_mem_lv0 m0    ;
  float_mem_lv1 m1[2] ;
  float_mem_lv3 m3[8] ;
  float_mem_lv3  e[8] ;
} float_simd ;

typedef union alignas(alignof(int_mem_lv0)) {
  int_vec      v    ;
  int_mem_lv0 m0    ;
  int_mem_lv1 m1[2] ;
  int_mem_lv2 m2[4] ;
  int_mem_lv3 m3[8] ;
  int_mem_lv3  e[8] ;
} int_simd ;

constexpr float_vec float_vec_uniform = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f} ;
constexpr int_vec   int_vec_uniform   = {1, 1, 1, 1, 1, 1, 1, 1} ;

constexpr float_vec float_vec_increment = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f} ;
constexpr int_vec   int_vec_increment   = {0, 1, 2, 3, 4, 5, 6, 7} ;


// *** Julia set ***
constexpr int MAX_ITER   =  255 ;
constexpr float z_Re_min = -0.8 ;
constexpr float z_Im_max =  0.8 ;
constexpr float z_Re_length = 1.6 ;
constexpr float z_Im_length = 1.6 ;
constexpr float c_Re = -0.62277 ;
constexpr float c_Im =  0.42193 ;
constexpr float d    = 4.0 ;

constexpr int_vec   MAX_ITER_IVEC = MAX_ITER *int_vec_uniform ;
constexpr float_vec z_Re_min_FVEC    = z_Re_min    *float_vec_uniform ;
constexpr float_vec z_Im_max_FVEC    = z_Im_max    *float_vec_uniform ;
constexpr float_vec z_Re_length_FVEC = z_Re_length *float_vec_uniform ;
constexpr float_vec z_Im_length_FVEC = z_Im_length *float_vec_uniform ;
constexpr float_vec c_Re_FVEC = c_Re *float_vec_uniform ;
constexpr float_vec c_Im_FVEC = c_Im *float_vec_uniform ;
constexpr float_vec d_FVEC    =    d *float_vec_uniform ;


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

  const float_vec N_RE_FVEC = static_cast<float>(N_RE) *float_vec_uniform ;
  const float_vec N_IM_FVEC = static_cast<float>(N_IM) *float_vec_uniform ;


  // *** Julia set ***
  for(long long int i=start ; i<start+num; i+=VEC_NUM_E)
  {
    float_simd ii, xi, yj, re, im, z_Re, z_Im ;
    int_simd   k, flag, next ;

    ii.v  = static_cast<float>(base+i)*float_vec_uniform + float_vec_increment ;
    yj.v  = ii.v/N_RE_FVEC ;
    yj.m0 = _mm256_floor_ps(yj.m0) ;
    xi.v  = ii.v - yj.v*N_RE_FVEC ;

    z_Re.v = z_Re_min_FVEC + ( xi.v/N_RE_FVEC ) * z_Re_length_FVEC ;
    z_Im.v = z_Im_max_FVEC - ( yj.v/N_IM_FVEC ) * z_Im_length_FVEC ; // Reversed

    // Iterate up to a maximum number or break if mod(z) > sqrt(d)
    k.v    = 0*int_vec_uniform ;
    flag.v =   int_vec_uniform ;
    next.v =   int_vec_uniform ;

    while( (flag.e[0]+flag.e[1]) != 0 )
    {
      // Note: 0 = false, -1 = True ***
      next.v = (next.v) ? ( (z_Re.v*z_Re.v + z_Im.v*z_Im.v)<4.0) : 0 ;

      re.v   = z_Re.v ;
      im.v   = z_Im.v ;
      z_Re.v =   re.v*re.v - im.v*im.v + c_Re_FVEC ;
      z_Im.v = 2*re.v*im.v + c_Im_FVEC ;

      // Not: k = k - (-1 or 0) = k + (1 or 0) --> ++k
      k.v  = k.v - next.v ;

      flag.v = (k.v<MAX_ITER_IVEC) && next.v ;

      // Accumulative sum
      flag.m1[0] = _mm_hadd_epi32(flag.m1[0], flag.m1[1]);
      flag.m2[0] = _mm_hadd_pi32(flag.m2[0], flag.m2[1]);
    }
    _mm256_stream_si256((int_mem_lv0*)&nIter[i], k.m0);  // AVX2
    //_mm256_store_epi32((void*)&nIter[i], k.m0);        // AVX512f
    //for(int j=0 ; j<VEC_NUM_E ; ++j)                   // Auto-vectorized
    //  nIter[i+j] = k.e[j] ;

  } // LOOP: Julia set


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

  if( npoint%VEC_NUM_E != 0 ){
    std::printf("Error:: Num points (%lld) of MPI proc (%d) must be divisible by %d.\n", npoint, mpi_id, VEC_NUM_E);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  nIter = static_cast<int*>(aligned_alloc(VEC_SIZE, sizeof(int)*npoint));


  // *** Julia set ***
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
