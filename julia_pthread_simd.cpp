/*
*  Julia set (C++)
*
*  = Explicit AVX2 SIMD + Pthread
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h>
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
long long int total_point ;


void *julia(void *detail)
{
  const int thread_id = *static_cast<int*>(detail);

  long long int num = static_cast<long long int>( ceil((double)total_point/num_threads) ) ;
  const long long int start = num*thread_id ;

  if( thread_id==num_threads-1 )
    num = total_point - num*thread_id ;

  const float_vec N_RE_FVEC = static_cast<float>(N_RE) *float_vec_uniform ;
  const float_vec N_IM_FVEC = static_cast<float>(N_IM) *float_vec_uniform ;


  // *** Julia set ***
  for(long long int i=start ; i<start+num; i+=VEC_NUM_E)
  {
    float_simd ii, xi, yj, re, im, z_Re, z_Im ;
    int_simd   k, flag, next ;

    ii.v  = static_cast<float>(i)*float_vec_uniform + float_vec_increment ;
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

  if( argc==4 ){
    num_threads = static_cast<int>(atoi(argv[1])) ;
    N_RE = static_cast<int>(atoi(argv[2])) ;
    N_IM = static_cast<int>(atoi(argv[3])) ;
  }else{
    std::printf("Incorrect input argument -- Need NumThread, N_RE and N_IM.\n");
    return 1;
  }

  total_point = static_cast<long long int>(N_RE)*static_cast<long long int>(N_IM) ;
  if( total_point%VEC_NUM_E != 0 ){
    std::printf("Error:: total number of points (%lld) must be divisible by %d.\n", total_point, VEC_NUM_E);
    return 1;
  }
  nIter = static_cast<int*>(aligned_alloc(VEC_SIZE, sizeof(int)*total_point));
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
