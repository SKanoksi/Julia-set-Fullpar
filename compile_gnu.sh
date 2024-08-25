#!/bin/bash

# For GNU, don't pass -march but pass -mavxXX
#FLAGS="-Ofast -fopt-info-vec-optimized -mavx2 "
FLAGS="-Ofast -mavx2 "
#FLAGS="-Ofast -mavx2 -fno-tree-vectorize "

g++ -o julia_cpp.exe ${FLAGS} julia.cpp

g++ -o julia_simd_cpp.exe ${FLAGS} julia_simd.cpp

g++ -o julia_omp_cpp.exe ${FLAGS} julia_omp.cpp -fopenmp

g++ -o julia_omp_simd_cpp.exe ${FLAGS} julia_omp_simd.cpp -fopenmp

g++ -o julia_pthread_cpp.exe ${FLAGS} julia_pthread.cpp -pthread

g++ -o julia_pthread_simd_cpp.exe ${FLAGS} julia_pthread_simd.cpp -pthread

mpic++ -o julia_mpi_cpp.exe ${FLAGS} julia_mpi.cpp

mpic++ -o julia_mpi_simd_cpp.exe ${FLAGS} julia_mpi_simd.cpp

mpic++ -o julia_mpi_omp_cpp.exe ${FLAGS} julia_mpi_omp.cpp -fopenmp

mpic++ -o julia_mpi_omp_simd_cpp.exe ${FLAGS} julia_mpi_omp_simd.cpp -fopenmp

mpic++ -o julia_mpi_pthread_cpp.exe ${FLAGS} julia_mpi_pthread.cpp -pthread

mpic++ -o julia_mpi_pthread_simd_cpp.exe ${FLAGS} julia_mpi_pthread_simd.cpp -pthread




