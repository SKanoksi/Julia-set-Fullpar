#!/bin/bash

NX=9000
NY=9000

export MPI_NUM=2
export OMP_NUM_THREADS=2
export CPU_NUM=$((${MPI_NUM}*${OMP_NUM_THREADS}))

export OMP_SCHEDULE=guided  # Since Julia set = Non-uniform workload



printf "\n*** *** Single core *** ***\n"

printf "\nC++: AUTO:\n"
./julia_cpp.exe $NX $NY

printf "\nC++: SIMD:\n"
./julia_simd_cpp.exe $NX $NY



printf "\n*** *** ${OMP_NUM_THREADS} cores *** ***\n"

printf "\nC++:  OMP: AUTO:\n"
./julia_omp_cpp.exe $NX $NY

printf "\nC++:  OMP: SIMD:\n"
./julia_omp_simd_cpp.exe $NX $NY

printf "\nC++:  PTHREAD: AUTO:\n"
./julia_pthread_cpp.exe ${OMP_NUM_THREADS} $NX $NY

printf "\nC++:  PTHREAD: SIMD:\n"
./julia_pthread_simd_cpp.exe ${OMP_NUM_THREADS} $NX $NY

printf "\nC++:  MPI: AUTO:\n"
mpirun -np ${OMP_NUM_THREADS} ./julia_mpi_cpp.exe $NX $NY

printf "\nC++:  MPI: SIMD:\n"
mpirun -np ${OMP_NUM_THREADS} ./julia_mpi_simd_cpp.exe $NX $NY



printf "\n*** *** ${CPU_NUM} cores *** ***\n"

printf "\nC++:  MPI:  OMP: AUTO:\n"
mpirun -np ${MPI_NUM} ./julia_mpi_omp_cpp.exe $NX $NY

printf "\nC++:  MPI:  OMP: SIMD:\n"
mpirun -np ${MPI_NUM} ./julia_mpi_omp_simd_cpp.exe $NX $NY

printf "\nC++:  MPI: PTHREAD: AUTO:\n"
mpirun -np ${MPI_NUM} ./julia_mpi_pthread_cpp.exe ${OMP_NUM_THREADS} $NX $NY

printf "\nC++:  MPI: PTHREAD: SIMD:\n"
mpirun -np ${MPI_NUM} ./julia_mpi_pthread_simd_cpp.exe ${OMP_NUM_THREADS} $NX $NY



printf "\n*** *** ${CPU_NUM} cores *** ***\n"

export OMP_NUM_THREADS=${CPU_NUM}

printf "\nC++:  OMP: AUTO:\n"
./julia_omp_cpp.exe $NX $NY

printf "\nC++:  OMP: SIMD:\n"
./julia_omp_simd_cpp.exe $NX $NY

printf "\nC++:  PTHREAD: AUTO:\n"
./julia_pthread_cpp.exe ${OMP_NUM_THREADS} $NX $NY

printf "\nC++:  PTHREAD: SIMD:\n"
./julia_pthread_simd_cpp.exe ${OMP_NUM_THREADS} $NX $NY

printf "\nC++:  MPI: AUTO:\n"
mpirun -np ${OMP_NUM_THREADS} ./julia_mpi_cpp.exe $NX $NY

printf "\nC++:  MPI: SIMD:\n"
mpirun -np ${OMP_NUM_THREADS} ./julia_mpi_simd_cpp.exe $NX $NY
