# Julia-set-Fullpar
Parallel implementations of julia set computation in C++

Version: 0.0.1

- ... = serial / auto vectorization
- SIMD = explicit AVX2 vectorization
- OMP = multi-threading using OpenMP
- Pthread = multi-threading using Pthread library
- MPI = multi-processing using MPI library

Compile and Run: \
see compile_gnu.sh and run.sh

Note: The workload of Julia set computation is non-unitorm. 
