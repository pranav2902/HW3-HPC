#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>
using namespace std;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, int n, long p,long* offset) {
  //int p = omp_get_num_threads();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  offset[0] = 0;
  prefix_sum[0] = 0;
  if (n == 0) return;
  #pragma omp parallel num_threads(p)
  {
      long t = omp_get_thread_num();
      long sum = 0;
      #pragma omp for schedule(static)
      for (long i=0; i<n; i++){
          sum += A[i];
          prefix_sum[i+1] = sum;
      }
  offset[t+1] = sum;
  }
  // Offset update
  for (long i=0; i<p; i++){
  offset[i+1] += offset[i];
  }
  // Update prefix sum
    #pragma omp parallel num_threads(p)
  {
    long t = omp_get_thread_num();
    #pragma omp for schedule(static)
      for (long i=1; i<n; i++){
        prefix_sum[i] += offset[t];
      }
  }
}

int main() {
  //Set number of threads here
  long num_threads[8] = {2,4,8,16,32,64,128,256};
  for(long i = 0;i < 8; i++){
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  long* offset = (long*) malloc(num_threads[i] * sizeof(long));

  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("#Threads: %d\n",num_threads[i]);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N, num_threads[i],offset);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  free(offset);
}
  return 0;
}
