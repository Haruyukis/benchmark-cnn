// -----------------------------------------------------------------------------
// * Name:       gemm.h
// * Purpose:    Provide a set of kernel for matrix multiplication
// * History:    Christophe Picard, Fall 2021
// -----------------------------------------------------------------------------

#pragma once

/// Parallelism on CPU
#include <omp.h>
#include <iostream>
/// ----------------------------------------------------------------------------
/// \fn void gemm_cpu_noblas_seq(T *&A, T *&B, T *&C, int M, int N, int K) 
/// \brief Compute sequential matrix product using three loops approach on CPU
/// \param A First matrix in the product
/// \param B Second matrix in the product
/// \param C Output matrix
/// \param M Number of rows of A
/// \param N Number of columns of B
/// \param K Number of rows of B
/// ----------------------------------------------------------------------------
template <typename T>
void gemm_cpu_noblas_seq(T *C, const T *A, const T *B, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T tmp = 0;
      for (int k = 0; k < K; k++) { 
        tmp += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = tmp;
    }
  }
}

/// ----------------------------------------------------------------------------
/// \fn void gemm_cpu_noblas_seq(T *&A, T *&B, T *&C, int M, int N, int K) 
/// \brief Compute parallel matrix product using three loops approach on CPU
/// \param A First matrix in the product
/// \param B Second matrix in the product
/// \param C Output matrix
/// \param M Number of rows of A
/// \param N Number of columns of B
/// \param K Number of rows of B
/// ----------------------------------------------------------------------------
template <typename T>
void gemm_cpu_noblas_par(T *&C, T *&A, T  *&B, int M, int N, int K) {
#pragma omp parallel for shared(A, B, C, N, M, K)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T tmp= T(0.0);
      for (int k = 0; k < K; k++) {
        tmp += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = tmp;
    }
  }
}

