#pragma once

#include <cusparse_v2.h>

class SpMvMethod {
public:
  virtual void run() = 0;
};

template <typename T> class CRSMethod : public SpMvMethod {
protected:
  int Nrow;
  double *AA;
  int *IA;
  int *JA;
  T x;
  double *y;

public:
  CRSMethod(int Nrow, double *AA, int *IA, int *JA, T x, double *y)
      : Nrow(Nrow), AA(AA), IA(IA), JA(JA), x(x), y(y) {}
};

class CRSMethodCPU : public CRSMethod<double *> {
public:
  void run();
  CRSMethodCPU(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : CRSMethod(Nrow, AA, IA, JA, x, y) {}
};

template <typename T> class CRSMethodGPU : public CRSMethod<T> {
public:
  void run();
  CRSMethodGPU(int Nrow, double *AA, int *IA, int *JA, T x, double *y)
      : CRSMethod<T>::CRSMethod(Nrow, AA, IA, JA, x, y) {}
};

class CudaSparse : public CRSMethod<double *> {
protected:
  cusparseHandle_t handle;
  int Ncol;
  int nnz;
  cusparseMatDescr_t descr = 0;

public:
  void run();
  CudaSparse(cusparseHandle_t handle, int Nrow, int Ncol, int nnz, double *AA,
             int *IA, int *JA, double *x, double *y);
};

template <typename T> class ELLPACKMethod : public SpMvMethod {
protected:
  int Nrow;
  int maxnzr;
  int *row_lengths;
  double **AS;
  int **JA;
  T x;
  double *y;

public:
  ELLPACKMethod(int Nrow, int maxnzr, int *row_lengths, double **AS, int **JA,
                T x, double *y)
      : Nrow(Nrow), maxnzr(maxnzr), row_lengths(row_lengths), AS(AS), JA(JA),
        x(x), y(y) {}
};

class ELLPACKMethodCPU : public ELLPACKMethod<double *> {
public:
  void run();
  ELLPACKMethodCPU(int Nrow, int maxnzr, int *row_lengths, double **AS,
                   int **JA, double *x, double *y)
      : ELLPACKMethod(Nrow, maxnzr, row_lengths, AS, JA, x, y) {}
};

template <typename T> class ELLPACKMethodGPU : public ELLPACKMethod<T> {
public:
  void run();
  ELLPACKMethodGPU(int Nrow, int maxnzr, int *row_lengths, double **AS,
                   int **JA, T x, double *y)
      : ELLPACKMethod<T>::ELLPACKMethod(Nrow, maxnzr, row_lengths, AS, JA, x,
                                        y) {}
};
