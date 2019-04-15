#pragma once

#include <cusparse_v2.h>

class SpMvMethod {
public:
  virtual void run() = 0;
};

class CRSMethod : public SpMvMethod {
protected:
  int Nrow;
  double *AA;
  int *IA;
  int *JA;
  double *x;
  double *y;

public:
  CRSMethod(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : Nrow(Nrow), AA(AA), IA(IA), JA(JA), x(x), y(y) {}
};

class CRSMethodCPU : public CRSMethod {
public:
  void run();
  CRSMethodCPU(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : CRSMethod(Nrow, AA, IA, JA, x, y) {}
};

class CRSMethodGPU : public CRSMethod {
public:
  void run();
  CRSMethodGPU(int Nrow, double *AA, int *IA, int *JA, double *x, double *y)
      : CRSMethod(Nrow, AA, IA, JA, x, y) {}
};

class CudaSparse : public CRSMethod {
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

class ELLPACKMethod : public SpMvMethod {
protected:
  int Nrow;
  int maxnzr;
  int *row_lengths;
  double *y;

public:
  ELLPACKMethod(int Nrow, int maxnzr, int *row_lengths, double *y)
      : Nrow(Nrow), maxnzr(maxnzr), row_lengths(row_lengths), y(y) {}
};

class ELLPACKMethod2DArray : public ELLPACKMethod {
protected:
  double **AS;
  int **JA;
  double *x;

public:
  ELLPACKMethod2DArray(int Nrow, int maxnzr, int *row_lengths, double **AS,
                       int **JA, double *x, double *y)
      : ELLPACKMethod(Nrow, maxnzr, row_lengths, y), AS(AS), JA(JA), x(x) {}
};

class ELLPACKMethodCPU : public ELLPACKMethod2DArray {
public:
  void run();
  ELLPACKMethodCPU(int Nrow, int maxnzr, int *row_lengths, double **AS,
                   int **JA, double *x, double *y)
      : ELLPACKMethod2DArray(Nrow, maxnzr, row_lengths, AS, JA, x, y) {}
};

class ELLPACKMethodGPUManaged : public ELLPACKMethod2DArray {
public:
  void run();
  ELLPACKMethodGPUManaged(int Nrow, int maxnzr, int *row_lengths, double **AS,
                          int **JA, double *x, double *y)
      : ELLPACKMethod2DArray(Nrow, maxnzr, row_lengths, AS, JA, x, y) {}
};

class ELLPACKMethodGPU : public ELLPACKMethod {
protected:
  double *AS;
  int *JA;
  size_t pitch_AS;
  size_t pitch_JA;
  cudaTextureObject_t x;

public:
  void run();
  ELLPACKMethodGPU(int Nrow, int maxnzr, int *row_lengths, double *AS,
                   size_t pitch_AS, int *JA, size_t pitch_JA,
                   cudaTextureObject_t x, double *y)
      : ELLPACKMethod(Nrow, maxnzr, row_lengths, y), AS(AS), pitch_AS(pitch_AS),
        JA(JA), pitch_JA(pitch_JA), x(x) {}
};
