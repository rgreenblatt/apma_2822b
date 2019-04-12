#pragma once
#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <stdio.h>

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

class ELLPACKMethod : public SpMvMethod {
protected:
  int Nrow;
  int maxnzr;
  double *AS;
  int *JA;
  double *x;
  double *y;

public:
  ELLPACKMethod(int Nrow, int maxnzr, double *AS, int *JA, double *x, double *y)
      : Nrow(Nrow), maxnzr(maxnzr), AS(AS), JA(JA), x(x), y(y) {}
};

class ELLPACKMethodCPU : public ELLPACKMethod {
public:
public:
  void run();
  ELLPACKMethodCPU(int Nrow, int maxnzr, double *AS, int *JA, double *x,
                   double *y)
      : ELLPACKMethod(Nrow, maxnzr, AS, JA, x, y) {}
};

class ELLPACKMethodGPU : public ELLPACKMethod {
public:
  void run();
  ELLPACKMethodGPU(int Nrow, int maxnzr, double *AS, int *JA, double *x,
                   double *y)
      : ELLPACKMethod(Nrow, maxnzr, AS, JA, x, y) {}
};

class CudaSparse : public SpMvMethod {
protected:
  cusparseHandle_t handle;
  int Nrow;
  int Ncol;
  int nnz;
  double *AA;
  int *IA;
  int *JA;
  double *x;
  double *y;
  cusparseMatDescr_t descr = 0;

public:
  void run();
  CudaSparse(cusparseHandle_t handle, int Nrow, int Ncol, int nnz, double *AA,
             int *IA, int *JA, double *x, double *y);
};
