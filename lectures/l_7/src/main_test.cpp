#include <omp.h>
#include <stdio.h>
#include <math.h>

double **mem_alloc2D(unsigned N_D1, unsigned N_D2); 
double fexact(double x, double y);
double compute_RHS(double x, double y);

int MIN(int a, int b) {return a<b?a:b; }


int main(){

  unsigned Nx, Ny;
  double **d2U, **U;
  double **Uexact;
  double **RHS;
  double *x, *y;
  double Lx, Ly, dx, dy;
  unsigned r,c;

  Lx = 1.0;
  Ly = 1.0;
  Nx = 20004;
  Ny = 20004;
  double TOL = 1.0e-4;

  dx = Lx/(Nx-1);
  dy = Ly/(Ny-1);

  x = new double[Nx];
  y = new double[Ny];
  d2U = mem_alloc2D(Ny,Nx); 
  U = mem_alloc2D(Ny,Nx);
  Uexact = mem_alloc2D(Ny,Nx);
  RHS = mem_alloc2D(Ny,Nx);


  for (unsigned i = 0; i < Nx; ++i) x[i] = i*dx;
  for (unsigned i = 0; i < Ny; ++i) y[i] = i*dy;


  //compute the exact solution
  for (r = 0; r < Ny; ++r){
    for (c = 0; c < Nx; ++c){
      U[r][c] = fexact(x[c],y[r]);
    }  
  }


/*  
  d^2/dx^2 =  (-1.0*U[r][c-2] + 16.0 * U[r][c-1] -30.0 *  U[r][c] + 16.0*  U[r][c+1] - 1.0*U[r][c+2] ) / (12.0*dx*dx);  
*/  


  double Am2x = -1.0/(12.0*dx*dx);
  double Am1x = 16.0/(12.0*dx*dx);
  double Ax   =-30.0/(12.0*dx*dx);
  double Ap1x = 16.0/(12.0*dx*dx);
  double Ap2x = -1.0/(12.0*dx*dx);

  double Am2y = -1.0/(12.0*dy*dy);
  double Am1y = 16.0/(12.0*dy*dy);
  double Ay   =-30.0/(12.0*dy*dy);
  double Ap1y = 16.0/(12.0*dy*dy);
  double Ap2y = -1.0/(12.0*dy*dy);


  double sizeGB = Nx*Ny*sizeof(double) / (1024.0*1024.0*1024.0);

  double t1 = omp_get_wtime();
  #pragma omp parallel for schedule(static,1) private(r,c) 
  for (unsigned r = 2; r < Ny-2; ++r){
    for (unsigned c = 2; c < Nx-2; ++c){
      d2U[r][c] =  Am2x*U[r][c-2] + Am1x*U[r][c-1] + Ax*U[r][c] + Ap1x*U[r][c+1] + Ap2x*U[r][c+2] +
                   Am2y*U[r-2][c] + Am1y*U[r-1][c] + Ay*U[r][c] + Ap1y*U[r+1][c] + Ap2y*U[r+2][c];  
    }
  }
  double t2 = omp_get_wtime();
  printf("stencil time : %g, effective BW = %g [GB/s]\n",t2-t1, sizeGB*2.0 / (t2-t1) );



  unsigned row_block_size = 128;
  unsigned col_block_size = 400;

  t1 = omp_get_wtime();
  #pragma omp parallel for collapse(2) 
  for (unsigned rb = 2; rb < Ny; rb = rb + row_block_size){   //ROW BLOCKING
    for (unsigned cb = 2; cb < Nx; cb = cb + col_block_size){ // COLUMN BLOCKING

       for (unsigned r  = rb; r < MIN(Ny-2,rb + row_block_size+1); ++r){
         for (unsigned c  = cb; c < MIN(Nx-2,cb + col_block_size+1); ++c){

            d2U[r][c] =  Am2x*U[r][c-2] + Am1x*U[r][c-1] + Ax*U[r][c] + Ap1x*U[r][c+1] + Ap2x*U[r][c+2] +
                   Am2y*U[r-2][c] + Am1y*U[r-1][c] + Ay*U[r][c] + Ap1y*U[r+1][c] + Ap2y*U[r+2][c];
         }
       }
    }  
  }
  t2 = omp_get_wtime();
  printf("2D blocking stencil Version 1 time : %g, effective BW = %g [GB/s]\n",t2-t1, sizeGB*2.0 / (t2-t1) );

  row_block_size = 500;
  col_block_size = 100;

  t1 = omp_get_wtime();
  #pragma omp parallel for collapse(1) schedule(static,1)
  for (unsigned cb = 2; cb < Nx; cb = cb + col_block_size){ // COLUMN BLOCKING
    for (unsigned rb = 2; rb < Ny; rb = rb + row_block_size){   //ROW BLOCKING

       for (unsigned r  = rb; r < MIN(Ny-2,rb + row_block_size+1); ++r){
         for (unsigned c  = cb; c < MIN(Nx-2,cb + col_block_size+1); ++c){

            d2U[r][c] =  Am2x*U[r][c-2] + Am1x*U[r][c-1] + Ax*U[r][c] + Ap1x*U[r][c+1] + Ap2x*U[r][c+2] +
                   Am2y*U[r-2][c] + Am1y*U[r-1][c] + Ay*U[r][c] + Ap1y*U[r+1][c] + Ap2y*U[r+2][c];
         }
       }
    }
  }
  t2 = omp_get_wtime();
  printf("2D blocking stencil Version 2 time : %g, effective BW = %g [GB/s]\n",t2-t1, sizeGB*2.0 / (t2-t1) );




  t1 = omp_get_wtime();
  #pragma omp parallel for 
  for (r = 0; r < Nx*Ny; ++r)  
     d2U[0][r]=U[0][r]; 
  t2 = omp_get_wtime();
  printf("memcopy time : %g, effective BW = %g [GB/s]\n",t2-t1, sizeGB*2.0 / (t2-t1) );





  return 0;
}

inline 
double fexact(double x, double y){
   return x*x + y*y;
} 

inline
double compute_RHS(double x, double y){
   return 4.0;
}



double **mem_alloc2D(unsigned N_D1, unsigned N_D2){
  double **U;
  U = new double*[N_D1];
  U[0] = new double[N_D1*N_D2];
  for (unsigned r = 1; r < N_D1; ++r)
     U[r] = U[0] + r*N_D2;
  return U;
}



