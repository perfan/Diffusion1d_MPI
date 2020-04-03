   // 
// diffusion1d_timestep.cc
//
// Time step module for 1d diffusion with periodic boundary conditions 
//
#include <mpi.h>
#include "diffusion1d_timestep.h"
#include <iostream>
// perform a single time step
void diffusion1d_timestep(rvector<double>& P, double D, double dt, double dx, int rank, int size, int Nlocal)
{     
    static rvector<double> laplacian;
    const int Nguards = 2;
    const int Nplusguard = P.size();
    const int N = Nplusguard - Nguards;  

    if (laplacian.size() != Nplusguard) laplacian = rvector<double>(Nplusguard);
    const double alpha = D*dt/(dx*dx);   
    // fill the first and last ghost cells for correct periodic boundary conditions
    int left = rank-1; if(left<0) left = size-1;
    int right = rank+1; if(right>=size) right = 0;

    // determining the first and last cells of each decompsed domain as the ghost cells
    const int guardleft = 0;
    const int guardright = Nlocal+1;

    // Data comminications between each chunk ghost cells
    MPI_Sendrecv(&P[1],1,MPI_DOUBLE,left,11,&P[guardright],1,MPI_DOUBLE,right,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
    MPI_Sendrecv(&P[Nlocal],1, MPI_DOUBLE,right,11,&P[guardleft],1, MPI_DOUBLE,left, 11, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
   
    // compute rhs
    for (int i = 1; i <= N; i++)
       laplacian[i] = P[i-1] + P[i+1] - 2*P[i];
    // apply change
    for (int i = 1; i <= N; i++)
       P[i] += alpha*laplacian[i];
}
