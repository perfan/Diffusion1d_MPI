// 
// diffusion1d_timestep.cc
//
// Time step module for 1d diffusion with periodic boundary conditions 
//

#include "diffusion1d_timestep.h"
#include <iostream>
// perform a single time step
void diffusion1d_timestep(rvector<double>& P, double D, double dt, double dx)
{     
    static rvector<double> laplacian;
    const int Nguards = 2;
    const int Nplusguard = P.size();
    const int N = Nplusguard - Nguards;    
    if (laplacian.size() != Nplusguard) laplacian = rvector<double>(Nplusguard);
    const double alpha = D*dt/(dx*dx);   
    // fill guard cells for correct periodic boundary conditions
    const int guardleft = 0;
    const int guardright = N+1;
    P[guardleft] = P[N];
    P[guardright] = P[1];
    //std::cout << P[guardright] << "\n";
    // compute rhs
    laplacian[0] = P[N+1] + P[1] - 2*P[0];
    laplacian[N+1] = P[0] + P[N] - 2*P[N+1];
    for (int i = 1; i < N+1; i++)
       laplacian[i] = P[i-1] + P[i+1] - 2*P[i];
    // apply change
    for (int i = 0; i < N+2; i++)
       P[i] += alpha*laplacian[i];
}
