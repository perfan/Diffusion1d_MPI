/// 
/// @file diffusion1d_timestep.h
///
/// @brief Time step module for 1d diffusion with periodic boundary conditions
///

#ifndef DIFFUSION1DTIMESTEPH
#define DIFFUSION1DTIMESTEPH

#include <rarray>

/// @brief Perform a single time step for the density field.
///
/// @param P   the density
/// @param D   the diffusion constant
/// @param dt  the time step to be used
/// @param dx  the spatial resolution
void diffusion1d_timestep(rvector<double>& P, double D, double dt, double dx, int rank, int size, int Nlocal);

#endif
