// 
// diffusion1d.cc
//
// 1d diffusion with periodic boundary conditions
//
// Compile with make using provided Makefile 
//

#include <mpi.h>
#include <fstream>
#include <rarray>
#include "diffusion1d_output.h"
#include "diffusion1d_timestep.h"
#include "parameters.h"
#include <iostream>
#include <string>

// the main function drives the simulation
int main(int argc, char *argv[]) 
{
  int size;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  // Simulation parameters
  double      L;  // system length
  double      D;  // diffusion constant
  double      T;  // time
  double      dx; // spatial resolution
  double      dt; // temporal resolution (time step)
  int         Z;  // number of walkers (dummy variable, not used)
  std::string datafile; // filename for output
  double      time_between_output;

  // Read parameters from a file given on the command line. 
  // If no file was given, use "params.ini".
  std::string paramFilename = argc>1?argv[1]:"params.ini";
  read_parameters(paramFilename, L, D, T, dx, dt, Z, datafile, time_between_output);
  
  // Compute derived parameters 
  const int numSteps = int(T/dt + 0.5);  // number of steps to take
  const int N = int(L/dx + 0.5);         // number of grid points
  const int Nguards = 2;                 // number of guard cells
  const int outputEvery = int(time_between_output/dt + 0.5); // how many steps between output
  const int outputcols = 48;             // number of columns for sparkline output
  const int Nlocal = N/size;             // determine number of point for this MPI process 
  // Allocate density data 
  rvector<double> P(Nlocal+Nguards);

  // Setup initial conditions for P
  P.fill(0.0);

  if (size % 2 == 0)
    if (rank == size/2) P[1] = 1.0;
  if (size % 2 == 1)
    if (rank == (size-1)/2) P[Nlocal/2+1] = 1.0;

  // Setup initial time
  double time = 0.0;    
  
  // Open a file for data output
  datafile = datafile + std::to_string(rank+1);
  std::ofstream file;
  diffusion1d_output_init(file, datafile);

  // Initial output
  diffusion1d_output(file, 0, time, P, outputcols);
  
  // Time evolution
  for (int step = 1; step <= numSteps; step++) {

    // Compute next time point
    diffusion1d_timestep(P, D, dt, dx, rank, size, Nlocal);

    // Update time
    time += dt;

    // Periodically add data to the file
    if (step % outputEvery == 0 and step > 0) 
        diffusion1d_output(file, step, time, P, outputcols);
  }
  
  // Close file
  diffusion1d_output_finish(file);

  // All done
  MPI_Finalize();
  return 0;
}
  
