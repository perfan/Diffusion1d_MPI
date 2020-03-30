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

// the main function drives the simulation
int main(int argc, char *argv[]) 
{
  MPI_Init(&argc, &argv);

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
  const int Nlocal = N;                  // determine number of point for this MPI process 
  // Allocate density data 
  rvector<double> P(Nlocal+Nguards);

  // Setup initial conditions for P
  P.fill(0.0);
  P[N/2+1] = 1.0;  // shift by one for the left guard cell

  // Setup initial time
  double time = 0.0;    

  // Open a file for data output
  std::ofstream file;
  diffusion1d_output_init(file, datafile);

  // Initial output
  diffusion1d_output(file, 0, time, P, outputcols);

  // Time evolution
  for (int step = 1; step <= numSteps; step++) {

    // Compute next time point
    diffusion1d_timestep(P, D, dt, dx);

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
  
