// 
// diffusion1d_output.cc
//
// Module for output for 1d diffusion 
//

#include "diffusion1d_output.h"
#include <iostream>
#include <iomanip>
#include <rarrayio>
#include "ticktock.h"
#include "sparkline.h"

static TickTock tt;

void diffusion1d_output_init(std::ofstream& file, std::string datafilename)
{
  // Setup output into a file of file name datafilename. 
  // A header will be written to screen as well.
  // Arguments:
  //   file: the file object to use to open a file
  //   datafilename: name of the file to open

  // First, open the file
  file.open(datafilename);
  // Start measuring elapsed time
  tt.tick();
  // Some initial output to screen
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
      std::cout << "   Step   Time    Walltime(s) Profile"  << std::endl;
}

void diffusion1d_output(std::ofstream& file, int step, double time, const rvector<double> &P, int outputcols)
{
  // Output into a file of file name datafilename. There will be other
  // output to screen as well, such as a little graph of the density.
  // Arguments:
  //   file:        the file object to use
  //   step:        number of the current time step
  //   time:        current value of the time
  //   P:           density array
  //   outputcols:  width of the output graph.

  // First, write rvector to file
  file << time << "\n" << P << "\n\n";
  // report to screen too (note: mpi_sparkline is a collective procedure)
  std::string sparks = mpi_sparkline(P, outputcols, 0, MPI_COMM_WORLD, 1, 1, true);
  if (sparks != "")
   std::cout<< std::fixed << std::setw(7) << step  << " "
            << std::fixed << std::setprecision(4) << std::setw(8) << time << "   "
            << std::fixed << std::setprecision(4) << std::setw(8) << tt.silent_tock() << "   "
            << sparks << std::endl;
}

// Finalize output
// Arguments:
//   file: the file object that was used
void diffusion1d_output_finish(std::ofstream& file)
{
  // Close file
  file.close();
}
