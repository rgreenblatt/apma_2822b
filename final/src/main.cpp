/*
//@HEADER
// ************************************************************************
//
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "minife_version.h"

#include "outstream.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "box.hpp"
#include "box_partition.hpp"
#include "box_utils.hpp"
#include "driver.hpp"
#include "parameters.hpp"
#include "utils.hpp"
#include "yaml_doc.hpp"

// The following macros should be specified as compile-macros in the
// makefile. They are defaulted here just in case...
#ifndef MINIFE_SCALAR
#define MINIFE_SCALAR double
#endif

// ************************************************************************

void add_params_to_yaml(YAML_Doc &doc, miniFE::Parameters &params);
void add_configuration_to_yaml(YAML_Doc &doc, int numprocs);
void add_timestring_to_yaml(YAML_Doc &doc);

//
// We will create a 'box' of size nx X ny X nz, partition it among processors,
// then call miniFE::driver which will use the partitioned box as the domain
// from which to assemble finite-element matrices into a global matrix and
// vector, then solve the linear-system using Conjugate Gradients.
//

int main(int argc, char **argv) {
  miniFE::Parameters params;
  miniFE::get_parameters(argc, argv, params);

  int numprocs = 1, myproc = 0;
  miniFE::initialize_mpi(argc, argv, numprocs, myproc);

  miniFE::timer_type start_time = miniFE::my_timer();

#ifdef MINIFE_DEBUG
  outstream(numprocs, myproc);
#endif

  // make sure each processor has the same parameters:
  miniFE::broadcast_parameters(params);

  Box global_box = {0, params.nx, 0, params.ny, 0, params.nz};
  std::vector<Box> local_boxes(static_cast<size_t>(numprocs));

  box_partition(0, numprocs, 2, global_box, &local_boxes[0]);

  Box &my_box = local_boxes[static_cast<size_t>(myproc)];

  int num_my_ids =
      miniFE::get_num_ids(my_box);
  int min_ids = num_my_ids;

#ifdef HAVE_MPI
  MPI_Datatype mpi_dtype = MPI_INT;
  MPI_Allreduce(&num_my_ids, &min_ids, 1, mpi_dtype, MPI_MIN, MPI_COMM_WORLD);
#endif

  if (min_ids == 0) {
    std::cout << "One or more processors have 0 equations. Not currently "
                 "supported. Exiting."
              << std::endl;

    miniFE::finalize_mpi();

    return 1;
  }

  std::ostringstream osstr;
  osstr << "miniFE." << params.nx << "x" << params.ny << "x" << params.nz;
#ifdef HAVE_MPI
  osstr << ".P" << numprocs;
#endif
  osstr << ".";
  if (params.name != "") {
    osstr << params.name << ".";
  }

  YAML_Doc doc("miniFE", MINIFE_VERSION, ".", osstr.str());
  if (myproc == 0) {
    add_params_to_yaml(doc, params);
    add_configuration_to_yaml(doc, numprocs);
    add_timestring_to_yaml(doc);
  }

  // Most of the program is performed in the 'driver' function, which is
  // templated on <Scalar>.
  // To run miniFE with float instead of double
  // change the template-parameters by changing the macro definitions in
  // the makefile or on the make command-line.

  int return_code =
      miniFE::driver<MINIFE_SCALAR>(global_box, my_box, params, doc);

  miniFE::timer_type total_time = miniFE::my_timer() - start_time;

  if (myproc == 0) {
    doc.add("Total Program Time", total_time);
    doc.generateYAML();
  }

  miniFE::finalize_mpi();

  return return_code;
}

void add_params_to_yaml(YAML_Doc &doc, miniFE::Parameters &params) {
  doc.add("Global Run Parameters", "");
  doc.get("Global Run Parameters")->add("dimensions", "");
  doc.get("Global Run Parameters")->get("dimensions")->add("nx", params.nx);
  doc.get("Global Run Parameters")->get("dimensions")->add("ny", params.ny);
  doc.get("Global Run Parameters")->get("dimensions")->add("nz", params.nz);
  doc.get("Global Run Parameters")
      ->add("load_imbalance", params.load_imbalance);
  if (params.mv_overlap_comm_comp == 1) {
    std::string val("1 (yes)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  } else {
    std::string val("0 (no)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  }
}

void add_configuration_to_yaml(YAML_Doc &doc, int numprocs) {
  doc.get("Global Run Parameters")->add("number of processors", numprocs);

  /*   doc.add("Platform",""); */
  /*   doc.get("Platform")->add("hostname",MINIFE_HOSTNAME); */
  /*   doc.get("Platform")->add("kernel name",MINIFE_KERNEL_NAME); */
  /*   doc.get("Platform")->add("kernel release",MINIFE_KERNEL_RELEASE); */
  /*   doc.get("Platform")->add("processor",MINIFE_PROCESSOR); */

  doc.add("Build", "");
  /* doc.get("Build")->add("CXX",MINIFE_CXX); */
  /* #if MINIFE_INFO != 0 */
  /* doc.get("Build")->add("compiler version",MINIFE_CXX_VERSION); */
  /* #endif */
  /* doc.get("Build")->add("CXXFLAGS",MINIFE_CXXFLAGS); */

  std::string using_mpi("no");
#ifdef HAVE_MPI
  using_mpi = "yes";
#endif
#ifdef _OPENMP
  doc.get("Build")->add("OMP_NUM_THREADS", omp_get_max_threads());
#else
  doc.get("Build")->add("OMP_NUM_THREADS", 1);
#endif
  doc.get("Build")->add("using MPI", using_mpi);
}

void add_timestring_to_yaml(YAML_Doc &doc) {
  std::time_t rawtime;
  struct tm *timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ostringstream osstr;
  osstr.fill('0');
  osstr << timeinfo->tm_year + 1900 << "-";
  osstr.width(2);
  osstr << timeinfo->tm_mon + 1 << "-";
  osstr.width(2);
  osstr << timeinfo->tm_mday << ", ";
  osstr.width(2);
  osstr << timeinfo->tm_hour << "-";
  osstr.width(2);
  osstr << timeinfo->tm_min << "-";
  osstr.width(2);
  osstr << timeinfo->tm_sec;
  std::string timestring = osstr.str();
  doc.add("Run Date/Time", timestring);
}
