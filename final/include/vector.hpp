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

#pragma once

#include <vector>

#include "mem_init_op.hpp"
#include "vector_uvm.hpp"

namespace miniFE {

template <typename Scalar> struct Vector {
  typedef Scalar ScalarType;

  Vector(int startIdx, int local_sz)
      : startIndex(startIdx), local_size(local_sz),
        coefs(static_cast<size_t>(local_size)) {
    MemInitOp<Scalar> mem_init;
    mem_init.ptr = &coefs[0];
    mem_init.n = static_cast<size_t>(local_size);

#pragma omp parallel for
    for (size_t i = 0; i < mem_init.n; ++i) {
      mem_init(i);
    }
  }

  ~Vector() {}

  int startIndex;
  int local_size;
  AllocVec<Scalar> coefs;
};

} // namespace miniFE
