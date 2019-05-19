#pragma once

#include <vector>

#include "vector_uvm.hpp"

using GlobalOrdinalAllocVec =
    std::vector<GlobalOrdinal, UMAllocator<GlobalOrdinal>>;
