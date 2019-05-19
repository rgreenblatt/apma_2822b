#pragma once

#include <cstdio>

template <class T> struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <class U> UMAllocator(const UMAllocator<U> &other);

  T *allocate(size_t n);

  void deallocate(T *p, size_t n);
};

template <class T, class U>
bool operator==(const UMAllocator<T> &, const UMAllocator<U> &);
template <class T, class U>
bool operator!=(const UMAllocator<T> &, const UMAllocator<U> &);
