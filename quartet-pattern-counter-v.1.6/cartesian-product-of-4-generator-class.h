//*******************************************************
// BSD license:
// Copyright 2026, Dr. Christoph Mayer, Nikita Kulikov, Dr. Iker Irisarri LIB, Bonn/Hamburg, Germany
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials provided 
//with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*******************************************************

#ifndef COMBINATIONS_4_OUTOF_N_GENERATOR_H
#define COMBINATIONS_4_OUTOF_N_GENERATOR_H

#include <iostream>
#include <vector>
#include <array>

#include <iostream>
using namespace std;

typedef std::array<uint16_t, 4> Quartet_Cartesian_Product_type;

// Highly efficient functions to compute the number of combinations
// C(n, 4) (n choose k) and to generate all these combinations for the
// numbers 0,1,2,...,n-1.

// Function to compute C(n, k)
inline long long binomialCoefficient4(int n)
{
  int k=4;
  if (k > n) return 0;
  if (k == 0 || k == n) return 1;
  
  // Since C(n, k) = C(n, n-k), we use the smaller k to optimize
  if (k > n - k)
    k = n - k;
  
  long long result = 1;
  for (int i = 0; i < k; ++i)
  {
    result *= (n - i);
    result /= (i + 1);
  }
  return result;
}


inline void print_vec(std::ostream &os, std::vector<int> v, const char *final = "\n")
{
  size_t N = v.size();
  if (N>0)
    os << v[0];
  for (size_t i=1; i<N; ++i)
    os << ", " << v[i];
  os << final;
}


class cartesianProductOf4_Generator
{
private:
  Quartet_Cartesian_Product_type state;
  
  uint16_t n;
  const uint16_t k = 4;
  bool started;
  
public:
  cartesianProductOf4_Generator(uint16_t n): n(n), started(false)
  {
    if (n < k)
    {
      std::cerr << "Undefined combination generator request. For n over 4, n must be >= 4. Exiting.\n";
      exit(-1);
    }
    
    for (uint16_t i=0; i<k; ++i)
      state[i] = i;
  }
  
  bool next(Quartet_Cartesian_Product_type &out)
  {
    if (!started)
    {
      started = true;
      out = state;
      return true;
    }
    
    // Find position to increment
    int i = k-1;
    while (i >= 0 && state[i] == n - k + i)
      --i;
    
    if (i < 0)
      return false; // exhausted
    
    ++state[i];
    for (int j = i + 1; j < k; ++j)
      state[j] = state[j - 1] + 1;
    
    out = state;
    return true;
  }

  size_t total_combinations() const
  {
    if (n<=4)
      return 0;
    return (size_t)n*(size_t)(n-1)*(size_t)(n-2)*(size_t)(n-3)/(size_t)24; // n*(n-1)*(n-2)*(n-3)/4/3/2/1;
  }
};


class CartesianProductOf4arrays_Generator
{
private:
  std::array<std::vector<int>, 4> arrays_;  // Own copies of input arrays
  std::array<int, 4> sizes_;                // Sizes of each array
  std::array<int, 4> indices_;              // Current position in each array
  bool done_;                                // True when all combinations exhausted

public:
  // Constructor: copies the input arrays for safe self-contained storage
  CartesianProductOf4arrays_Generator(const int* a1, int N1,
                                      const int* a2, int N2,
                                      const int* a3, int N3,
                                      const int* a4, int N4)
  : sizes_{N1, N2, N3, N4}, indices_{0, 0, 0, 0}
  {
    // Copy input arrays to internal storage
    arrays_[0].assign(a1, a1 + N1);
    arrays_[1].assign(a2, a2 + N2);
    arrays_[2].assign(a3, a3 + N3);
    arrays_[3].assign(a4, a4 + N4);

    // Check if any array is empty
    done_ = (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0);
  }

  // Get next combination, returns false when exhausted
  bool next(Quartet_Cartesian_Product_type &out)
  {
    if (done_)
    {
      return false;
    }

    // Fill current combination
    out[0] = arrays_[0][indices_[0]];
    out[1] = arrays_[1][indices_[1]];
    out[2] = arrays_[2][indices_[2]];
    out[3] = arrays_[3][indices_[3]];

    // Increment indices (rightmost index changes fastest, like an odometer)
    indices_[3]++;

    // Handle carry-over (when an index reaches its limit)
    if (indices_[3] >= sizes_[3]) {
      indices_[3] = 0;
      indices_[2]++;

      if (indices_[2] >= sizes_[2]) {
        indices_[2] = 0;
        indices_[1]++;

        if (indices_[1] >= sizes_[1]) {
          indices_[1] = 0;
          indices_[0]++;

          if (indices_[0] >= sizes_[0]) {
            // All combinations exhausted
            done_ = true;
          }
        }
      }
    }
    return true;
  }

  // Check if more combinations are available
  bool has_next() const
  {
    return !done_;
  }

  // Reset iterator to beginning
  void reset()
  {
    indices_ = {0, 0, 0, 0};
    done_ = (sizes_[0] == 0 || sizes_[1] == 0 ||
             sizes_[2] == 0 || sizes_[3] == 0);
  }

  // Get total number of combinations
  size_t total_combinations() const
  {
    return static_cast<size_t>(sizes_[0]) *
           static_cast<size_t>(sizes_[1]) *
           static_cast<size_t>(sizes_[2]) *
           static_cast<size_t>(sizes_[3]);
  }

  // Get current combination without advancing
  bool current(int* comb) const
  {
    if (done_)
    {
      return false;
    }
    comb[0] = arrays_[0][indices_[0]];
    comb[1] = arrays_[1][indices_[1]];
    comb[2] = arrays_[2][indices_[2]];
    comb[3] = arrays_[3][indices_[3]];
    return true;
  }

  // Get the sizes of the input arrays
  const std::array<int, 4>& get_sizes() const
  {
    return sizes_;
  }

  void print_debug(std::ostream &os)
  {
    os << "DEBUG Information stored in CartesianProductOf4arrays_Generator object:\n";
    os << "Partition1: ";
    print_vec(os, arrays_[0]);
    os << "Partition2: ";
    print_vec(os, arrays_[1]);
    os << "Partition3: ";
    print_vec(os, arrays_[2]);
    os << "Partition4: ";
    print_vec(os, arrays_[3]);
    os << "Sizes: " << sizes_[0] << ", " << sizes_[1] << ", " << sizes_[2] << ", " << sizes_[3] << "\n";
  }
};


#endif
