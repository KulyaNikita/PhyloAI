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

// ============================================================================
// Quartet Pattern Counter Library, version 1.6
// Pure C++ class exposted to Python using bybind11
// ============================================================================
#ifndef QuartetPatternCounter_core_H
#define QuartetPatternCounter_core_H

#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <cstdint>  // For int32_t
#include <unordered_map>
#include "cartesian-product-of-4-generator-class.h"
#include <optional>
#include <climits>
#include <cstdint>

typedef float float_type_freq;   // If this is changed, also the numpy array passed to the functions has to be adapted.
typedef uint16_t cartesianProductIndex_type;

extern int global_verbosity;
class specializedThreadsCollection;


// Stateful processor class
class QuartetPatternCounter
{
private:
  std::unordered_map<std::string, std::string> fasta_map_string;
  std::vector<std::string>             seq_names;
  std::vector<const char *>            pointers_to_sequences;
  uint32_t                             alignment_length; // Used in count_four_sequences_patterns_*
  const std::string                     current_filename;
  const char                           dataType;
  uint32_t                             num_cols; // the frequency arrays are
  size_t                               call_count;
  double                               total_processing_time;
  std::optional<cartesianProductOf4_Generator>       combination_gen_full_dataset;
  std::optional<CartesianProductOf4arrays_Generator> combination_gen_four_lists_quartets;
  int32_t                              num_sequences;
  int                                  numThreads;

  uint8_t                              generator_type; // 0: uninitialised, 1: all, 2: four_cluster_quartet_frequencies



private:
  specializedThreadsCollection create_specializedThreadsCollection_4nextBatch(float_type_freq*, uint32_t*, size_t, size_t &, uint16_t *);

public:
  // Constructor - all patterns: Set generator_type to 0
  QuartetPatternCounter(const std::string &filename, char dataType='D', int numThreads=4, int verbosity=0);

  // Constructor - all patterns: Set generator_type to 0
  QuartetPatternCounter(const std::string &filename,
                        const int* partition1, size_t N1,
                        const int* partition2, size_t N2,
                        const int* partition3, size_t N3,
                        const int* partition4, size_t N4,
                        char dataType='D', int numThreads=4, int verbosity=0);

  void supply_data_and_construct_generator_for_four_cluster_quartet_analyses()
  {

  }
  //    // Constructor - four cluster mapping quartets: Set generator_type to 1
  //    QuartetPatternCounter(const std::string &filename=std::string(),  char dataType='D', int numThreads=4, int verbosity=0);

  // Get statistics about loaded file
  std::pair<size_t, unsigned> get_dataset_characteristics() const;

  // Process data (can be called multiple times with different arrays or the same array)
  // Returns the number of quartets that have been analyses.
  //    size_t next_quartet_pattern_frequencies(float_type_freq* freq, size_t N_freq);
  size_t next_quartet_pattern_frequencies(float_type_freq* freq, size_t N_freq, uint16_t *quartet_array=nullptr);

  // Process data (can be called multiple times with different arrays or the same array)
  // Returns the number of quartets that have been analyses.
  //    size_t next_quartet_pattern_frequencies_with_lengths(float_type_freq *freq, uint32_t *lengths, size_t N_freq);
  size_t next_quartet_pattern_frequencies_with_lengths(float_type_freq *freq, uint32_t *lengths, size_t N_freq, uint16_t *quartet_array=nullptr);

  size_t get_num_cols() const;

  // State queries
  size_t get_call_count() const;
  double get_total_processing_time() const;
};


#endif // QuartetPatternCounter_core_H
