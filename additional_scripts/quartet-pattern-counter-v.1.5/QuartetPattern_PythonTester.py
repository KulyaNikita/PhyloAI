# *******************************************************
#  BSD license:
#  Copyright 2026, Dr. Christoph Mayer, Nikita Kulikov, Dr. Iker Irisarri LIB, Bonn/Hamburg, Germany
# 
#  Redistribution and use in source and binary forms, with or without modification, are
#  permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice, this list of
#  conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of 
#  conditions and the following disclaimer in the documentation and/or other materials provided 
# with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to 
#  endorse or promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *******************************************************


import numpy as np
import QuartetPatternCounterModule
import itertools


# Define your symbols
bases = ['A', 'C', 'G', 'T']

## Set the repeat value (e.g., for 3-character combinations)
#n = 3

## Generate the product
#product_gen = itertools.product(bases, repeat=n)

## Print the results
#for combo in product_gen:
#    print("".join(combo))

def print_freqs(freqs):
  product_gen = itertools.product(bases, repeat=4)
  i = 0
  for combo in product_gen:
    print("(","".join(combo), ",", freqs[i], ")", sep='')
    i=i+1


# ============================================================================
# Testing and using the new quartet pattern counter
# ============================================================================
def test_simple_5():
  print("=== Testing and using the new quartet pattern counter with simple_5.fas ===\n")
  numThreads = 4
  dataType = 'D'
  verbose = 5

# Create a QuartetPatternCounter instance
  filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5.fas"
  filename = "/Daten/Bio/Tagungen/2025 St-Andrews_Meeting/Workshop-and-Lecture/Hands-On-Phylogenetics/Machine-Learning-Phylogenomics-Workshop-St-Andrews/concatenated_alignment_of_frogs_genes.fas"
  QPcounter = QuartetPatternCounterModule.QuartetPatternCounter(filename, dataType, numThreads, verbose)
  print(QPcounter.get_dataset_characteristics())

  batch_size = 5000
  freq_array = np.empty((batch_size, 256), dtype=np.float32)

  N = QPcounter.next_quartet_pattern_frequencies(freq_array)
  batch = 0
  while N!=0:
    print("Number of quartets determined: ", N)
    for i in range(N):
      print("Combination: ", i+1+batch*batch_size)
#      print_freqs(freq_array[i])
    batch +=1
    N = QPcounter.next_quartet_pattern_frequencies(freq_array)


def test_simple_5_gap():
  print("=== Testing and using the new quartet pattern counter with simple_5.fas ===\n")
  numThreads = 4
  dataType = 'D'
  verbose = 5

# Create a QuartetPatternCounter instance
  filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5_gap.fas"
#  filename = "/Daten/Bio/Tagungen/2025 St-Andrews_Meeting/Workshop-and-Lecture/Hands-On-Phylogenetics/Machine-Learning-Phylogenomics-Workshop-St-Andrews/concatenated_alignment_of_frogs_genes.fas"
  QPcounter = QuartetPatternCounterModule.QuartetPatternCounter(filename, dataType, numThreads, verbose)
  print(QPcounter.get_dataset_characteristics())

  batch_size = 5000
  freq_array = np.empty((batch_size, 256), dtype=np.float32)
  length_array = np.empty((batch_size), dtype=np.uint32)

  N = QPcounter.next_quartet_pattern_frequencies_with_lengths(freq_array, length_array)
  batch = 0
  while N!=0:
    print("Number of quartets determined: ", N)
    for i in range(N):
      print("Combination: ", i+1+batch*batch_size, " length: ", length_array[i+batch*batch_size]) 
#      print_freqs(freq_array[i])
    batch +=1
    N = QPcounter.next_quartet_pattern_frequencies(freq_array)

def test_gap_four_index_sets(filename, indexset1, indexset2, indexset3, indexset4):
  print("=== Testing and using the new quartet pattern counter using effective length mode with four index sets ===\n")
  print("Filename: ", filename)
  numThreads = 1
  dataType = 'D'
  verbose = 20

  # Test four index sets case:
  print("\n********\nFour index sets case\n********")
  print(indexset1)
  print(indexset2)
  print(indexset3)
  print(indexset4)
  QPcounter_fc = QuartetPatternCounterModule.QuartetPatternCounter(filename, indexset1, indexset2, indexset3, indexset4, dataType, numThreads, verbose)
  print("Dataset sizes (Number of sequences, alignment length): ", QPcounter_fc.get_dataset_characteristics())

  batch_size = 5000
  freq_array = np.empty((batch_size, 256), dtype=np.float32)
  length_array = np.empty((batch_size), dtype=np.uint32)
  pattern_array = np.empty((batch_size, 4), dtype=np.uint16)

  N = QPcounter_fc.next_quartet_pattern_frequencies_with_lengths_with_patterns(freq_array, length_array, pattern_array)
  batch = 0
  while N!=0:
    print("Number of quartets determined: ", N)
    for i in range(N):
      print("Combination: ", i+1+batch*batch_size, " length: ", length_array[i+batch*batch_size], " pattern: ", pattern_array[i]) 
      print_freqs(freq_array[i])
    batch +=1
    N = QPcounter_fc.next_quartet_pattern_frequencies_with_lengths_with_patterns(freq_array, length_array, pattern_array)


###### MAIN:

#test_simple_5()
# test_simple_5_gap()

filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-1.5/quartet-pattern-counter-xcode-v1.5/quartet-pattern-counter-xcode-v1.5/simple_5_gap.fas"
# filename = "/Daten/Bio/Tagungen/2025 St-Andrews_Meeting/Workshop-and-Lecture/Hands-On-Phylogenetics/Machine-Learning-Phylogenomics-Workshop-St-Andrews/concatenated_alignment_of_frogs_genes.fas"

arr1 = np.array([0]) 
arr2 = np.array([1,4])
arr3 = np.array([2])
arr4 = np.array([9])
test_gap_four_index_sets(filename, arr1, arr2, arr3, arr4)

