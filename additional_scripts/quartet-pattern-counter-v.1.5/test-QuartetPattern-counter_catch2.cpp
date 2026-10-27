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
// File: test_compute_catch2.cpp
// Unit tests using Catch2 framework (header-only version)
// ============================================================================

#include "QuartetPattern-counter-core.hpp"
#include <fstream>
#include <vector>
#include "symbol-cartesian-product_string.h"

// Helper to create test file
void create_test_file(const std::string& filename, const std::vector<double>& data) {
  std::ofstream file(filename);
  for (double val : data) {
    file << val << "\n";
  }
  file.close();
}

void print_pattern_frequencies_and_lengths_DEBUG(std::ostream &os, float_type_freq *freq, uint32_t *lengths, char dataType, size_t N_combinations, uint16_t *patterns=nullptr)
{
  cartesian_product    cp;
  vector<string>       all_patterns;
  int                  num_cols;

  if (dataType == 'P')
  {
    cp.operator()(4, "ARNDCQEGHILKMFPSTWYV", all_patterns);
    num_cols = 160000;
  }
  else
  {
    cp.operator()(4, "ACGT", all_patterns);
    num_cols = 256;
  }

  if (lengths)
    for (size_t r = 0; r < N_combinations; ++r)
    {
      os << "Combination: " << r+1 << "\t" << " length: " << lengths[r] << endl;
      if (patterns)
        os << "Combination indices: " << patterns[0+r*4] << "," << patterns[1+r*4] << "," << patterns[2+r*4] << "," << patterns[3+r*4] << "\n";
      for (int i=0; i<num_cols; ++i)
      {
        os << all_patterns[i] << "\t" << freq[r*num_cols+i] << endl;
      }
    }
  else
    for (size_t r = 0; r < N_combinations; ++r)
    {
      os << "Combination: " << r+1 << endl;
      for (int i=0; i<num_cols; ++i)
      {
        os << all_patterns[i] << "\t" << freq[r*num_cols+i] << endl;
      }
    }
}


//#define CATCH_CONFIG_MAIN  // Tells Catch2 to provide main()
#include "catch.hpp"
// #include "/Daten/Programmieren/C++/Bib/Catch2-single-header/catch.hpp"       // For header-only single file version
                                                                             // OR if using catch_amalgamated.hpp:

#ifdef CATCH_CONFIG_MAIN
// #include "catch_amalgamated.hpp"

TEST_CASE("DataProcessor construction", "[DataProcessor][constructor]") {
  SECTION("DNA data constructor") {
//    std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple.fas";
    std::string filename = "simple.fas";
    DataProcessor data_DNA(filename);
    CHECK(data_DNA.get_num_cols() == 256);
    CHECK(data_DNA.get_call_count() == 0);
    CHECK(data_DNA.get_dataset_characteristics()==std::pair<int,int>(4,5));
  }
}

  TEST_CASE("DataProcessor construct thread collection", "[DataProcessor][construct thread collection]") {
    SECTION("DNA data constructor") {
//      std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple.fas";
      std::string filename = "simple.fas";
      DataProcessor data_DNA(filename);
      float_type_freq *freqs = new float_type_freq[10];
      data_DNA.next_quartet_pattern_frequencies(freqs, 10);
  }



/*
  SECTION("Constructor with custom num_cols") {
    DataProcessor processor(512);
    CHECK(processor.get_num_cols() == 512);
    CHECK_FALSE(processor.has_file_loaded());
  }
 */
}

#endif


/*
TEST_CASE("DataProcessor construction", "[DataProcessor][constructor]") {
  SECTION("Default constructor") {
    DataProcessor processor;
    CHECK(processor.get_num_cols() == 256);
    CHECK_FALSE(processor.has_file_loaded());
    CHECK(processor.get_call_count() == 0);
  }

  SECTION("Constructor with custom num_cols") {
    DataProcessor processor(512);
    CHECK(processor.get_num_cols() == 512);
    CHECK_FALSE(processor.has_file_loaded());
  }
}

TEST_CASE("DataProcessor file loading", "[DataProcessor][load]") {
  DataProcessor processor;

  SECTION("Load valid file") {
    std::vector<double> test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    create_test_file("test_load.txt", test_data);

    processor.load_file("test_load.txt");

    CHECK(processor.has_file_loaded());
    auto file_data = processor.get_file_data();
    CHECK(file_data.size() == 5);
    CHECK(file_data[0] == Catch::Detail::Approx(1.0));
    CHECK(file_data[4] == Catch::Detail::Approx(5.0));
  }

  SECTION("Load empty file") {
    create_test_file("test_empty_load.txt", {});

    processor.load_file("test_empty_load.txt");

    CHECK(processor.has_file_loaded()); // File is loaded, just empty
    CHECK(processor.get_file_data().empty());
  }

  SECTION("Load nonexistent file throws") {
    REQUIRE_THROWS_AS(
                      processor.load_file("nonexistent.txt"),
                      std::runtime_error
                      );
    CHECK_FALSE(processor.has_file_loaded());
  }

  SECTION("Load multiple files in sequence") {
    std::vector<double> data1 = {1.0, 2.0};
    std::vector<double> data2 = {10.0, 20.0, 30.0};
    create_test_file("test_file1.txt", data1);
    create_test_file("test_file2.txt", data2);

    processor.load_file("test_file1.txt");
    CHECK(processor.get_file_data().size() == 2);

    processor.load_file("test_file2.txt");
    CHECK(processor.get_file_data().size() == 3);
    CHECK(processor.get_file_data()[0] == Catch::Detail::Approx(10.0));
  }
}

TEST_CASE("DataProcessor statistics", "[DataProcessor][stats]") {
  DataProcessor processor;

  SECTION("Stats on loaded file") {
    std::vector<double> test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    create_test_file("test_stats_proc.txt", test_data);

    processor.load_file("test_stats_proc.txt");
    auto [mean, min_val, max_val] = processor.compute_file_stats();

    CHECK(mean == Catch::Detail::Approx(3.0));
    CHECK(min_val == Catch::Detail::Approx(1.0));
    CHECK(max_val == Catch::Detail::Approx(5.0));
  }

  SECTION("Stats before loading file") {
    auto [mean, min_val, max_val] = processor.compute_file_stats();
    CHECK(mean == Catch::Detail::Approx(0.0));
  }
}

TEST_CASE("DataProcessor process_data", "[DataProcessor][process]") {
  DataProcessor processor(256);

  std::vector<double> file_data = {10.0, 20.0, 30.0};
  create_test_file("test_process_class.txt", file_data);
  processor.load_file("test_process_class.txt");

  double arr1[] = {1.0, 2.0, 3.0};
  double arr2[] = {4.0, 5.0, 6.0};
  double arr3[] = {7.0, 8.0, 9.0};
  double arr4[] = {0.1, 0.2, 0.3};

  size_t N = 3;
  size_t num_cols = 256;

  std::vector<double> output1(N * num_cols);
  std::vector<double> output2(N * num_cols);
  std::vector<double> output3(N * num_cols);

  SECTION("Basic processing") {
    processor.process_data(
                           arr1, 3, arr2, 3, arr3, 3, arr4, 3,
                           output1.data(), output2.data(), output3.data(), N
                           );

    // Check first element
    CHECK(output1[0] == Catch::Detail::Approx(1.0 + 4.0 + 10.0 + 0));

    // Check call count incremented
    CHECK(processor.get_call_count() == 1);

    // Check processing time tracked
//    std::cout << "Value: " << processor.get_total_processing_time() << std::endl;
    CHECK(processor.get_total_processing_time() >= 0.0);
  }

  SECTION("Multiple calls increment state") {
    processor.process_data(arr1, 3, arr2, 3, arr3, 3, arr4, 3,
                           output1.data(), output2.data(), output3.data(), N);

    CHECK(processor.get_call_count() == 1);
    double time1 = processor.get_total_processing_time();

    processor.process_data(arr1, 3, arr2, 3, arr3, 3, arr4, 3,
                           output1.data(), output2.data(), output3.data(), N);

    CHECK(processor.get_call_count() == 2);
    CHECK(processor.get_total_processing_time() >= time1);
  }

  SECTION("Processing without file loaded") {
    DataProcessor empty_processor;

    empty_processor.process_data(arr1, 3, arr2, 3, arr3, 3, arr4, 3,
                                 output1.data(), output2.data(), output3.data(), N);

    // Should work, file_val will be 0
    CHECK(output1[0] == Catch::Detail::Approx(1.0 + 4.0 + 0.0 + 0));
    CHECK(empty_processor.get_call_count() == 1);
  }
}

TEST_CASE("DataProcessor configuration", "[DataProcessor][config]") {
  DataProcessor processor;

  SECTION("Change num_cols") {
    CHECK(processor.get_num_cols() == 256);

    processor.set_num_cols(512);
    CHECK(processor.get_num_cols() == 512);

    processor.set_num_cols(128);
    CHECK(processor.get_num_cols() == 128);
  }

  SECTION("num_cols affects output shape") {
    std::vector<double> file_data = {5.0};
    create_test_file("test_cols.txt", file_data);
    processor.load_file("test_cols.txt");

    double arr[] = {1.0};
    std::vector<double> output1(128);
    std::vector<double> output2(128);
    std::vector<double> output3(128);

    processor.set_num_cols(128);
    processor.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                           output1.data(), output2.data(), output3.data(), 1);

    // Check that all 128 columns are filled
    CHECK(output1[0] == Catch::Detail::Approx(1.0 + 1.0 + 5.0 + 0));    // j=0
    CHECK(output1[127] == Catch::Detail::Approx(1.0 + 1.0 + 5.0 + 127)); // j=127
  }
}

TEST_CASE("DataProcessor reset", "[DataProcessor][reset]") {
  DataProcessor processor;

  std::vector<double> file_data = {10.0, 20.0};
  create_test_file("test_reset.txt", file_data);
  processor.load_file("test_reset.txt");

  double arr[] = {1.0};
  std::vector<double> output1(256);
  std::vector<double> output2(256);
  std::vector<double> output3(256);

  processor.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                         output1.data(), output2.data(), output3.data(), 1);
  processor.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                         output1.data(), output2.data(), output3.data(), 1);

  SECTION("Reset clears all state") {
    CHECK(processor.has_file_loaded());
    CHECK(processor.get_call_count() == 2);
    CHECK(processor.get_total_processing_time() >= 0.0);

    processor.reset();

    CHECK_FALSE(processor.has_file_loaded());
    CHECK(processor.get_call_count() == 0);
    CHECK(processor.get_total_processing_time() == Catch::Detail::Approx(0.0));
    CHECK(processor.get_file_data().empty());
  }

  SECTION("Can use processor after reset") {
    processor.reset();

    processor.load_file("test_reset.txt");
    CHECK(processor.has_file_loaded());

    processor.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                           output1.data(), output2.data(), output3.data(), 1);
    CHECK(processor.get_call_count() == 1);
  }
}

TEST_CASE("DataProcessor caching", "[DataProcessor][cache]") {
  DataProcessor processor(10); // Small for testing

  std::vector<double> file_data = {5.0};
  create_test_file("test_cache.txt", file_data);
  processor.load_file("test_cache.txt");

  double arr[] = {1.0, 2.0};
  size_t N = 2;
  std::vector<double> output1(N * 10);
  std::vector<double> output2(N * 10);
  std::vector<double> output3(N * 10);

  processor.process_data(arr, 2, arr, 2, arr, 2, arr, 2,
                         output1.data(), output2.data(), output3.data(), N);

  SECTION("Last results are cached") {
    const auto& cached1 = processor.get_last_result1();
    const auto& cached2 = processor.get_last_result2();
    const auto& cached3 = processor.get_last_result3();

    CHECK(cached1.size() == N * 10);
    CHECK(cached2.size() == N * 10);
    CHECK(cached3.size() == N * 10);

    // Verify cached values match output
    CHECK(cached1[0] == Catch::Detail::Approx(output1[0]));
    CHECK(cached1[19] == Catch::Detail::Approx(output1[19]));
  }

  SECTION("Cache updates on new call") {
    double arr_new[] = {10.0, 20.0};
    processor.process_data(arr_new, 2, arr_new, 2, arr_new, 2, arr_new, 2,
                           output1.data(), output2.data(), output3.data(), N);

    const auto& cached = processor.get_last_result1();
    // New values should be different
    CHECK(cached[0] == Catch::Detail::Approx(10.0 + 10.0 + 5.0 + 0));
  }

  SECTION("Reset clears cache") {
    CHECK_FALSE(processor.get_last_result1().empty());

    processor.reset();

    CHECK(processor.get_last_result1().empty());
    CHECK(processor.get_last_result2().empty());
    CHECK(processor.get_last_result3().empty());
  }
}

TEST_CASE("DataProcessor state isolation", "[DataProcessor][isolation]") {
  SECTION("Multiple independent processors") {
    DataProcessor proc1(256);
    DataProcessor proc2(128);

    std::vector<double> data1 = {10.0};
    std::vector<double> data2 = {20.0};
    create_test_file("test_proc1.txt", data1);
    create_test_file("test_proc2.txt", data2);

    proc1.load_file("test_proc1.txt");
    proc2.load_file("test_proc2.txt");

    CHECK(proc1.get_file_data()[0] == Catch::Detail::Approx(10.0));
    CHECK(proc2.get_file_data()[0] == Catch::Detail::Approx(20.0));
    CHECK(proc1.get_num_cols() == 256);
    CHECK(proc2.get_num_cols() == 128);

    double arr[] = {1.0};
    std::vector<double> output(256);

    proc1.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                       output.data(), output.data(), output.data(), 1);

    CHECK(proc1.get_call_count() == 1);
    CHECK(proc2.get_call_count() == 0); // Unaffected
  }
}

TEST_CASE("DataProcessor edge cases", "[DataProcessor][edge]") {
  DataProcessor processor;

  SECTION("Process with N=0") {
    std::vector<double> output1, output2, output3;
    double arr[] = {1.0};

    // Should not crash
    processor.process_data(arr, 1, arr, 1, arr, 1, arr, 1,
                           output1.data(), output2.data(), output3.data(), 0);

    CHECK(processor.get_call_count() == 1);
  }

  SECTION("Large N value") {
    std::vector<double> file_data = {5.0};
    create_test_file("test_large.txt", file_data);
    processor.load_file("test_large.txt");

    size_t N = 10000;
    std::vector<double> arr(N, 1.0);
    std::vector<double> output1(N * 256);
    std::vector<double> output2(N * 256);
    std::vector<double> output3(N * 256);

    processor.process_data(arr.data(), N, arr.data(), N, arr.data(), N, arr.data(), N,
                           output1.data(), output2.data(), output3.data(), N);

    CHECK(processor.get_call_count() == 1);
    CHECK(output1[0] == Catch::Detail::Approx(1.0 + 1.0 + 5.0 + 0));
  }
}

*/


#ifndef CATCH_CONFIG_MAIN

using namespace std;

void test_without_lengths_DNA()
{
  size_t count_res;

  cerr << "Welcome to the test main function.\n";
  cerr << "Testing DNA input sequence file.\n";

//  std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5_gap.fas";
  std::string filename = "simple_5_gap.fas";
  QuartetPatternCounter qpc(filename, 'D', 1, 10);
  float_type_freq *freqs = new float_type_freq[10*256];
  count_res = qpc.next_quartet_pattern_frequencies(freqs, 10);

  cout << "Number of quartet combinations for which frequencies have been stored: " << count_res << endl;
  print_pattern_frequencies_and_lengths_DEBUG(cout, freqs, nullptr, 'D', count_res);
}


void test_with_lengths_DNA()
{
  size_t count_res;

//  std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5_gap.fas";
  std::string filename = "simple_5_gap.fas";
  cerr << "Testing DNA input sequence file:\n";
  cerr << filename << endl;

   QuartetPatternCounter qpc(filename, 'D', 1, 10);
  float_type_freq *freqs = new float_type_freq[10*256];
  uint32_t *lengths = new uint32_t[10];
  count_res = qpc.next_quartet_pattern_frequencies_with_lengths(freqs, lengths, 10);

  cout << "Number of quartet combinations for which frequencies have been stored: " << count_res << endl;
  print_pattern_frequencies_and_lengths_DEBUG(cout, freqs, lengths, 'D', count_res);
}

void test_without_lengths_Protein()
{
  size_t count_res;

//  std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5_gap.fas";
  std::string filename = "simple_5_gap.fas";
  cerr << "Testing protein input sequence file:\n";
  cerr << filename << endl;

  QuartetPatternCounter qpc(filename, 'P', 1, 10);
  float_type_freq *freqs = new float_type_freq[10*160000];
  count_res = qpc.next_quartet_pattern_frequencies(freqs, 10);

  cout << "Number of quartet combinations for which frequencies have been stored: " << count_res << endl;
  print_pattern_frequencies_and_lengths_DEBUG(cout, freqs, nullptr, 'P', count_res);
}

void test_with_lengths_Protein()
{
  size_t count_res;

  std::string filename = "simple_5_gap.fas";
//  std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-xcode/quartet-pattern-counter-xcode/simple_5_gap.fas";
  cerr << "Testing protein input sequence file:\n";
  cerr << filename << endl;

  QuartetPatternCounter qpc(filename, 'P', 1, 10);
  float_type_freq *freqs = new float_type_freq[10*160000];
  uint32_t *lengths = new uint32_t[10];
  count_res = qpc.next_quartet_pattern_frequencies_with_lengths(freqs, lengths, 10);

  cout << "Number of quartet combinations for which frequencies have been stored: " << count_res << endl;
  print_pattern_frequencies_and_lengths_DEBUG(cout, freqs, lengths, 'P', count_res);
}

void test_four_cluster_quartet_freq_without_lengths_DNA()
{
  size_t count_res;

//  std::string filename = "/Daten/Programmieren/C++/Programme/quartet_pattern_counter/quartet-pattern-counter-1.5/quartet-pattern-counter-xcode-v1.5/quartet-pattern-counter-xcode-v1.5/simple_5_gap.fas";
  std::string filename = "simple_5_gap.fas";
  cerr << "Testing four cluster quartet selection. Testing DNA input sequence file:\n";
  cerr << filename << endl;

  /*
  int part1[] = {0,4};
  int part2[] = {1};
  int part3[] = {2};
  int part4[] = {3};
  int size1 = 2, size2 = 1, size3 = 1, size4 = 1;
  */

  int part1[] = {0,3};
  int part2[] = {1};
  int part3[] = {2};
  int part4[] = {4};
  int size1 = 2, size2 = 1, size3 = 1, size4 = 1;

  QuartetPatternCounter qpc(filename, part1, size1, part2, size2, part3, size3, part4, size4, 'D', 1, 10);
  float_type_freq *freqs = new float_type_freq[10*256];
  uint32_t *lengths = new uint32_t[10];
  uint16_t *patterns = new uint16_t [10*4];
  count_res = qpc.next_quartet_pattern_frequencies_with_lengths(freqs, lengths, 10, patterns);

  cout << "Number of quartet combinations for which frequencies have been stored: " << count_res << endl;
  print_pattern_frequencies_and_lengths_DEBUG(cout, freqs, lengths, 'D', count_res, patterns);
}

int main()
{
//  test_without_lengths_DNA();
//  test_with_lengths_DNA();

//  test_without_lengths_Protein();
//  test_with_lengths_Protein();

//  test_four_cluster_quartet_freq_without_lengths_DNA();

  std::cout << "Welcome to the main function in the test-QuartetPatternCounter-catch2.cpp file:\n";
  test_four_cluster_quartet_freq_without_lengths_DNA();
}

#endif
