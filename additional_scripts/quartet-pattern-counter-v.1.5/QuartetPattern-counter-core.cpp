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
// File: compute_core.cpp
// Implementation
// ============================================================================
#include "QuartetPattern-counter-core.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <unordered_map>
#include "basic-DNA-RNA-AA-routines.h"
#include <thread>

int global_verbosity = 10;

//*********************************************
// Helper functions. Not exposed globally.
//*********************************************
int32_t count_four_sequences_patterns_DNA(const char *s1, const char *s2, const char *s3, const char *s4,
                                          unsigned alignment_length,
                                          float_type_freq *pfreqs)
{
  unsigned char d1,d2,d3,d4;
  unsigned sites = 0;
  const int dataType_freq_len = 256;
  
  // This is 1kb large. Should run fast even if created millions of times.
  // This version is 100% thread safe.
  unsigned ufreq[4][4][4][4] = {}; // Inits with 0's, according to C++ standard.
  
  for (unsigned i=0; i<alignment_length; ++i)
  {
    d1 = s1[i];
    d2 = s2[i];
    d3 = s3[i];
    d4 = s4[i];
    
    if (d1>3 || d2>3 || d3>3 || d4>3 ) // Filter sites with gaps or invalid symbols
      continue;
    ++ufreq[d1][d2][d3][d4];
    ++sites;
  }
  
  if (sites == 0)
    return 0;
  
  unsigned         *Pu = &ufreq[0][0][0][0];
  float_type_freq  *Pf = &pfreqs[0];
  for (int i=0; i<dataType_freq_len; ++i)
  {
    *Pf = *Pu/(float_type_freq)sites;
    ++Pu;
    ++Pf;
  }
  return sites;
}


int32_t count_four_sequences_patterns_AA(const char *s1, const char *s2, const char *s3, const char *s4,
                                         unsigned alignment_length,
                                         float_type_freq *pfreqs)
{
  unsigned char d1,d2,d3,d4;
  unsigned sites = 0;
  const int dataType_freq_len = 160000;
  
  // TODO:
  //   unsigned ufreq[20][20][20][20] = {0}; // C-Style array might be too large. Crashes with Thread 2: EXC_BAD_ACCESS (code=2, address=0x70000f2ec000)
  
  //  unsigned ****ufreq = new unsigned***[20];
  //  for (int i = 0; i < 20; ++i) {
  //    ufreq[i] = new unsigned**[20];
  //    for (int j = 0; j < 20; ++j) {
  //      ufreq[i][j] = new unsigned*[20];
  //      for (int k = 0; k < 20; ++k) {
  //        ufreq[i][j][k] = new unsigned[20]();
  //      }
  //    }
  //  }

  // Be careful: this is 640kb large, should be more efficient if static,
  // but then needs thread_local - otherwise all threads write to the same array.
  thread_local static unsigned ufreq[20][20][20][20]; // Create one for each thread
  memset(ufreq, 0, sizeof(ufreq));                    // Init for each call to function.

  //** Alternative to the previous two lines: If the above causes problems with size on the stack.
  // thread_local static std::unique_ptr<unsigned[][20][20][20]> ufreq(new unsigned[20][20][20][20]);  // Created once, no init.
  // memset(ufreq.get(), 0, 20*20*20*20*sizeof(unsigned));  // Zero when needed
  //
  //** Bad version. Multiple threads write to the same array.
  // static unsigned (*ufreq)[20][20][20] = new unsigned[20][20][20][20];
  // memset(ufreq, 0, 160000);
  
  for (unsigned i=0; i<alignment_length; ++i)
  {
    d1 = s1[i];
    d2 = s2[i];
    d3 = s3[i];
    d4 = s4[i];
    
    //    cout << d1 << " " << d2 << " " << d3 << " " << d4 << "\n";
    
    if (d1>19 || d2>19 || d3>19 || d4>19 )  // Filter sites with gaps or invalid symbols
      continue;
    ++ufreq[d1][d2][d3][d4];
    ++sites;
  }
  
  unsigned         *Pu = &ufreq[0][0][0][0];
  float_type_freq  *Pf = &pfreqs[0];
  for (int i=0; i<dataType_freq_len; ++i)
  {
    *Pf = *Pu/(float_type_freq)sites;
    ++Pu;
    ++Pf;
  }
  
  return sites;
}


struct single_analysis_data
{
  cartesianProductIndex_type i0;
  cartesianProductIndex_type i1;
  cartesianProductIndex_type i2;
  cartesianProductIndex_type i3;
  float_type_freq      *freqs;  // Is only allowed to access dataType_freq_len values.
  uint32_t             *sites;
  vector<const char *> &pointers_to_sequences;
  unsigned             seqlen;
  const char           dataType;
  //  static int           dataType_freq_len; //TODO: Not used yet.
  
  single_analysis_data(cartesianProductIndex_type i0, cartesianProductIndex_type i1, cartesianProductIndex_type i2, cartesianProductIndex_type i3, float_type_freq *freqs,
                       vector<const char *> &pointers_to_sequences, unsigned seqlen, uint32_t *psites, char dataType):
  i0(i0), i1(i1), i2(i2), i3(i3), freqs(freqs), sites(psites), pointers_to_sequences(pointers_to_sequences), seqlen(seqlen), dataType(dataType)
  {}
  
  void print_debug_info_single_analysis(const char *str, ostream &os)
  {
    os << str << "(" << i0 << "," << i1 << "," << i2 << "," << i3 << ")";
  }
  
  void run_single_analysis()
  {
    const bool store = (sites != nullptr);
    int result;
    
    if (dataType == 'P')
      //    if (1)
      result = count_four_sequences_patterns_AA(pointers_to_sequences[i0],
                                                pointers_to_sequences[i1],
                                                pointers_to_sequences[i2],
                                                pointers_to_sequences[i3],
                                                seqlen, freqs);
    else if (dataType == 'D')
      result = count_four_sequences_patterns_DNA(pointers_to_sequences[i0],
                                                 pointers_to_sequences[i1],
                                                 pointers_to_sequences[i2],
                                                 pointers_to_sequences[i3],
                                                 seqlen, freqs);
    //    print_debug_info("#DEBUG INFO FOR THIS ANALYSIS: #", cout);
    else
    {
      std::cerr << "Unknown dataType: " << dataType << "\n";
      exit(0);
    }
    
    if (store)
      *sites = result;
  }
  
};

class thread_data
{
private:
  vector<single_analysis_data> all_analyses;
  
public:
  void add(single_analysis_data &sa)
  {
    all_analyses.push_back(sa);
  }
  
  void print_debug_info_thread_data(const char *str, ostream &os)
  {
    for (auto a: all_analyses)
    {
      a.print_debug_info_single_analysis(str, os);
    }
  }
  
  void run_all_analyses()
  {
    //    cout << "Hi from run_all_analyses(). Size of all_analyses:  " << all_analyses.size() << endl;
    for (auto a: all_analyses )
    {
      //      cout << "Starting analysis!\n";
      a.run_single_analysis();
    }
  }
};

class specializedThreadsCollection
{
private:
  vector<thread_data> th_data;
  unsigned cyclic_add_index;
  unsigned num_threads;
  
public:
  specializedThreadsCollection(int N):th_data(N), cyclic_add_index(0), num_threads(N)
  {}
  
  void add(single_analysis_data &sad)
  {
    th_data[cyclic_add_index].add(sad);
    ++cyclic_add_index;
    if (cyclic_add_index == num_threads)
      cyclic_add_index = 0;
  }
  
  void print_debug_info_thread_collection(ostream &os)
  {
    os << "\n### Debug information for thread collection:" << '\n';
    os << "##  Number of threads: " << num_threads << '\n';
    os << "#   Tasks assigned to the threads:\n";
    for (int i=0; i< (int)th_data.size(); ++i)
    {
      os << "#     Thread:" << i << ":";
      
      th_data[i].print_debug_info_thread_data(" ", os);
      os << '\n';
    }
    if (th_data.size() == 0)
      os << "-\n";
    os << "### Finished printing Debug information for thread collection.\n\n";
  }
  
  void run_thread(int i)
  {
    //    cout << "Hi from run_thread: " << i << '\n';
    th_data[i].run_all_analyses();
  }
  
  void run_all_threads_non_parallel()
  {
    for (unsigned i = 0; i < num_threads; ++i)
    {
      run_thread(i);
    }
  }
  
  void run_all_threads_and_join()
  {
    std::vector<std::thread> threads;
    
    // Example:
    //    void Test::runMultiThread()
    //    {
    //     std::thread t1(&Test::calculate, this,  0, 10);
    //     std::thread t2(&Test::calculate, this, 11, 20);
    //     t1.join();
    //     t2.join();
    //    }
    
    
    try
    {
      for (unsigned i = 0; i < num_threads; ++i)
      {
        // We are using the std::thread constructor that allows binding a member function to an object instance. The syntax is:
        // std::thread(&Class::MemberFunction, object_pointer, arg1, arg2, ...)
        // We pass the index i. This tells the thread which data it should work on:
        //  cout << "Calling threads.emplace_back for thread " << i << endl;
        threads.emplace_back(&specializedThreadsCollection::run_thread, this, i);
      }
      
      // Join the threads with the main thread
      for (auto& th : threads)
      {
        if (th.joinable())
        {
          if (global_verbosity >= 5)
            std::cerr << "Joining thread." << std::endl;
          th.join();  // Wait for the thread to finish
        }
        else
        {
          std::cerr << "Warning: Thread not joinable. Not sure this runs correctly.\n";
        }
      }
    }
    catch (const std::system_error& e)
    {
      std::cerr << "Thread creation/join error: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
      std::cerr << "Exception: " << e.what() << std::endl;
    }
  }
};


void simple_recode_string_DNA(std::string &str)
{
  for (auto i = str.begin(); i != str.end(); ++i)
    *i = DNA2simpleIndex(*i);
}

void simple_recode_string_AA(std::string &str)
{
  for (auto i = str.begin(); i != str.end(); ++i)
    *i = AA2simpleIndex(*i);
}

void read_fasta_file_into_map_remove_spaces(const std::string &filename,
                                            std::unordered_map<std::string, std::string> &fastamap,
                                            std::vector<std::string> &namevec)
{
  std::ifstream is;
  is.open(filename);
  std::string line, lastseqstring, lastname;
  
  fastamap.clear();
  
  if (is.fail())
  {
    return;
  }
  
  while (getline(is,line))
  {
    if (line[0] == '>')
    {
      if (!lastseqstring.empty())
      {
        fastamap[lastname] = lastseqstring;
        namevec.push_back(lastname);
      }
      lastname = line.substr(1, std::string::npos);
      lastseqstring.clear();
    }
    else
    {
      line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
      lastseqstring.append(line);
    }
  }
  if (!lastseqstring.empty())
  {
    fastamap[lastname] = lastseqstring;
    namevec.push_back(lastname);
  }
  is.close();
}

void read_fasta_file(const std::string &filename,
                     std::unordered_map<std::string, std::string> &fastamap,
                     std::vector<std::string> &namevec,
                     std::vector<const char *> &pointers_to_sequences,
                     char dataType)
{
  read_fasta_file_into_map_remove_spaces(filename, fastamap, namevec);
  if (namevec.empty())
  {
    std::cerr << "No sequnces read from input file. Exiting.\n";
    exit(0);
  }
  
  if (dataType == 'D')
  {
    for (unsigned i=0; i<namevec.size(); ++i)
    {
      //       cout << ">>" << namevec[i] << '\n'
      //            << fastamap[namevec[i]] << '\n';
      std::string &tmp = fastamap[namevec[i]];
      simple_recode_string_DNA(tmp);
      pointers_to_sequences.push_back(tmp.c_str());
    }
  }
  else
  {
    for (unsigned i=0; i<namevec.size(); ++i)
    {
      //       cout << ">>" << namevec[i] << '\n'
      //            << fastamap[namevec[i]] << '\n';
      std::string &tmp = fastamap[namevec[i]];
      simple_recode_string_AA(tmp);
      pointers_to_sequences.push_back(tmp.c_str());
    }
  }
  // DONE: Read sequence information and store in data structures
}



  
  // ============================================================================
  // DataProcessor implementation
  // ============================================================================
  
  QuartetPatternCounter::QuartetPatternCounter(const std::string &filename, char dataType, int numThreads, int verbosity)
  : current_filename(filename), dataType(dataType), call_count(0), total_processing_time(0.0), numThreads(numThreads), generator_type(1)
  {
    global_verbosity = verbosity;
    
    if (dataType == 'D')
      num_cols = 256;
    else if (dataType == 'P')
      num_cols = 160000;
    else
    {
      std::cerr << "Unknown datattype argument in DataProcessor constructor: " << dataType << "\n";
      exit(-1);
    }
    
    read_fasta_file(current_filename, fasta_map_string, seq_names, pointers_to_sequences, dataType);

    num_sequences = (int32_t)seq_names.size();
    alignment_length        = (uint32_t)fasta_map_string[seq_names[0]].length();

    if (generator_type == 1) // all generator
    {
      // Construct our generator that enumerates all patterns:
      combination_gen_full_dataset.emplace(num_sequences);
    }

    long long num_combinations = binomialCoefficient4((uint32_t)num_sequences);
    if (num_combinations > INT_MAX)
    {
      cout << "Number of combinations will be > INT_MAX. Please rewrite program to make this possible.\n";
      exit(-1);
    }
    
    if (global_verbosity >= 5)
    {
      std::cerr << "INFO Call to constructor DataProcessor. Stored data:\n"
      << "INFO filename: " << current_filename << "\nINFO dataType: " << dataType << "\nINFO numThreads: " << numThreads
      << "\nINFO num_sequences: " << num_sequences << "\nINFO seq_lengths: " << alignment_length << "\n";
    }
  }

  QuartetPatternCounter::QuartetPatternCounter(const std::string &filename,
                                               const int32_t* partition1, size_t N1,
                                               const int32_t* partition2, size_t N2,
                                               const int32_t* partition3, size_t N3,
                                               const int32_t* partition4, size_t N4,
                                               char dataType, int numThreads, int verbosity)
  : current_filename(filename), dataType(dataType), call_count(0), total_processing_time(0.0), numThreads(numThreads), generator_type(2)
  {
    global_verbosity = verbosity;

    if (dataType == 'D')
      num_cols = 256;
    else if (dataType == 'P')
      num_cols = 160000;
    else
    {
      std::cerr << "Unknown datattype argument in DataProcessor constructor: " << dataType << "\n";
      exit(-1);
    }

    read_fasta_file(current_filename, fasta_map_string, seq_names, pointers_to_sequences, dataType);

    num_sequences = (int32_t)seq_names.size();
    alignment_length        = (uint32_t)fasta_map_string[seq_names[0]].length();

    for (size_t i=0; i<N1; ++i)
      if (partition1[i] < 0 || partition1[i] >= num_sequences)
      {
        std::cerr << "ERROR: One of the indices in list 1 is out of rang: (" << 0 << "," << num_sequences-1 << "): " << partition1[i] << '\n';
        exit(-1);
      }
    for (size_t i=0; i<N2; ++i)
      if (partition2[i] < 0 || partition2[i] >= num_sequences)
      {
        std::cerr << "ERROR: One of the indices in list 2 is out of rang (" << 0 << "," << num_sequences-1 << "): " << partition2[i] << '\n';
        exit(-1);
      }
    for (size_t i=0; i<N3; ++i)
      if (partition3[i] < 0 || partition3[i] >= num_sequences)
      {
        std::cerr << "ERROR: One of the indices in list 3 is out of rang (" << 0 << "," << num_sequences-1 << "): " << partition3[i] << '\n';
        exit(-1);
      }
    for (size_t i=0; i<N4; ++i)
      if (partition4[i] < 0 || partition4[i] >= num_sequences)
      {
        std::cerr << "ERROR: One of the indices in list 4 is out of rang (" << 0 << "," << num_sequences-1 << "): " << partition4[i] << '\n';
        exit(-1);
      }

    // Construct our generator that enumerates all four_cluster_quartets:
    combination_gen_four_lists_quartets.emplace(partition1, N1,
                                                  partition2, N2,
                                                  partition3, N3,
                                                  partition4, N4);

    long long num_combinations = combination_gen_four_lists_quartets->total_combinations();
    if (num_combinations > INT_MAX)
    {
      cout << "Number of combinations will be > INT_MAX. Please rewrite program to make this possible.\n";
      exit(-1);
    }

    if (global_verbosity >= 5)
    {
      std::cerr << "INFO Call to constructor DataProcessor. Stored data:\n"
      << "INFO filename: " << current_filename << "\nINFO dataType: " << dataType << "\nINFO numThreads: " << numThreads
      << "\nINFO num_sequences: " << num_sequences << "\nINFO alignment_length: " << alignment_length << "\n";
    }
    if (global_verbosity >= 10)
    {
      combination_gen_four_lists_quartets->print_debug(std::cerr);
    }
  }

  
  specializedThreadsCollection QuartetPatternCounter::create_specializedThreadsCollection_4nextBatch(float_type_freq* freq, uint32_t* lengths, size_t N_freq, size_t &N_freq_computed, uint16_t *patterns)
  {
    specializedThreadsCollection thread_collection(numThreads);
    Quartet_Cartesian_Product_type comb;
    
    //   std::cout << num_cols << std::endl;
    
    unsigned combination_counter = 0;
    uint32_t *pointer_length_value_this_element;

    if (generator_type == 1)
    {
      while (combination_counter < N_freq && combination_gen_full_dataset->next(comb))
      {
        if (global_verbosity >= 5)
        {
          // The following does not make sense when using treads:
          //      cout << "Working on combination: " << "(" << combination_counter+1 << "/" << num_combinations << ") "
          //      << i0 << " " << i1 << " " << i2 << " " << i3 << "\n";

          if (global_verbosity >= 10)
          {
            //          cout << "Sequences:\n";
            //          print_coded_sequence(pointers_to_sequences[comb[0]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[1]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[2]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[3]], seqlen); cout << '\n';
          }
        }

        if (patterns)
          memcpy(patterns+combination_counter*sizeof(Quartet_Cartesian_Product_type), comb.data(), sizeof(Quartet_Cartesian_Product_type));

        if (lengths)
          pointer_length_value_this_element = lengths+combination_counter;
        else
          pointer_length_value_this_element = nullptr;
        float_type_freq *freq_element = freq+combination_counter*num_cols;
        single_analysis_data sad(comb[0], comb[1], comb[2], comb[3], freq_element, pointers_to_sequences, alignment_length, pointer_length_value_this_element, dataType);
        thread_collection.add(sad);
        ++combination_counter;
      }
    }
    else if (generator_type == 2)
    {
      while (combination_counter < N_freq && combination_gen_four_lists_quartets->next(comb))
      {
        if (global_verbosity >= 5)
        {
          // The following does not make sense when using treads:
          //      cout << "Working on combination: " << "(" << combination_counter+1 << "/" << num_combinations << ") "
          //      << i0 << " " << i1 << " " << i2 << " " << i3 << "\n";

          if (global_verbosity >= 10)
          {
            //          cout << "Sequences:\n";
            //          print_coded_sequence(pointers_to_sequences[comb[0]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[1]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[2]], seqlen); cout << '\n';
            //          print_coded_sequence(pointers_to_sequences[comb[3]], seqlen); cout << '\n';
          }
        }

        if (patterns)
          memcpy(patterns+combination_counter*4, comb.data(), sizeof(Quartet_Cartesian_Product_type));

        if (lengths)
          pointer_length_value_this_element = lengths+combination_counter;
        else
          pointer_length_value_this_element = nullptr;
        float_type_freq *freq_element = freq+combination_counter*num_cols;
        single_analysis_data sad(comb[0], comb[1], comb[2], comb[3], freq_element, pointers_to_sequences, alignment_length, pointer_length_value_this_element, dataType);
        thread_collection.add(sad);
        ++combination_counter;
      }
    }
    else
    {
      cerr << "Unexpected value fo generator_type: " << generator_type << " Exiting.\n";
      exit(-1);
    }
    
    N_freq_computed = combination_counter;
    return thread_collection;
  }
  
  
  
  
  size_t QuartetPatternCounter::next_quartet_pattern_frequencies(float_type_freq* freq, size_t N_freq, uint16_t *patterns)
  {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t N_freq_computed = 0;
    specializedThreadsCollection sTC = create_specializedThreadsCollection_4nextBatch(freq, nullptr, N_freq, N_freq_computed, patterns);

    if (N_freq_computed == 0)
    {
      ++call_count;
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      total_processing_time += elapsed.count();

      if (global_verbosity>0)
        cout << "Finished determing quartet pattern frequencies for this call to next_quartet_pattern_frequencies.\n";

      return 0;
    }

    if (global_verbosity >= 5)
      sTC.print_debug_info_thread_collection(cout);
    
    if (global_verbosity>0)
      cout << "Starting to determine quartet pattern frequencies in next_quartet_pattern_frequencies.\n";
    
    if (numThreads == 1)
      sTC.run_all_threads_non_parallel();
    else
      sTC.run_all_threads_and_join();
    
    /*
     if (global_verbosity >= 10)
     {
     cartesian_product    cp;
     vector<string>       all_patterns;
     
     if (dataType_protein)
     cp.operator()(4, "ARNDCQEGHILKMFPSTWYV", all_patterns);
     else
     cp.operator()(4, "ACGT", all_patterns);
     
     for (int i=0; i<num_combinations; ++i)
     {
     cout << "# Next Combination: " << i << "\n";
     for (int j=0; j<dataType_freq_len; ++j)
     cout << j << " " << all_patterns[j] << " " << allfreqs[i*dataType_freq_len+j] << endl;
     }
     }
     */
    
    if (global_verbosity>0)
      cout << "Finished determing quartet pattern frequencies in next_quartet_pattern_frequencies.\n";

    ++call_count;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_processing_time += elapsed.count();
    
    return N_freq_computed;
  }
  
  size_t QuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths(float_type_freq *freq, uint32_t *lengths, size_t N_freq, uint16_t *patterns)
  {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t N_freq_computed = 0;
    specializedThreadsCollection sTC = create_specializedThreadsCollection_4nextBatch(freq, lengths, N_freq, N_freq_computed, patterns);

    if (N_freq_computed == 0)
    {
      ++call_count;
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      total_processing_time += elapsed.count();

      return 0;
    }

    if (global_verbosity >= 5)
      sTC.print_debug_info_thread_collection(cout);
    
    if (global_verbosity>0)
      cout << "Starting to determine quartet pattern frequencies in next_quartet_pattern_frequencies_with_lengths.\n";
    
    if (numThreads == 1)
      sTC.run_all_threads_non_parallel();
    else
      sTC.run_all_threads_and_join();
    
    /*
     if (global_verbosity >= 10)
     {
     cartesian_product    cp;
     vector<string>       all_patterns;
     
     if (dataType_protein)
     cp.operator()(4, "ARNDCQEGHILKMFPSTWYV", all_patterns);
     else
     cp.operator()(4, "ACGT", all_patterns);
     
     for (int i=0; i<num_combinations; ++i)
     {
     cout << "# Next Combination: " << i << "\n";
     for (int j=0; j<dataType_freq_len; ++j)
     cout << j << " " << all_patterns[j] << " " << allfreqs[i*dataType_freq_len+j] << endl;
     }
     }
     */
    
    if (global_verbosity>0)
      cout << "Finished determing quartet pattern frequencies in next_quartet_pattern_frequencies_with_lengths.\n";

    ++call_count;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_processing_time += elapsed.count();
    
    return N_freq_computed;
  }
  
  // Get statistics about loaded file
  std::pair<size_t, unsigned> QuartetPatternCounter::get_dataset_characteristics() const
  {
    return make_pair(num_sequences, alignment_length);
  }
  
  size_t QuartetPatternCounter::get_num_cols() const {
    return num_cols;
  }
  
  size_t QuartetPatternCounter::get_call_count() const {
    return call_count;
  }
  
  double QuartetPatternCounter::get_total_processing_time() const {
    return total_processing_time;
  }
  

