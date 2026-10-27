#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <climits>
#include "symbol-cartesian-product_string.h"
#include "numpy.h"
#include "combinations_k_outof_n_generator.h"
#include "basic-DNA-RNA-AA-routines.h"
#include <thread>

#define PROGVERSION "1.4b"
#define PROGNAME    "quartet-pattern-counter"

// Has to be adapted for protein

using namespace std;

unsigned short global_verbosity = 1;
unsigned short number_of_threads = 1;

void Usage_and_Exit()
{
  cerr << "Usage: pattern_counter_quartett_trees [-p][-O][-t num][-V][-v num] infile.fas outfile.npy\n";
  cerr << "       -p: specify to assume protein sequences. Default: Assume nucleotide sequences.\n";
  cerr << "       -O: only print the patterns in their natural order.\n";
  cerr << "       -t: Number of threads that shall be used.\n";
  cerr << "       -V: Print version and exit.\n";
  cerr << "       -v: Verbosity (Default 1) 0: Only error messages, 1: minimal progress, 5: thread informaiton, 10: all output.\n";
  exit(-1);
}

void welcome_message(ostream &os, const char *infile, const char *outfile, bool dataType_protein, unsigned short verbosity, unsigned short num_threads)
{
  os << "Welcome to the " PROGNAME " program, version " PROGVERSION " by Christoph Mayer\n";
  os << "This analysis was started with the following paramters:\n";
  os << "Input fasta file:       " << infile << '\n';
  os << "Output npy file:        " << outfile << '\n';
  os << "DNA or Amino acid mode: " << (dataType_protein ? "AA\n" : "DNA\n");
  os << "Number of threads:      " << num_threads << '\n';
  os << "Verbosity:              " << verbosity << '\n';
}

void count_four_sequences_patterns_DNA(const char *s1, const char *s2, const char *s3, const char *s4,
                                       unsigned seqlen,
                                       float *pfreqs)
{
  unsigned char d1,d2,d3,d4;
  unsigned sites = 0;
  const int dataType_freq_len = 256;

  // TODO:
  unsigned ufreq[4][4][4][4] = {0};

  for (unsigned i=0; i<seqlen; ++i)
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
    return;

  unsigned *Pu = &ufreq[0][0][0][0];
  float    *Pf = &pfreqs[0];
  for (int i=0; i<dataType_freq_len; ++i)
  {
    *Pf = *Pu/(float)sites;
    ++Pu;
    ++Pf;
  }
}

void count_four_sequences_patterns_AA(const char *s1, const char *s2, const char *s3, const char *s4,
                                      unsigned seqlen,
                                      float *pfreqs)
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

//  auto ufreq = new unsigned[20][20][20][20]{};
  unsigned (*ufreq)[20][20][20] = new unsigned[20][20][20][20];

  memset(ufreq, 0, sizeof(unsigned[20][20][20][20]));


  for (unsigned i=0; i<seqlen; ++i)
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

  if (sites == 0)
  {
    delete[] ufreq;
    return;
  }

  unsigned *Pu = &ufreq[0][0][0][0];
  float    *Pf = &pfreqs[0];
  for (int i=0; i<dataType_freq_len; ++i)
  {
    *Pf = *Pu/(float)sites;
    ++Pu;
    ++Pf;
  }
  delete[] ufreq;
}


struct single_analysis_data
{
  unsigned short       i0;
  unsigned short       i1;
  unsigned short       i2;
  unsigned short       i3;
  float                *freqs;  // Is only allowed to access dataType_freq_len floats.
  vector<const char *> &pointers_to_sequences;
  unsigned             seqlen;
  static bool          dataType_protein;
  //  static int           dataType_freq_len; //TODO: Not used yet.

  single_analysis_data(unsigned short i0, unsigned short i1, unsigned short i2, unsigned short i3, float *freqs,
                       vector<const char *> &pointers_to_sequences, unsigned seqlen):
  i0(i0), i1(i1), i2(i2), i3(i3), freqs(freqs), pointers_to_sequences(pointers_to_sequences), seqlen(seqlen)
  {}

  void print_debug_info_single_analysis(const char *str, ostream &os)
  {
    cout << str << "(" << i0 << "," << i1 << "," << i2 << "," << i3 << ")";
  }

  void run_single_analysis()
  {
    // Has to be adapted for protein sequences:


    if (dataType_protein)
      //    if (1)
      count_four_sequences_patterns_AA(pointers_to_sequences[i0],
                                       pointers_to_sequences[i1],
                                       pointers_to_sequences[i2],
                                       pointers_to_sequences[i3],
                                       seqlen, freqs);
    else
      count_four_sequences_patterns_DNA(pointers_to_sequences[i0],
                                        pointers_to_sequences[i1],
                                        pointers_to_sequences[i2],
                                        pointers_to_sequences[i3],
                                        seqlen, freqs);
    //    print_debug_info("#DEBUG INFO FOR THIS ANALYSIS: #", cout);
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
  unsigned num_threads;
  vector<thread_data> th_data;
  unsigned short cyclic_add_index;

public:
  specializedThreadsCollection(int N):th_data(N), cyclic_add_index(0), num_threads(N)
  {}

  void add(single_analysis_data &sad)
  {
    th_data[cyclic_add_index].add(sad);
    cyclic_add_index = (cyclic_add_index+1)%num_threads;
  }

  void print_debug_info_thread_collection(ostream &os)
  {
    os << "\n### Debug information for thread collection:" << '\n';
    os << "##  Number of threads: " << num_threads << '\n';
    os << "#   Tasks assigned to the threads:\n";
    for (int i=0; i<th_data.size(); ++i)
    {
      os << "#     Thread:" << i << ":";

      th_data[i].print_debug_info_thread_data(" ", os);
      os << '\n';
    }
    if (th_data.size() == 0)
      os << "-\n";
  }

  void run_thread(int i)
  {
    //    cout << "Hi from run_thread: " << i << '\n';
    th_data[i].run_all_analyses();
  }

  void run_all_threads_non_parallel()
  {
    for (int i = 0; i < num_threads; ++i)
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
      for (int i = 0; i < num_threads; ++i)
      {
        // Pass the thread id and some parameter (for demonstration, we use i as param)
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


// This function is used simply to verify that the order of the patterns stored in the array is
// the same for the two approaches to get all patterns.

void print_coded_sequence(const char *str, int len)
{
  for (int i=0; i<len; ++i)
    cout << (int)str[i];
}


void print_pattern_order(bool dataType_protein)
{
  cartesian_product    cp;
  vector<string>       all_patterns1, all_patterns2;

  if (dataType_protein)
    cp.operator()(4, "ARNDCQEGHILKMFPSTWYV", all_patterns1);
  else
    cp.operator()(4, "ACGT", all_patterns1);

  cout << "The default order is the natural order when computing a cartesian product.\n";
  cout << "This order is also produced by the itertools Python library with the command:\n";
  cout << "itertools.product(letters, repeat=4), where letters is either \"ACGT\" or\n";
  cout << "\"ARNDCQEGHILKMFPSTWYV.\n";

  if (0)
  {
    cout << "Order of the cartesian product:\n";
    for (int i=0; i<all_patterns1.size(); ++i)
    {
      cout << all_patterns1[i] << endl;
    }
  }

  cout << "Natural order of patterns used in this software:\n";
  if (dataType_protein)
  {
    for (int i1=0; i1<20; ++i1)
      for (int i2=0; i2<20; ++i2)
        for (int i3=0; i3<20; ++i3)
          for (int i4=0; i4<20; ++i4)
          {
            string tmp;
            tmp.push_back(simpleIndex2ascii_AA[i1]);
            tmp.push_back(simpleIndex2ascii_AA[i2]);
            tmp.push_back(simpleIndex2ascii_AA[i3]);
            tmp.push_back(simpleIndex2ascii_AA[i4]);
            all_patterns2.push_back(tmp);
          }
  }
  else
  {
    for (int i1=0; i1<4; ++i1)
      for (int i2=0; i2<4; ++i2)
        for (int i3=0; i3<4; ++i3)
          for (int i4=0; i4<4; ++i4)
          {
            string tmp;
            tmp.push_back(simpleIndex2ascii_DNA[i1]);
            tmp.push_back(simpleIndex2ascii_DNA[i2]);
            tmp.push_back(simpleIndex2ascii_DNA[i3]);
            tmp.push_back(simpleIndex2ascii_DNA[i4]);
            all_patterns2.push_back(tmp);
          }
  }

  cout << "Natural order form cartesian product\n";
  for (int i=0; i<all_patterns2.size(); ++i)
  {
    cout << all_patterns2[i] << endl;
  }
}


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

void save_array_of_freq_to_file(const char *filename, int size, int freqs_len, float *pallfreqs)
{
  //  int shape[2] = {size,freqs_len};
  //  int dim = 2;
  aoba::SaveArrayAsNumpy(filename, size, freqs_len, pallfreqs);
}



void read_fasta_file_into_map_remove_spaces(const char *filename,
                                            unordered_map<string, string> &fastamap,
                                            vector<string> &namevec)
{
  ifstream is;
  is.open(filename);
  string line, lastseqstring, lastname;

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
      lastname = line.substr(1, string::npos);
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


bool single_analysis_data::dataType_protein = false;

int main(int argc, char **argv)
{
  bool  dataType_protein  = false;
  int   dataType_freq_len = 256;


  char *filename = NULL;
  char *outfilename = NULL;

  if (argc < 2 || argc > 8)
    Usage_and_Exit();

  int i,j;
  for (i=1, j=1; i<argc; ++i)
  {
    if (string(argv[i]) == string("-O") )
    {
      print_pattern_order(false);
      exit(0);
    }
    else if (string(argv[i]) == string("-p") )
    {
      dataType_protein  = true;
      single_analysis_data::dataType_protein = true;
      dataType_freq_len = 160000;
    }
    else if (string(argv[i]) == string("-V") )
    {
      cerr << "Version of " << PROGNAME << " Version " << PROGVERSION << '\n';
      exit(0);
    }
    else if (string(argv[i]) == string("-t") )
    {
      ++i;
      if (i == argc)
      {
        cerr << "The -t paramter requires an integer parameter.\n";
        Usage_and_Exit();
      }

      char* p;
      number_of_threads = (int)strtol(argv[i], &p, 10);
      if (!p)
      {
        cerr << "The -t option requires an integer parameter.\n";
        Usage_and_Exit();
      }
    }
    else if (string(argv[i]) == string("-v") )
    {
      ++i;
      if (i == argc)
      {
        cerr << "The -v paramter requires an integer parameter .\n";
        Usage_and_Exit();
      }

      char* p;
      global_verbosity = (int)strtol(argv[i], &p, 10);
      if (!p)
      {
        cerr << "The -t option requires an integer parameter.\n";
        Usage_and_Exit();
      }
    }
    else if (j==1)
    {
      filename = argv[i];
      ++j;
    }
    else if (j == 2)
    {
      outfilename = argv[i];
      ++j;
    }
    else
    {
      cerr << "Unknown or additional unnamed parameter: " << argv[i] << "\n";
      Usage_and_Exit();
    }
  }

  if (global_verbosity > 0)
    welcome_message(cout, filename, outfilename, dataType_protein, global_verbosity, number_of_threads);

  unordered_map<string, string> fasta_map_string;
  vector<const char *>      pointers_to_sequences;
  //  unordered_map<string, unsigned> pattern_counter;
  //  vector<string> all_patterns;
  vector<string> seq_names;

  // Read sequence information and store in data structures

  if (global_verbosity>0)
    cout << "\nReading input data.\n";

  read_fasta_file_into_map_remove_spaces(filename, fasta_map_string, seq_names);

  if (seq_names.empty())
  {
    cerr << "No sequnces read from input file. Exiting.\n";
    exit(0);
  }

  unsigned       seqlen = 0;
  unsigned short k      = 4;
  unsigned short n      = seq_names.size();

  seqlen = (unsigned) fasta_map_string[seq_names[0]].length();

  if (dataType_protein)
  {
    for (unsigned i=0; i<n; ++i)
    {
      //       cout << ">>" << seq_names[i] << '\n'
      //            << fasta_map_string[seq_names[i]] << '\n';
      string &tmp = fasta_map_string[seq_names[i]];
      simple_recode_string_AA(tmp);
      pointers_to_sequences.push_back(tmp.c_str());
    }
  }
  else
  {
    for (unsigned i=0; i<n; ++i)
    {
      //       cout << ">>" << seq_names[i] << '\n'
      //            << fasta_map_string[seq_names[i]] << '\n';
      string &tmp = fasta_map_string[seq_names[i]];
      simple_recode_string_DNA(tmp);
      pointers_to_sequences.push_back(tmp.c_str());
    }
  }
  // DONE: Read sequence information and store in data structures

  combination_generator gen = combination_generator(k, n);
  vector<unsigned short> comb;

  long long num_combinations = binomialCoefficient(k, n);
  if (num_combinations > INT_MAX)
  {
    cout << "Too many combinations. Rewrite program to make this possible.\n";
  }


  float * allfreqs = new float[num_combinations*dataType_freq_len];  // GNU extension

  if (global_verbosity>0)
    cout << "Assigning quartets to threads.\n";

  specializedThreadsCollection thread_collection(number_of_threads);

  unsigned combination_counter = 0;
  while (!(comb = gen.next()).empty())
  {
    unsigned short i0 = comb[0]-1;
    unsigned short i1 = comb[1]-1;
    unsigned short i2 = comb[2]-1;
    unsigned short i3 = comb[3]-1;

    if (global_verbosity >= 5)
    {
      // The following does not make sense when using treads:
      //      cout << "Working on combination: " << "(" << combination_counter+1 << "/" << num_combinations << ") "
      //      << i0 << " " << i1 << " " << i2 << " " << i3 << "\n";

      if (global_verbosity >= 10)
      {
        cout << "Sequences:\n";
        print_coded_sequence(pointers_to_sequences[i0], seqlen); cout << '\n';
        print_coded_sequence(pointers_to_sequences[i1], seqlen); cout << '\n';
        print_coded_sequence(pointers_to_sequences[i2], seqlen); cout << '\n';
        print_coded_sequence(pointers_to_sequences[i3], seqlen); cout << '\n';
      }
    }

    single_analysis_data sad(i0, i1, i2, i3, &allfreqs[combination_counter*dataType_freq_len], pointers_to_sequences, seqlen);

    thread_collection.add(sad);

    /*
     count_four_sequences_patterns_DNA(pointers_to_sequences[i0],
     pointers_to_sequences[i1],
     pointers_to_sequences[i2],
     pointers_to_sequences[i3],
     seqlen, allfreqs[combination_counter]);
     */

    ++combination_counter;
  }

  if (global_verbosity >= 5)
    thread_collection.print_debug_info_thread_collection(cout);

  if (global_verbosity>0)
    cout << "Started determing quartet pattern frequencies.\n";

  if (number_of_threads == 1)
    thread_collection.run_all_threads_non_parallel();
  else
    thread_collection.run_all_threads_and_join();

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

  save_array_of_freq_to_file(outfilename, combination_counter, dataType_freq_len, allfreqs);

  if (global_verbosity>0)
    cout << "Finished determing quartet pattern frequencies.\n";

  //  get_pattern_map(four_sequences, all_patterns, len, pattern_counter, dataType_protein);
  //  write_pattern_fequencies_to_file(all_patterns, pattern_counter, len, outfilename, dataType_protein);

  /*
   if (stringType == 1)
   {
   map<faststring, faststring> fasta_map_string;
   map<string, unsigned> pattern_counter;
   vector<faststring> seq_names;
   unsigned len = 0;

   read_fasta_file_into_map(filename, fasta_map_string, seq_names);

   for (unsigned i=0; i<seq_names.size(); ++i)
   {
   //       cout << ">>" << seq_names[i] << '\n' << fasta_map_string[seq_names[i]] << '\n';
   faststring &tmp = fasta_map_string[seq_names[i]];
   if (i==0)
   len = tmp.length();
   four_sequences[i] = tmp.c_str();
   }
   }

   if (stringType == 2)
   {
   map<faststring, faststring> fasta_map_string;
   map<string, unsigned> pattern_counter;
   vector<faststring> seq_names;

   CSequences2 *seqs;
   read_fasta_file_into_map(filename, &seqs);

   for (unsigned i=0; i<seqs->GetTaxaNum(); ++i)
   {
   //       cout << ">>" << seq_names[i] << '\n' << fasta_map_string[seq_names[i]] << '\n';
   seqs->get_Seq_Data(i);
   }
   }
   */

  exit(0);


}
