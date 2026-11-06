// Copyright 2026, Dr. Christoph Mayer, Nikita Kulikov, Dr. Iker Irisarri LIB, Bonn/Hamburg, Germany
// License: see License.txt

// quartet-pattern-counter v1.7.1, stand-alone program
// Command line front end to the counting core used by the Python module. It has the command
// line and output of quartet-pattern-counter v1.1 and can replace it.
//
// Usage: quartet-pattern-counter [-p] infile.fas outfile.npy
//
// The input file must contain exactly four aligned sequences. The output is a float32 numpy
// vector with the relative frequencies of all 256 (DNA) or 160000 (protein) site patterns.
// Patterns are ordered lexicographically by their symbols (A<C<G<T, amino acids alphabetically),
// as in v1.1. Sites with a gap, ambiguity code or other invalid symbol in any of the four
// sequences are skipped. Lower case symbols are counted as upper case. Frequencies are relative
// to the number of counted sites. On such input this differs from v1.1, which mixed invalid
// patterns into the output vector.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "QuartetPattern-counter-core.hpp"

// Defined in QuartetPattern-counter-core.cpp, not exported in the header.
extern char order_of_AA_site_patterns;
int32_t count_four_sequences_patterns_DNA(const char *s1, const char *s2, const char *s3, const char *s4,
                                          unsigned alignment_length, float_type_freq *pfreqs);
int32_t count_four_sequences_patterns_AA(const char *s1, const char *s2, const char *s3, const char *s4,
                                         unsigned alignment_length, float_type_freq *pfreqs);
void read_fasta_file(const std::string &filename,
                     std::unordered_map<std::string, std::string> &fastamap,
                     std::vector<std::string> &namevec,
                     std::vector<const char *> &pointers_to_sequences,
                     char dataType);

// Writes a 1-dimensional float32 numpy array (.npy format version 1.0).
bool save_float_vector_as_numpy(const char *filename, const std::vector<float> &v)
{
  static_assert(sizeof(float) == 4, "float must be 32 bit");
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(v.size()) + ",), }";
  // Magic (6) + version (2) + header length (2) + header + '\n' must be a multiple of 64 bytes.
  size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header.push_back('\n');

  FILE *fp = fopen(filename, "wb");
  if (!fp)
    return false;
  const char magic[] = "\x93NUMPY\x01\x00";
  uint16_t hlen = header.size();
  unsigned char hlen_le[2] = {(unsigned char)(hlen & 0xff), (unsigned char)(hlen >> 8)};
  bool ok = fwrite(magic, 1, 8, fp) == 8
         && fwrite(hlen_le, 1, 2, fp) == 2
         && fwrite(header.data(), 1, header.size(), fp) == header.size()
         && fwrite(v.data(), sizeof(float), v.size(), fp) == v.size(); // Assumes a little-endian host.
  return fclose(fp) == 0 && ok;
}

int main(int argc, char **argv)
{
  bool dataType_protein = false;
  const char *filename = nullptr;
  const char *outfilename = nullptr;

  for (int i=1; i<argc; ++i)
  {
    if (std::string(argv[i]) == "-p")
      dataType_protein = true;
    else if (!filename)
      filename = argv[i];
    else if (!outfilename)
      outfilename = argv[i];
    else
    {
      filename = nullptr;
      i = argc; // Too many arguments.
    }
  }
  if (!filename || !outfilename)
  {
    std::cerr << "Usage: quartet-pattern-counter [-p] infile.fas outfile.npy\n";
    exit(-1);
  }

  order_of_AA_site_patterns = 'a'; // Alphabetical, the order of v1.1.
  std::unordered_map<std::string, std::string> fastamap;
  std::vector<std::string> names;
  std::vector<const char *> seqs;
  try
  {
    // Checks for duplicate names and equal lengths.
    read_fasta_file(filename, fastamap, names, seqs, dataType_protein ? 'P' : 'D');
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << '\n';
    exit(-1);
  }

  if (names.size() != 4)
  {
    std::cerr << "Error: The input file " << filename << " must contain exactly four sequences. Found: "
              << names.size() << " sequences.\n";
    exit(-1);
  }
  size_t len = fastamap[names[0]].length();

  std::vector<float_type_freq> freqs(dataType_protein ? 160000 : 256);
  int32_t sites = dataType_protein
    ? count_four_sequences_patterns_AA(seqs[0], seqs[1], seqs[2], seqs[3], len, freqs.data())
    : count_four_sequences_patterns_DNA(seqs[0], seqs[1], seqs[2], seqs[3], len, freqs.data());
  if (sites == 0)
  {
    std::cerr << "Error: The input file " << filename << " contains no site without gaps or ambiguous symbols.\n";
    exit(-1);
  }
  if (!save_float_vector_as_numpy(outfilename, freqs))
  {
    std::cerr << "Error: Could not write the output file " << outfilename << ".\n";
    exit(-1);
  }
  return 0;
}
