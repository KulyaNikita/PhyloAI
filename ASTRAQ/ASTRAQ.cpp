// ASTRAQ.cpp
//
// A from-scratch, original implementation of ASTRAL's core optimization
// problem (Mirarab et al. 2014 Bioinformatics; Mirarab & Warnow 2015;
// Zhang et al. 2018 BMC Bioinformatics "ASTRAL-III"), adapted for input
// that already consists of (possibly weighted) quartet trees, instead of
// full multi-leaf gene trees.
//
// ============================== THE MATH ===================================
// ASTRAL seeks the species tree T whose induced quartets are shared as much
// as possible with the input, subject to: every bipartition of T must come
// from a constrained candidate set X (this is what makes it polynomial-time
// instead of NP-hard). The recursion is:
//
//   DP(A) = max over bipartitions A = A1 (sqcup) A2, with A1,A2 in X, of
//             DP(A1) + DP(A2) + score(A1, A2, N \ A)
//
//   score(A1, A2, A3) = sum over input quartets "a,b | x,y" with
//                        a in A1, b in A2, x in A3, y in A3, of that
//                        quartet's weight
//
// score(A1,A2,A3) is exactly the number (or weight) of quartets that the
// tree edge separating A=(A1|A2) from A3 will display, no matter how A1, A2,
// A3 are further resolved internally -- that's the key fact that makes the
// recursion correct and is the same fact ASTRAL itself relies on.
//
// ============================ WHAT CHANGES WITHOUT GENE TREES ===============
// Full gene trees give ASTRAL two things this code must do differently:
//
//   1. A constrained search space X. ASTRAL builds X largely from the
//      bipartitions that literally appear in input gene trees (plus a
//      heuristic completion: greedy consensus, similarity/NJ-based extra
//      trees, etc.). You don't have gene trees, so this code instead builds
//      X from your quartets directly: it derives a quartet-based distance
//      matrix and runs Neighbor-Joining (optionally several randomized
//      times) to generate candidate trees, and unions their bipartitions
//      into X. You can also feed in any guide/constraint trees you do have
//      via -g to enrich X (this is the analogue of ASTRAL's -e/-f options).
//      THIS is the step where "-p" would have mattered for ASTRAL (it
//      bounds a polytomy-resolution heuristic in gene-tree-based completion)
//      -- it has no equivalent here because there are no polytomies or gene
//      trees to complete from.
//
//   2. Efficient quartet counting without materializing C(n,4) quartets per
//      gene tree (ASTRAL-II's O(n) traversal trick). You don't need this:
//      your quartets ARE your explicit input, so score() just scans your
//      quartet list directly. This is simpler and exact, but means runtime
//      scales with (size of X)^2 * (number of input quartets) -- fine for
//      small/moderate inputs; for very large quartet sets you'd want to
//      index quartets by taxon-pair to avoid full scans (noted at bottom).
//
// ============================== INPUT FORMAT ================================
// Quartets file (one per line), EITHER:
//   (a) Newick quartet trees, e.g.   (A,B,(C,D));            -> weight 1
//                                    (A,B,(C,D));   12.5     -> weight 12.5
//       Any resolved 4-leaf rooted-or-unrooted Newick is accepted; the code
//       finds the size-2 clade (the cherry) to identify the topology.
//   (b) Plain format:  A B C D weight     -> means topology AB|CD, weight
//       weight may be omitted (defaults to 1).
// Lines starting with # are comments; blank lines ignored.
//
// Optional guide trees file (-g), one Newick tree per line (any number of
// leaves), used only to add extra candidate bipartitions to X.
//
// ============================== USAGE ========================================
//   g++ -O2 -std=c++17 -o ASTRAQ ASTRAQ.cpp
//   ./ASTRAQ -q quartets.txt [-g guide_trees.txt] [-r 8] [-o out.tre]
//
//   -r N   number of extra randomized NJ restarts used to enrich the search
//          space (default 8). More restarts = larger X = closer to the true
//          ASTRAL search space, at the cost of runtime.
// =============================================================================

#include <iostream>     // std::cerr, std::cout
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>      // std::istringstream
#include <string>       // std::string, std::to_string, std::getline
#include <vector>       // std::vector
#include <array>        // std::array
#include <utility>      // std::pair, std::swap, std::move
#include <map>          // std::map
#include <set>          // std::set
#include <unordered_map>// std::unordered_map
#include <algorithm>    // std::sort, std::unique, std::min, std::max, std::shuffle
#include <numeric>      // std::iota
#include <functional>   // std::function
#include <limits>       // std::numeric_limits
#include <random>       // std::mt19937, std::uniform_real_distribution
#include <cmath>        // std::llround, std::fabs
#include <cctype>       // std::isspace, std::isalnum, std::isdigit
#include <cstdint>      // std::uint64_t
#include <cstdio>       // std::snprintf
#ifdef _OPENMP
#include <omp.h>        // parallel DP (optional; compile with -fopenmp)
#endif
#include "CLI11.hpp"
using namespace std;

#define PROG_NAME "ASTRAQ"
#define PROG_VERSION "0.5.8"

void displayWelcomeMessage() {
  std::cout << "Welcome to " PROG_NAME ", version " PROG_VERSION "\n\n";
}


// ---------------------------------------------------------------- BitSet ---
struct BitSet
{
  int n = 0;
  vector<uint64_t> w;
  BitSet() {}
  explicit BitSet(int n_) : n(n_), w((n_ + 63) / 64, 0) {}
  void set(int i) { w[i >> 6] |= (1ULL << (i & 63)); }
  bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
  int popcount() const
  {
    int c = 0;
    for (auto x : w) c += __builtin_popcountll(x);
    return c;
  }
  bool any() const
  {
    for (auto x : w) if (x) return true;
    return false;
  }
  bool isSubsetOf(const BitSet& o) const
  {
    for (size_t i = 0; i < w.size(); ++i)
      if ((w[i] & ~o.w[i]) != 0) return false;
    return true;
  }
  BitSet operator&(const BitSet& o) const { BitSet r(n); for (size_t i=0;i<w.size();++i) r.w[i]=w[i]&o.w[i]; return r; }
  BitSet operator|(const BitSet& o) const { BitSet r(n); for (size_t i=0;i<w.size();++i) r.w[i]=w[i]|o.w[i]; return r; }
  BitSet operator^(const BitSet& o) const { BitSet r(n); for (size_t i=0;i<w.size();++i) r.w[i]=w[i]^o.w[i]; return r; }
  bool operator==(const BitSet& o) const { return w == o.w; }

  // Not used here:
  void reset(int i)
  {
    w[i >> 6] &= ~(1ULL << (i & 63));
  }

  void flip(int i)
  {
    w[i >> 6] ^= (1ULL << (i & 63));
  }

  void flip()
  {
    for (auto& x : w)
      x = ~x;
    mask_unused();
  }

  void mask_unused()
  {
    int rem = n & 63; // n % 64
    if (rem != 0)
      w.back() &= (1ULL << rem) - 1;
  }

  BitSet operator~() const
  {
    BitSet r(*this);

    for (auto& x : r.w)
      x = ~x;
    r.mask_unused();
    return r;
  }

  void setAll()
  {
    for (auto& x : w)
      x = ~0ULL;
    mask_unused();
  }

  void resetAll()
  {
    for (auto& x : w)
      x = 0;
  }




};

struct BitSetHash
{
  size_t operator()(const BitSet& b) const
  {
    size_t h = 1469598103934665603ULL;
    for (auto x : b.w) { h ^= x; h *= 1099511628211ULL; }
    return h;
  }
};

// Total order on BitSets, used only to pick a canonical representative of the
// unordered pair {B, C} so co-optimal splits are counted once, not twice.
static bool lexSmaller(const BitSet& a, const BitSet& b)
{
  for (size_t i = 0; i < a.w.size(); ++i)
  {
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
  }
  return false;
}

// --------------------------------------------------------------- Quartet ---
struct Quartet
{
  int a, b, x, y; // topology: a,b paired together  vs  x,y paired together
  double weight;
};

// ----------------------------------------------------------- Newick read ---
// Minimal recursive-descent Newick parser. Returns the leafset of the whole
// tree and appends the leafset of every internal clade (incl. the root) to
// `clades`. Does not support quoted taxon names or NHX comments.
struct NewickParser
{
  const string& s;
  size_t pos = 0;
  unordered_map<string, int>& nameToId;
  int n;
  NewickParser(const string& str, unordered_map<string, int>& m, int n_)
  : s(str), nameToId(m), n(n_) {}

  void skipWs() { while (pos < s.size() && isspace((unsigned char)s[pos])) pos++; }

  void skipLabelAndLength()
  {
    skipWs();
    // optional internal-node label (rare) then optional :branchlength
    while (pos < s.size() && s[pos] != ',' && s[pos] != ')' &&
           s[pos] != ';' && s[pos] != ':' && !isspace((unsigned char)s[pos]))
      pos++;
    skipWs();
    if (pos < s.size() && s[pos] == ':') {
      pos++;
      while (pos < s.size() &&
             (isdigit((unsigned char)s[pos]) || s[pos]=='.' || s[pos]=='-' ||
              s[pos]=='e' || s[pos]=='E' || s[pos]=='+'))
        pos++;
    }
  }

  BitSet parseNode(vector<BitSet>& clades)
  {
    skipWs();
    if (pos < s.size() && s[pos] == '(')
    {
      pos++;
      BitSet leafset(n);
      while (true)
      {
        BitSet child = parseNode(clades);
        leafset = leafset | child;
        skipWs();
        if (pos < s.size() && s[pos] == ',') { pos++; continue; }
        break;
      }
      skipWs();
      if (pos < s.size() && s[pos] == ')') pos++;
      skipLabelAndLength();
      clades.push_back(leafset);
      return leafset;
    }
    else
    {
      size_t start = pos;
      while (pos < s.size() && s[pos] != ',' && s[pos] != ')' &&
             s[pos] != ':' && s[pos] != ';' && !isspace((unsigned char)s[pos]))
        pos++;
      string name = s.substr(start, pos - start);
      skipLabelAndLength();
      BitSet leafset(n);
      auto it = nameToId.find(name);
      if (it != nameToId.end()) leafset.set(it->second);
      return leafset;
    }
  }
};

static vector<BitSet> newickToClades(const string& newick, unordered_map<string,int>& nameToId, int n)
{
  vector<BitSet> clades;
  NewickParser p(newick, nameToId, n);
  BitSet root = p.parseNode(clades);
  return clades; // includes the root clade (== full leafset of this tree)
}

// trim helper
static string trim(const string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// -------------------------------------------------------- global taxon set
unordered_map<string,int> nameToId;
vector<string> idToName;
int N = 0;

// Two-phase taxon registration:
//   Phase 1 (collectName): record every name seen, no ID yet.
//   Phase 2 (finaliseNames): sort names alphabetically, assign IDs in that
//            order, then call getId() normally for the rest of the program.
// This makes ID assignment completely independent of the order in which taxa
// appear in the input file, so two files with identical quartet topologies
// but different line orderings always produce the same distance matrix, the
// same NJst trees, and the same random-subset draws for a given seed.
set<string> collectedNames;

void collectName(const string& name) { collectedNames.insert(name); }

void finaliseNames() {
  // collectedNames is already sorted (std::set is ordered)
  for (auto& name : collectedNames) {
    nameToId[name] = N++;
    idToName.push_back(name);
  }
}

int getId(const string& name) {
  auto it = nameToId.find(name);
  if (it != nameToId.end()) return it->second;
  // Should not happen after finaliseNames() — but handle gracefully
  int id = N++;
  nameToId[name] = id;
  idToName.push_back(name);
  return id;
}

// ------------------------------------------------------------ read quartets
//
// Canonical form: always store the quartet so that the SMALLER taxon id of
// the two pairs is in the (a,b) slot.  Concretely: min(a,b) < min(x,y).
// This makes the representation independent of:
//   - input file order
//   - whether the file uses rooted or unrooted Newick
//   - which cherry the Newick parser happens to find first
// Without this, two files containing identical unrooted quartet topologies
// but written in different formats produce different NJst distance matrices
// (because buildDistanceMatrix only counts agree[a][b] and agree[x][y],
// not the symmetric counterparts) and therefore different output trees.
static Quartet canonicalise(int a, int b, int x, int y, double w) {
  // ensure min(a,b) <= min(x,y); swap pairs if needed
  if (min(a,b) > min(x,y)) { swap(a,x); swap(b,y); }
  // within each pair, put the smaller id first (cosmetic, not required)
  if (a > b) swap(a,b);
  if (x > y) swap(x,y);
  return {a, b, x, y, w};
}

vector<Quartet> readQuartets(const string& path) {
  ifstream f(path);
  if (!f) { cerr << "Cannot open quartets file: " << path << "\n"; exit(1); }
  vector<Quartet> out;
  string line;
  // first pass: collect all taxon names without assigning IDs yet,
  // so that IDs can be assigned alphabetically in finaliseNames().
  vector<string> rawLines;
  while (getline(f, line)) {
    string t = trim(line);
    if (t.empty() || t[0] == '#') continue;
    rawLines.push_back(t);
    if (t[0] != '(') {
      istringstream iss(t);
      string a,b,x,y; iss >> a >> b >> x >> y;
      collectName(a); collectName(b); collectName(x); collectName(y);
    } else {
      size_t semi = t.find(';');
      string treepart = (semi == string::npos) ? t : t.substr(0, semi+1);
      size_t i = 0;
      while (i < treepart.size()) {
        if (isalnum((unsigned char)treepart[i]) || treepart[i]=='_' || treepart[i]=='.' || treepart[i]=='-') {
          size_t j = i;
          while (j < treepart.size() && (isalnum((unsigned char)treepart[j])||treepart[j]=='_'||treepart[j]=='.'||treepart[j]=='-')) j++;
          collectName(treepart.substr(i, j-i));
          i = j;
        } else i++;
      }
    }
  }

  // assign IDs alphabetically -- deterministic regardless of file order
  finaliseNames();
  // second pass: actually parse each line into a Quartet using stable ids
  for (auto& t : rawLines) {
    if (t[0] != '(') {
      istringstream iss(t);
      string a,b,x,y; double w = 1.0;
      iss >> a >> b >> x >> y;
      if (!(iss >> w)) w = 1.0;
      out.push_back(canonicalise(getId(a), getId(b), getId(x), getId(y), w));
    } else {
      size_t semi = t.find(';');
      string treepart = (semi==string::npos) ? t : t.substr(0, semi+1);
      string rest = (semi==string::npos) ? "" : t.substr(semi+1);
      double w = 1.0;
      { istringstream iss(rest); if (!(iss >> w)) w = 1.0; }
      vector<BitSet> clades = newickToClades(treepart, nameToId, N);
      BitSet pair2(N); bool found = false;
      for (auto& c : clades) {
        if (c.popcount() == 2) { pair2 = c; found = true; break; }
      }
      if (!found) {
        cerr << "Warning: unresolved/unsupported quartet line skipped: " << t << "\n";
        continue;
      }
      // full leafset of this quartet tree is the largest clade found
      BitSet full(N);
      for (auto& c : clades) if (c.popcount() > full.popcount()) full = c;
      BitSet other2 = full ^ pair2;
      int p[2], o[2], pi=0, oi=0;
      for (int i=0;i<N;i++) {
        if (pair2.test(i)) p[pi++]=i;
        else if (other2.test(i)) o[oi++]=i;
      }
      if (pi==2 && oi==2)
        out.push_back(canonicalise(p[0], p[1], o[0], o[1], w));
      else
        cerr << "Warning: could not extract a clean quartet from: " << t << "\n";
    }
  }
  return out;
}

// ------------------------------------------------------- guide tree input
void addGuideTreeBipartitions(const string& path, vector<BitSet>& X) {
  ifstream f(path);
  if (!f) { cerr << "Cannot open guide trees file: " << path << "\n"; exit(1); }
  string line;
  while (getline(f, line)) {
    string t = trim(line);
    if (t.empty() || t[0]=='#') continue;
    // register any unseen taxa first (rare for guide trees, but be safe)
    size_t i=0;
    while (i<t.size()) {
      if (isalnum((unsigned char)t[i])||t[i]=='_'||t[i]=='.'||t[i]=='-') {
        size_t j=i; while (j<t.size() && (isalnum((unsigned char)t[j])||t[j]=='_'||t[j]=='.'||t[j]=='-')) j++;
        getId(t.substr(i,j-i)); i=j;
      } else i++;
    }
    vector<BitSet> clades = newickToClades(t, nameToId, N);
    for (auto& c : clades) {
      int pc = c.popcount();
      if (pc >= 2 && pc <= N-2) X.push_back(c);
    }
  }
}

// ------------------------------------------------------- distance + NJ -----
// Build a quartet-topology-based distance: fraction of (weighted) quartets
// containing both i and j in which i,j are NOT placed together. This is a
// simple heuristic distance (not the formal statistically-consistent
// estimator some quartet-NJ methods use) -- its only job here is to produce
// reasonable candidate trees to seed/enrich the search space X, not to be
// the final answer itself (the DP step is what actually optimizes).
vector<vector<double>> buildDistanceMatrix(const vector<Quartet>& qs, int n) {
  vector<vector<double>> agree(n, vector<double>(n,0.0)), total(n, vector<double>(n,0.0));
  for (auto& q : qs) {
    int t[4] = {q.a,q.b,q.x,q.y};
    for (int i=0;i<4;i++) for (int j=i+1;j<4;j++) {
      int u=t[i], v=t[j];
      total[u][v]+=q.weight; total[v][u]+=q.weight;
      bool together = (u==q.a||u==q.b) ? (v==q.a||v==q.b) : (v==q.x||v==q.y);
      if (together) { agree[u][v]+=q.weight; agree[v][u]+=q.weight; }
    }
  }
  vector<vector<double>> d(n, vector<double>(n,0.0));
  for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
    if (i==j) { d[i][j]=0; continue; }
    if (total[i][j] > 0) d[i][j] = 1.0 - agree[i][j]/total[i][j];
    else d[i][j] = 0.5; // no direct evidence; neutral default
  }
  return d;
}

// --------------------------------------------------------- jitter policy ---
// Determine the jitter amplitude used to perturb the distance matrix before
// each NJst restart. Isolated here so the policy can be changed in one place.
//
// Two modes:
//   * absolute (jitterAbs >= 0): use exactly that amplitude (the legacy
//     fixed +/-0.05 behaviour). Chosen when the user passes --jitter-abs.
//   * adaptive (jitterAbs < 0): amplitude = alpha * IQR(off-diagonal
//     distances). The inter-quartile range is a ROBUST measure of the spread
//     of the bulk of distances: unlike the standard deviation it is not
//     inflated by the extreme distances (0 and 1) that dominate clean data,
//     and unlike the median inter-value gap it does not collapse to ~0 when
//     conflicting quartets create many near-tied distances. Empirically it
//     keeps the fraction of NJ merge-orders that get reordered roughly stable
//     (~90% clean, ~60-90% conflicted) instead of over-perturbing clean data
//     or under-perturbing conflicted data.
//
// alpha ~ 0.3 is a reasonable default (see --jitter-alpha).
struct JitterParams {
  double alpha;      // multiplier for the adaptive (IQR) mode
  double jitterAbs;  // >=0 forces this absolute amplitude; <0 => adaptive
};

double computeJitterAmplitude(const vector<vector<double>>& D, int n,
                              const JitterParams& jp) {
  if (jp.jitterAbs >= 0.0) return jp.jitterAbs;   // fixed / legacy mode
  // adaptive: alpha * IQR of the off-diagonal distances
  vector<double> v;
  v.reserve((size_t)n * (n - 1) / 2);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      v.push_back(D[i][j]);
  if (v.empty()) return 0.0;
  sort(v.begin(), v.end());
  auto pct = [&](double p) {
    size_t idx = (size_t)(p * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
  };
  double iqr = pct(0.75) - pct(0.25);
  return jp.alpha * iqr;
}

// Write the distance matrix to a file (or stderr if path == "-"), for the
// user to inspect the magnitude of distances and the effect of jitter.
void dumpDistanceMatrix(const vector<vector<double>>& D, int n,
                        const vector<string>& names, const string& path,
                        double jitterAmp) {
  ostream* os;
  ofstream fout;
  if (path == "-") { os = &cerr; }
  else { fout.open(path); os = &fout; }
  (*os) << "# distance matrix (" << n << " taxa); jitter amplitude in use: +/-"
  << jitterAmp << "\n#     ";
  for (int j = 0; j < n; j++) (*os) << "\t" << names[j];
  (*os) << "\n";
  for (int i = 0; i < n; i++) {
    (*os) << names[i];
    for (int j = 0; j < n; j++) {
      char buf[32];
      snprintf(buf, sizeof(buf), "\t%.4f", D[i][j]);
      (*os) << buf;
    }
    (*os) << "\n";
  }
  if (path != "-") cerr << "Wrote distance matrix to " << path << "\n";
}

// Neighbor-Joining that returns the set of clades (bipartitions) formed
// along the way, rather than an explicit tree object -- that's all we need
// to populate the search space X.
void neighborJoiningClades(vector<vector<double>> D, int n, mt19937& rng, double jitter, vector<BitSet>& outClades) {
  if (jitter > 0) {
    for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) {
      double noise = uniform_real_distribution<double>(-jitter, jitter)(rng);
      D[i][j] += noise; D[j][i] = D[i][j];
    }
  }
  vector<int> active;
  vector<BitSet> leafsetOf;
  for (int i=0;i<n;i++) { active.push_back(i); BitSet b(n); b.set(i); leafsetOf.push_back(b); }
  vector<vector<double>> d = D; // indexed by position in `active`-style growing array; we keep a map id->row
  // Use a map from "node id" to row/col in a dynamic distance table.
  unordered_map<int,int> idToRow;
  vector<vector<double>> table = D;
  vector<int> rowToId(n);
  for (int i=0;i<n;i++) { idToRow[i]=i; rowToId[i]=i; }
  int nextId = n;
  vector<bool> alive(n, true);
  int numActive = n;

  while (numActive > 2) {
    vector<int> liveIds;
    for (auto& kv : idToRow) if (alive[ (kv.second<(int)alive.size())?kv.second:0 ]) {} // unused, kept simple below
    // gather live node ids
    liveIds.clear();
    for (auto& kv : idToRow) liveIds.push_back(kv.first);
    // compute r_i
    unordered_map<int,double> r;
    for (int idI : liveIds) {
      double s=0;
      for (int idJ : liveIds) if (idJ!=idI) s += table[idToRow[idI]][idToRow[idJ]];
      r[idI]=s;
    }
    int m = (int)liveIds.size();
    double bestQ = numeric_limits<double>::infinity();
    int bi=-1, bj=-1;
    for (size_t a=0;a<liveIds.size();++a) for (size_t b=a+1;b<liveIds.size();++b) {
      int idI=liveIds[a], idJ=liveIds[b];
      double q = (m-2)*table[idToRow[idI]][idToRow[idJ]] - r[idI] - r[idJ];
      if (q < bestQ) { bestQ=q; bi=idI; bj=idJ; }
    }
    // merge bi,bj into new node
    int newId = nextId++;
    BitSet merged = leafsetOf[bi] | leafsetOf[bj];
    leafsetOf.push_back(merged);
    if (merged.popcount()>=2 && merged.popcount()<=n-2) outClades.push_back(merged);
    // extend table with a new row/col for newId
    int newRow = (int)table.size();
    for (auto& row : table) row.push_back(0.0);
    table.push_back(vector<double>(newRow+1, 0.0));
    for (int idK : liveIds) {
      if (idK==bi || idK==bj) continue;
      double dik = table[idToRow[bi]][idToRow[idK]];
      double djk = table[idToRow[bj]][idToRow[idK]];
      double dij = table[idToRow[bi]][idToRow[bj]];
      double newd = 0.5*(dik+djk-dij);
      if (newd < 0) newd = 0;
      table[newRow][idToRow[idK]] = newd;
      table[idToRow[idK]][newRow] = newd;
    }
    idToRow.erase(bi); idToRow.erase(bj);
    idToRow[newId] = newRow;
    numActive--;
  }
  // final two active nodes merge implicitly into the full set; no need to
  // record it since the full taxon set is always added to X separately.
}

// -------------------------------------------------------- random-subset NJ --
// ASTRAL-III enrichment trick (Zhang et al. 2018 §"Search space"):
// Draw a random subset S of taxa, run NJ on the D[S×S] submatrix, and lift
// the resulting bipartitions back to the full N-taxon BitSet space. This is
// qualitatively different from jittering the full-taxon NJ: jitter produces
// slightly different trees on the same n taxa (useful for tie-breaking), while
// the subset trick can reveal bipartitions that are *impossible* to appear in
// any full-taxon NJ tree because competing signal from the excluded taxa
// suppresses them. Running multiple rounds with different random subsets
// (ASTRAL-III does 3) therefore adds genuinely new candidates to X.
//
// Subset size: we use ceil(N/2), matching ASTRAL-III's heuristic. Sizes
// smaller than 4 are skipped since NJ needs at least 4 taxa to produce a
// non-trivial bipartition.
void subsetNJClades(const vector<vector<double>>& D, int N, mt19937& rng,
                    int subsetSize, vector<BitSet>& outClades) {
  if (subsetSize < 4) return;
  subsetSize = min(subsetSize, N);

  // draw subsetSize distinct taxa uniformly at random
  vector<int> perm(N);
  iota(perm.begin(), perm.end(), 0);
  shuffle(perm.begin(), perm.end(), rng);
  vector<int> S(perm.begin(), perm.begin() + subsetSize);
  // S[i] is the global taxon id of the i-th taxon in the subset

  // build submatrix (indexed 0..subsetSize-1)
  int m = subsetSize;
  vector<vector<double>> sub(m, vector<double>(m, 0.0));
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      sub[i][j] = D[S[i]][S[j]];

  // run NJ on the submatrix; track leafsets using GLOBAL taxon ids so the
  // resulting BitSets are immediately valid entries for the full-N X.
  unordered_map<int,int> idToRow;
  vector<vector<double>> table = sub;
  // leafsetOf[nodeId] = full-N BitSet of leaves under that node
  unordered_map<int, BitSet> leafsetOf;
  for (int i = 0; i < m; i++) {
    idToRow[i] = i;
    BitSet b(N); b.set(S[i]); // global taxon id
    leafsetOf[i] = b;
  }
  int nextId = m, numActive = m;

  while (numActive > 2) {
    vector<int> live;
    for (auto& kv : idToRow) live.push_back(kv.first);

    unordered_map<int,double> r;
    for (int u : live) {
      double s = 0;
      for (int v : live) if (v != u) s += table[idToRow[u]][idToRow[v]];
      r[u] = s;
    }

    int sz = (int)live.size();
    double bestQ = numeric_limits<double>::infinity();
    int bi = -1, bj = -1;
    for (int a = 0; a < sz; a++) for (int b = a+1; b < sz; b++) {
      int u = live[a], v = live[b];
      double q = (sz-2) * table[idToRow[u]][idToRow[v]] - r[u] - r[v];
      if (q < bestQ) { bestQ = q; bi = u; bj = v; }
    }

    int newId = nextId++;
    BitSet merged = leafsetOf[bi] | leafsetOf[bj];
    leafsetOf[newId] = merged;

    // only add if it's a proper non-trivial bipartition in the FULL tree:
    // both sides must have ≥2 global taxa, meaning popcount ≥2 and ≤N-2
    int pc = merged.popcount();
    if (pc >= 2 && pc <= N - 2) outClades.push_back(merged);

    int nr = (int)table.size();
    for (auto& row : table) row.push_back(0.0);
    table.push_back(vector<double>(nr + 1, 0.0));
    for (int k : live) {
      if (k == bi || k == bj) continue;
      double dik = table[idToRow[bi]][idToRow[k]];
      double djk = table[idToRow[bj]][idToRow[k]];
      double dij = table[idToRow[bi]][idToRow[bj]];
      double nd = max(0.0, 0.5*(dik + djk - dij));
      table[nr][idToRow[k]] = nd;
      table[idToRow[k]][nr] = nd;
    }
    idToRow.erase(bi); idToRow.erase(bj);
    idToRow[newId] = nr;
    numActive--;
  }
}

// --------------------------------------------------------- taxon index -----
// Incremental / indexed scoring: instead of scanning every input quartet for
// every candidate split (O(Q) per call), index quartets by the taxa they
// touch. For a split (A1,A2,A3) we then only walk the quartet-lists of taxa
// in the SMALLER of A1/A2, turning the per-call cost into roughly
// O(sum of degree(t) for t in the smaller side), where degree(t) is how many
// input quartets mention t. This is a big win precisely when the quartet set
// is sparse (low average degree); for dense, gene-tree-scale quartet sets it
// degenerates back toward O(Q) per call, which is why ASTRAL itself instead
// Structure-of-Arrays layout for the quartet index, tuned for the
// bandwidth-bound scoreSplit loop.
//
// The hot inner loop streams, per entry, the three taxon ids (partner, c1, c2)
// and tests them against bitsets; the weight is used only on the ~4.5% of
// entries that pass. So we store the ids in three PARALLEL uint16 arrays that
// are streamed contiguously (6 bytes/entry instead of the old 24-byte struct),
// and keep the weight in a SEPARATE "cold" uint16 array touched only on a hit.
// This ~4x reduction in streamed bytes directly raises throughput on a
// memory-bandwidth-limited loop, and stacks with threading / NUMA interleaving.
//
// Width choices:
//   * ids as uint16: taxon indices are 0..N-1; N<=65535 covers every realistic
//     case (target is ~1000 taxa). If you ever exceed 65535 taxa, widen IdT.
//   * weight as uint16: a quartet weight is a count (e.g. #gene trees showing
//     it); 65535 is ample. Rescale inputs if you somehow exceed it.
//   * ACCUMULATORS are uint64, NOT uint32: a single scoreSplit can return up to
//     ~1.5e10 at N=1000 (>uint32), and dp values sum to up to C(N,4) ~ 4e10.
//     uint32 would silently overflow. The accumulators are few and
//     cache-resident, so their width costs no streaming bandwidth.
typedef uint16_t IdT;
typedef uint16_t WeightT;
typedef uint64_t ScoreT;

struct TaxonIndex {
  // For taxon t, its entries occupy [start[t], start[t+1]) in the flat arrays.
  vector<int>     start;      // size n+1 (offsets; int is fine, total < 2^31)
  vector<IdT>     partner;    // hot
  vector<IdT>     c1;         // hot
  vector<IdT>     c2;         // hot
  vector<WeightT> weight;     // cold (read only on a passing entry)
};
TaxonIndex idx;

void buildTaxonIndex(const vector<Quartet>& qs, int n) {
  // Two entries per quartet (co-pair a->b and b->a); count per taxon first.
  idx.start.assign(n + 1, 0);
  for (auto& q : qs) { idx.start[q.a + 1]++; idx.start[q.b + 1]++; }
  for (int i = 0; i < n; i++) idx.start[i + 1] += idx.start[i];
  size_t total = idx.start[n];
  idx.partner.resize(total);
  idx.c1.resize(total);
  idx.c2.resize(total);
  idx.weight.resize(total);
  // Fill using a moving cursor per taxon.
  vector<int> cur(idx.start.begin(), idx.start.end() - 1);
  auto put = [&](int t, int partner, int c1, int c2, double w) {
    int p = cur[t]++;
    idx.partner[p] = (IdT)partner;
    idx.c1[p]      = (IdT)c1;
    idx.c2[p]      = (IdT)c2;
    idx.weight[p]  = (WeightT)(w + 0.5);   // round to nearest; weights are counts
  };
  for (auto& q : qs) {
    // Index ONLY the co-pair (q.a, q.b): the two taxa that must straddle the
    // A1|A2 split for a quartet to contribute. Indexing q.x/q.y too would
    // double-count.
    put(q.a, q.b, q.x, q.y, q.weight);
    put(q.b, q.a, q.x, q.y, q.weight);
  }
}

// Cache of "which taxa are in this cluster" so we don't re-scan a BitSet's
// bits every time the same cluster is used as one side of a candidate split
// (clusters in X get reused across many DP merge attempts).
unordered_map<BitSet, vector<int>, BitSetHash> membersCache;
const vector<int>& membersOfCluster(const BitSet& b) {
  auto it = membersCache.find(b);
  if (it != membersCache.end()) return it->second;
  vector<int> v;
  for (int i = 0; i < b.n; i++) if (b.test(i)) v.push_back(i);
  return membersCache.emplace(b, move(v)).first->second;
}

// Weight contributed by splitting cluster A into (A1, A2), with A3 = L \ A the
// complement. This is the ASTRAL/Bryant-Steel per-split term W(...): the number
// (weight) of input quartets that the newly created tree edge -- the one
// separating A1|A2 within A -- displays for the first time.
//
// RULE (validated against brute force on complete quartet sets):
//   A quartet ab|cd is counted here iff ONE of its two pairs lies wholly
//   inside a single part, and the OTHER pair has exactly one member in each
//   of the two remaining parts. Three configurations satisfy this:
//     (i)   co-pair spans A1 & A2, other pair inside A3
//     (ii)  co-pair spans A1 & A3, other pair inside A2
//     (iii) co-pair spans A2 & A3, other pair inside A1
//   All three must be counted. An earlier version implemented only case (i);
//   that silently scored 0 for whole clades (any clade whose quartets are
//   "contained" in A1/A2 rather than A3), so the DP's argmax was wrong even
//   when every needed bipartition was present in X.
//
// COUNTING MULTIPLICITY: each displayed quartet is attributed to exactly ONE
// tree edge -- its unique separating edge -- so summing this term over all
// splits of a tree equals the number of displayed quartets exactly (verified:
// a 6-taxon tree's 15 quartets sum to 15, not 30). dp[full] is therefore the
// true satisfied-quartet weight and is reported WITHOUT any halving.
//
// (Bryant & Steel 2001 Eq. 10 give an equivalent total via a different
// attribution -- each quartet counted at the cluster where its co-pair first
// unites, rather than at its separating edge. The totals match, but the
// per-split values differ, so their formula is NOT a drop-in for this DP,
// which is built around separating-edge attribution.)
//
// PERFORMANCE: the index is keyed on the co-pair, so each case iterates one
// endpoint of the straddling co-pair; we always iterate the SMALLER of the two
// straddled parts so per-call cost stays proportional to the smaller side.
// Iterating the larger part -- especially A3, which near the root is almost all
// taxa -- was the cause of an earlier version-to-version slowdown.
ScoreT scoreSplit(const BitSet& A1, const BitSet& A2, const BitSet& A3) {
  const vector<int>& m1 = membersOfCluster(A1);
  const vector<int>& m2 = membersOfCluster(A2);
  const vector<int>& m3 = membersOfCluster(A3);
  ScoreT s = 0;

  // Stream taxon t's entries from the SoA arrays. The three id arrays are the
  // hot, contiguously-streamed data (2 bytes each); weight is read only on a
  // passing entry (cold). A local helper keeps the three cases identical.
  auto accumulate = [&](const vector<int>& S, const BitSet& O,
                        const BitSet& P1, const BitSet& P2) {
    for (int t : S) {
      int lo = idx.start[t], hi = idx.start[t + 1];
      const IdT* pp = idx.partner.data();
      const IdT* pc1 = idx.c1.data();
      const IdT* pc2 = idx.c2.data();
      for (int p = lo; p < hi; p++) {
        if (O.test(pp[p]) && P1.test(pc1[p]) && P2.test(pc2[p]))
          s += idx.weight[p];   // cold read, only on a hit
      }
    }
  };

  // (i) co-pair spans A1|A2, other pair inside A3.
  {
    bool it1 = m1.size() <= m2.size();
    accumulate(it1 ? m1 : m2, it1 ? A2 : A1, A3, A3);
  }
  // (ii) co-pair spans A1|A3, other pair inside A2.
  {
    bool it1 = m1.size() <= m3.size();
    accumulate(it1 ? m1 : m3, it1 ? A3 : A1, A2, A2);
  }
  // (iii) co-pair spans A2|A3, other pair inside A1.
  {
    bool it2 = m2.size() <= m3.size();
    accumulate(it2 ? m2 : m3, it2 ? A3 : A2, A1, A1);
  }

  return s;
}

int main(int argc, char** argv) {
  string quartetsPath, guidePath, outPath;
  int restarts = 8;   // jittered full-taxon NJ rounds
  int subsets  = 8;   // random-subset NJ rounds (ASTRAL-III trick)
  int seed     = 12345;
  string allTreesPath;
  double jitterAlpha = 0.3;   // adaptive jitter = alpha * IQR(distances)
  double jitterAbs   = -1.0;  // >=0 overrides adaptive with a fixed amplitude
  string dumpDistPath;        // if set, write the distance matrix here
  bool verbose = false;       // print |X| after each construction step
  int threads = 0;            // 0 => OpenMP default; >0 => that many threads

  CLI::App app{
    "ASTRAQ -- Quartet Amalgamation via Dynamic Programming\n"
    "Infers a species tree from weighted quartet trees by maximising\n"
    "quartet score over a candidate bipartition set X built from NJst.\n"
    "Reference: analogous to ASTRAL (Mirarab et al. 2014; Zhang et al. 2018)\n"
    "           with X seeded by NJst (Liu & Yu 2011) instead of gene trees."
  };

  app.set_version_flag("--version", PROG_VERSION);
  app.get_formatter()->column_width(40);

  app.add_option("-q,--quartets", quartetsPath,
                 "Input quartet trees file (required).\n"
                 "One quartet per line, either:\n"
                 "  plain:  A B C D [weight]   (A,B grouped vs C,D; weight defaults to 1)\n"
                 "  Newick: (A,B,(C,D));  [weight]")
  ->required()
  ->check(CLI::ExistingFile);

  app.add_option("-g,--guide", guidePath,
                 "Guide trees file (optional, Newick, one per line).\n"
                 "Bipartitions are extracted and added to the search space X,\n"
                 "supplementing those produced by NJst runs.")
  ->check(CLI::ExistingFile);

  app.add_option("-o,--output", outPath,
                 "Output file for the species tree in Newick format.\n"
                 "If omitted, the tree is written to stdout.");

  app.add_option("-r,--restarts", restarts,
                 "Number of jittered full-taxon NJst restarts (default: 8).\n"
                 "Each restart adds small random noise to the distance matrix\n"
                 "before running NJst, producing alternative bipartitions for X.")
  ->check(CLI::NonNegativeNumber);

  app.add_option("-s,--subsets", subsets,
                 "Number of random-subset NJst rounds (default: 8).\n"
                 "Each round draws ceil(n/2) random taxa, runs NJst on the\n"
                 "submatrix, and lifts the bipartitions to the full taxon set.\n"
                 "Analogous to the polytomy-resolution enrichment in ASTRAL-III.")
  ->check(CLI::NonNegativeNumber);

  app.add_option("--seed", seed,
                 "Random seed for NJst jitter and subset sampling (default: 12345).")
  ->check(CLI::NonNegativeNumber);

  app.add_option("--all-trees", allTreesPath,
                 "If set, write ALL distinct co-optimal trees (Newick, one per line)\n"
                 "to this file. Enumeration is capped; see stderr for whether the cap\n"
                 "was hit. Useful for diagnosing whether sparse data determines the\n"
                 "topology uniquely.");

  app.add_option("--jitter-alpha", jitterAlpha,
                 "Adaptive jitter multiplier (default: 0.3). The per-restart distance\n"
                 "perturbation amplitude is alpha * IQR(distances), where IQR is the\n"
                 "inter-quartile range of the pairwise distances -- a robust spread\n"
                 "measure that stays sensible across clean, sparse, and conflicted data.\n"
                 "Larger alpha => more diverse NJst restarts => larger search space X.")
  ->check(CLI::NonNegativeNumber);

  app.add_option("--jitter-abs", jitterAbs,
                 "Escape hatch: use a FIXED jitter amplitude (e.g. 0.05, the legacy\n"
                 "behaviour) instead of the adaptive alpha*IQR rule. When set, overrides\n"
                 "--jitter-alpha. Negative (default) means use the adaptive rule.");

  app.add_option("--dump-distances", dumpDistPath,
                 "If set, write the (unjittered) NJst distance matrix to this file\n"
                 "('-' for stderr), then continue normally. The header line also reports\n"
                 "the jitter amplitude that will be applied per restart, so you can see\n"
                 "whether it is large relative to the spread of distances.");

  app.add_flag("-v,--verbose", verbose,
               "Print the running size of the candidate set X after each construction\n"
               "step (clean NJst, each jitter/subset round, guide trees, intersection\n"
               "closure, and each dedupe), to see which steps actually enrich X.");

  app.add_option("-t,--threads", threads,
                 "Number of threads for the parallel DP (default: 0 = OpenMP's own\n"
                 "default, which is all hardware threads unless OMP_NUM_THREADS is set).\n"
                 "Ignored if the program was built without -fopenmp. For this\n"
                 "memory-bandwidth-bound DP, setting this to the number of PHYSICAL\n"
                 "cores often matches or beats using every hyperthread.")
  ->check(CLI::NonNegativeNumber);

  CLI11_PARSE(app, argc, argv);

  // ---- threads: configure and report the parallel DP thread count ----
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
  // Query the size of the team that a parallel region will actually get.
  int effThreads = 1;
#pragma omp parallel
  {
#pragma omp single
    effThreads = omp_get_num_threads();
  }
  cerr << "Parallel DP: using " << effThreads << " thread"
  << (effThreads == 1 ? "" : "s");
  if (threads > 0)      cerr << " (--threads " << threads << ")";
  else                  cerr << " (OpenMP default; set --threads or OMP_NUM_THREADS to change)";
  cerr << ".\n";
#else
  if (threads > 0)
    cerr << "Note: --threads " << threads
    << " ignored -- this build has no OpenMP (compile with -fopenmp).\n";
  cerr << "Parallel DP: 1 thread (built without OpenMP).\n";
#endif

  vector<Quartet> quartets = readQuartets(quartetsPath);
  cerr << "Read " << quartets.size() << " quartets over " << N << " taxa.\n";
  if (N < 4) { cerr << "Need at least 4 taxa.\n"; return 1; }
  buildTaxonIndex(quartets, N);

  // ---- build search space X ----
  vector<BitSet> Xvec;

  // Helper: when --verbose, report |X| and the change since the last step.
  size_t lastX = 0;
  auto vstep = [&](const string& label) {
    if (!verbose) return;
    long long delta = (long long)Xvec.size() - (long long)lastX;
    cerr << "  [X] " << Xvec.size() << " clusters"
    << " (" << (delta >= 0 ? "+" : "") << delta << ")"
    << "  after " << label << "\n";
    lastX = Xvec.size();
  };

  for (int i=0;i<N;i++) { BitSet b(N); b.set(i); Xvec.push_back(b); }
  BitSet full(N); for (int i=0;i<N;i++) full.set(i);
  Xvec.push_back(full);
  vstep("singletons + full set");

  auto D = buildDistanceMatrix(quartets, N);
  mt19937 rng(seed);

  // Determine the per-restart jitter amplitude (fixed or adaptive alpha*IQR).
  JitterParams jp{jitterAlpha, jitterAbs};
  double jitterAmp = computeJitterAmplitude(D, N, jp);
  if (jitterAbs >= 0.0)
    cerr << "Jitter amplitude: +/-" << jitterAmp << "  (fixed, --jitter-abs)\n";
  else
    cerr << "Jitter amplitude: +/-" << jitterAmp
    << "  (adaptive: " << jitterAlpha << " * IQR of distances)\n";

  if (!dumpDistPath.empty())
    dumpDistanceMatrix(D, N, idToName, dumpDistPath, jitterAmp);

  // 1. One clean full-taxon NJ (baseline guide tree)
  neighborJoiningClades(D, N, rng, 0.0, Xvec);
  vstep("clean NJst");

  // 2. Jittered full-taxon NJ restarts: vary distances to break ties and
  //    produce alternative full-taxon bipartitions. Amplitude is chosen by
  //    computeJitterAmplitude (see that function for the policy rationale).
  for (int r = 0; r < restarts; r++) {
    neighborJoiningClades(D, N, rng, jitterAmp, Xvec);
    vstep("jitter restart " + to_string(r + 1) + "/" + to_string(restarts));
  }

  // 3. Random-subset NJ rounds (ASTRAL-III enrichment trick):
  //    Each round draws a random subset S of ceil(N/2) taxa, runs NJ on
  //    the D[S×S] submatrix, and lifts the bipartitions to the full N-taxon
  //    BitSet. These can produce bipartitions that no full-taxon NJ ever
  //    would, because excluding irrelevant taxa removes the competing signal
  //    that suppresses a true bipartition in the full-taxon tree.
  int subsetSize = max(4, (N + 1) / 2);
  for (int s = 0; s < subsets; s++) {
    subsetNJClades(D, N, rng, subsetSize, Xvec);
    vstep("subset round " + to_string(s + 1) + "/" + to_string(subsets));
  }

  cerr << "X before dedup: " << Xvec.size() << " clusters"
  << " (full-NJ: 1, jitter: " << restarts
  << ", subset: " << subsets << ").\n";

  if (!guidePath.empty()) { addGuideTreeBipartitions(guidePath, Xvec); vstep("guide trees"); }

  // ---- initial dedupe before intersection closure ----
  auto dedupe = [&]() {
    sort(Xvec.begin(), Xvec.end(), [](const BitSet&a, const BitSet&b){
      if (a.popcount()!=b.popcount()) return a.popcount()<b.popcount();
      return a.w < b.w;
    });
    Xvec.erase(unique(Xvec.begin(), Xvec.end(),
                      [](const BitSet&a,const BitSet&b){return a==b;}), Xvec.end());
  };
  dedupe();
  vstep("dedupe");

  // ---- pairwise intersection closure ----
  // For every pair of overlapping clusters A, B already in X (neither a
  // subset of the other, intersection non-empty and non-trivial), add A∩B.
  // Motivation: suppose NJ run 1 produces clade {B,C,D} and NJ run 2
  // produces {C,D,E}; their intersection {C,D} might be the true clade in
  // the species tree but neither run produced it directly.  One round of
  // pairwise intersections recovers it.  We do exactly one round (not
  // iterating until stable) to keep cost at O(|X|^2 * n/64).
  {
    size_t base = Xvec.size();
    for (size_t i = 0; i < base; i++)
      for (size_t j = i+1; j < base; j++) {
        const BitSet& A = Xvec[i];
        const BitSet& B = Xvec[j];
        // Skip if one is a subset of the other (no new cluster produced)
        // or if they are disjoint (intersection is empty / trivial).
        if (A.isSubsetOf(B) || B.isSubsetOf(A)) continue;
        BitSet inter = A & B;
        int pc = inter.popcount();
        if (pc < 2 || pc > N-2) continue;   // trivial
        Xvec.push_back(inter);
      }
    size_t added = Xvec.size() - base;
    if (added > 0)
      cerr << "Intersection closure added " << added
      << " candidate clusters.\n";
  }
  vstep("intersection closure");
  
  {
    size_t base = Xvec.size();
    BitSet L(N);
    for (int i = 0; i < N; i++) L.set(i);
    for (size_t i = 0; i < base; i++) {
      BitSet comp = L ^ Xvec[i];
      int pc = comp.popcount();
      if (pc >= 1 && pc <= N-1) Xvec.push_back(comp);
    }
  }
  
  // ---- final dedupe ----
  dedupe();
  vstep("final dedupe");
  cerr << "Search space X has " << Xvec.size() << " candidate clusters.\n";

  // ---- DP ----
  // dp[A]   = best achievable quartet score for cluster A using splits in X
  // optSplits[A] = every split {B, A\B} that achieves dp[A] (Bryant & Steel's
  //           Dopt: keep all maximisers, not just one). This lets us
  //           enumerate ALL optimal trees, not merely count or return one.
  //
  // PARALLELISM. The outer loop over clusters A has a real dependency:
  // dp[A] reads dp[B], dp[C] for sub-clusters, which must already be done.
  // Because Xvec is sorted by cluster size, all clusters of a given size are
  // mutually independent -- their sub-clusters are strictly smaller and thus
  // already computed. We therefore process Xvec in contiguous same-size
  // BANDS, parallelising the clusters within each band (an OpenMP barrier
  // falls naturally between bands). Within one cluster the inner loop over
  // candidate splits B is also independent, but we parallelise at the
  // cluster level since that gives coarser tasks and avoids nested reduction.
  //
  // To make the parallel region data-race-free:
  //   * dp/optSplits are index-keyed vectors (one slot per Xvec position);
  //     distinct clusters write distinct slots.
  //   * a BitSet->index map (xIndex) is built once, single-threaded, so
  //     look-ups of C = A^B are read-only during parallel work.
  //   * membersCache is pre-populated for every cluster AND its complement
  //     single-threaded, so scoreSplit()'s membersOfCluster() only READS a
  //     stable map inside the parallel region (concurrent inserts into a
  //     std::unordered_map would be undefined behaviour).
  unordered_map<BitSet,int,BitSetHash> xIndex;
  xIndex.reserve(Xvec.size()*2);
  for (int i = 0; i < (int)Xvec.size(); i++) xIndex[Xvec[i]] = i;

  // Pre-populate membersCache single-threaded (clusters + complements).
  for (auto& A : Xvec) {
    membersOfCluster(A);
    membersOfCluster(full ^ A);
  }

  vector<ScoreT> dpVal(Xvec.size(), 0);
  vector<char>   dpDone(Xvec.size(), 0);   // char, not bool: parallel writes
  // to distinct elements must not
  // share a byte (vector<bool> would)
  vector<vector<pair<BitSet,BitSet>>> optSplitsVec(Xvec.size());

  // Precompute popcounts (Xvec is size-sorted).
  vector<int> pcOf(Xvec.size());
  for (int i = 0; i < (int)Xvec.size(); i++) pcOf[i] = Xvec[i].popcount();

  // ---- Phase 1: enumerate all scoreable tripartitions, then score them in
  //      one flat parallel loop. ----
  //
  // Rationale: scoreSplit dominates runtime (~98%) and reads NO DP state --
  // only the quartet index and the three bitsets -- so every split's score
  // can be computed before any dp combination. Enumerating all valid triples
  // into one flat worklist and scoring them with a single parallel_for gives
  // perfect load balancing. The previous per-band parallelism under-filled
  // cores badly: the expensive root-level clusters (large m1/m2, tiny A3)
  // live in bands of only 1-6 clusters, so most cores sat idle exactly where
  // the work was heaviest. The flat list has no such imbalance and no
  // per-band barriers.
  //
  // A triple is "valid" (its score is potentially usable) iff:
  //   B in X, B subset of A, B is the smaller half of {B, A\B},
  //   C = A\B is in X.
  // The dpDone checks are NOT applied here -- they depend on processing
  // order and belong to the combine phase. Scoring a split whose children
  // turn out unreachable is harmless (the combine phase simply won't use a
  // child that has no value), and costs only a little extra scoring; in
  // practice with a size-sorted X every child of A is smaller than A and
  // will have been solved, so essentially all scored triples are used.
  struct Triple { int ai; int bIdx; int cIdx; };
  vector<Triple> work;
  // reserve a rough estimate to avoid reallocs
  work.reserve(Xvec.size() * 8);

  for (int ai = 0; ai < (int)Xvec.size(); ai++) {
    int pc = pcOf[ai];
    if (pc <= 1) continue;                  // singletons: no split
    const BitSet& A = Xvec[ai];
    for (int bi = 0; bi < (int)Xvec.size(); bi++) {
      int bpc = pcOf[bi];
      if (bpc == 0 || bpc >= pc) continue;
      const BitSet& B = Xvec[bi];
      if (!B.isSubsetOf(A)) continue;
      BitSet C = A ^ B;
      if (!( (bpc * 2 < pc) ||
            (bpc * 2 == pc && lexSmaller(B, C)) )) continue; // smaller-half only
      auto itC = xIndex.find(C);
      if (itC == xIndex.end()) continue;   // C not in X
      work.push_back({ai, bi, itC->second});
    }
  }

  // Score every triple in parallel (embarrassingly parallel, no barriers).
  vector<ScoreT> triScore(work.size(), 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (long long w = 0; w < (long long)work.size(); w++) {
    const Triple& t = work[w];
    const BitSet& A = Xvec[t.ai];
    const BitSet& B = Xvec[t.bIdx];
    const BitSet& C = Xvec[t.cIdx];
    BitSet A3 = full ^ A;
    triScore[w] = scoreSplit(B, C, A3);
  }

  // ---- Phase 2: combine. Group the (already-scored) triples by their
  //      cluster A and, walking clusters in size order so children are
  //      solved first, compute dp[A] = max over its triples of
  //      dpVal[B] + dpVal[C] + triScore. This phase does only cheap
  //      additions and comparisons (~2% of former DP time), so it is left
  //      serial for simplicity and determinism. ----
  //
  // Build, per cluster A, the index range of its triples. Because `work`
  // was filled in increasing ai order, each cluster's triples are contiguous.
  vector<pair<int,int>> triRange(Xvec.size(), {0, 0}); // [begin, end) into work
  {
    size_t w = 0;
    for (int ai = 0; ai < (int)Xvec.size(); ai++) {
      size_t begin = w;
      while (w < work.size() && work[w].ai == ai) w++;
      triRange[ai] = { (int)begin, (int)w };
    }
  }

  // Scores are exact non-negative integers now (ScoreT is unsigned), so we
  // track "has any valid split" with a flag rather than a negative sentinel,
  // and compare with == rather than an epsilon.
  // Clusters must be processed smallest-first so dpVal[B], dpVal[C] are ready.
  // Xvec is size-sorted, so a single forward pass suffices.
  for (int ai = 0; ai < (int)Xvec.size(); ai++) {
    int pc = pcOf[ai];
    if (pc <= 1) { dpVal[ai] = 0; dpDone[ai] = 1; continue; }

    ScoreT best = 0;
    bool haveSplit = false;
    auto [wb, we] = triRange[ai];
    for (int w = wb; w < we; w++)
    {
      const Triple& t = work[w];
      if (!dpDone[t.bIdx] || !dpDone[t.cIdx]) continue;
      ScoreT val = dpVal[t.bIdx] + dpVal[t.cIdx] + triScore[w];
      if (!haveSplit || val > best) { best = val; haveSplit = true; }
    }
    if (!haveSplit)
    {
      dpVal[ai] = 0;
      dpDone[ai] = 1;   // no valid split
      continue;
    }
    vector<pair<BitSet,BitSet>> splits;
    for (int w = wb; w < we; w++) {
      const Triple& t = work[w];
      if (!dpDone[t.bIdx] || !dpDone[t.cIdx]) continue;
      ScoreT val = dpVal[t.bIdx] + dpVal[t.cIdx] + triScore[w];
      if (val != best) continue;
      splits.push_back({ Xvec[t.bIdx], Xvec[t.cIdx] });
    }
    dpVal[ai] = best;
    optSplitsVec[ai] = std::move(splits);
    dpDone[ai] = 1;
  }

  // Emit the "not rich enough" warning single-threaded (kept out of the
  // parallel region to avoid interleaved output).
  for (size_t i = 0; i < Xvec.size(); i++)
  {
    if (pcOf[i] > 1 && optSplitsVec[i].empty() && dpVal[i] == 0)
    {
      cerr << "Warning: no valid split found in X for a cluster of size "
      << pcOf[i]
      << " -- search space is not rich enough (try increasing -r or supplying -g guide trees).\n";
      break; // one warning is enough
    }
  }

  // Adapters so the rest of the code (enumeration, scoring) can look up by
  // BitSet as before.
  auto dpLookup = [&](const BitSet& A) -> ScoreT
  {
    auto it = xIndex.find(A);
    return (it == xIndex.end()) ? 0 : dpVal[it->second];
  };
  auto optSplitsLookup = [&](const BitSet& A) -> const vector<pair<BitSet,BitSet>>*
  {
    auto it = xIndex.find(A);
    if (it == xIndex.end()) return nullptr;
    return &optSplitsVec[it->second];
  };

  // ---- enumerate optimal trees ----
  // A rooted-cluster DP can express the same unrooted tree under several
  // rootings (the root may sit on any internal edge). We therefore enumerate
  // candidate trees and DEDUPE by their unrooted bipartition set, which is
  // the canonical identity of an unrooted topology. Enumeration is capped:
  // co-optimal trees can be exponentially many when data is sparse, and that
  // abundance is itself the diagnostic signal we want to report.
  const size_t ENUM_CAP = 1000;

  // enumRooted(A): up to ENUM_CAP Newick strings for optimal subtrees on A.
  // Memoised. Each string has no trailing ';'.
  unordered_map<BitSet, vector<string>, BitSetHash> memo;
  bool truncated = false;
  function<const vector<string>&(const BitSet&)> enumRooted =
  [&](const BitSet& A) -> const vector<string>& {
    auto m = memo.find(A);
    if (m != memo.end()) return m->second;
    vector<string> out;
    if (A.popcount() == 1) {
      for (int i = 0; i < N; i++) if (A.test(i)) { out.push_back(idToName[i]); break; }
      return memo.emplace(A, std::move(out)).first->second;
    }
    const vector<pair<BitSet,BitSet>>* spl = optSplitsLookup(A);
    if (spl == nullptr || spl->empty()) {
      // no recorded resolution: emit a flat (star) clade as a fallback
      string s = "("; bool first = true;
      for (int i = 0; i < N; i++) if (A.test(i)) { if(!first) s += ","; s += idToName[i]; first=false; }
      out.push_back(s + ")");
      return memo.emplace(A, std::move(out)).first->second;
    }
    for (auto& sp : *spl)
    {
      const vector<string>& L = enumRooted(sp.first);
      const vector<string>& R = enumRooted(sp.second);
      for (const string& ls : L) {
        for (const string& rs : R) {
          if (out.size() >= ENUM_CAP) { truncated = true; break; }
          out.push_back("(" + ls + "," + rs + ")");
        }
        if (out.size() >= ENUM_CAP) { truncated = true; break; }
      }
      if (out.size() >= ENUM_CAP) { truncated = true; break; }
    }
    return memo.emplace(A, std::move(out)).first->second;
  };

  // Helper: canonical unrooted bipartition-set signature of a Newick string,
  // used to collapse different rootings of the same unrooted tree.
  auto signature = [&](const string& nwk) -> string
  {
    // parse the newick into clades over the full leaf set
    vector<BitSet> clades = newickToClades(nwk + ";", nameToId, N);
    set<string> bips;
    for (auto& c : clades)
    {
      int pc = c.popcount();
      if (pc < 2 || pc > N - 2) continue;
      // canonical side: the half not containing leaf 0 (fixed reference)
      BitSet side = c.test(0) ? (full ^ c) : c;
      string key;
      for (auto w : side.w) key += to_string(w) + ",";
      bips.insert(key);
    }
    string sig;
    for (auto& b : bips) sig += b + "|";
    return sig;
  };

  const vector<string>& rootedTrees = enumRooted(full);
  // Dedupe by unrooted signature.
  vector<string> distinctTrees;
  {
    set<string> seenSig;
    for (const string& t : rootedTrees)
    {
      string sig = signature(t);
      if (seenSig.insert(sig).second)
        distinctTrees.push_back(t);
    }
  }

  string tree = distinctTrees.front() + ";";

  ScoreT totalWeight = 0;
  for (auto& q : quartets) totalWeight += (WeightT)(q.weight + 0.5);
  ScoreT score = dpLookup(full);

  cerr << "Quartet score (weighted): " << score
       << "  out of  " << totalWeight
       << "  (" << (totalWeight>0 ? 100.0*(double)score/(double)totalWeight : 0.0) << "%)\n";

  if (truncated)
    cerr << "Co-optimal trees in X: >= " << distinctTrees.size()
         << " (enumeration capped at " << ENUM_CAP
         << " rooted assemblies; data underdetermines the topology)\n";
  else
    cerr << "Co-optimal trees in X: " << distinctTrees.size()
         << (distinctTrees.size() > 1
            ? "  (topology not uniquely determined by the quartets)\n"
            : "  (unique optimum)\n");

  if (!allTreesPath.empty())
  {
    ofstream out(allTreesPath);
    for (const string& t : distinctTrees) out << t << ";\n";
    cerr << "Wrote " << distinctTrees.size()
         << " distinct co-optimal tree(s) to " << allTreesPath << "\n";
  }

  if (!outPath.empty())
  {
    ofstream out(outPath);
    out << tree << "\n";
    cerr << "Wrote species tree to " << outPath << "\n";
  }
  else
  {
    cout << tree << "\n";
  }
  return 0;
}
