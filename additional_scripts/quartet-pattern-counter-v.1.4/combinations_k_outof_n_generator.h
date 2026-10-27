#include <iostream>
#include <vector>

#include <iostream>
using namespace std;

// Function to compute C(n, k) (n choose k)
long long binomialCoefficient(int k, int n)
{
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


class combination_generator
{
  std::vector<unsigned short> numbers;
  std::vector<unsigned short> state;

  unsigned short k;
  unsigned short n;
  bool started;

 public:
 combination_generator(unsigned short k, unsigned short n):numbers(std::vector<unsigned short>(n)), state(std::vector<unsigned short>(k)), k(k), n(n), started(false)
  {
    for (unsigned short i=0; i<n; ++i)
      numbers[i] = i+1;
    for (unsigned short i=0; i<k; ++i)
      state[i] = i+1;
  }

  const std::vector<unsigned short> &next()
  {
    if (!started)
    {
      started = true;
      return state;
    }

    for (unsigned short l=0; l<k; ++l)
    {
      if (state[k-1-l] == n-l)
        state.pop_back();
      else
        break;
    }
    if (state.empty())
      return state;

    ++state.back();

    unsigned short v = state.size();
    for (unsigned short i=v; i<k; ++i)
    {
      state.push_back(state[i-1]+1);
    }

    return state;
  }
};
