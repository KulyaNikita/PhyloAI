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

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// typedef string mystringType;
typedef string mystringType;


class cartesian_product
{
private:
  void cartesian_product_for_2strings(const vector<mystringType> &a, const vector<mystringType> &b, vector<mystringType> &res)
  {
    res.clear();
    for (unsigned i=0; i<a.size(); ++i)
    {
      for (unsigned j=0; j<b.size(); ++j)
      {
        mystringType tmp = a[i] + b[j];
        res.push_back(tmp);
      }
    }
  }
  
  void nth_cartesian_product_of_symbolset(unsigned nth, const mystringType symbols, vector<mystringType> &res)
  {
    res.clear();
    
    vector<mystringType> start, tmp_res, tmp_res2;
    for (unsigned i=0; i < symbols.size(); ++i)
    {
      // string:
      mystringType tmp = mystringType(1u,symbols[i]);
      // faststring:
      //      mystringType tmp = mystringType(symbols[i], 1u);
      
      start.push_back(tmp);
    }
    
    //    cout << "DEBUG: start" << endl;
    //    print_container(cout, start, "", ",","");
    //    cout << "DEBUG: start -- END" << endl;
    
    tmp_res = start;
    for (unsigned j=1; j<nth; ++j)
    {
      cartesian_product_for_2strings(tmp_res, start, tmp_res2);
      tmp_res = tmp_res2;
    }
    
    res = tmp_res2;
  }
  
  
public:
  void operator()(unsigned nth, const mystringType symbols, vector<mystringType> &res)
  {
    nth_cartesian_product_of_symbolset(nth, symbols, res);
  }
};



