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
// with the distribution.
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
// File: QP_python_bindings.cpp
// Python bindings for QuartetPatternCounter class
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "QuartetPattern-counter-core.hpp"

namespace py = pybind11;
using partition_array = py::array_t<int, py::array::c_style | py::array::forcecast>;

// Python wrapper for QuartetPatternCounter class
// The output arrays are filled in place. They must have exactly this dtype and be C-contiguous.
// Together with py::arg().noconvert() this makes pybind11 reject other arrays instead of
// silently filling a temporary converted copy.
typedef py::array_t<float_type_freq, py::array::c_style> freq_array;
typedef py::array_t<uint32_t,        py::array::c_style> length_array;
typedef py::array_t<uint16_t,        py::array::c_style> pattern_array;

class PyQuartetPatternCounter
{
  private:
    QuartetPatternCounter myQPcounter;

    void check_columns(const py::buffer_info &buf_freq) const
    {
        if (buf_freq.shape[0] == 0)
            throw py::value_error("batch_size must be positive");
        if ((size_t)buf_freq.shape[1] != myQPcounter.get_num_cols())
        {
            throw py::value_error("Frequency array must have shape (batch_size, " + std::to_string(myQPcounter.get_num_cols()) +
                                     ") for this data type, but has " + std::to_string(buf_freq.shape[1]) + " columns.");
        }
    }
    
    static QuartetPatternCounter create_QuartetPatternCounter(
        const std::string &filename,
        partition_array arr1, 
        partition_array arr2,
        partition_array arr3, 
        partition_array arr4,
        char dataType, int numThreads, int verbose
        )
    {
        py::buffer_info buf1 = arr1.request();
        py::buffer_info buf2 = arr2.request();
        py::buffer_info buf3 = arr3.request();
        py::buffer_info buf4 = arr4.request();       

        // Error checking
        if (buf1.ndim != 1) throw py::value_error("arr1 must be 1D");
        if (buf2.ndim != 1) throw py::value_error("arr2 must be 1D");
        if (buf3.ndim != 1) throw py::value_error("arr3 must be 1D");
        if (buf4.ndim != 1) throw py::value_error("arr4 must be 1D");

        size_t size1 = buf1.size;
        size_t size2 = buf2.size;
        size_t size3 = buf3.size;
        size_t size4 = buf4.size;
        
        // Extract pointers and sizes
        const int* ptr1 = static_cast<const int*>(buf1.ptr);
        const int* ptr2 = static_cast<const int*>(buf2.ptr);
        const int* ptr3 = static_cast<const int*>(buf3.ptr);
        const int* ptr4 = static_cast<const int*>(buf4.ptr);
        
        // Construct and return C++ object
        return QuartetPatternCounter(filename, ptr1, size1, ptr2, size2, ptr3, size3, ptr4, size4, dataType, numThreads, verbose);
    }
    
    
public:
    // Constructor for computing all quartet frequencies
    PyQuartetPatternCounter(const std::string &filename, char dataType, int numThreads, int verbose) : myQPcounter(filename, dataType, numThreads, verbose)
    {}

    // Constructor for computing four cluster quartet frequencies
    PyQuartetPatternCounter(const std::string &filename,
                partition_array partition1, partition_array partition2,
                partition_array partition3, partition_array partition4,
                char dataType, int numThreads, int verbose) : myQPcounter(create_QuartetPatternCounter(filename, partition1, partition2, partition3, partition4,
                                                                          dataType, numThreads, verbose))
    {}
   
    
   
    std::pair<int32_t, int32_t> get_dataset_characteristics() const
    {
        return myQPcounter.get_dataset_characteristics();
    }

    // *************************************************
    // We can implement different variants of this function by implementing different parameter lists
    // or by using py::optional<py::array_t<double>> array_opt. 
    // Finally, I decided to implement this using different paramters lists, since this can improve code optimisation by the compiler. 
    // *************************************************

    int32_t next_quartet_pattern_frequencies(freq_array output)
    {
        // Get buffer info from the numpy array
        auto buf_freq = output.request(true);
        
        // Verify it's 2D
        if (buf_freq.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: Output array must be 2D with shape (batch_size, 256 or 160000).");
        }
        
        size_t rows_freq = buf_freq.shape[0];
        check_columns(buf_freq);
//        size_t cols = buf_freq.shape[1];
        
        // Get pointer to the array data (no copy!)
        float_type_freq* ptr_freq = static_cast<float_type_freq*>(buf_freq.ptr);
        
        // Fill the array in-place
        return myQPcounter.next_quartet_pattern_frequencies(ptr_freq, rows_freq);
    }


    int32_t next_quartet_pattern_frequencies_with_patterns(freq_array output, pattern_array patterns)
    {
        // Get buffer info from the numpy array
        auto buf_freq = output.request(true);
        auto buf_patt = patterns.request(true);
        
        // Verify it's 2D
        if (buf_freq.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: Output array must be 2D with shape (batch_size, 256 or 160000).");
        }
        if (buf_patt.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: pattern array must be 2D with shape (batch_size, 4)");
        }
        if (buf_patt.shape[1] != 4)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: pattern array must have shape (batch_size, 4)");
        }
          
        size_t rows_freq = buf_freq.shape[0];
        check_columns(buf_freq);

        if (rows_freq != (size_t)buf_patt.shape[0])
        {
           throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths: "
                                    "Pattern array has to have the same size as the frequency array in the first dimension.");
        }

//        size_t cols = buf_freq.shape[1];

        // Get pointer to the array data (no copy!)
        float_type_freq* ptr_freq = static_cast<float_type_freq*>(buf_freq.ptr);
        uint16_t*        ptr_patt = static_cast<uint16_t*>(buf_patt.ptr);

        // Fill the array in-place
        return myQPcounter.next_quartet_pattern_frequencies(ptr_freq, rows_freq, ptr_patt);
    }


    int32_t next_quartet_pattern_frequencies_with_lengths(freq_array frequencies, length_array lengths)
    {
        // Get buffer info from the numpy array
        auto buf_freq = frequencies.request(true);
        auto buf_len = lengths.request(true);
        
        // Verify it's 2D
        if (buf_freq.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: Output array must be 2D with shape (batch_size, 256 or 160000).");
        }
        if (buf_len.ndim != 1)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: lenth array must be 1D with shape (batch_size)");
        }
        
        size_t rows_freq = buf_freq.shape[0];
        check_columns(buf_freq);
//        size_t cols = buf_freq.shape[1];
        
        if (rows_freq != (size_t)buf_len.shape[0])
        {
           throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths: "
                                    "Lengths array has to have the same size as the frequency "
                                    "array in the first dimension.");
        }
        
        // Get pointer to the array data (no copy!)
        float_type_freq*  ptr_freq = static_cast<float_type_freq*>(buf_freq.ptr);
        uint32_t*         ptr_len  = static_cast<uint32_t*>(buf_len.ptr);
        
        // Fill the array in-place
        return myQPcounter.next_quartet_pattern_frequencies_with_lengths(ptr_freq, ptr_len, rows_freq);
    }

    int32_t next_quartet_pattern_frequencies_with_lengths_with_patterns(freq_array frequencies,
                                                                        length_array lengths,
                                                                        pattern_array patterns)
    {
        // Get buffer info from the numpy array
        auto buf_freq = frequencies.request(true);
        auto buf_len  = lengths.request(true);
        auto buf_patt = patterns.request(true);
        
        // Verify it's 2D
        if (buf_freq.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: Output array must be 2D with shape (batch_size, 256 or 160000).");
        }
        if (buf_patt.ndim != 2)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: pattern array must be 2D with shape (batch_size, 4)");
        }
        if (buf_patt.shape[1] != 4)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: pattern array must have shape (batch_size, 4)");
        }
        if (buf_len.ndim != 1)
        {
            throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies: lenth array must be 1D with shape (batch_size)");
        }
        
        size_t rows_freq = buf_freq.shape[0];
        check_columns(buf_freq);
//        size_t cols = buf_len.shape[1];
        
        if (rows_freq != (size_t)buf_len.shape[0])
        {
           throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths: "
                                    "Lengths array has to have the same size as the frequency array in the first dimension.");
        }

        if (rows_freq != (size_t)buf_patt.shape[0])
        {
           throw py::value_error("In PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths: "
                                    "Pattern array has to have the same size as the frequency array in the first dimension.");
        }
        
        // Get pointer to the array data (no copy!)
        float_type_freq*  ptr_freq = static_cast<float_type_freq*>(buf_freq.ptr);
        uint32_t*         ptr_len  = static_cast<uint32_t*>(buf_len.ptr);
        uint16_t*         ptr_patt = static_cast<uint16_t*>(buf_patt.ptr);
        
        // Fill the array in-place
        return myQPcounter.next_quartet_pattern_frequencies_with_lengths(ptr_freq, ptr_len, rows_freq, ptr_patt);
    }

    
    size_t get_num_cols() const {
        return myQPcounter.get_num_cols();
    }
    
    size_t get_call_count() const {
        return myQPcounter.get_call_count();
    }
    
    float_type_freq get_total_processing_time() const {
        return myQPcounter.get_total_processing_time();
    }
    
};


PYBIND11_MODULE(QuartetPatternCounterModule, m) {
    m.doc() = "QPC 1.7.1: batched quartet frequencies and effective lengths";
    m.attr("__version__") = "1.7.1";
    m.attr("AA_PATTERN_ORDER") = "ACDEFGHIKLMNPQRSTVWY";
    m.attr("DNA_PATTERN_ORDER") = "ACGT";
    
    // Expose the QuartetPatternCounter class
    py::class_<PyQuartetPatternCounter>(m, "QuartetPatternCounter")
        // Bind constructor for computing all quartet frequencies
        .def(py::init<std::string, char, int, int>(), 
             py::arg("filename"),
             py::arg("data_type"),
             py::arg("numThreads"),
             py::arg("verbose"),
             "Create a new QuartetPatternCounter Object for computing all quartet frequencies")

        // Bind constructor for computing four cluster quartet frequencies
        .def(py::init<std::string,
                     partition_array, partition_array,
                     partition_array, partition_array, char, int, int>(), 
             py::arg("filename"),
             py::arg("partition1"), py::arg("partition2"), py::arg("partition3"), py::arg("partition4"),
             py::arg("data_type"),
             py::arg("numThreads"),
             py::arg("verbose"),
             "Create a new QuartetPatternCounter Object for four cluster quartet frequencies")
  
        .def("next_quartet_pattern_frequencies", &PyQuartetPatternCounter::next_quartet_pattern_frequencies,
             "Fill a pre-allocated numpy array in-place with pattern frequencies (no copy)",
             py::arg("freq").noconvert())            
        .def("next_quartet_pattern_frequencies_with_patterns", &PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_patterns,
             "Fill a pre-allocated numpy array in-place with pattern frequencies (no copy)",
             py::arg("freq").noconvert(), py::arg("patterns").noconvert())

        .def("next_quartet_pattern_frequencies_with_lengths", &PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths,
             "Fill a pre-allocated numpy array in-place with pattern frequencies and effective lengths(no copy)",
             py::arg("freq").noconvert(), py::arg("lengths").noconvert())
        .def("next_quartet_pattern_frequencies_with_lengths_with_patterns", &PyQuartetPatternCounter::next_quartet_pattern_frequencies_with_lengths_with_patterns,
             "Fill a pre-allocated numpy array in-place with pattern frequencies and effective lengths(no copy)",
             py::arg("freq").noconvert(), py::arg("lengths").noconvert(), py::arg("patterns").noconvert())
        
        .def("get_num_cols", &PyQuartetPatternCounter::get_num_cols)
        .def("get_dataset_characteristics", &PyQuartetPatternCounter::get_dataset_characteristics,
             "Get number of sequences and sequnce lengths")
    
     
     
     
     
         
/* // OLD VERSION: Version using only functions, no classes.
        .def("load_file", &PyQuartetPatternCounter::load_file,
             "Load data from file",
             py::arg("filename"))
        .def("compute_file_stats", &PyQuartetPatternCounter::compute_file_stats,
             "Compute mean, min, max of loaded file")
        .def("process_data", &PyQuartetPatternCounter::process_data,
             "Process arrays with loaded file data",
             py::arg("arr1"),
             py::arg("arr2"),
             py::arg("arr3"),
             py::arg("arr4"))
        .def("set_num_cols", &PyQuartetPatternCounter::set_num_cols,
             "Set number of output columns",
             py::arg("num_cols"))
        .def("get_num_cols", &PyQuartetPatternCounter::get_num_cols,
             "Get number of output columns")
        .def("get_call_count", &PyQuartetPatternCounter::get_call_count,
             "Get number of process_data calls")
        .def("get_total_processing_time", &PyQuartetPatternCounter::get_total_processing_time,
             "Get total processing time in seconds")
        .def("reset", &PyQuartetPatternCounter::reset,
             "Reset all state")
        .def("process_data", &PyQuartetPatternCounter::process_data,
             "Process arrays with loaded file data",
             py::arg("arr1"),
             py::arg("arr2"),
             py::arg("arr3"),
             py::arg("arr4"))
        .def("assign_data", &PyQuartetPatternCounter::assign_data,
             "Fill a pre-allocated numpy array in-place (no copy)",
             py::arg("output"))
*/
         ;
             

}
