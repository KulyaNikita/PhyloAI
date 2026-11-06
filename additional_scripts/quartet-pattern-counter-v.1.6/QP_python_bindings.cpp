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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "QuartetPattern-counter-core.hpp"

namespace py = pybind11;
using Partition = py::array_t<int, py::array::c_style | py::array::forcecast>;

class PyQuartetPatternCounter {
    QuartetPatternCounter counter;

    template <typename T>
    static void check_output(const py::array &array, const char *name) {
        if (!array.dtype().is(py::dtype::of<T>()))
            throw py::type_error(std::string(name) + " has the wrong dtype");
        if (!(array.flags() & py::array::c_style))
            throw py::value_error(std::string(name) + " must be C-contiguous");
        if (!array.writeable())
            throw py::value_error(std::string(name) + " must be writable");
    }

    size_t check_frequencies(const py::array &freq) const {
        check_output<float_type_freq>(freq, "freq");
        if (freq.ndim() != 2 || freq.shape(1) != counter.get_num_cols())
            throw py::value_error("freq must have shape (batch_size, " +
                                  std::to_string(counter.get_num_cols()) + ")");
        if (freq.shape(0) == 0)
            throw py::value_error("batch_size must be positive");
        return freq.shape(0);
    }

    static void check_lengths(const py::array &lengths, size_t rows) {
        check_output<uint32_t>(lengths, "lengths");
        if (lengths.ndim() != 1 || static_cast<size_t>(lengths.shape(0)) != rows)
            throw py::value_error("lengths must have shape (batch_size,)");
    }

    static void check_patterns(const py::array &patterns, size_t rows) {
        check_output<uint16_t>(patterns, "patterns");
        if (patterns.ndim() != 2 || static_cast<size_t>(patterns.shape(0)) != rows ||
            patterns.shape(1) != 4)
            throw py::value_error("patterns must have shape (batch_size, 4)");
    }

    static void check_config(char type, int threads) {
        if (type != 'D' && type != 'P')
            throw py::value_error("data_type must be 'D' (DNA) or 'P' (protein)");
        if (threads < 1)
            throw py::value_error("numThreads must be positive");
    }

    static QuartetPatternCounter make_all(const std::string &file, char type,
                                          int threads, int verbose) {
        check_config(type, threads);
        return QuartetPatternCounter(file, type, threads, verbose);
    }

    static QuartetPatternCounter make_clusters(const std::string &file,
            Partition p1, Partition p2, Partition p3, Partition p4,
            char type, int threads, int verbose) {
        check_config(type, threads);
        for (const auto &part : {p1, p2, p3, p4}) {
            if (part.ndim() != 1)
                throw py::value_error("Each partition must be one-dimensional");
        }
        return QuartetPatternCounter(file, p1.data(), p1.size(), p2.data(), p2.size(),
                                     p3.data(), p3.size(), p4.data(), p4.size(),
                                     type, threads, verbose);
    }

public:
    PyQuartetPatternCounter(const std::string &file, char type, int threads, int verbose)
        : counter(make_all(file, type, threads, verbose)) {}

    PyQuartetPatternCounter(const std::string &file,
            Partition p1, Partition p2, Partition p3, Partition p4,
            char type, int threads, int verbose)
        : counter(make_clusters(file, p1, p2, p3, p4, type, threads, verbose)) {}

    size_t frequencies(py::array freq) {
        auto rows = check_frequencies(freq);
        return counter.next_quartet_pattern_frequencies(
            static_cast<float_type_freq *>(freq.mutable_data()), rows);
    }
    size_t frequencies_patterns(py::array freq, py::array patterns) {
        auto rows = check_frequencies(freq);
        check_patterns(patterns, rows);
        return counter.next_quartet_pattern_frequencies(
            static_cast<float_type_freq *>(freq.mutable_data()), rows,
            static_cast<uint16_t *>(patterns.mutable_data()));
    }
    size_t frequencies_lengths(py::array freq, py::array lengths) {
        auto rows = check_frequencies(freq);
        check_lengths(lengths, rows);
        return counter.next_quartet_pattern_frequencies_with_lengths(
            static_cast<float_type_freq *>(freq.mutable_data()),
            static_cast<uint32_t *>(lengths.mutable_data()), rows);
    }
    size_t frequencies_lengths_patterns(py::array freq, py::array lengths,
                                       py::array patterns) {
        auto rows = check_frequencies(freq);
        check_lengths(lengths, rows);
        check_patterns(patterns, rows);
        return counter.next_quartet_pattern_frequencies_with_lengths(
            static_cast<float_type_freq *>(freq.mutable_data()),
            static_cast<uint32_t *>(lengths.mutable_data()), rows,
            static_cast<uint16_t *>(patterns.mutable_data()));
    }
    auto characteristics() const { return counter.get_dataset_characteristics(); }
    size_t columns() const { return counter.get_num_cols(); }
};

PYBIND11_MODULE(QuartetPatternCounterModule, m) {
    m.doc() = "QPC 1.6: batched quartet frequencies and effective lengths";
    m.attr("__version__") = "1.6.1";
    m.attr("AA_PATTERN_ORDER") = "ACDEFGHIKLMNPQRSTVWY";
    m.attr("DNA_PATTERN_ORDER") = "ACGT";
    py::class_<PyQuartetPatternCounter>(m, "QuartetPatternCounter")
        .def(py::init<const std::string &, char, int, int>(),
             py::arg("filename"), py::arg("data_type"),
             py::arg("numThreads"), py::arg("verbose"))
        .def(py::init<const std::string &, Partition, Partition, Partition,
                      Partition, char, int, int>(),
             py::arg("filename"), py::arg("partition1"), py::arg("partition2"),
             py::arg("partition3"), py::arg("partition4"), py::arg("data_type"),
             py::arg("numThreads"), py::arg("verbose"))
        .def("next_quartet_pattern_frequencies", &PyQuartetPatternCounter::frequencies,
             py::arg("freq").noconvert())
        .def("next_quartet_pattern_frequencies_with_patterns",
             &PyQuartetPatternCounter::frequencies_patterns,
             py::arg("freq").noconvert(), py::arg("patterns").noconvert())
        .def("next_quartet_pattern_frequencies_with_lengths",
             &PyQuartetPatternCounter::frequencies_lengths,
             py::arg("freq").noconvert(), py::arg("lengths").noconvert())
        .def("next_quartet_pattern_frequencies_with_lengths_with_patterns",
             &PyQuartetPatternCounter::frequencies_lengths_patterns,
             py::arg("freq").noconvert(), py::arg("lengths").noconvert(),
             py::arg("patterns").noconvert())
        .def("get_dataset_characteristics", &PyQuartetPatternCounter::characteristics)
        .def("get_num_cols", &PyQuartetPatternCounter::columns);
}
