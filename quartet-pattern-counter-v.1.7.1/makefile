# Makefile for building both C++ tests and Python module

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
#CXXFLAGS = -std=c++17 -g3 -ggdb -Wall -Wextra


# For C++ unit tests (simple macros - no dependencies)
#test: test_compute
#	./test_compute

#test_compute: test_compute.cpp QuartetPattern-counter-core.cpp QuartetPattern-counter-core.hpp
#	$(CXX) $(CXXFLAGS) -o test_compute test_compute.cpp QuartetPattern-counter-core.cpp

# Default target: the stand-alone program. Needs only a C++17 compiler (no pybind11, no Catch2).
# Same command line and output as quartet-pattern-counter v1.1: [-p] infile.fas outfile.npy
standalone: quartet-pattern-counter

quartet-pattern-counter: quartet-pattern-counter.cpp QuartetPattern-counter-core.cpp QuartetPattern-counter-core.hpp basic-DNA-RNA-AA-routines.h cartesian-product-of-4-generator-class.h
	$(CXX) $(CXXFLAGS) -pthread -o quartet-pattern-counter quartet-pattern-counter.cpp QuartetPattern-counter-core.cpp

# Run both tests and build Python module
all: python run_test-QP-counter_catch2

# For Catch2 tests (requires Catch2 installed)
run_test-QP-counter_catch2: test_QP-counter_catch2
	@echo "*****************************"
	@echo "** Run test_compute_catch2 **"
	@echo "*****************************"
#	./test_QP-counter_catch2 -s
	./test_QP-counter_catch2

test_QP-counter_catch2: test-QuartetPattern-counter_catch2.cpp QuartetPattern-counter-core.cpp QuartetPattern-counter-core.hpp
	$(CXX) $(CXXFLAGS) -o test_QP-counter_catch2 test-QuartetPattern-counter_catch2.cpp QuartetPattern-counter-core.cpp

# For Python module (requires pybind11)
python: 
	python3 setup.py build_ext --inplace

# Clean build artifacts
clean:
	rm -f quartet-pattern-counter test_QP-counter_catch2 *.o *.so *.pyc
	rm -rf build __pycache__ *.egg-info
	rm -f test_*.txt



.PHONY: standalone test test-QP-counter_catch2 python clean all


