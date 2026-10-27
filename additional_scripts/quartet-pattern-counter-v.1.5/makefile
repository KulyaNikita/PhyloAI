# Makefile for building both C++ tests and Python module

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
#CXXFLAGS = -std=c++17 -g3 -ggdb -Wall -Wextra


# For C++ unit tests (simple macros - no dependencies)
#test: test_compute
#	./test_compute

#test_compute: test_compute.cpp QuartetPattern-counter-core.cpp QuartetPattern-counter-core.hpp
#	$(CXX) $(CXXFLAGS) -o test_compute test_compute.cpp QuartetPattern-counter-core.cpp

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
	rm -f test_QP-counter_catch2 *.o *.so *.pyc
	rm -rf build __pycache__ *.egg-info
	rm -f test_*.txt



.PHONY: test test-QP-counter_catch2 python clean all


