PYTHON ?= python3

.PHONY: all python test
all: python

python:
	$(PYTHON) -m pip install --no-deps --no-build-isolation .

test:
	$(PYTHON) -m unittest discover -s tests -v
