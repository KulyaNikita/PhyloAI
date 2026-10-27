# ASTRAQ

This is the local version 0.5.8 source used in the empirical analyses, previously
stored as `QuADP-v0.5.8/QuADP.cpp`. It is distributed here as `ASTRAQ.cpp` and
compiled as `ASTRAQ`. Only the program name and example compile/run commands
have been renamed; the reconstruction algorithm is unchanged.

The installer compiles a C++17 executable without OpenMP and installs it into
the selected environment. Manual compilation:

```bash
c++ -std=c++17 -O3 ASTRAQ.cpp -o ASTRAQ
./ASTRAQ --help
./ASTRAQ -q quartets.txt -o species.tre
```

`CLI11.hpp` is the accompanying CLI11 header and retains its original license
notice. See `ASTRAQ --help` for supported quartet formats and weights.
