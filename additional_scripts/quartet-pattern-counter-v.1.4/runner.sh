### Print pattern test order --- Test done
# ./quartet-pattern-counter-v1.4 -O > order_test.txt

### Simple test: 2 patterns only --- Test done
time ./quartet-pattern-counter-v1.4    in4.fas out_in4_v1.4.npy
time ./quartet-pattern-counter-v1.4 -p in4.fas out_in4_v1.4-p.npy

### Big quartet test: 100m sites
time ./quartet-pattern-counter-v1.4    /Daten/Programmieren/C++/Programme/quartet_pattern_counter/simulated-trees/sim-sequences_100m.fas out_im-sequences_100m_v1.4.npy
time ./quartet-pattern-counter-v1.4 -p /Daten/Programmieren/C++/Programme/quartet_pattern_counter/simulated-trees/sim-sequences_100m.fas out_im-sequences_100m_v1.4-p.npy

### 20 Taxa
time ./quartet-pattern-counter-v1.4 -t 4    /Daten/Programmieren/C++/Programme/quartet_pattern_counter/simulated-trees/sim-sequences_20taxa.fas out_sim-sequences_20taxa_v1.4.npy
time ./quartet-pattern-counter-v1.4 -t 4 -p /Daten/Programmieren/C++/Programme/quartet_pattern_counter/simulated-trees/sim-sequences_20taxa.fas out_sim-sequences_20taxa_v1.4-p.npy
