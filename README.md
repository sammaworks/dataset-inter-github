Objective. Benchmark dense matrix multiplication computed as repeated row–column dot products C=A⋅BC=A\cdot BC=A⋅B across languages and execution modes.
Implementations (per sheet).
Python: Sequential; “Threaded” (threads used, compute kernel unchanged).
Rust: Sequential; Parallel; AVX; AVX Parallel.
C# (.NET): Sequential; Parallel; AVX; AVX Parallel.
Environment. Runs were executed on multiple similar-spec x86-64 hosts matching the profile shown: 12th-Gen Intel® Core™ i7-1255U, 10 cores / 12 logical processors, base 1.70 GHz, single socket; caches L1 = 928 KB, L2 = 6.5 MB, L3 = 12.0 MB; 16 GB RAM; NVMe SSD; integrated Intel UHD Graphics; virtualization enabled. (The Task Manager screenshot indicates Windows OS.) AVX support is required for the AVX variants.
Workload & Sizes. Dense real-valued N×NN\times NN×N matrices in contiguous storage; N∈{500,1000,…,5000}N \in \{500,1000,\ldots,5000\}N∈{500,1000,…,5000}. Each element of CCC is computed as the dot product of one row of AAA and one column of BBB.
Trial Design. 180 trials for every (method, size). In sequential modes, all 180 trials run serially. In parallelized modes, the same 180 trials are divided among the available CPU workers on the executing host (i.e., by logical processor count), keeping total work constant while wall-clock time reflects concurrency.
Measurement & Logging. Each trial’s wall-clock time (ms) is recorded into the spreadsheet columns shown and plotted against NNN; a secondary axis is used where needed to place slower and faster curves on the same chart.

