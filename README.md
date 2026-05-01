# mini_mp

`mini_mp` is a single-header C++20 multiple-precision integer implementation.
The implementation is scalar by default: SIMD code paths are disabled unless
`MINI_MP_ENABLE_SIMD` is explicitly enabled.

## Benchmark Notes

The numbers below compare `mini_mp` with `mini-gmp` from the GMP 6.3.0 source
tree. SIMD was disabled for `mini_mp`, and compiler auto-vectorization was also
disabled for both builds.

Build flags used for the benchmark:

```text
-O3 -fno-tree-vectorize -DNDEBUG -DMINI_MP_ENABLE_SIMD=0
```

Test machine:

```text
CPU: Intel(R) Core(TM) Ultra 7 265K, 20 logical processors
OS: Microsoft Windows NT 10.0.26300.0
Compiler: MinGW g++ 15.2.0, x86_64-win32-seh
mini_mp limb size: 64 bits
mini-gmp limb size in this MinGW build: 32 bits
```

Times are nanoseconds per operation. Lower is better. The ratio is
`mini_mp / mini-gmp`.

## Scalar Benchmark Results

| Case | mini_mp | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit multiply | 1,713.7 | 26,278.7 | 0.065 |
| 4096-bit square | 2,592.4 | 26,107.8 | 0.099 |
| 4096-bit truncate divide+remainder | 2,008.0 | 5,168.5 | 0.389 |
| 4096-bit remainder | 1,227.5 | 5,109.0 | 0.240 |
| 4096-bit gcd | 107,202.0 | 250,849.0 | 0.427 |
| 4096-bit parse base 16 | 1,521.4 | 1,773.2 | 0.858 |
| 4096-bit format base 16 | 422.2 | 531.0 | 0.795 |
| 16384-bit multiply | 28,690.0 | 425,533.3 | 0.067 |
| 16384-bit square | 38,826.7 | 423,106.7 | 0.092 |
| 16384-bit truncate divide+remainder | 24,393.3 | 73,240.0 | 0.333 |
| 16384-bit remainder | 13,150.0 | 73,666.7 | 0.179 |
| 16384-bit gcd | 2,339,025.0 | 3,584,837.5 | 0.652 |
| 16384-bit parse base 16 | 6,088.6 | 12,019.2 | 0.507 |
| 16384-bit format base 16 | 1,604.4 | 2,134.0 | 0.752 |
| binomial(1000, 500) | 14,554.0 | 115,128.0 | 0.126 |
| factorial(1000) | 14,446.0 | 201,726.0 | 0.072 |

The 16384-bit cases are included because they exercise the larger-number
dispatch paths: multiplication, squaring, division, gcd, and base conversion.
With SIMD disabled, `mini_mp` is ahead on all listed workloads in this
environment.
