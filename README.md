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

This table uses `MINI_MP_ENABLE_NTT=0`. Division rows use a dividend of the
listed bit size and a divisor of half that bit size. `powm65537` uses a
same-size odd modulus and exponent 65537.

| Case | mini_mp | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit multiply | 1,825.7 | 26,438.6 | 0.069 |
| 4096-bit square | 3,134.3 | 26,122.9 | 0.120 |
| 4096-bit truncate divide+remainder | 2,070.0 | 5,200.0 | 0.398 |
| 4096-bit remainder | 1,350.0 | 5,185.0 | 0.260 |
| 4096-bit gcd | 104,955.6 | 247,511.1 | 0.424 |
| 4096-bit parse base 16 | 1,483.3 | 1,820.7 | 0.815 |
| 4096-bit format base 16 | 422.0 | 645.3 | 0.654 |
| 4096-bit powm65537 | 158,500.0 | 788,166.7 | 0.201 |
| 16384-bit multiply | 28,483.3 | 425,150.0 | 0.067 |
| 16384-bit square | 45,816.7 | 424,316.7 | 0.108 |
| 16384-bit truncate divide+remainder | 26,150.0 | 77,600.0 | 0.337 |
| 16384-bit remainder | 12,850.0 | 73,300.0 | 0.175 |
| 16384-bit gcd | 2,401,200.0 | 3,579,200.0 | 0.671 |
| 16384-bit parse base 16 | 6,084.4 | 12,002.2 | 0.507 |
| 16384-bit format base 16 | 1,573.3 | 2,313.3 | 0.680 |
| 16384-bit powm65537 | 2,400,000.0 | 12,545,500.0 | 0.191 |
| binomial(1000, 500) | 13,957.5 | 122,391.7 | 0.114 |
| factorial(1000) | 14,070.0 | 194,683.3 | 0.072 |

The 16384-bit cases are included because they exercise the larger-number
dispatch paths: multiplication, squaring, division, gcd, and base conversion.
With SIMD disabled, `mini_mp` is ahead on all listed workloads in this
environment.

## NTT Multiplication Addendum

The following numbers were measured with `MINI_MP_ENABLE_NTT=1` and SIMD still
disabled. They call the direct NTT backend after a warm-up multiply, so root
table setup is not included in the timed region. On this test machine the
runtime tuner leaves NTT out of the default dispatch path, but the direct NTT
backend becomes useful at larger sizes.

| Case | mini_mp direct NTT | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit multiply | 108,325.0 | 26,031.2 | 4.161 |
| 4096-bit square | 61,343.8 | 26,193.8 | 2.342 |
| 16384-bit multiply | 629,900.0 | 422,450.0 | 1.491 |
| 16384-bit square | 439,325.0 | 428,400.0 | 1.026 |
| 65536-bit multiply | 3,262,400.0 | 6,911,800.0 | 0.472 |
| 65536-bit square | 2,111,600.0 | 6,910,800.0 | 0.306 |
