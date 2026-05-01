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
| 4096-bit multiply | 1,672.9 | 26,810.0 | 0.062 |
| 4096-bit square | 1,722.9 | 26,960.0 | 0.064 |
| 4096-bit truncate divide+remainder | 2,165.0 | 5,190.0 | 0.417 |
| 4096-bit remainder | 1,300.0 | 5,065.0 | 0.257 |
| 4096-bit gcd | 104,044.4 | 248,177.8 | 0.419 |
| 4096-bit parse base 16 | 1,614.0 | 1,884.7 | 0.856 |
| 4096-bit format base 16 | 428.7 | 625.3 | 0.686 |
| 4096-bit powm65537 | 153,416.7 | 791,166.7 | 0.194 |
| 16384-bit multiply | 28,650.0 | 418,483.3 | 0.068 |
| 16384-bit square | 28,050.0 | 420,366.7 | 0.067 |
| 16384-bit truncate divide+remainder | 25,250.0 | 72,800.0 | 0.347 |
| 16384-bit remainder | 12,850.0 | 72,550.0 | 0.177 |
| 16384-bit gcd | 2,302,200.0 | 3,545,800.0 | 0.649 |
| 16384-bit parse base 16 | 6,051.1 | 12,591.1 | 0.481 |
| 16384-bit format base 16 | 1,524.4 | 2,306.7 | 0.661 |
| 16384-bit powm65537 | 2,351,500.0 | 12,556,000.0 | 0.187 |
| binomial(1000, 500) | 13,732.5 | 120,558.3 | 0.114 |
| factorial(1000) | 13,965.0 | 193,675.0 | 0.072 |

The 16384-bit cases are included because they exercise the larger-number
dispatch paths: multiplication, squaring, division, gcd, and base conversion.
With SIMD disabled, `mini_mp` is ahead on all listed workloads in this
environment.

## NTT Multiplication Addendum

The following diagnostic numbers were measured with `MINI_MP_ENABLE_NTT=1` and
SIMD still disabled. They call the direct NTT backend after a warm-up multiply,
so root table setup is not included in the timed region. On this test machine
the runtime tuner leaves NTT out of the default dispatch path. The direct NTT
backend is not recommended at 4096 or 16384 bits here; it only becomes useful at
larger sizes.

| Case | mini_mp direct NTT | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit multiply | 108,325.0 | 26,031.2 | 4.161 |
| 4096-bit square | 61,343.8 | 26,193.8 | 2.342 |
| 16384-bit multiply | 629,900.0 | 422,450.0 | 1.491 |
| 16384-bit square | 439,325.0 | 428,400.0 | 1.026 |
| 65536-bit multiply | 3,262,400.0 | 6,911,800.0 | 0.472 |
| 65536-bit square | 2,111,600.0 | 6,910,800.0 | 0.306 |
