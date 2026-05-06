# mini_mp

`mini_mp` is a single-header C++20 multiple-precision integer and rational
library. It is portable C++ by default. SIMD is disabled unless
`MINI_MP_ENABLE_SIMD` is explicitly enabled, and the benchmark data below was
collected with SIMD and NTT disabled.

## Files

- `mini_mp.hpp`: the library header.
- `benchtofile.cpp`: runs full autotuning and writes a tuned starter source
  file containing the selected parameters.
- `README.md`: usage notes and current benchmark data.

## Build

Use a C++20 compiler:

```text
g++ -O3 -std=c++20 -DNDEBUG -DMINI_MP_ENABLE_SIMD=0 -DMINI_MP_ENABLE_NTT=0 app.cpp -o app
```

Useful compile-time options:

```text
MINI_MP_ENABLE_AUTOTUNE=0  disable runtime tuning
MINI_MP_ENABLE_SIMD=0      keep scalar code only
MINI_MP_ENABLE_NTT=0       disable the NTT backend
MINI_MP_NO_EXCEPT          use assert/abort style error handling
MINI_MP_INLINE_LIMBS=N     set the small inline limb capacity
```

## Autotuning

Autotuning is enabled by default. The default mode is fast tuning, intended to
finish well under one second while still selecting good thresholds for the
current machine.

Runtime controls:

```text
MINI_MP_AUTOTUNE=fast      fast tuning, the default
MINI_MP_AUTOTUNE=full      wider tuning for best benchmark results
MINI_MP_AUTOTUNE=off       skip runtime tuning
```

Code controls:

```cpp
#include "mini_mp.hpp"

int main(){
	mini_mp::autotune_fast();
	mini_mp::autotune_full();
}
```

`autotune_full()` can upgrade a process that already ran fast tuning. It is
intended for benchmark runs, long-running services, and generating pinned
parameter templates.

## BigInt Usage

```cpp
#include "mini_mp.hpp"
#include <iostream>

int main(){
	using mini_mp::BigInt;

	BigInt a=BigInt::parse("12345678901234567890",10);
	BigInt b=BigInt::parse("fedcba9876543210",16);
	BigInt c=(a*b)+(a<<17)-b;

	std::cout<<c.to_string(10)<<"\n";
	std::cout<<c.to_string(16)<<"\n";
}
```

Construction and conversion:

```cpp
mini_mp::BigInt a;
mini_mp::BigInt b(123);
auto c=mini_mp::BigInt::from_u64(123456789);
auto d=mini_mp::BigInt::parse("-101010",2);
std::string s=d.to_string(10);
std::uint64_t small=c.to_u64();
```

Integer operators:

```cpp
a+b;
a-b;
a*b;
a/b;
a%b;
a<<128;
a>>64;
a&b;
a|b;
a^b;
~a;
```

Integer helpers:

```cpp
a.is_zero();
a.is_one();
a.is_neg();
a.is_even();
a.is_odd();
a.bit_length();
a.ctz();
a.mod_u32(1000000007u);
```

Number theory and exact operations:

```cpp
using mini_mp::BigInt;

BigInt g=mini_mp::gcd(a,b);
BigInt l=mini_mp::lcm(a,b);
BigInt p=mini_mp::pow(a,20);
BigInt m=mini_mp::modpow(a,BigInt::from_u64(65537),mod);
BigInt q=mini_mp::divexact(a*b,a);
BigInt inv=mini_mp::invert(a,mod);
bool ok=mini_mp::invert(&inv,a,mod);
BigInt r=mini_mp::isqrt(a);
BigInt k=mini_mp::iroot(a,3);
bool sq=mini_mp::is_square(a);
bool pp=mini_mp::is_ppow(a);
int prime=mini_mp::pr_prime(a,25);
BigInt np=mini_mp::next_prime(a,25);
BigInt f=mini_mp::factorial(1000);
BigInt bin=mini_mp::binomial(1000,500);
```

Division variants:

```cpp
auto tr=mini_mp::tdiv_qr(a,b);
auto fr=mini_mp::fdiv_qr(a,b);
auto cr=mini_mp::cdiv_qr(a,b);

BigInt tq=mini_mp::tdiv_q(a,b);
BigInt trm=mini_mp::tdiv_r(a,b);
BigInt fq=mini_mp::fdiv_q(a,b);
BigInt frm=mini_mp::fdiv_r(a,b);
BigInt cq=mini_mp::cdiv_q(a,b);
BigInt crm=mini_mp::cdiv_r(a,b);
```

`tdiv` truncates toward zero, `fdiv` floors, and `cdiv` ceilings.

## BigRat Usage

```cpp
#include "mini_mp.hpp"
#include <iostream>

int main(){
	using mini_mp::BigRat;

	BigRat a=BigRat::parse("355/113");
	BigRat b=BigRat::parse("1.25");
	BigRat c=(a+b)*BigRat(3);

	std::cout<<c.to_string(10)<<"\n";
}
```

`BigRat` supports `+`, `-`, `*`, `/`, comparisons, parsing from integer,
decimal, and `num/den` forms, and normalized output with `to_string(base)`.

## benchtofile

`benchtofile.cpp` runs full tuning, then writes a standalone starter source
file with the measured parameter values pinned in code. This is useful when
startup time matters or when a deployment target should use a known tuning
profile.

Build:

```text
g++ -O3 -std=c++20 -DNDEBUG -DMINI_MP_ENABLE_SIMD=0 -DMINI_MP_ENABLE_NTT=0 benchtofile.cpp -o benchtofile
```

Run:

```text
benchtofile tuned_app.cpp
```

If no output path is given, it writes `mini_mp_tuned_template.cpp`.

The generated file contains:

- `app_mp::tune_once()`: installs the measured thresholds once.
- `app_mp::BigInt` and `app_mp::BigRat` aliases.
- `app_mp::parse()`: a small parsing wrapper that calls `tune_once()`.
- A minimal `main()` showing multiplication and decimal output.

Build the generated file as a normal C++20 program:

```text
g++ -O3 -std=c++20 -DNDEBUG -DMINI_MP_ENABLE_SIMD=0 -DMINI_MP_ENABLE_NTT=0 tuned_app.cpp -o tuned_app
```

## Benchmark Notes

The numbers below compare `mini_mp` with `mini-gmp` from the GMP 6.3.0 source
tree. SIMD was disabled for `mini_mp`, NTT was disabled, and compiler
auto-vectorization was disabled for both builds.

Build flags used for the benchmark:

```text
-O3 -fno-tree-vectorize -DNDEBUG -DMINI_MP_ENABLE_SIMD=0 -DMINI_MP_ENABLE_NTT=0
```

Test machine:

```text
CPU: Intel(R) Core(TM) Ultra 7 265K class, 20 logical processors
OS: Microsoft Windows NT 10.0.26300.0
Compiler: MinGW g++ 15.2.0, x86_64-win32-seh
mini_mp limb size: 64 bits
mini-gmp limb size in this MinGW build: 32 bits
```

Times are nanoseconds per operation. Lower is better. Ratio is
`mini_mp / mini-gmp`.

## Fast Autotune Benchmark

Fast tuning completed in about 65 ms on the test machine.

Tuned parameters:

```text
kar=48 krec=48 srec=80 kdif=96 gcd_sm=256 gcd_lg=96
```

| Case | mini_mp | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit parse base 10 | 1,243 | 16,043 | 0.077 |
| 4096-bit format base 10 | 14,083 | 840,750 | 0.017 |
| 4096-bit add | 50 | 91 | 0.549 |
| 4096-bit sub | 50 | 100 | 0.500 |
| 4096-bit multiply | 1,743 | 26,057 | 0.067 |
| 4096-bit square | 1,000 | 27,214 | 0.037 |
| 4096-bit tdiv_qr | 2,000 | 5,833 | 0.343 |
| 4096-bit gcd | 80,583 | 253,250 | 0.318 |
| 4096-bit powm65537 | 68,917 | 809,583 | 0.085 |
| 16384-bit parse base 10 | 13,600 | 227,760 | 0.060 |
| 16384-bit format base 10 | 154,000 | 12,411,800 | 0.012 |
| 16384-bit add | 133 | 311 | 0.428 |
| 16384-bit sub | 122 | 411 | 0.297 |
| 16384-bit multiply | 20,240 | 435,520 | 0.046 |
| 16384-bit square | 10,680 | 422,680 | 0.025 |
| 16384-bit tdiv_qr | 24,600 | 75,200 | 0.327 |
| 16384-bit gcd | 1,136,600 | 3,537,400 | 0.321 |
| 16384-bit powm65537 | 895,000 | 12,515,800 | 0.072 |
| 32768-bit parse base 10 | 51,300 | 876,800 | 0.059 |
| 32768-bit format base 10 | 442,500 | 48,273,500 | 0.009 |
| 32768-bit add | 311 | 622 | 0.500 |
| 32768-bit sub | 289 | 822 | 0.352 |
| 32768-bit multiply | 45,800 | 1,739,500 | 0.026 |
| 32768-bit square | 33,800 | 1,701,400 | 0.020 |
| 32768-bit tdiv_qr | 93,500 | 288,000 | 0.325 |
| 32768-bit gcd | 7,222,000 | 13,875,500 | 0.520 |
| 32768-bit powm65537 | 3,401,000 | 50,414,000 | 0.067 |

## Full Autotune Benchmark

Full tuning completed in about 6.9 seconds on the test machine.

Tuned parameters:

```text
kar=128 krec=40 srec=96 kdif=128 gcd_sm=256 gcd_lg=192
```

| Case | mini_mp | mini-gmp | Ratio |
| --- | ---: | ---: | ---: |
| 4096-bit parse base 10 | 1,200 | 15,357 | 0.078 |
| 4096-bit format base 10 | 14,167 | 809,333 | 0.018 |
| 4096-bit add | 50 | 91 | 0.549 |
| 4096-bit sub | 50 | 105 | 0.476 |
| 4096-bit multiply | 1,843 | 26,800 | 0.069 |
| 4096-bit square | 1,071 | 25,943 | 0.041 |
| 4096-bit tdiv_qr | 2,333 | 5,583 | 0.418 |
| 4096-bit gcd | 78,083 | 248,667 | 0.314 |
| 4096-bit powm65537 | 68,167 | 810,250 | 0.084 |
| 16384-bit parse base 10 | 13,640 | 224,560 | 0.061 |
| 16384-bit format base 10 | 128,000 | 12,388,000 | 0.010 |
| 16384-bit add | 133 | 311 | 0.428 |
| 16384-bit sub | 300 | 1,111 | 0.270 |
| 16384-bit multiply | 14,160 | 424,880 | 0.033 |
| 16384-bit square | 10,600 | 432,440 | 0.025 |
| 16384-bit tdiv_qr | 25,400 | 75,800 | 0.335 |
| 16384-bit gcd | 1,147,400 | 3,545,000 | 0.324 |
| 16384-bit powm65537 | 899,000 | 12,620,200 | 0.071 |
| 32768-bit parse base 10 | 51,500 | 882,800 | 0.058 |
| 32768-bit format base 10 | 378,500 | 48,345,500 | 0.008 |
| 32768-bit add | 244 | 644 | 0.379 |
| 32768-bit sub | 244 | 822 | 0.297 |
| 32768-bit multiply | 45,800 | 1,710,700 | 0.027 |
| 32768-bit square | 31,500 | 1,706,800 | 0.018 |
| 32768-bit tdiv_qr | 94,500 | 285,500 | 0.331 |
| 32768-bit gcd | 6,968,500 | 13,827,500 | 0.504 |
| 32768-bit powm65537 | 3,362,500 | 50,216,000 | 0.067 |
