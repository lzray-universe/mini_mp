// SPDX-License-Identifier: MIT
/*
MIT License

Copyright (c) 2026 lzray-universe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MINI_MP_HPP_INCLUDED
#define MINI_MP_HPP_INCLUDED
#pragma once

/*
mini_mp.hpp

Single-file C++20 mini big integer / rational framework.

Goals:
- Header-only style.
- No assembly, standard library only.
- Extension-ready architecture for schoolbook -> Karatsuba -> NTT.

Usage:
  #include "mini_mp.hpp"
  using mini_mp::BigInt;
  using mini_mp::BigRat;

Optional knobs:
- MINI_MP_NO_EXCEPT            : disable exceptions (abort/assert fallback).
- MINI_MP_ENABLE_NTT           : default 0. When 1, compile NTT path.
- MINI_MP_ENABLE_AUTOTUNE      : default 1. One-time runtime micro tuning.
- MINI_MP_BENCH_MAIN           : emit benchmark main() at end of file.
- MINI_MP_IMPLEMENTATION       : kept for stb-style compatibility (default 1).

*/

#include<algorithm>
#include<array>
#include<atomic>
#include<bit>
#include<cctype>
#include<chrono>
#include<cmath>
#include<cstddef>
#include<cstdint>
#include<cstdlib>
#include<cstring>
#include<initializer_list>
#include<iterator>
#include<limits>
#include<memory>
#include<mutex>
#include<numeric>
#include<random>
#include<stdexcept>
#include<string>
#include<string_view>
#include<tuple>
#include<type_traits>
#include<utility>
#include<vector>

#if (defined(__x86_64__)||defined(__i386__)||defined(_M_X64)||                 \
	 defined(_M_IX86))&&                                                         \
	(defined(__GNUC__)||defined(__clang__)||defined(_MSC_VER))
#include<immintrin.h>
#endif

#ifndef MINI_MP_IMPLEMENTATION
#define MINI_MP_IMPLEMENTATION 1
#endif

#ifndef MINI_MP_ENABLE_NTT
#define MINI_MP_ENABLE_NTT 0
#endif

#ifndef MINI_MP_ENABLE_AUTOTUNE
#define MINI_MP_ENABLE_AUTOTUNE 1
#endif

#ifndef MINI_MP_ENABLE_SIMD
#define MINI_MP_ENABLE_SIMD 0
#endif

#ifndef MINI_MP_INLINE_LIMBS
#define MINI_MP_INLINE_LIMBS 8
#endif

#ifndef MINI_MP_ASSERT
#include<cassert>
#define MINI_MP_ASSERT(expr) assert(expr)
#endif

namespace mini_mp{

using limb_t=std::uint64_t;
static_assert(sizeof(limb_t)==8,"mini_mp requires 64-bit limbs.");

#if defined(_MSC_VER)&&defined(_M_X64)
#define MINI_MP_DETAIL_USE_MSVC_INTRIN 1
#include<intrin.h>
#else
#define MINI_MP_DETAIL_USE_MSVC_INTRIN 0
#endif

#ifndef MINI_MP_FORCE_NO_INT128
#define MINI_MP_FORCE_NO_INT128 0
#endif

#ifndef MINI_MP_DETAIL_HAS_UINT128
#if !MINI_MP_FORCE_NO_INT128&&defined(__SIZEOF_INT128__)&&!defined(_MSC_VER)
#define MINI_MP_DETAIL_HAS_UINT128 1
#else
#define MINI_MP_DETAIL_HAS_UINT128 0
#endif
#endif

#if defined(__x86_64__)&&(defined(__GNUC__)||defined(__clang__))&&             \
	!defined(_MSC_VER)
#define MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY 1
#else
#define MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY 0
#endif

namespace detail{

[[noreturn]] inline void throw_dom(const char*msg){
#ifdef MINI_MP_NO_EXCEPT
	(void)msg;
	MINI_MP_ASSERT(false&&"mini_mp domain_error");
	std::abort();
#else
	throw std::domain_error(msg);
#endif
}

[[noreturn]] inline void throw_inv(const char*msg){
#ifdef MINI_MP_NO_EXCEPT
	(void)msg;
	MINI_MP_ASSERT(false&&"mini_mp invalid_argument");
	std::abort();
#else
	throw std::invalid_argument(msg);
#endif
}

[[noreturn]] inline void throw_ovf(const char*msg){
#ifdef MINI_MP_NO_EXCEPT
	(void)msg;
	MINI_MP_ASSERT(false&&"mini_mp overflow_error");
	std::abort();
#else
	throw std::overflow_error(msg);
#endif
}

inline std::uint64_t bswap64(std::uint64_t x) noexcept{
#if defined(_MSC_VER)
	return _byteswap_uint64(x);
#elif defined(__GNUC__)||defined(__clang__)
	return __builtin_bswap64(x);
#else
	return ((x&0x00000000000000ffULL)<<56)|
		   ((x&0x000000000000ff00ULL)<<40)|
		   ((x&0x0000000000ff0000ULL)<<24)|
		   ((x&0x00000000ff000000ULL)<<8)|
		   ((x&0x000000ff00000000ULL)>>8)|
		   ((x&0x0000ff0000000000ULL)>>24)|
		   ((x&0x00ff000000000000ULL)>>40)|
		   ((x&0xff00000000000000ULL)>>56);
#endif
}

namespace vecab{

enum class vid : std::uint32_t{
	scal=0,
	x86=1,
	arm=2,
	riscv=3,
	power=4,
	mips=5,
	other=6,
};

#if defined(__x86_64__)||defined(__i386__)||defined(_M_X64)||defined(_M_IX86)
inline constexpr vid kId=vid::x86;
#define MINI_MP_VEC_X86 1
#elif defined(__aarch64__)||defined(__arm__)||defined(_M_ARM64)||defined(_M_ARM)
inline constexpr vid kId=vid::arm;
#define MINI_MP_VEC_X86 0
#elif defined(__riscv)
inline constexpr vid kId=vid::riscv;
#define MINI_MP_VEC_X86 0
#elif defined(__powerpc64__)||defined(__powerpc__)||defined(_M_PPC)
inline constexpr vid kId=vid::power;
#define MINI_MP_VEC_X86 0
#elif defined(__mips__)||defined(__mips64)
inline constexpr vid kId=vid::mips;
#define MINI_MP_VEC_X86 0
#else
inline constexpr vid kId=vid::other;
#define MINI_MP_VEC_X86 0
#endif

#if MINI_MP_ENABLE_SIMD&&defined(__AVX512F)
#define MINI_MP_VEC_W8 1
#else
#define MINI_MP_VEC_W8 0
#endif

#if MINI_MP_ENABLE_SIMD&&!MINI_MP_VEC_W8&&defined(__AVX2__)
#define MINI_MP_VEC_W4 1
#else
#define MINI_MP_VEC_W4 0
#endif

#if MINI_MP_ENABLE_SIMD&&!MINI_MP_VEC_W8&&!MINI_MP_VEC_W4&&                 \
	(defined(__SSE2__)||defined(_M_X64)||defined(_M_IX86_FP)||              \
	 defined(__ARM_NEON)||defined(__ARM_NEON__)||defined(__aarch64__)||     \
	 defined(__riscv_vector)||defined(__ALTIVEC__)||defined(__VSX__)||      \
	 defined(__mips_msa))
#define MINI_MP_VEC_W2 1
#else
#define MINI_MP_VEC_W2 0
#endif

inline constexpr std::size_t kW=
	MINI_MP_VEC_W8?std::size_t(8)
				 : (MINI_MP_VEC_W4?std::size_t(4)
								  : (MINI_MP_VEC_W2?std::size_t(2)
													:std::size_t(1)));

#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
using v8u64=std::uint64_t __attribute__((vector_size(64)));
inline v8u64 ld8(const std::uint64_t*p) noexcept{
	v8u64 v{};
	std::memcpy(&v,p,sizeof(v));
	return v;
}
inline void st8(std::uint64_t*p,v8u64 v) noexcept{
	std::memcpy(p,&v,sizeof(v));
}
#endif

#if MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
using v4u64=std::uint64_t __attribute__((vector_size(32)));
inline v4u64 ld4(const std::uint64_t*p) noexcept{
	v4u64 v{};
	std::memcpy(&v,p,sizeof(v));
	return v;
}
inline void st4(std::uint64_t*p,v4u64 v) noexcept{
	std::memcpy(p,&v,sizeof(v));
}
#endif

#if MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
using v2u64=std::uint64_t __attribute__((vector_size(16)));
inline v2u64 ld2(const std::uint64_t*p) noexcept{
	v2u64 v{};
	std::memcpy(&v,p,sizeof(v));
	return v;
}
inline void st2(std::uint64_t*p,v2u64 v) noexcept{
	std::memcpy(p,&v,sizeof(v));
}
#endif

inline bool eq_n(const std::uint64_t*a,const std::uint64_t*b,
				 std::size_t n) noexcept{
	std::size_t i=0;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const v8u64 d=ld8(a+i)^ld8(b+i);
		if((d[0]|d[1]|d[2]|d[3]|d[4]|d[5]|d[6]|d[7])!=0)
			return false;
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const __m512i va=
			_mm512_loadu_si512(reinterpret_cast<const void*>(a+i));
		const __m512i vb=
			_mm512_loadu_si512(reinterpret_cast<const void*>(b+i));
		const __m512i vd=_mm512_xor_si512(va,vb);
		alignas(64) std::uint64_t t[8];
		_mm512_storeu_si512(reinterpret_cast<void*>(t),vd);
		if((t[0]|t[1]|t[2]|t[3]|t[4]|t[5]|t[6]|t[7])!=0)
			return false;
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const v4u64 d=ld4(a+i)^ld4(b+i);
		if((d[0]|d[1]|d[2]|d[3])!=0)
			return false;
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const __m256i va=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
		const __m256i vb=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
		const __m256i vd=_mm256_xor_si256(va,vb);
		if(_mm256_testz_si256(vd,vd)==0)
			return false;
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+2<=n;i+=2){
		const v2u64 d=ld2(a+i)^ld2(b+i);
		if((d[0]|d[1])!=0)
			return false;
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	const __m128i z=_mm_setzero_si128();
	for(;i+2<=n;i+=2){
		const __m128i va=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i));
		const __m128i vb=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+i));
		const __m128i vd=_mm_xor_si128(va,vb);
		if(_mm_movemask_epi8(_mm_cmpeq_epi8(vd,z))!=0xFFFF)
			return false;
	}
#endif
	for(;i<n;++i){
		if(a[i]!=b[i])
			return false;
	}
	return true;
}

inline bool z_n(const std::uint64_t*a,std::size_t n) noexcept{
	std::size_t i=0;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const v8u64 v=ld8(a+i);
		if((v[0]|v[1]|v[2]|v[3]|v[4]|v[5]|v[6]|v[7])!=0)
			return false;
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const __m512i v=
			_mm512_loadu_si512(reinterpret_cast<const void*>(a+i));
		alignas(64) std::uint64_t t[8];
		_mm512_storeu_si512(reinterpret_cast<void*>(t),v);
		if((t[0]|t[1]|t[2]|t[3]|t[4]|t[5]|t[6]|t[7])!=0)
			return false;
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const v4u64 v=ld4(a+i);
		if((v[0]|v[1]|v[2]|v[3])!=0)
			return false;
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const __m256i v=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
		if(_mm256_testz_si256(v,v)==0)
			return false;
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+2<=n;i+=2){
		const v2u64 v=ld2(a+i);
		if((v[0]|v[1])!=0)
			return false;
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	const __m128i z=_mm_setzero_si128();
	for(;i+2<=n;i+=2){
		const __m128i v=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i));
		if(_mm_movemask_epi8(_mm_cmpeq_epi8(v,z))!=0xFFFF)
			return false;
	}
#endif
	for(;i<n;++i){
		if(a[i]!=0)
			return false;
	}
	return true;
}

inline int cmp_n(const std::uint64_t*a,const std::uint64_t*b,
				 std::size_t n) noexcept{
	std::size_t i=n;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=8){
		i-=8;
		const v8u64 va=ld8(a+i);
		const v8u64 vb=ld8(b+i);
		const v8u64 d=va^vb;
		if((d[0]|d[1]|d[2]|d[3]|d[4]|d[5]|d[6]|d[7])!=0){
			for(int k=7;k>=0;--k){
				if(va[k]!=vb[k])
					return (va[k]<vb[k])?-1:1;
			}
		}
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	while(i>=8){
		i-=8;
		const __m512i va=
			_mm512_loadu_si512(reinterpret_cast<const void*>(a+i));
		const __m512i vb=
			_mm512_loadu_si512(reinterpret_cast<const void*>(b+i));
		alignas(64) std::uint64_t ta[8];
		alignas(64) std::uint64_t tb[8];
		_mm512_storeu_si512(reinterpret_cast<void*>(ta),va);
		_mm512_storeu_si512(reinterpret_cast<void*>(tb),vb);
		if((ta[0]^tb[0])|(ta[1]^tb[1])|(ta[2]^tb[2])|(ta[3]^tb[3])|
		   (ta[4]^tb[4])|(ta[5]^tb[5])|(ta[6]^tb[6])|(ta[7]^tb[7])){
			for(int k=7;k>=0;--k){
				if(ta[k]!=tb[k])
					return (ta[k]<tb[k])?-1:1;
			}
		}
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=4){
		i-=4;
		const v4u64 va=ld4(a+i);
		const v4u64 vb=ld4(b+i);
		const v4u64 d=va^vb;
		if((d[0]|d[1]|d[2]|d[3])!=0){
			for(int k=3;k>=0;--k){
				if(va[k]!=vb[k])
					return (va[k]<vb[k])?-1:1;
			}
		}
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	while(i>=4){
		i-=4;
		const __m256i va=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
		const __m256i vb=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
		alignas(32) std::uint64_t ta[4];
		alignas(32) std::uint64_t tb[4];
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(ta),va);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(tb),vb);
		if((ta[0]^tb[0])|(ta[1]^tb[1])|(ta[2]^tb[2])|(ta[3]^tb[3])){
			for(int k=3;k>=0;--k){
				if(ta[k]!=tb[k])
					return (ta[k]<tb[k])?-1:1;
			}
		}
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=2){
		i-=2;
		const v2u64 va=ld2(a+i);
		const v2u64 vb=ld2(b+i);
		const v2u64 d=va^vb;
		if((d[0]|d[1])!=0){
			for(int k=1;k>=0;--k){
				if(va[k]!=vb[k])
					return (va[k]<vb[k])?-1:1;
			}
		}
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	while(i>=2){
		i-=2;
		const __m128i va=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i));
		const __m128i vb=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+i));
		alignas(16) std::uint64_t ta[2];
		alignas(16) std::uint64_t tb[2];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(ta),va);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(tb),vb);
		if((ta[0]^tb[0])|(ta[1]^tb[1])){
			for(int k=1;k>=0;--k){
				if(ta[k]!=tb[k])
					return (ta[k]<tb[k])?-1:1;
			}
		}
	}
#endif
	while(i>0){
		--i;
		if(a[i]!=b[i])
			return (a[i]<b[i])?-1:1;
	}
	return 0;
}

inline std::size_t nsz_n(const std::uint64_t*a,std::size_t n) noexcept{
	std::size_t i=n;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=8){
		i-=8;
		const v8u64 v=ld8(a+i);
		if((v[0]|v[1]|v[2]|v[3]|v[4]|v[5]|v[6]|v[7])!=0){
			for(int k=7;k>=0;--k){
				if(v[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	while(i>=8){
		i-=8;
		const __m512i v=
			_mm512_loadu_si512(reinterpret_cast<const void*>(a+i));
		alignas(64) std::uint64_t t[8];
		_mm512_storeu_si512(reinterpret_cast<void*>(t),v);
		if((t[0]|t[1]|t[2]|t[3]|t[4]|t[5]|t[6]|t[7])!=0){
			for(int k=7;k>=0;--k){
				if(t[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=4){
		i-=4;
		const v4u64 v=ld4(a+i);
		if((v[0]|v[1]|v[2]|v[3])!=0){
			for(int k=3;k>=0;--k){
				if(v[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	while(i>=4){
		i-=4;
		const __m256i v=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
		alignas(32) std::uint64_t t[4];
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(t),v);
		if((t[0]|t[1]|t[2]|t[3])!=0){
			for(int k=3;k>=0;--k){
				if(t[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	while(i>=2){
		i-=2;
		const v2u64 v=ld2(a+i);
		if((v[0]|v[1])!=0){
			for(int k=1;k>=0;--k){
				if(v[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	while(i>=2){
		i-=2;
		const __m128i v=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i));
		alignas(16) std::uint64_t t[2];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(t),v);
		if((t[0]|t[1])!=0){
			for(int k=1;k>=0;--k){
				if(t[k]!=0)
					return i+static_cast<std::size_t>(k)+1;
			}
		}
	}
#endif
	while(i>0){
		--i;
		if(a[i]!=0)
			return i+1;
	}
	return 0;
}

inline void and_n(std::uint64_t*rp,const std::uint64_t*ap,
				  const std::uint64_t*bp,std::size_t n) noexcept{
	std::size_t i=0;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		st8(rp+i,ld8(ap+i)&ld8(bp+i));
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const __m512i va=
			_mm512_loadu_si512(reinterpret_cast<const void*>(ap+i));
		const __m512i vb=
			_mm512_loadu_si512(reinterpret_cast<const void*>(bp+i));
		const __m512i vr=_mm512_and_si512(va,vb);
		_mm512_storeu_si512(reinterpret_cast<void*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		st4(rp+i,ld4(ap+i)&ld4(bp+i));
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const __m256i va=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap+i));
		const __m256i vb=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp+i));
		const __m256i vr=_mm256_and_si256(va,vb);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+2<=n;i+=2){
		st2(rp+i,ld2(ap+i)&ld2(bp+i));
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	for(;i+2<=n;i+=2){
		const __m128i va=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(ap+i));
		const __m128i vb=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(bp+i));
		const __m128i vr=_mm_and_si128(va,vb);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(rp+i),vr);
	}
#endif
	for(;i<n;++i){
		rp[i]=ap[i]&bp[i];
	}
}

inline void or_n(std::uint64_t*rp,const std::uint64_t*ap,
				 const std::uint64_t*bp,std::size_t n) noexcept{
	std::size_t i=0;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		st8(rp+i,ld8(ap+i)|ld8(bp+i));
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const __m512i va=
			_mm512_loadu_si512(reinterpret_cast<const void*>(ap+i));
		const __m512i vb=
			_mm512_loadu_si512(reinterpret_cast<const void*>(bp+i));
		const __m512i vr=_mm512_or_si512(va,vb);
		_mm512_storeu_si512(reinterpret_cast<void*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		st4(rp+i,ld4(ap+i)|ld4(bp+i));
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const __m256i va=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap+i));
		const __m256i vb=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp+i));
		const __m256i vr=_mm256_or_si256(va,vb);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+2<=n;i+=2){
		st2(rp+i,ld2(ap+i)|ld2(bp+i));
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	for(;i+2<=n;i+=2){
		const __m128i va=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(ap+i));
		const __m128i vb=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(bp+i));
		const __m128i vr=_mm_or_si128(va,vb);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(rp+i),vr);
	}
#endif
	for(;i<n;++i){
		rp[i]=ap[i]|bp[i];
	}
}

inline void xor_n(std::uint64_t*rp,const std::uint64_t*ap,
				  const std::uint64_t*bp,std::size_t n) noexcept{
	std::size_t i=0;
#if MINI_MP_VEC_W8&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		st8(rp+i,ld8(ap+i)^ld8(bp+i));
	}
#elif MINI_MP_VEC_W8&&MINI_MP_VEC_X86
	for(;i+8<=n;i+=8){
		const __m512i va=
			_mm512_loadu_si512(reinterpret_cast<const void*>(ap+i));
		const __m512i vb=
			_mm512_loadu_si512(reinterpret_cast<const void*>(bp+i));
		const __m512i vr=_mm512_xor_si512(va,vb);
		_mm512_storeu_si512(reinterpret_cast<void*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W4&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		st4(rp+i,ld4(ap+i)^ld4(bp+i));
	}
#elif MINI_MP_VEC_W4&&MINI_MP_VEC_X86
	for(;i+4<=n;i+=4){
		const __m256i va=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ap+i));
		const __m256i vb=
			_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp+i));
		const __m256i vr=_mm256_xor_si256(va,vb);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(rp+i),vr);
	}
#elif MINI_MP_VEC_W2&&(defined(__GNUC__)||defined(__clang__))&&!MINI_MP_VEC_X86
	for(;i+2<=n;i+=2){
		st2(rp+i,ld2(ap+i)^ld2(bp+i));
	}
#elif MINI_MP_VEC_W2&&MINI_MP_VEC_X86&&(defined(__SSE2__)||defined(_M_X64)|| \
										   defined(_M_IX86_FP))
	for(;i+2<=n;i+=2){
		const __m128i va=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(ap+i));
		const __m128i vb=
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(bp+i));
		const __m128i vr=_mm_xor_si128(va,vb);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(rp+i),vr);
	}
#endif
	for(;i<n;++i){
		rp[i]=ap[i]^bp[i];
	}
}

}





struct u128{
	std::uint64_t lo;
	std::uint64_t hi;
};

inline u128 mul_u64(std::uint64_t a,std::uint64_t b) noexcept{
#if MINI_MP_DETAIL_USE_MSVC_INTRIN
	unsigned __int64 hi=0;
	const unsigned __int64 lo=_umul128(static_cast<unsigned __int64>(a),
									   static_cast<unsigned __int64>(b),&hi);
	return {static_cast<std::uint64_t>(lo),static_cast<std::uint64_t>(hi)};
#elif MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 p=
		static_cast<unsigned __int128>(a)*static_cast<unsigned __int128>(b);
	return {static_cast<std::uint64_t>(p),static_cast<std::uint64_t>(p>>64)};
#else
	const std::uint64_t a0=static_cast<std::uint32_t>(a);
	const std::uint64_t a1=a>>32;
	const std::uint64_t b0=static_cast<std::uint32_t>(b);
	const std::uint64_t b1=b>>32;

	const std::uint64_t p0=a0*b0;
	const std::uint64_t p1=a0*b1;
	const std::uint64_t p2=a1*b0;
	const std::uint64_t p3=a1*b1;

	const std::uint64_t mid=(p0>>32)+static_cast<std::uint32_t>(p1)+
							static_cast<std::uint32_t>(p2);
	const std::uint64_t lo=(p0&0xffffffffull)|(mid<<32);
	const std::uint64_t hi=p3+(p1>>32)+(p2>>32)+(mid>>32);
	return {lo,hi};
#endif
}

inline std::uint64_t addc_u64(std::uint64_t carry_in,std::uint64_t a,
								  std::uint64_t b,std::uint64_t*out) noexcept{
	MINI_MP_ASSERT(out!=nullptr);
	MINI_MP_ASSERT(carry_in<=1);
#if MINI_MP_DETAIL_USE_MSVC_INTRIN
	unsigned __int64 tmp=0;
	const unsigned char c=_addcarry_u64(static_cast<unsigned char>(carry_in),
										static_cast<unsigned __int64>(a),
										static_cast<unsigned __int64>(b),&tmp);
	*out=static_cast<std::uint64_t>(tmp);
	return static_cast<std::uint64_t>(c);
#elif MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY
	unsigned long long tmp=0;
	const unsigned char c=_addcarry_u64(
		static_cast<unsigned char>(carry_in),static_cast<unsigned long long>(a),
		static_cast<unsigned long long>(b),&tmp);
	*out=static_cast<std::uint64_t>(tmp);
	return static_cast<std::uint64_t>(c);
#elif MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 s=static_cast<unsigned __int128>(a)+
							  static_cast<unsigned __int128>(b)+
							  static_cast<unsigned __int128>(carry_in);
	*out=static_cast<std::uint64_t>(s);
	return static_cast<std::uint64_t>(s>>64);
#else
	const std::uint64_t s=a+b;
	const std::uint64_t c1=(s<a)?1u:0u;
	const std::uint64_t s2=s+carry_in;
	const std::uint64_t c2=(s2<s)?1u:0u;
	*out=s2;
	return c1|c2;
#endif
}

inline std::uint64_t subb_u64(std::uint64_t borrow_in,std::uint64_t a,
								   std::uint64_t b,std::uint64_t*out) noexcept{
	MINI_MP_ASSERT(out!=nullptr);
	MINI_MP_ASSERT(borrow_in<=1);
#if MINI_MP_DETAIL_USE_MSVC_INTRIN
	unsigned __int64 tmp=0;
	const unsigned char c=_subborrow_u64(static_cast<unsigned char>(borrow_in),
										 static_cast<unsigned __int64>(a),
										 static_cast<unsigned __int64>(b),&tmp);
	*out=static_cast<std::uint64_t>(tmp);
	return static_cast<std::uint64_t>(c);
#elif MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY
	unsigned long long tmp=0;
	const unsigned char c=_subborrow_u64(
		static_cast<unsigned char>(borrow_in),
		static_cast<unsigned long long>(a),
		static_cast<unsigned long long>(b),&tmp);
	*out=static_cast<std::uint64_t>(tmp);
	return static_cast<std::uint64_t>(c);
#elif MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 aa=static_cast<unsigned __int128>(a);
	const unsigned __int128 bb=static_cast<unsigned __int128>(b)+
							   static_cast<unsigned __int128>(borrow_in);
	const unsigned __int128 d=aa-bb;
	*out=static_cast<std::uint64_t>(d);
	return (aa<bb)?1u:0u;
#else
	const std::uint64_t d=a-b;
	const std::uint64_t b1=(a<b)?1u:0u;
	const std::uint64_t d2=d-borrow_in;
	const std::uint64_t b2=(d<borrow_in)?1u:0u;
	*out=d2;
	return b1|b2;
#endif
}



inline std::uint64_t udiv_qft(std::uint64_t hi,std::uint64_t lo,
									  std::uint64_t d,
									  std::uint64_t*rem_ptr) noexcept{
	MINI_MP_ASSERT(rem_ptr!=nullptr);
	MINI_MP_ASSERT(d!=0);
	MINI_MP_ASSERT(hi<d);

	std::uint64_t q=0;
	std::uint64_t r=hi;
	for(int bit=63;bit>=0;--bit){
		const std::uint64_t in_bit=(lo>>bit)&1ull;
		const std::uint64_t overflow=(r>>63);
		const std::uint64_t shifted=(r<<1)|in_bit;

		if(overflow!=0||shifted>=d){
			std::uint64_t new_r=0;
			std::uint64_t b=subb_u64(0,shifted,d,&new_r);
			if(overflow!=0){
				std::uint64_t hi_tmp=0;
				b=subb_u64(b,1,0,&hi_tmp);
				MINI_MP_ASSERT(b==0&&hi_tmp==0);
			}else{
				MINI_MP_ASSERT(b==0);
			}
			r=new_r;
			q|=(std::uint64_t(1)<<bit);
		}else{
			r=shifted;
		}
	}

	MINI_MP_ASSERT(r<d);
	*rem_ptr=r;
	return q;
}

inline std::uint64_t udiv128(std::uint64_t hi,std::uint64_t lo,
								 std::uint64_t d,std::uint64_t*rem) noexcept{
	std::uint64_t local_rem=0;
	std::uint64_t*rem_ptr=(rem!=nullptr)?rem:&local_rem;
	if(d==0){
		*rem_ptr=0;
		MINI_MP_ASSERT(false&&"udiv128: division by zero");
		return 0;
	}
#if MINI_MP_DETAIL_USE_MSVC_INTRIN
	
	
	
#if !defined(__clang__)
	if(hi<d){
		unsigned __int64 r=0;
		const unsigned __int64 q=_udiv128(static_cast<unsigned __int64>(hi),
										  static_cast<unsigned __int64>(lo),
										  static_cast<unsigned __int64>(d),&r);
		*rem_ptr=static_cast<std::uint64_t>(r);
		return static_cast<std::uint64_t>(q);
	}
#endif
	if(hi>=d){
		
		
		const std::uint64_t hi_rem=hi%d;
		return udiv_qft(hi_rem,lo,d,rem_ptr);
	}
	return udiv_qft(hi,lo,d,rem_ptr);
#elif MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 num=(static_cast<unsigned __int128>(hi)<<64)|
								static_cast<unsigned __int128>(lo);
	const std::uint64_t q=static_cast<std::uint64_t>(num/d);
	*rem_ptr=static_cast<std::uint64_t>(num%d);
	return q;
#else
	if(hi>=d){
		const std::uint64_t hi_rem=hi%d;
		return udiv_qft(hi_rem,lo,d,rem_ptr);
	}
	return udiv_qft(hi,lo,d,rem_ptr);
#endif
}

inline bool is0_u128(u128 v) noexcept{ return v.lo==0&&v.hi==0; }

inline u128 add_u128(u128 a,u128 b) noexcept{
	u128 out{};
	const std::uint64_t c=addc_u64(0,a.lo,b.lo,&out.lo);
	(void)addc_u64(c,a.hi,b.hi,&out.hi);
	return out;
}

inline u128 add128_64(u128 a,std::uint64_t b) noexcept{
	u128 out{};
	const std::uint64_t c=addc_u64(0,a.lo,b,&out.lo);
	out.hi=a.hi+c;
	return out;
}

inline u128 shr_u128(u128 v,unsigned bits) noexcept{
	if(bits==0)
		return v;
	if(bits>=128)
		return {0,0};
	if(bits>=64){
		const unsigned s=bits-64;
		return {v.hi>>s,0};
	}
	return {
		(v.lo>>bits)|(v.hi<<(64u-bits)),
		(v.hi>>bits),
	};
}

inline u128 shl_u128(u128 v,unsigned bits) noexcept{
	if(bits==0)
		return v;
	if(bits>=128)
		return {0,0};
	if(bits>=64){
		const unsigned s=bits-64;
		return {0,v.lo<<s};
	}
	return {
		v.lo<<bits,
		(v.hi<<bits)|(v.lo>>(64u-bits)),
	};
}


inline std::uint64_t mulad64(std::uint64_t a,std::uint64_t b,
								 std::uint64_t add,
								 std::uint64_t*carry) noexcept{
	MINI_MP_ASSERT(carry!=nullptr);
#if MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 prod=static_cast<unsigned __int128>(a)*b+
								 static_cast<unsigned __int128>(add)+
								 static_cast<unsigned __int128>(*carry);
	*carry=static_cast<std::uint64_t>(prod>>64);
	return static_cast<std::uint64_t>(prod);
#else
	const u128 p=mul_u64(a,b);

	std::uint64_t t0=0;
	const std::uint64_t c1=addc_u64(0,p.lo,add,&t0);
	std::uint64_t out=0;
	const std::uint64_t c2=addc_u64(0,t0,*carry,&out);

	std::uint64_t hi=0;
	const std::uint64_t c3=addc_u64(0,p.hi,c1,&hi);
	const std::uint64_t c4=addc_u64(0,hi,c2,&hi);
	MINI_MP_ASSERT(c3==0&&c4==0);

	*carry=hi;
	return out;
#endif
}

template<class T,std::size_t InlineN> class SmallLimbs{
	static_assert(InlineN>0,"SmallLimbs requires InlineN > 0");

  public:
	using value_type=T;
	using size_type=std::size_t;
	using iterator=T*;
	using const_iterator=const T*;

	SmallLimbs() noexcept : size_(0),cap_(InlineN),ptr_(inl_data()){}

	SmallLimbs(size_type n,const T&value=T()) : SmallLimbs(){ resize(n,value); }

	template<class It,
			 class=std::enable_if_t<!std::is_integral_v<std::decay_t<It>>>>
	SmallLimbs(It first,It last) : SmallLimbs(){
		assign(first,last);
	}

	SmallLimbs(std::initializer_list<T> init)
		: SmallLimbs(init.begin(),init.end()){}

	SmallLimbs(const SmallLimbs&other) : SmallLimbs(){
		assign(other);
	}

	SmallLimbs(SmallLimbs&&other) noexcept : SmallLimbs(){
		move_from(std::move(other));
	}

	~SmallLimbs(){ rel_heap(); }

	SmallLimbs&operator=(const SmallLimbs&other){
		if(this!=&other){
			assign(other);
		}
		return *this;
	}

	SmallLimbs&operator=(SmallLimbs&&other) noexcept{
		if(this!=&other){
			rel_heap();
			size_=0;
			cap_=InlineN;
			ptr_=inl_data();
			move_from(std::move(other));
		}
		return *this;
	}

	SmallLimbs&operator=(std::initializer_list<T> init){
		assign(init.begin(),init.end());
		return *this;
	}

	size_type size() const noexcept{ return size_; }
	size_type capacity() const noexcept{ return cap_; }
	bool empty() const noexcept{ return size_==0; }

	T*data() noexcept{ return ptr_; }
	const T*data() const noexcept{ return ptr_; }

	iterator begin() noexcept{ return ptr_; }
	const_iterator begin() const noexcept{ return ptr_; }
	const_iterator cbegin() const noexcept{ return ptr_; }

	iterator end() noexcept{ return ptr_+static_cast<std::ptrdiff_t>(size_); }
	const_iterator end() const noexcept{
		return ptr_+static_cast<std::ptrdiff_t>(size_);
	}
	const_iterator cend() const noexcept{
		return ptr_+static_cast<std::ptrdiff_t>(size_);
	}

	T&operator[](size_type i) noexcept{ return ptr_[i]; }
	const T&operator[](size_type i) const noexcept{ return ptr_[i]; }

	T&back() noexcept{ return ptr_[size_-1]; }
	const T&back() const noexcept{ return ptr_[size_-1]; }

	void clear() noexcept{ size_=0; }

	void reserve(size_type n){
		if(n<=cap_)
			return;
		grow_to(n);
	}

	void resize(size_type n,const T&value=T()){
		if(n<=size_){
			size_=n;
			return;
		}
		reserve(n);
		for(size_type i=size_;i<n;++i){
			ptr_[i]=value;
		}
		size_=n;
	}

	void resize_uninit(size_type n){
		static_assert(std::is_trivially_copyable_v<T>,
					  "resize_uninit requires trivially copyable limbs");
		reserve(n);
		size_=n;
	}

	void push_back(const T&v){
		reserve(size_+1);
		ptr_[size_]=v;
		++size_;
	}

	void push_back(T&&v){
		reserve(size_+1);
		ptr_[size_]=std::move(v);
		++size_;
	}

	void pop_back(){
		MINI_MP_ASSERT(size_>0);
		--size_;
	}

	void assign(size_type n,const T&value){
		clear();
		reserve(n);
		for(size_type i=0;i<n;++i){
			ptr_[i]=value;
		}
		size_=n;
	}

	template<class It,
			 class=std::enable_if_t<!std::is_integral_v<std::decay_t<It>>>>
	void assign(It first,It last){
		using cat=typename std::iterator_traits<It>::iterator_category;
		if constexpr(std::is_base_of_v<std::random_access_iterator_tag,cat>){
			const auto diff=last-first;
			MINI_MP_ASSERT(diff>=0);
			const size_type n=static_cast<size_type>(diff);
			clear();
			reserve(n);
			if constexpr(std::is_trivially_copyable_v<T>){
				if(n!=0){
					std::memcpy(ptr_,std::to_address(first),n*sizeof(T));
				}
			}else{
				for(size_type i=0;i<n;++i){
					ptr_[i]=first[
						static_cast<typename std::iterator_traits<It>::
									 difference_type>(i)];
				}
			}
			size_=n;
		}else{
			SmallLimbs tmp;
			for(It it=first;it!=last;++it){
				tmp.push_back(*it);
			}
			swap(tmp);
		}
	}

	void assign(std::initializer_list<T> init){
		assign(init.begin(),init.end());
	}

	iterator insert(iterator pos,size_type count,const T&value){
		const size_type idx=static_cast<size_type>(pos-begin());
		if(count==0)
			return begin()+static_cast<std::ptrdiff_t>(idx);

		const size_type old_size=size_;
		reserve(old_size+count);
		for(size_type i=old_size;i>idx;--i){
			ptr_[i+count-1]=ptr_[i-1];
		}
		for(size_type i=0;i<count;++i){
			ptr_[idx+i]=value;
		}
		size_=old_size+count;
		return begin()+static_cast<std::ptrdiff_t>(idx);
	}

	template<class It,
			 class=std::enable_if_t<!std::is_integral_v<std::decay_t<It>>>>
	iterator insert(iterator pos,It first,It last){
		SmallLimbs tmp(first,last);
		return ins_impl(pos,tmp.begin(),tmp.end());
	}

	void swap(SmallLimbs&other) noexcept{
		if(this==&other)
			return;

		if(is_heap()&&other.is_heap()){
			std::swap(size_,other.size_);
			std::swap(cap_,other.cap_);
			std::swap(ptr_,other.ptr_);
			return;
		}

		SmallLimbs tmp(std::move(*this));
		*this=std::move(other);
		other=std::move(tmp);
	}

	friend bool operator==(const SmallLimbs&a,const SmallLimbs&b) noexcept{
		if(a.size_!=b.size_)
			return false;
		if(a.size_==0)
			return true;
		if constexpr(std::is_same_v<T,std::uint64_t>){
			return vecab::eq_n(reinterpret_cast<const std::uint64_t*>(a.ptr_),
							   reinterpret_cast<const std::uint64_t*>(b.ptr_),
							   a.size_);
		}
		for(size_type i=0;i<a.size_;++i){
			if(a.ptr_[i]!=b.ptr_[i])
				return false;
		}
		return true;
	}

	friend bool operator!=(const SmallLimbs&a,const SmallLimbs&b) noexcept{
		return !(a==b);
	}

  private:
	alignas(T) unsigned char inl_buf_[sizeof(T)*InlineN];
	size_type size_;
	size_type cap_;
	T*ptr_;

	T*inl_data() noexcept{ return reinterpret_cast<T*>(inl_buf_); }

	const T*inl_data() const noexcept{
		return reinterpret_cast<const T*>(inl_buf_);
	}

	bool is_heap() const noexcept{ return ptr_!=inl_data(); }

	void rel_heap() noexcept{
		if(is_heap()){
			delete[] ptr_;
		}
	}

	void grow_to(size_type new_cap){
		T*new_ptr=new T[new_cap];
		if constexpr(std::is_trivially_copyable_v<T>){
			if(size_!=0){
				std::memcpy(new_ptr,ptr_,size_*sizeof(T));
			}
		}else{
			for(size_type i=0;i<size_;++i){
				new_ptr[i]=ptr_[i];
			}
		}
		rel_heap();
		ptr_=new_ptr;
		cap_=new_cap;
	}

	void assign(const SmallLimbs&other){
		clear();
		reserve(other.size_);
		if constexpr(std::is_trivially_copyable_v<T>){
			if(other.size_!=0){
				std::memcpy(ptr_,other.ptr_,other.size_*sizeof(T));
			}
		}else{
			for(size_type i=0;i<other.size_;++i){
				ptr_[i]=other.ptr_[i];
			}
		}
		size_=other.size_;
	}

	void move_from(SmallLimbs&&other) noexcept{
		if(other.is_heap()){
			size_=other.size_;
			cap_=other.cap_;
			ptr_=other.ptr_;
			other.ptr_=other.inl_data();
			other.size_=0;
			other.cap_=InlineN;
			return;
		}

		size_=other.size_;
		for(size_type i=0;i<size_;++i){
			ptr_[i]=std::move(other.ptr_[i]);
		}
		other.size_=0;
	}

	iterator ins_impl(iterator pos,const T*first,const T*last){
		const size_type idx=static_cast<size_type>(pos-begin());
		const size_type count=static_cast<size_type>(last-first);
		if(count==0)
			return begin()+static_cast<std::ptrdiff_t>(idx);

		const size_type old_size=size_;
		reserve(old_size+count);
		for(size_type i=old_size;i>idx;--i){
			ptr_[i+count-1]=ptr_[i-1];
		}
		for(size_type i=0;i<count;++i){
			ptr_[idx+i]=first[i];
		}
		size_=old_size+count;
		return begin()+static_cast<std::ptrdiff_t>(idx);
	}
};

template<class T,std::size_t InlineN>
inline void swap(SmallLimbs<T,InlineN>&a,SmallLimbs<T,InlineN>&b) noexcept{
	a.swap(b);
}

using limbs_t=SmallLimbs<limb_t,MINI_MP_INLINE_LIMBS>;

inline void trim_lz(limbs_t&x){
	while(!x.empty()&&x.back()==0){
		x.pop_back();
	}
}

inline void norm_sign(int&sign,limbs_t&x){
	trim_lz(x);
	if(x.empty()){
		sign=0;
	}else{
		sign=(sign<0)?-1:1;
	}
}

inline int cmp_abs(const limbs_t&a,const limbs_t&b) noexcept{
	if(a.size()!=b.size()){
		return (a.size()<b.size())?-1:1;
	}
	return vecab::cmp_n(a.data(),b.data(),a.size());
}

inline int cmp_abs_dbl(const limbs_t&a,const limbs_t&b) noexcept{
	MINI_MP_ASSERT(!b.empty());
	const std::size_t bn=b.size();
	const std::size_t dn=bn+((b.back()>>63u)!=0u?1u:0u);
	if(a.size()!=dn)
		return (a.size()<dn)?-1:1;
	for(std::size_t k=dn;k>0;--k){
		const std::size_t i=k-1u;
		const std::uint64_t carry=(i==0)?0u:(b[i-1u]>>63u);
		const std::uint64_t dbl=(i<bn)?((b[i]<<1u)|carry):carry;
		if(a[i]!=dbl)
			return (a[i]<dbl)?-1:1;
	}
	return 0;
}

inline std::size_t bit_length(const limbs_t&x) noexcept{
	if(x.empty())
		return 0;
	const limb_t ms=x.back();
	MINI_MP_ASSERT(ms!=0);
	const unsigned lz=std::countl_zero(ms);
	return (x.size()-1)*64+(64u-lz);
}

inline std::size_t ctz(const limbs_t&x) noexcept{
	for(std::size_t i=0;i<x.size();++i){
		if(x[i]!=0){
			return i*64+std::countr_zero(x[i]);
		}
	}
	return 0;
}

inline bool test_bit(const limbs_t&x,std::size_t bit_index) noexcept{
	const std::size_t limb_index=bit_index/64;
	const unsigned bit_off=static_cast<unsigned>(bit_index%64);
	if(limb_index>=x.size())
		return false;
	return ((x[limb_index]>>bit_off)&1u)!=0;
}

inline std::size_t popcount_abs(const limbs_t&x) noexcept{
	std::size_t out=0;
	for(limb_t v : x)
		out+=std::popcount(v);
	return out;
}

inline limb_t high_mask_from(std::size_t bit_index) noexcept{
	const unsigned off=static_cast<unsigned>(bit_index&63u);
	return off==0u?~limb_t(0):(~limb_t(0)<<off);
}

inline std::size_t bit_npos() noexcept{
	return std::numeric_limits<std::size_t>::max();
}

inline std::size_t scan1_abs(const limbs_t&x,
							 std::size_t start_bit) noexcept{
	std::size_t i=start_bit/64u;
	if(i>=x.size())
		return bit_npos();
	limb_t limb=x[i]&high_mask_from(start_bit);
	for(;;){
		if(limb!=0)
			return i*64u+std::countr_zero(limb);
		++i;
		if(i>=x.size())
			return bit_npos();
		limb=x[i];
	}
}

inline std::size_t scan0_abs(const limbs_t&x,
							 std::size_t start_bit) noexcept{
	std::size_t i=start_bit/64u;
	if(i>=x.size())
		return start_bit;
	limb_t limb=(~x[i])&high_mask_from(start_bit);
	for(;;){
		if(limb!=0)
			return i*64u+std::countr_zero(limb);
		++i;
		if(i>=x.size())
			return i*64u;
		limb=~x[i];
	}
}

inline std::size_t first_nonzero_limb(const limbs_t&x) noexcept{
	std::size_t i=0;
	while(i<x.size()&&x[i]==0)
		++i;
	return i;
}

inline limb_t sub1_limb_at_known(const limbs_t&x,std::size_t limb_index,
								 std::size_t first_nz) noexcept{
	if(limb_index>=x.size())
		return 0;
	if(limb_index<first_nz)
		return ~limb_t(0);
	if(limb_index==first_nz)
		return x[limb_index]-1u;
	return x[limb_index];
}

inline std::size_t scan0_sub1_abs(const limbs_t&x,
								  std::size_t start_bit) noexcept{
	std::size_t i=start_bit/64u;
	if(i>=x.size())
		return start_bit;
	const std::size_t first_nz=first_nonzero_limb(x);
	limb_t limb=(~sub1_limb_at_known(x,i,first_nz))&
		high_mask_from(start_bit);
	for(;;){
		if(limb!=0)
			return i*64u+std::countr_zero(limb);
		++i;
		if(i>=x.size())
			return i*64u;
		limb=~sub1_limb_at_known(x,i,first_nz);
	}
}

inline std::size_t scan1_sub1_abs(const limbs_t&x,
								  std::size_t start_bit) noexcept{
	std::size_t i=start_bit/64u;
	if(i>=x.size())
		return bit_npos();
	const std::size_t first_nz=first_nonzero_limb(x);
	limb_t limb=sub1_limb_at_known(x,i,first_nz)&
		high_mask_from(start_bit);
	for(;;){
		if(limb!=0)
			return i*64u+std::countr_zero(limb);
		++i;
		if(i>=x.size())
			return bit_npos();
		limb=sub1_limb_at_known(x,i,first_nz);
	}
}

inline std::mt19937_64&default_rng(){
	static thread_local std::mt19937_64 rng=[](){
		std::random_device rd;
		std::array<std::uint32_t,8> seed{};
		for(auto&v : seed)
			v=rd();
		std::seed_seq seq(seed.begin(),seed.end());
		return std::mt19937_64(seq);
	}();
	return rng;
}

template<class URBG>
inline limb_t rand_limb(URBG&rng){
	using res_t=typename URBG::result_type;
	if constexpr(std::is_unsigned_v<res_t>&&
				 URBG::min()==res_t(0)&&
				 ((URBG::max()&(URBG::max()+res_t(1)))==res_t(0))){
		constexpr unsigned rb=std::bit_width(URBG::max());
		if constexpr(rb>=64u){
			return static_cast<limb_t>(rng());
		}else{
			limb_t out=0;
			unsigned filled=0;
			while(filled<64u){
				out|=static_cast<limb_t>(rng())<<filled;
				filled=static_cast<unsigned>(filled+rb);
			}
			return out;
		}
	}else{
		std::uniform_int_distribution<limb_t> dist(
			std::numeric_limits<limb_t>::min(),
			std::numeric_limits<limb_t>::max());
		return dist(rng);
	}
}

template<class URBG>
inline limbs_t rand_limbs_bits(std::size_t bits,URBG&rng){
	if(bits==0)
		return {};
	if(bits>std::numeric_limits<std::size_t>::max()-63u)
		throw_ovf("rand_bits: too many bits");
	const std::size_t n=(bits+63u)/64u;
	limbs_t out;
	out.resize_uninit(n);
	for(std::size_t i=0;i<n;++i)
		out[i]=rand_limb(rng);
	const unsigned hi_bits=static_cast<unsigned>(bits&63u);
	if(hi_bits!=0u)
		out.back()&=(limb_t(1)<<hi_bits)-1u;
	trim_lz(out);
	return out;
}

template<class URBG>
inline std::uint64_t rand_u64_bits(unsigned bits,URBG&rng){
	if(bits==0u)
		return 0;
	if(bits>=64u)
		return rand_limb(rng);
	using res_t=typename URBG::result_type;
	if constexpr(std::is_unsigned_v<res_t>&&
				 URBG::min()==res_t(0)&&
				 ((URBG::max()&(URBG::max()+res_t(1)))==res_t(0))){
		constexpr unsigned rb=std::bit_width(URBG::max());
		std::uint64_t out=0;
		unsigned filled=0;
		while(filled<bits){
			out|=static_cast<std::uint64_t>(rng())<<filled;
			filled=static_cast<unsigned>(filled+rb);
		}
		return out&((std::uint64_t(1)<<bits)-1u);
	}else{
		return rand_limb(rng)&((std::uint64_t(1)<<bits)-1u);
	}
}

template<class URBG>
inline std::uint64_t rand_u64_below(std::uint64_t limit,URBG&rng){
	MINI_MP_ASSERT(limit!=0);
	if(limit==1u)
		return 0;
	if(std::has_single_bit(limit))
		return rand_u64_bits(static_cast<unsigned>(
			std::bit_width(limit-1u)),rng)&(limit-1u);
	const unsigned bits=std::bit_width(limit-1u);
	for(;;){
		const std::uint64_t x=rand_u64_bits(bits,rng);
		if(x<limit)
			return x;
	}
}

inline bool pow2_abs(const limbs_t&x) noexcept{
	bool seen=false;
	for(limb_t v : x){
		if(v==0)
			continue;
		if(!std::has_single_bit(v)||seen)
			return false;
		seen=true;
	}
	return seen;
}

inline std::size_t pow2_exp_abs(const limbs_t&x) noexcept{
	MINI_MP_ASSERT(pow2_abs(x));
	for(std::size_t i=0;i<x.size();++i){
		if(x[i]!=0)
			return i*64u+std::countr_zero(x[i]);
	}
	return 0;
}

inline bool one_abs(const limbs_t&x) noexcept{
	return x.size()==1u&&x[0]==1u;
}

inline limbs_t low_bits_abs(const limbs_t&x,std::size_t bits){
	if(x.empty()||bits==0)
		return {};
	const std::size_t limb_n=bits/64u;
	const unsigned bit_n=static_cast<unsigned>(bits%64u);
	const std::size_t take=std::min(
		x.size(),limb_n+((bit_n==0)?0u:1u));
	limbs_t out;
	out.assign(x.begin(),x.begin()+static_cast<std::ptrdiff_t>(take));
	if(bit_n!=0&&limb_n<out.size()){
		out[limb_n]&=(limb_t(1)<<bit_n)-1u;
	}
	trim_lz(out);
	return out;
}

inline limbs_t bit_slice_abs(const limbs_t&x,std::size_t first,
							 std::size_t bits){
	if(x.empty()||bits==0)
		return {};
	const std::size_t li=first/64u;
	const unsigned bi=static_cast<unsigned>(first%64u);
	if(li>=x.size())
		return {};
	const std::size_t take=(bits+63u)/64u;
	limbs_t out(take,0);
	for(std::size_t i=0;i<take;++i){
		const std::size_t src=li+i;
		std::uint64_t lo=src<x.size()?x[src]:0u;
		if(bi!=0){
			lo>>=bi;
			if(src+1u<x.size())
				lo|=x[src+1u]<<(64u-bi);
		}
		out[i]=lo;
	}
	const unsigned rem=static_cast<unsigned>(bits%64u);
	if(rem!=0)
		out.back()&=(limb_t(1)<<rem)-1u;
	trim_lz(out);
	return out;
}

inline void set_bit(limbs_t&x,std::size_t bit_index){
	const std::size_t limb_index=bit_index/64;
	const unsigned bit_off=static_cast<unsigned>(bit_index%64);
	if(x.size()<=limb_index){
		x.resize(limb_index+1,0);
	}
	x[limb_index]|=(limb_t(1)<<bit_off);
}

inline std::size_t lb_nz(const std::uint64_t*ap,std::size_t n){
	return vecab::nsz_n(ap,n);
}

inline bool lb_zero(const std::uint64_t*ap,std::size_t n) noexcept{
	return vecab::z_n(ap,n);
}

inline int lb_cmp(const std::uint64_t*ap,const std::uint64_t*bp,
				   std::size_t n) noexcept{
	return vecab::cmp_n(ap,bp,n);
}

inline std::uint64_t add_1n(std::uint64_t*rp,const std::uint64_t*ap,
							   std::size_t n,std::uint64_t b){
	if(n==0)
		return b;
#if MINI_MP_DETAIL_USE_MSVC_INTRIN||MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY
	unsigned long long out=0;
	unsigned char c=_addcarry_u64(0,static_cast<unsigned long long>(ap[0]),
								  static_cast<unsigned long long>(b),&out);
	rp[0]=static_cast<std::uint64_t>(out);
	if(c==0){
		if(rp!=ap&&n>1){
			std::memcpy(rp+1,ap+1,(n-1)*sizeof(std::uint64_t));
		}
		return 0;
	}
	for(std::size_t i=1;i<n;++i){
		c=_addcarry_u64(c,static_cast<unsigned long long>(ap[i]),0,&out);
		rp[i]=static_cast<std::uint64_t>(out);
		if(c==0){
			if(rp!=ap&&i+1<n){
				std::memcpy(rp+static_cast<std::ptrdiff_t>(i+1),
							ap+static_cast<std::ptrdiff_t>(i+1),
							(n-(i+1))*sizeof(std::uint64_t));
			}
			return 0;
		}
	}
	return static_cast<std::uint64_t>(c);
#else
	std::uint64_t r=ap[0]+b;
	rp[0]=r;
	std::uint64_t carry=(r<ap[0])?1u:0u;
	if(carry==0){
		if(rp!=ap&&n>1){
			std::memcpy(rp+1,ap+1,(n-1)*sizeof(std::uint64_t));
		}
		return 0;
	}
	for(std::size_t i=1;i<n;++i){
		const std::uint64_t t=ap[i]+carry;
		rp[i]=t;
		carry=(t<ap[i])?1u:0u;
		if(carry==0){
			if(rp!=ap&&i+1<n){
				std::memcpy(rp+static_cast<std::ptrdiff_t>(i+1),
							ap+static_cast<std::ptrdiff_t>(i+1),
							(n-(i+1))*sizeof(std::uint64_t));
			}
			return 0;
		}
	}
	return carry;
#endif
}

inline std::uint64_t add_nn(std::uint64_t*rp,const std::uint64_t*ap,
							   const std::uint64_t*bp,std::size_t n){
	std::uint64_t carry=0;
	std::size_t i=0;
	for(;i+4u<=n;i+=4u){
		std::uint64_t out=0;
		carry=addc_u64(carry,ap[i],bp[i],&out);
		rp[i]=out;
		carry=addc_u64(carry,ap[i+1u],bp[i+1u],&out);
		rp[i+1u]=out;
		carry=addc_u64(carry,ap[i+2u],bp[i+2u],&out);
		rp[i+2u]=out;
		carry=addc_u64(carry,ap[i+3u],bp[i+3u],&out);
		rp[i+3u]=out;
	}
	for(;i<n;++i){
		std::uint64_t out=0;
		carry=addc_u64(carry,ap[i],bp[i],&out);
		rp[i]=out;
	}
	return carry;
}

inline std::uint64_t add_nm(std::uint64_t*rp,const std::uint64_t*ap,
							 std::size_t an,const std::uint64_t*bp,
							 std::size_t bn){
	MINI_MP_ASSERT(an>=bn);
	if(bn==0){
		if(rp!=ap&&an!=0){
			std::memcpy(rp,ap,an*sizeof(std::uint64_t));
		}
		return 0;
	}
	if(bn==1){
		return add_1n(rp,ap,an,bp[0]);
	}
	if(an==bn){
		return add_nn(rp,ap,bp,an);
	}
	std::uint64_t carry=0;
	carry=add_nn(rp,ap,bp,bn);
	if(carry==0){
		if(rp!=ap&&an>bn){
			std::memcpy(rp+static_cast<std::ptrdiff_t>(bn),
						ap+static_cast<std::ptrdiff_t>(bn),
						(an-bn)*sizeof(std::uint64_t));
		}
		return 0;
	}
	if(an==bn)
		return 1;
	carry=add_1n(rp+static_cast<std::ptrdiff_t>(bn),
					ap+static_cast<std::ptrdiff_t>(bn),an-bn,carry);
	return carry;
}

inline std::uint64_t sub_1n(std::uint64_t*rp,const std::uint64_t*ap,
							   std::size_t n,std::uint64_t b){
	if(n==0)
		return (b!=0)?1u:0u;
	if(b==0){
		if(rp!=ap){
			std::memcpy(rp,ap,n*sizeof(std::uint64_t));
		}
		return 0;
	}
#if MINI_MP_DETAIL_USE_MSVC_INTRIN||MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY
	unsigned long long out=0;
	unsigned char c=_subborrow_u64(0,static_cast<unsigned long long>(ap[0]),
								   static_cast<unsigned long long>(b),&out);
	rp[0]=static_cast<std::uint64_t>(out);
	if(c==0){
		if(rp!=ap&&n>1){
			std::memcpy(rp+1,ap+1,(n-1)*sizeof(std::uint64_t));
		}
		return 0;
	}
	for(std::size_t i=1;i<n;++i){
		c=_subborrow_u64(c,static_cast<unsigned long long>(ap[i]),0,&out);
		rp[i]=static_cast<std::uint64_t>(out);
		if(c==0){
			if(rp!=ap&&i+1<n){
				std::memcpy(rp+static_cast<std::ptrdiff_t>(i+1),
							ap+static_cast<std::ptrdiff_t>(i+1),
							(n-(i+1))*sizeof(std::uint64_t));
			}
			return 0;
		}
	}
	return static_cast<std::uint64_t>(c);
#else
	std::uint64_t r=ap[0]-b;
	rp[0]=r;
	std::uint64_t borrow=(ap[0]<b)?1u:0u;
	if(borrow==0){
		if(rp!=ap&&n>1){
			std::memcpy(rp+1,ap+1,(n-1)*sizeof(std::uint64_t));
		}
		return 0;
	}
	for(std::size_t i=1;i<n;++i){
		const std::uint64_t t=ap[i]-borrow;
		rp[i]=t;
		borrow=(ap[i]<borrow)?1u:0u;
		if(borrow==0){
			if(rp!=ap&&i+1<n){
				std::memcpy(rp+static_cast<std::ptrdiff_t>(i+1),
							ap+static_cast<std::ptrdiff_t>(i+1),
							(n-(i+1))*sizeof(std::uint64_t));
			}
			return 0;
		}
	}
	return borrow;
#endif
}

inline std::uint64_t sub_nn(std::uint64_t*rp,const std::uint64_t*ap,
							   const std::uint64_t*bp,std::size_t n){
	std::uint64_t borrow=0;
	std::size_t i=0;
	for(;i+4u<=n;i+=4u){
		std::uint64_t out=0;
		borrow=subb_u64(borrow,ap[i],bp[i],&out);
		rp[i]=out;
		borrow=subb_u64(borrow,ap[i+1u],bp[i+1u],&out);
		rp[i+1u]=out;
		borrow=subb_u64(borrow,ap[i+2u],bp[i+2u],&out);
		rp[i+2u]=out;
		borrow=subb_u64(borrow,ap[i+3u],bp[i+3u],&out);
		rp[i+3u]=out;
	}
	for(;i<n;++i){
		std::uint64_t out=0;
		borrow=subb_u64(borrow,ap[i],bp[i],&out);
		rp[i]=out;
	}
	return borrow;
}

inline std::uint64_t sub_nm(std::uint64_t*rp,const std::uint64_t*ap,
							 std::size_t an,const std::uint64_t*bp,
							 std::size_t bn){
	MINI_MP_ASSERT(an>=bn);
	if(bn==0){
		if(rp!=ap&&an!=0){
			std::memcpy(rp,ap,an*sizeof(std::uint64_t));
		}
		return 0;
	}
	std::uint64_t borrow=0;
	borrow=sub_nn(rp,ap,bp,bn);
	if(an==bn)
		return borrow;
	borrow=sub_1n(rp+static_cast<std::ptrdiff_t>(bn),
					 ap+static_cast<std::ptrdiff_t>(bn),an-bn,borrow);
	return borrow;
}

inline std::uint64_t mul_1n(std::uint64_t*rp,const std::uint64_t*ap,
							   std::size_t n,std::uint64_t v){
	std::uint64_t carry=0;
#if MINI_MP_DETAIL_HAS_UINT128
	std::size_t i=0;
	for(;i+4u<=n;i+=4u){
		unsigned __int128 prod=
			static_cast<unsigned __int128>(ap[i])*v+carry;
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+1u])*v+carry;
		rp[i+1u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+2u])*v+carry;
		rp[i+2u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+3u])*v+carry;
		rp[i+3u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
	}
	for(;i<n;++i){
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(carry);
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
	}
#else
	for(std::size_t i=0;i<n;++i){
		const u128 p=mul_u64(ap[i],v);
		std::uint64_t out=0;
		const std::uint64_t c=addc_u64(0,p.lo,carry,&out);
		rp[i]=out;
		std::uint64_t hi=0;
		const std::uint64_t c2=addc_u64(0,p.hi,c,&hi);
		MINI_MP_ASSERT(c2==0);
		carry=hi;
	}
#endif
	return carry;
}

inline std::uint64_t am_1n(std::uint64_t*rp,const std::uint64_t*ap,
								  std::size_t n,std::uint64_t v){
	std::uint64_t carry=0;
#if MINI_MP_DETAIL_HAS_UINT128
	std::size_t i=0;
	for(;i+4u<=n;i+=4u){
		unsigned __int128 prod=
			static_cast<unsigned __int128>(ap[i])*v+
			static_cast<unsigned __int128>(rp[i])+carry;
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+1u])*v+
			 static_cast<unsigned __int128>(rp[i+1u])+carry;
		rp[i+1u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+2u])*v+
			 static_cast<unsigned __int128>(rp[i+2u])+carry;
		rp[i+2u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
		prod=static_cast<unsigned __int128>(ap[i+3u])*v+
			 static_cast<unsigned __int128>(rp[i+3u])+carry;
		rp[i+3u]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
	}
	for(;i<n;++i){
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(rp[i])+
									 static_cast<unsigned __int128>(carry);
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
	}
#else
	for(std::size_t i=0;i<n;++i){
		const u128 p=mul_u64(ap[i],v);
		std::uint64_t t0=0;
		const std::uint64_t c1=addc_u64(0,p.lo,rp[i],&t0);
		std::uint64_t out=0;
		const std::uint64_t c2=addc_u64(0,t0,carry,&out);
		rp[i]=out;

		std::uint64_t hi=0;
		const std::uint64_t c3=addc_u64(0,p.hi,c1,&hi);
		const std::uint64_t c4=addc_u64(0,hi,c2,&hi);
		MINI_MP_ASSERT((c3&c4)==0);
		carry=hi;
	}
#endif
	return carry;
}

inline std::uint64_t sm_1n(std::uint64_t*rp,const std::uint64_t*ap,
								  std::size_t n,std::uint64_t v){
	std::uint64_t borrow=0;
#if MINI_MP_DETAIL_HAS_UINT128
	std::size_t i=0;
	for(;i+4u<=n;i+=4u){
		unsigned __int128 prod=
			static_cast<unsigned __int128>(ap[i])*v+borrow;
		std::uint64_t lo=static_cast<std::uint64_t>(prod);
		std::uint64_t hi=static_cast<std::uint64_t>(prod>>64);
		std::uint64_t cur=rp[i];
		rp[i]=cur-lo;
		borrow=hi+((cur<lo)?1u:0u);

		prod=static_cast<unsigned __int128>(ap[i+1u])*v+borrow;
		lo=static_cast<std::uint64_t>(prod);
		hi=static_cast<std::uint64_t>(prod>>64);
		cur=rp[i+1u];
		rp[i+1u]=cur-lo;
		borrow=hi+((cur<lo)?1u:0u);

		prod=static_cast<unsigned __int128>(ap[i+2u])*v+borrow;
		lo=static_cast<std::uint64_t>(prod);
		hi=static_cast<std::uint64_t>(prod>>64);
		cur=rp[i+2u];
		rp[i+2u]=cur-lo;
		borrow=hi+((cur<lo)?1u:0u);

		prod=static_cast<unsigned __int128>(ap[i+3u])*v+borrow;
		lo=static_cast<std::uint64_t>(prod);
		hi=static_cast<std::uint64_t>(prod>>64);
		cur=rp[i+3u];
		rp[i+3u]=cur-lo;
		borrow=hi+((cur<lo)?1u:0u);
	}
	for(;i<n;++i){
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(borrow);
		const std::uint64_t lo=static_cast<std::uint64_t>(prod);
		const std::uint64_t hi=static_cast<std::uint64_t>(prod>>64);
		const std::uint64_t cur=rp[i];
		const std::uint64_t out=cur-lo;
		const std::uint64_t b=(cur<lo)?1u:0u;
		rp[i]=out;
		borrow=hi+b;
	}
#else
	for(std::size_t i=0;i<n;++i){
		const u128 p=mul_u64(ap[i],v);
		std::uint64_t lo=0;
		const std::uint64_t c1=addc_u64(0,p.lo,borrow,&lo);
		std::uint64_t out=0;
		const std::uint64_t b1=subb_u64(0,rp[i],lo,&out);
		rp[i]=out;
		std::uint64_t hi=0;
		const std::uint64_t c2=addc_u64(0,p.hi,c1,&hi);
		std::uint64_t hi2=0;
		const std::uint64_t c3=addc_u64(0,hi,b1,&hi2);
		MINI_MP_ASSERT((c2&c3)==0);
		borrow=hi2;
	}
#endif
	return borrow;
}

inline void add_abs_to(limbs_t&out,const limbs_t&a,const limbs_t&b){
	const std::size_t na=a.size();
	const std::size_t nb=b.size();
	if(na==0){
		out=b;
		return;
	}
	if(nb==0){
		out=a;
		return;
	}
	if(na<=2&&nb<=2){
		const std::uint64_t a0=a[0];
		const std::uint64_t b0=b[0];
		const std::uint64_t a1=(na==2)?a[1]:0;
		const std::uint64_t b1=(nb==2)?b[1]:0;
		std::uint64_t s0=0;
		const std::uint64_t c0=addc_u64(0,a0,b0,&s0);
		std::uint64_t s1=0;
		const std::uint64_t c1=addc_u64(c0,a1,b1,&s1);
		out.clear();
		out.reserve(3);
		out.push_back(s0);
		if(s1!=0||c1!=0){
			out.push_back(s1);
		}
		if(c1!=0){
			out.push_back(c1);
		}
		return;
	}
	const limbs_t*ap=&a;
	const limbs_t*bp=&b;
	if(nb>na){
		ap=&b;
		bp=&a;
	}
	const std::size_t an=ap->size();
	const std::size_t bn=bp->size();
	out.resize_uninit(an+1);
	const std::uint64_t carry=
		(an==bn)?add_nn(out.data(),ap->data(),bp->data(),an)
				:add_nm(out.data(),ap->data(),an,bp->data(),bn);
	out[an]=carry;
	const std::size_t out_n=an+(carry!=0?1u:0u);
	out.resize(out_n);
}

inline void sub_abs_to(limbs_t&out,const limbs_t&a,const limbs_t&b){
	MINI_MP_ASSERT(cmp_abs(a,b)>=0);
	if(b.empty()){
		out=a;
		return;
	}
	out.resize_uninit(a.size());
	const std::uint64_t borrow=
		sub_nm(out.data(),a.data(),a.size(),b.data(),b.size());
	MINI_MP_ASSERT(borrow==0);
	out.resize(lb_nz(out.data(),out.size()));
}

inline limbs_t add_abs(const limbs_t&a,const limbs_t&b){
	limbs_t out;
	add_abs_to(out,a,b);
	return out;
}

inline void add_abs_ip(limbs_t&a,const limbs_t&b){
	const std::size_t na=a.size();
	const std::size_t nb=b.size();
	if(nb==0)
		return;
	if(na==0){
		a=b;
		return;
	}
	if(&a==&b){
		a.reserve(na+1);
		const std::uint64_t carry=add_nn(a.data(),a.data(),a.data(),na);
		if(carry!=0){
			a.push_back(carry);
		}
		return;
	}

	if(na>=nb){
		a.reserve(na+1);
		const std::uint64_t carry=
			(na==nb)?add_nn(a.data(),a.data(),b.data(),na)
					:add_nm(a.data(),a.data(),na,b.data(),nb);
		if(carry!=0){
			a.push_back(carry);
		}
		return;
	}

	
	const std::size_t old_na=na;
	a.reserve(nb+1);
	a.resize(nb);
	std::uint64_t carry=0;
	if(old_na!=0){
		carry=add_nn(a.data(),a.data(),b.data(),old_na);
	}
	carry=
		add_1n(a.data()+static_cast<std::ptrdiff_t>(old_na),
				  b.data()+static_cast<std::ptrdiff_t>(old_na),nb-old_na,carry);
	if(carry!=0){
		a.push_back(carry);
	}
}

inline limbs_t sub_abs(const limbs_t&a,const limbs_t&b){
	limbs_t out;
	sub_abs_to(out,a,b);
	return out;
}

inline void sub_abs_ip(limbs_t&a,const limbs_t&b){
	MINI_MP_ASSERT(cmp_abs(a,b)>=0);
	if(b.empty())
		return;
	const std::uint64_t borrow=
		sub_nm(a.data(),a.data(),a.size(),b.data(),b.size());
	MINI_MP_ASSERT(borrow==0);
	a.resize(lb_nz(a.data(),a.size()));
}

inline void add_one_ip(limbs_t&x){
	if(x.empty()){
		x.push_back(1);
		return;
	}
	x.reserve(x.size()+1);
	const std::uint64_t carry=add_1n(x.data(),x.data(),x.size(),1);
	if(carry!=0)
		x.push_back(carry);
}

inline limbs_t sub_one(limbs_t x){
	MINI_MP_ASSERT(!x.empty());
	const std::uint64_t borrow=sub_1n(x.data(),x.data(),x.size(),1);
	MINI_MP_ASSERT(borrow==0);
	trim_lz(x);
	return x;
}

inline limbs_t bit_and_abs(const limbs_t&a,const limbs_t&b){
	const std::size_t n=std::min(a.size(),b.size());
	limbs_t out;
	if(n==0)
		return out;
	out.resize_uninit(n);
	vecab::and_n(out.data(),a.data(),b.data(),n);
	trim_lz(out);
	return out;
}

inline limbs_t bit_or_abs(const limbs_t&a,const limbs_t&b){
	if(a.empty())
		return b;
	if(b.empty())
		return a;
	const limbs_t*ap=&a;
	const limbs_t*bp=&b;
	if(ap->size()<bp->size())
		std::swap(ap,bp);
	const std::size_t an=ap->size();
	const std::size_t bn=bp->size();
	limbs_t out;
	out.resize_uninit(an);
	vecab::or_n(out.data(),ap->data(),bp->data(),bn);
	if(an>bn){
		std::memcpy(out.data()+static_cast<std::ptrdiff_t>(bn),
					ap->data()+static_cast<std::ptrdiff_t>(bn),
					(an-bn)*sizeof(limb_t));
	}
	return out;
}

inline limbs_t bit_xor_abs(const limbs_t&a,const limbs_t&b){
	if(a.empty())
		return b;
	if(b.empty())
		return a;
	const limbs_t*ap=&a;
	const limbs_t*bp=&b;
	if(ap->size()<bp->size())
		std::swap(ap,bp);
	const std::size_t an=ap->size();
	const std::size_t bn=bp->size();
	limbs_t out;
	out.resize_uninit(an);
	vecab::xor_n(out.data(),ap->data(),bp->data(),bn);
	if(an>bn){
		std::memcpy(out.data()+static_cast<std::ptrdiff_t>(bn),
					ap->data()+static_cast<std::ptrdiff_t>(bn),
					(an-bn)*sizeof(limb_t));
	}
	trim_lz(out);
	return out;
}

inline limbs_t bit_andnot_abs(const limbs_t&a,const limbs_t&mask){
	limbs_t out;
	if(a.empty())
		return out;
	out.resize_uninit(a.size());
	const std::size_t mn=mask.size();
	std::size_t i=0;
	for(;i<a.size()&&i<mn;++i)
		out[i]=a[i]&~mask[i];
	for(;i<a.size();++i)
		out[i]=a[i];
	trim_lz(out);
	return out;
}

inline limbs_t bit_notand_abs(const limbs_t&a,const limbs_t&mask){
	limbs_t out;
	if(mask.empty())
		return out;
	out.resize_uninit(mask.size());
	const std::size_t an=a.size();
	std::size_t i=0;
	for(;i<mask.size()&&i<an;++i)
		out[i]=(~a[i])&mask[i];
	for(;i<mask.size();++i)
		out[i]=mask[i];
	trim_lz(out);
	return out;
}

inline limbs_t bit_and_pn_abs(const limbs_t&a,const limbs_t&b){
	limbs_t out;
	if(a.empty())
		return out;
	out.resize_uninit(a.size());
	std::uint64_t borrow=1;
	for(std::size_t i=0;i<a.size();++i){
		std::uint64_t mask=0;
		if(i<b.size()){
			const std::uint64_t bi=b[i];
			mask=bi-borrow;
			borrow=(bi<borrow)?1u:0u;
		}else{
			borrow=0;
		}
		out[i]=a[i]&~mask;
	}
	trim_lz(out);
	return out;
}

inline limbs_t bit_or_pn_mag(const limbs_t&a,const limbs_t&b){
	limbs_t out;
	out.resize_uninit(b.size());
	std::uint64_t borrow=1;
	std::uint64_t carry=1;
	for(std::size_t i=0;i<b.size();++i){
		const std::uint64_t bi=b[i];
		const std::uint64_t mask=bi-borrow;
		borrow=(bi<borrow)?1u:0u;
		const std::uint64_t av=(i<a.size())?a[i]:0u;
		const std::uint64_t v=(~av)&mask;
		const std::uint64_t sum=v+carry;
		carry=(sum<carry)?1u:0u;
		out[i]=sum;
	}
	if(carry!=0)
		out.push_back(carry);
	trim_lz(out);
	return out;
}

inline limbs_t bit_xor_pn_mag(const limbs_t&a,const limbs_t&b){
	const std::size_t n=std::max(a.size(),b.size());
	limbs_t out;
	out.resize_uninit(n);
	std::uint64_t borrow=1;
	std::uint64_t carry=1;
	for(std::size_t i=0;i<n;++i){
		std::uint64_t mask=0;
		if(i<b.size()){
			const std::uint64_t bi=b[i];
			mask=bi-borrow;
			borrow=(bi<borrow)?1u:0u;
		}else{
			borrow=0;
		}
		const std::uint64_t av=(i<a.size())?a[i]:0u;
		const std::uint64_t v=av^mask;
		const std::uint64_t sum=v+carry;
		carry=(sum<carry)?1u:0u;
		out[i]=sum;
	}
	if(carry!=0)
		out.push_back(carry);
	trim_lz(out);
	return out;
}

inline void twos_neg(limbs_t&x){
	for(auto&v : x)
		v=~v;
	std::uint64_t carry=1;
	for(std::size_t i=0;i<x.size();++i){
		std::uint64_t out=0;
		carry=addc_u64(carry,x[i],0,&out);
		x[i]=out;
		if(carry==0)
			break;
	}
}

inline limbs_t to_2sc(const limbs_t&mag,int sign,
								  std::size_t wid_limb){
	limbs_t out(wid_limb,0);
	const std::size_t ncopy=std::min(wid_limb,mag.size());
	for(std::size_t i=0;i<ncopy;++i)
		out[i]=mag[i];
	if(sign<0)
		twos_neg(out);
	return out;
}

inline std::pair<int,limbs_t> from_2sc(limbs_t tc){
	if(tc.empty())
		return {0,{}};
	const bool negative=((tc.back()>>63)!=0);
	if(!negative){
		trim_lz(tc);
		return {tc.empty()?0:1,std::move(tc)};
	}

	twos_neg(tc);
	trim_lz(tc);
	if(tc.empty())
		return {0,{}};
	return {-1,std::move(tc)};
}

inline void shl_into(limbs_t&out,const limbs_t&x,unsigned s){
	MINI_MP_ASSERT(s<64u);
	if(x.empty()){
		out.clear();
		return;
	}
	if(s==0u){
		if(&out!=&x){
			out.resize_uninit(x.size());
			std::memcpy(out.data(),x.data(),x.size()*sizeof(limb_t));
		}
		return;
	}

	if(&out==&x){
		const std::size_t old_size=out.size();
		out.resize_uninit(old_size+1);
		std::uint64_t carry=0;
		for(std::size_t i=0;i<old_size;++i){
			const std::uint64_t cur=out[i];
			out[i]=(cur<<s)|carry;
			carry=cur>>(64u-s);
		}
		out[old_size]=carry;
		trim_lz(out);
		return;
	}

	out.resize_uninit(x.size()+1);
	std::uint64_t carry=0;
	for(std::size_t i=0;i<x.size();++i){
		const std::uint64_t cur=x[i];
		out[i]=(cur<<s)|carry;
		carry=cur>>(64u-s);
	}
	out[x.size()]=carry;
	trim_lz(out);
}

inline void shl_into(limbs_t&out,const limbs_t&x,std::size_t shift_bits){
	if(shift_bits<64){
		shl_into(out,x,static_cast<unsigned>(shift_bits));
		return;
	}
	if(x.empty()){
		out.clear();
		return;
	}
	if(&out==&x){
		limbs_t tmp=x;
		shl_into(out,tmp,shift_bits);
		return;
	}

	const std::size_t limb_shift=shift_bits/64;
	const unsigned bit_shift=static_cast<unsigned>(shift_bits%64);
	const std::size_t out_size=x.size()+limb_shift+(bit_shift?1:0);
	out.resize_uninit(out_size);

	for(std::size_t i=0;i<limb_shift;++i){
		out[i]=0;
	}

	if(bit_shift==0){
		for(std::size_t i=0;i<x.size();++i){
			out[i+limb_shift]=x[i];
		}
	}else{
		std::uint64_t carry=0;
		for(std::size_t i=0;i<x.size();++i){
			const std::uint64_t cur=x[i];
			out[i+limb_shift]=(cur<<bit_shift)|carry;
			carry=cur>>(64u-bit_shift);
		}
		out[x.size()+limb_shift]=carry;
	}
	trim_lz(out);
}

inline void shl_ip(limbs_t&x,std::size_t shift_bits){
	if(x.empty()||shift_bits==0)
		return;
	const std::size_t limb_shift=shift_bits/64;
	const unsigned bit_shift=static_cast<unsigned>(shift_bits%64);
	const std::size_t old_size=x.size();
	const std::size_t new_size=old_size+limb_shift+(bit_shift?1u:0u);
	x.resize_uninit(new_size);
	if(limb_shift!=0){
		std::memmove(x.data()+static_cast<std::ptrdiff_t>(limb_shift),
					 x.data(),old_size*sizeof(limb_t));
		std::memset(x.data(),0,limb_shift*sizeof(limb_t));
	}
	if(bit_shift==0)
		return;

	std::uint64_t carry=0;
	const std::size_t begin=limb_shift;
	const std::size_t end=limb_shift+old_size;
	for(std::size_t i=begin;i<end;++i){
		const std::uint64_t cur=x[i];
		x[i]=(cur<<bit_shift)|carry;
		carry=cur>>(64u-bit_shift);
	}
	x[end]=carry;
	trim_lz(x);
}

inline limbs_t shl_bits(const limbs_t&x,std::size_t shift_bits){
	limbs_t out;
	shl_into(out,x,shift_bits);
	return out;
}

inline void shr_ip(limbs_t&x,unsigned s){
	MINI_MP_ASSERT(s<64u);
	if(x.empty()||s==0u)
		return;
	std::uint64_t carry=0;
	for(std::size_t src=x.size();src>0;--src){
		const std::uint64_t cur=x[src-1];
		x[src-1]=(cur>>s)|carry;
		carry=cur<<(64u-s);
	}
	trim_lz(x);
}

inline void shr_ip(limbs_t&x,std::size_t shift_bits){
	if(x.empty()||shift_bits==0)
		return;
	if(shift_bits<64){
		shr_ip(x,static_cast<unsigned>(shift_bits));
		return;
	}
	const std::size_t limb_shift=shift_bits/64;
	const unsigned bit_shift=static_cast<unsigned>(shift_bits%64);
	if(limb_shift>=x.size()){
		x.clear();
		return;
	}

	const std::size_t out_size=x.size()-limb_shift;
	if(limb_shift!=0){
		std::memmove(x.data(),
					 x.data()+static_cast<std::ptrdiff_t>(limb_shift),
					 out_size*sizeof(limb_t));
		x.resize(out_size);
	}
	if(bit_shift==0){
		trim_lz(x);
		return;
	}

	std::uint64_t carry=0;
	for(std::size_t src=x.size();src>0;--src){
		const std::uint64_t cur=x[src-1];
		x[src-1]=(cur>>bit_shift)|carry;
		carry=cur<<(64u-bit_shift);
	}
	trim_lz(x);
}

inline void shr_into(limbs_t&out,const limbs_t&x,std::size_t shift_bits){
	if(x.empty()){
		out.clear();
		return;
	}
	if(shift_bits==0){
		if(&out!=&x){
			out.resize_uninit(x.size());
			std::memcpy(out.data(),x.data(),x.size()*sizeof(limb_t));
		}
		return;
	}
	if(&out==&x){
		shr_ip(out,shift_bits);
		return;
	}
	if(shift_bits<64){
		const unsigned s=static_cast<unsigned>(shift_bits);
		const unsigned lshift=64u-s;
		out.resize_uninit(x.size());
		std::size_t i=0;
		for(;i+1u<x.size();++i){
			out[i]=(x[i]>>s)|(x[i+1u]<<lshift);
		}
		out[i]=x[i]>>s;
		trim_lz(out);
		return;
	}
	const std::size_t limb_shift=shift_bits/64;
	const unsigned bit_shift=static_cast<unsigned>(shift_bits%64);
	if(limb_shift>=x.size()){
		out.clear();
		return;
	}

	const std::size_t out_size=x.size()-limb_shift;
	out.resize_uninit(out_size);
	if(bit_shift==0){
		std::memcpy(out.data(),
					x.data()+static_cast<std::ptrdiff_t>(limb_shift),
					out_size*sizeof(limb_t));
		trim_lz(out);
		return;
	}

	const unsigned rshift=64u-bit_shift;
	for(std::size_t i=0;i<out_size;++i){
		const std::size_t src=i+limb_shift;
		const std::uint64_t lo=x[src]>>bit_shift;
		const std::uint64_t hi=
			(src+1u<x.size())?(x[src+1u]<<rshift):0u;
		out[i]=lo|hi;
	}
	trim_lz(out);
}

inline limbs_t shr_bits(const limbs_t&x,std::size_t shift_bits){
	limbs_t out;
	shr_into(out,x,shift_bits);
	return out;
}

inline void muladd_sm(limbs_t&x,std::uint32_t mul,std::uint32_t add){
	if(mul==0){
		x.clear();
		if(add!=0)
			x.push_back(add);
		return;
	}
	if(x.empty()){
		if(add!=0)
			x.push_back(add);
		return;
	}

	std::uint64_t carry=add;
	for(std::size_t i=0;i<x.size();++i){
#if MINI_MP_DETAIL_HAS_UINT128
		const unsigned __int128 z=
			static_cast<unsigned __int128>(x[i])*
			static_cast<std::uint64_t>(mul)+carry;
		x[i]=static_cast<std::uint64_t>(z);
		carry=static_cast<std::uint64_t>(z>>64);
#else
		x[i]=mulad64(x[i],static_cast<std::uint64_t>(mul),0,&carry);
#endif
	}
	if(carry!=0){
		x.push_back(carry);
	}
}

inline void muladd_lb(limbs_t&x,std::uint64_t mul,std::uint64_t add){
	if(mul==0){
		x.clear();
		if(add!=0)
			x.push_back(add);
		return;
	}
	if(x.empty()){
		if(add!=0)
			x.push_back(add);
		return;
	}

	std::uint64_t carry=add;
	for(std::size_t i=0;i<x.size();++i){
#if MINI_MP_DETAIL_HAS_UINT128
		const unsigned __int128 z=
			static_cast<unsigned __int128>(x[i])*mul+carry;
		x[i]=static_cast<std::uint64_t>(z);
		carry=static_cast<std::uint64_t>(z>>64);
#else
		x[i]=mulad64(x[i],mul,0,&carry);
#endif
	}
	if(carry!=0){
		x.push_back(carry);
	}
}

inline std::uint32_t div_small(limbs_t&x,std::uint32_t d){
	MINI_MP_ASSERT(d!=0);
	if(x.empty())
		return 0;

	std::uint64_t rem=0;
	for(std::size_t i=x.size();i>0;--i){
		const std::uint64_t q=udiv128(rem,x[i-1],d,&rem);
		x[i-1]=q;
	}
	trim_lz(x);
	return static_cast<std::uint32_t>(rem);
}

inline limbs_t div_limb(const limbs_t&x,std::uint64_t d,std::uint64_t*rem_out){
	MINI_MP_ASSERT(rem_out!=nullptr);
	MINI_MP_ASSERT(d!=0);
	if(x.empty()){
		*rem_out=0;
		return {};
	}
	if(d==1){
		*rem_out=0;
		return x;
	}
	if(std::has_single_bit(d)){
		*rem_out=x[0]&(d-1u);
		return shr_bits(x,std::countr_zero(d));
	}
	if(x.size()==1){
		const std::uint64_t v=x[0];
		const std::uint64_t q=v/d;
		*rem_out=v%d;
		if(q==0)
			return {};
		return limbs_t{q};
	}
	if(x.size()==2){
		const std::uint64_t q1=x[1]/d;
		std::uint64_t rem1=x[1]%d;
		std::uint64_t rem=0;
		const std::uint64_t q0=udiv128(rem1,x[0],d,&rem);
		*rem_out=rem;
		if(q1==0){
			if(q0==0)
				return {};
			return limbs_t{q0};
		}
		return limbs_t{q0,q1};
	}
	limbs_t q;
	q.resize_uninit(x.size());
	std::uint64_t rem=0;
	for(std::size_t i=x.size();i>0;--i){
		q[i-1]=udiv128(rem,x[i-1],d,&rem);
	}
	trim_lz(q);
	*rem_out=rem;
	return q;
}

inline std::uint64_t div_limb_ip(limbs_t&x,std::uint64_t d){
	MINI_MP_ASSERT(d!=0);
	if(x.empty())
		return 0;
	if(d==1)
		return 0;
	if(std::has_single_bit(d)){
		const std::uint64_t rem=x[0]&(d-1u);
		shr_ip(x,static_cast<unsigned>(std::countr_zero(d)));
		return rem;
	}
	if(x.size()==1){
		const std::uint64_t v=x[0];
		const std::uint64_t q=v/d;
		const std::uint64_t rem=v%d;
		if(q==0)
			x.clear();
		else
			x[0]=q;
		return rem;
	}
	if(x.size()==2){
		const std::uint64_t q1=x[1]/d;
		std::uint64_t rem1=x[1]%d;
		std::uint64_t rem=0;
		const std::uint64_t q0=udiv128(rem1,x[0],d,&rem);
		x[0]=q0;
		if(q1==0){
			x.resize(q0==0?0u:1u);
		}else{
			x[1]=q1;
		}
		return rem;
	}
	std::uint64_t rem=0;
	for(std::size_t i=x.size();i>0;--i){
		x[i-1]=udiv128(rem,x[i-1],d,&rem);
	}
	trim_lz(x);
	return rem;
}

inline std::uint64_t mod_limb(const limbs_t&x,std::uint64_t d){
	MINI_MP_ASSERT(d!=0);
	if(x.empty())
		return 0;
	if(d==1)
		return 0;
	if(std::has_single_bit(d))
		return x[0]&(d-1u);
	if(x.size()==1)
		return x[0]%d;
	if(x.size()==2){
		const std::uint64_t rem1=x[1]%d;
		std::uint64_t rem=0;
		(void)udiv128(rem1,x[0],d,&rem);
		return rem;
	}
	std::uint64_t rem=0;
	for(std::size_t i=x.size();i>0;--i){
		(void)udiv128(rem,x[i-1],d,&rem);
	}
	return rem;
}

inline std::uint32_t mod_small(const limbs_t&x,std::uint32_t d){
	return static_cast<std::uint32_t>(
		mod_limb(x,static_cast<std::uint64_t>(d)));
}

template<std::size_t M>
inline bool sqr_res_ok(std::uint32_t r){
	static const std::array<unsigned char,M> tb=[](){
		std::array<unsigned char,M> out{};
		for(std::size_t i=0;i<M;++i)
			out[(i*i)%M]=1;
		return out;
	}();
	return tb[r%M]!=0;
}

inline std::uint32_t pow_u32(std::uint32_t base,std::size_t exp){
	std::uint64_t acc=1;
	for(std::size_t i=0;i<exp;++i){
		acc*=static_cast<std::uint64_t>(base);
		MINI_MP_ASSERT(acc<=std::numeric_limits<std::uint32_t>::max());
	}
	return static_cast<std::uint32_t>(acc);
}

inline std::pair<std::uint32_t,std::size_t> chunk_par(std::uint32_t base){
	MINI_MP_ASSERT(base>=2);
	std::uint64_t mul=1;
	std::size_t len=0;
	while(mul*base<=std::numeric_limits<std::uint32_t>::max()){
		mul*=base;
		++len;
	}
	if(len==0)
		return {base,1};
	return {static_cast<std::uint32_t>(mul),len};
}

inline std::pair<std::uint64_t,std::size_t>
chunk_par_lb(std::uint32_t base){
	MINI_MP_ASSERT(base>=2);
	std::uint64_t mul=1;
	std::size_t len=0;
	const std::uint64_t maxv=std::numeric_limits<std::uint64_t>::max();
	while(mul<=maxv/base){
		mul*=base;
		++len;
	}
	if(len==0)
		return {base,1};
	return {mul,len};
}

inline void mulbig_in(const limbs_t&a,const limbs_t&b,
					  limbs_t&out);
inline std::pair<limbs_t,limbs_t> dvmk_absl(const limbs_t&u,
											const limbs_t&v);

inline constexpr std::uint64_t kD10Base=10000000000000000000ULL;
inline constexpr std::size_t kD10Len=19;

inline bool prs_d10_part(std::string_view digits,std::size_t pos,
						 std::size_t len,std::uint64_t*v){
	std::uint64_t out_v=0;
	while(len>=4u){
		const unsigned d0=static_cast<unsigned>(digits[pos]-'0');
		const unsigned d1=static_cast<unsigned>(digits[pos+1]-'0');
		const unsigned d2=static_cast<unsigned>(digits[pos+2]-'0');
		const unsigned d3=static_cast<unsigned>(digits[pos+3]-'0');
		if(d0>9u||d1>9u||d2>9u||d3>9u)
			return false;
		out_v=out_v*10000u+
			  static_cast<std::uint64_t>(d0*1000u+d1*100u+d2*10u+d3);
		pos+=4u;
		len-=4u;
	}
	while(len!=0u){
		const unsigned d=static_cast<unsigned>(digits[pos]-'0');
		if(d>9u)
			return false;
		out_v=out_v*10u+static_cast<std::uint64_t>(d);
		++pos;
		--len;
	}
	*v=out_v;
	return true;
}

inline bool prs_d10_chunks(std::string_view digits,
						   std::vector<std::uint64_t>&chunks){
	chunks.clear();
	if(digits.empty())
		return false;

	chunks.reserve((digits.size()+kD10Len-1u)/kD10Len);
	std::size_t pos=0;
	std::size_t first_len=digits.size()%kD10Len;
	if(first_len==0)
		first_len=kD10Len;

	std::uint64_t v=0;
	if(!prs_d10_part(digits,pos,first_len,&v))
		return false;
	chunks.push_back(v);
	pos+=first_len;

	while(pos<digits.size()){
		if(!prs_d10_part(digits,pos,kD10Len,&v))
			return false;
		chunks.push_back(v);
		pos+=kD10Len;
	}
	return true;
}

inline u128 d10_pair_val(std::uint64_t hi,std::uint64_t lo){
	u128 v=mul_u64(hi,kD10Base);
	std::uint64_t out=0;
	const std::uint64_t cy=addc_u64(0,v.lo,lo,&out);
	v.lo=out;
	v.hi+=cy;
	return v;
}

inline void set_u128(limbs_t&out,u128 v){
	out.clear();
	if(v.lo!=0||v.hi!=0)
		out.push_back(v.lo);
	if(v.hi!=0)
		out.push_back(v.hi);
}

inline void muladd_d10_pair(limbs_t&x,std::uint64_t hi,
							std::uint64_t lo){
#if MINI_MP_DETAIL_HAS_UINT128
	const u128 add=d10_pair_val(hi,lo);
	if(x.empty()){
		set_u128(x,add);
		return;
	}
	const u128 mul=mul_u64(kD10Base,kD10Base);
	const std::uint64_t c0=mul.lo;
	const std::uint64_t c1=mul.hi;
	unsigned __int128 carry=add.lo;
	for(std::size_t i=0;i<x.size();++i){
		const std::uint64_t xi=x[i];
		const unsigned __int128 term=
			static_cast<unsigned __int128>(xi)*c0+carry;
		x[i]=static_cast<std::uint64_t>(term);
		carry=(term>>64)+static_cast<unsigned __int128>(xi)*c1;
		if(i==0)
			carry+=add.hi;
	}
	while(carry!=0){
		x.push_back(static_cast<std::uint64_t>(carry));
		carry>>=64;
	}
	trim_lz(x);
#else
	muladd_lb(x,kD10Base,hi);
	muladd_lb(x,kD10Base,lo);
#endif
}

inline bool prs_d10_seq(std::string_view digits,limbs_t&out){
	out.clear();
	if(digits.empty())
		return false;

	out.reserve((digits.size()*3402u)/(64u*1024u)+2u);
	const std::size_t chunk_count=(digits.size()+kD10Len-1u)/kD10Len;
	std::size_t pos=0;
	std::size_t first_len=digits.size()%kD10Len;
	if(first_len==0)
		first_len=kD10Len;

	std::uint64_t v=0;
	if(!prs_d10_part(digits,pos,first_len,&v))
		return false;
	pos+=first_len;
	if((chunk_count&1u)!=0u){
		if(v!=0)
			out.push_back(v);
	}else{
		std::uint64_t lo=0;
		if(!prs_d10_part(digits,pos,kD10Len,&lo))
			return false;
		pos+=kD10Len;
		set_u128(out,d10_pair_val(v,lo));
	}

	while(pos<digits.size()){
		std::uint64_t hi=0;
		std::uint64_t lo=0;
		if(!prs_d10_part(digits,pos,kD10Len,&hi))
			return false;
		pos+=kD10Len;
		if(!prs_d10_part(digits,pos,kD10Len,&lo))
			return false;
		pos+=kD10Len;
		muladd_d10_pair(out,hi,lo);
	}
	trim_lz(out);
	return true;
}

inline void prs_d10_lin(const std::vector<std::uint64_t>&chunks,
						std::size_t lo,std::size_t hi,limbs_t&out){
	out.clear();
	for(std::size_t i=lo;i<hi;++i)
		muladd_lb(out,kD10Base,chunks[i]);
	trim_lz(out);
}

inline void mk_d10_parse_pows(std::size_t chunks,
							  std::vector<limbs_t>&pows){
	pows.clear();
	if(chunks<=1u)
		return;
	limbs_t base;
	base.push_back(kD10Base);
	pows.push_back(std::move(base));
	while((std::size_t(1)<<pows.size())<chunks){
		limbs_t sq;
		mulbig_in(pows.back(),pows.back(),sq);
		pows.push_back(std::move(sq));
	}
}

inline void prs_d10_rec(const std::vector<std::uint64_t>&chunks,
						std::size_t lo,std::size_t hi,
						const std::vector<limbs_t>&pows,
						std::size_t leaf,limbs_t&out){
	const std::size_t n=hi-lo;
	if(n==0){
		out.clear();
		return;
	}
	if(n<=leaf||n==1u){
		prs_d10_lin(chunks,lo,hi,out);
		return;
	}

	std::size_t low_n=std::bit_floor((n+1u)/2u);
	if(low_n==0)
		low_n=1;
	const std::size_t mid=hi-low_n;
	const unsigned lvl=std::countr_zero(low_n);

	limbs_t hi_v;
	limbs_t lo_v;
	prs_d10_rec(chunks,lo,mid,pows,leaf,hi_v);
	prs_d10_rec(chunks,mid,hi,pows,leaf,lo_v);

	if(hi_v.empty()){
		out=std::move(lo_v);
		return;
	}
	limbs_t prod;
	mulbig_in(hi_v,pows[lvl],prod);
	add_abs_to(out,prod,lo_v);
}

inline bool prs_d10(std::string_view digits,limbs_t*out,
					std::size_t leaf=
						std::numeric_limits<std::size_t>::max()){
	MINI_MP_ASSERT(out!=nullptr);
	out->clear();
	const std::size_t chunk_count=(digits.size()+kD10Len-1u)/kD10Len;
	if(leaf==std::numeric_limits<std::size_t>::max()||
	   chunk_count<leaf){
		return prs_d10_seq(digits,*out);
	}

	std::vector<std::uint64_t> chunks;
	if(!prs_d10_chunks(digits,chunks))
		return false;
	if(chunks.empty())
		return true;

	out->reserve((digits.size()*3402u)/(64u*1024u)+2u);

	std::vector<limbs_t> pows;
	mk_d10_parse_pows(chunks.size(),pows);
	if(pows.empty()){
		prs_d10_lin(chunks,0,chunks.size(),*out);
		return true;
	}
	prs_d10_rec(chunks,0,chunks.size(),pows,leaf,*out);
	trim_lz(*out);
	return true;
}

inline void app_d10(std::string&out,std::uint64_t v){
	if(v==0){
		out.push_back('0');
		return;
	}
	char buf[20];
	std::size_t n=0;
	while(v!=0){
		const std::uint64_t q=v/10u;
		buf[n++]=static_cast<char>('0'+(v-q*10u));
		v=q;
	}
	while(n>0)
		out.push_back(buf[--n]);
}

inline void app_d10_pad(std::string&out,std::uint64_t v){
	char buf[kD10Len];
	for(std::size_t i=0;i<kD10Len;++i){
		const std::uint64_t q=v/10u;
		buf[kD10Len-1u-i]=
			static_cast<char>('0'+(v-q*10u));
		v=q;
	}
	out.append(buf,buf+kD10Len);
}

inline void app_d10_zero(std::string&out,std::size_t chunks){
	out.append(chunks*kD10Len,'0');
}

inline std::size_t d10_span(std::size_t lvl){
	if(lvl>=std::numeric_limits<std::size_t>::digits)
		throw_ovf("d10_span: size overflow");
	return std::size_t(1)<<lvl;
}

inline void to_str_d10_bc(std::string&out,const limbs_t&x,bool pad,
						  std::size_t span){
	limbs_t tmp=x;
	std::vector<std::uint64_t> chunks;
	chunks.reserve(tmp.size()*2u+1u);
	while(!tmp.empty())
		chunks.push_back(div_limb_ip(tmp,kD10Base));
	if(chunks.empty())
		chunks.push_back(0);

	if(pad){
		if(span<chunks.size())
			span=chunks.size();
		app_d10_zero(out,span-chunks.size());
		for(std::size_t i=chunks.size();i>0;--i)
			app_d10_pad(out,chunks[i-1]);
		return;
	}

	app_d10(out,chunks.back());
	for(std::size_t i=chunks.size();i>1;--i)
		app_d10_pad(out,chunks[i-2]);
}

inline void to_str_d10_rec(std::string&out,const limbs_t&x,
						   const std::vector<limbs_t>&pows,
						   std::size_t lvl,bool pad,
						   std::size_t leaf){
	if(lvl==0||x.size()<leaf){
		to_str_d10_bc(out,x,pad,d10_span(lvl));
		return;
	}

	const limbs_t&div=pows[lvl-1u];
	auto qr=dvmk_absl(x,div);
	const limbs_t&q=qr.first;
	const limbs_t&r=qr.second;
	if(!pad&&q.empty()){
		to_str_d10_rec(out,r,pows,lvl-1u,false,leaf);
		return;
	}
	to_str_d10_rec(out,q,pows,lvl-1u,pad,leaf);
	to_str_d10_rec(out,r,pows,lvl-1u,true,leaf);
}

inline void mk_d10_pows(const limbs_t&x,std::vector<limbs_t>&pows){
	pows.clear();
	limbs_t base;
	base.push_back(kD10Base);
	pows.push_back(std::move(base));
	while(cmp_abs(pows.back(),x)<=0){
		limbs_t sq;
		mulbig_in(pows.back(),pows.back(),sq);
		pows.push_back(std::move(sq));
	}
}

inline std::string to_str_d10(const limbs_t&x,int sign,std::size_t leaf){
	if(x.empty())
		return "0";

	std::string out;
	const std::size_t approx_digits=(bit_length(x)*1234u)/4096u+1u;
	out.reserve(approx_digits+((sign<0)?1u:0u));
	if(sign<0)
		out.push_back('-');

	if(x.size()<leaf){
		to_str_d10_bc(out,x,false,0);
		return out;
	}

	std::vector<limbs_t> pows;
	mk_d10_pows(x,pows);
	if(pows.size()<=1u){
		to_str_d10_bc(out,x,false,0);
		return out;
	}
	to_str_d10_rec(out,x,pows,pows.size()-1u,false,leaf);
	return out;
}

inline int ch_to_dig(char c) noexcept;
inline char dig_to_ch(unsigned d) noexcept;

inline int hex_to_dig(char c) noexcept{
	const unsigned uc=static_cast<unsigned char>(c);
	unsigned v=uc-static_cast<unsigned>('0');
	if(v<10u)
		return static_cast<int>(v);
	v=(uc|0x20u)-static_cast<unsigned>('a');
	if(v<6u)
		return static_cast<int>(v+10u);
	return -1;
}

inline bool ispow2_32(std::uint32_t x) noexcept{
	return x!=0u&&(x&(x-1u))==0u;
}

inline unsigned log2p2_32(std::uint32_t x) noexcept{
	MINI_MP_ASSERT(ispow2_32(x));
	return std::countr_zero(x);
}

inline std::uint64_t mulmod64(std::uint64_t a,std::uint64_t b,
								 std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(mod!=0);
#if MINI_MP_DETAIL_HAS_UINT128
	return static_cast<std::uint64_t>(
		(static_cast<unsigned __int128>(a)*b)%mod);
#else
	const u128 p=mul_u64(a,b);
	std::uint64_t rem=0;
	(void)udiv128(p.hi,p.lo,mod,&rem);
	return rem;
#endif
}

inline std::uint64_t addmod64_red(std::uint64_t a,std::uint64_t b,
								  std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(mod!=0);
	MINI_MP_ASSERT(a<mod&&b<mod);
	std::uint64_t lo=0;
	const std::uint64_t hi=addc_u64(0,a,b,&lo);
	if(hi!=0)
		return lo-mod;
	if(lo>=mod)
		lo-=mod;
	return lo;
}

inline std::uint64_t addmod64(std::uint64_t a,std::uint64_t b,
							  std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(mod!=0);
	return addmod64_red(a%mod,b%mod,mod);
}

inline bool invmod64(std::uint64_t*out,std::uint64_t a,
					 std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(out!=nullptr);
	MINI_MP_ASSERT(mod>1);
	a%=mod;
	if(a==0)
		return false;

#if MINI_MP_DETAIL_HAS_UINT128
	__int128 t=0;
	__int128 nt=1;
	std::uint64_t r=mod;
	std::uint64_t nr=a;
	while(nr!=0){
		const std::uint64_t q=r/nr;
		const __int128 tt=t-static_cast<__int128>(q)*nt;
		t=nt;
		nt=tt;
		const std::uint64_t rr=r-q*nr;
		r=nr;
		nr=rr;
	}
	if(r!=1u)
		return false;
	if(t<0)
		t+=mod;
	*out=static_cast<std::uint64_t>(t);
	return true;
#else
	std::uint64_t t=0;
	std::uint64_t nt=1;
	std::uint64_t r=mod;
	std::uint64_t nr=a;
	while(nr!=0){
		const std::uint64_t q=r/nr;
		const std::uint64_t qnt=mulmod64(q%mod,nt,mod);
		const std::uint64_t tt=(t>=qnt)?(t-qnt):(mod-(qnt-t));
		t=nt;
		nt=tt;
		const std::uint64_t rr=r-q*nr;
		r=nr;
		nr=rr;
	}
	if(r!=1u)
		return false;
	*out=t;
	return true;
#endif
}

inline std::uint64_t invmod64_coprime(std::uint64_t a,
									  std::uint64_t mod) noexcept{
	std::uint64_t out=0;
	const bool ok=invmod64(&out,a,mod);
	MINI_MP_ASSERT(ok);
	return out;
}

inline std::uint64_t invodd264(std::uint64_t n) noexcept{
	MINI_MP_ASSERT((n&1u)!=0u);
	std::uint64_t x=1;
	
	for(int i=0;i<6;++i){
		x*=(2u-n*x);
	}
	return x;
}

inline bool divex_lb(limbs_t&x,std::uint64_t d){
	MINI_MP_ASSERT(d!=0);
	if(x.empty()||d==1)
		return true;
	if((d&1u)==0u){
		const unsigned sh=std::countr_zero(d);
		const std::uint64_t mask=(std::uint64_t(1)<<sh)-1u;
		if((x[0]&mask)!=0u)
			return false;
		shr_ip(x,sh);
		d>>=sh;
		if(d==1)
			return true;
	}

	const std::uint64_t inv=invodd264(d);
	std::uint64_t carry=0;
	for(std::size_t i=0;i<x.size();++i){
		std::uint64_t lo=0;
		const std::uint64_t borrow=subb_u64(0,x[i],carry,&lo);
		const std::uint64_t q=lo*inv;
		const u128 p=mul_u64(q,d);
		MINI_MP_ASSERT(p.lo==lo);
		std::uint64_t next=0;
		const std::uint64_t cy=addc_u64(0,p.hi,borrow,&next);
		MINI_MP_ASSERT(cy==0);
		x[i]=q;
		carry=next;
	}
	if(carry!=0)
		return false;
	trim_lz(x);
	return true;
}

inline bool divex_odd(const limbs_t&num,const limbs_t&den,limbs_t&out){
	MINI_MP_ASSERT(!den.empty());
	MINI_MP_ASSERT((den[0]&1u)!=0u);
	out.clear();
	if(num.empty())
		return true;
	if(cmp_abs(num,den)<0)
		return false;
	if(den.size()==1){
		out=num;
		return divex_lb(out,den[0]);
	}

	const std::size_t qn=num.size()-den.size()+1u;
	limbs_t rem=num;
	rem.push_back(0);
	out.resize(qn,0);

	const std::uint64_t inv=invodd264(den[0]);
	for(std::size_t i=0;i<qn;++i){
		const std::uint64_t q=rem[i]*inv;
		out[i]=q;
		if(q!=0){
			std::uint64_t cy=sm_1n(
				rem.data()+static_cast<std::ptrdiff_t>(i),
				den.data(),den.size(),q);
			std::size_t k=i+den.size();
			while(cy!=0&&k<rem.size()){
				std::uint64_t v=0;
				cy=subb_u64(0,rem[k],cy,&v);
				rem[k]=v;
				++k;
			}
			if(cy!=0)
				return false;
		}
		MINI_MP_ASSERT(rem[i]==0);
	}
	for(std::size_t i=qn;i<rem.size();++i){
		if(rem[i]!=0)
			return false;
	}
	trim_lz(out);
	return true;
}

struct Mont64Ctx{
	std::uint64_t mod;
	std::uint64_t nprime; 
	std::uint64_t r2;	  
};

inline Mont64Ctx mk_m64ctx(std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(mod>1&&(mod&1u)!=0u);
	Mont64Ctx c{};
	c.mod=mod;
	c.nprime=0u-invodd264(mod);

	std::uint64_t r=0;
	(void)udiv128(1u,0u,mod,&r); 
	c.r2=mulmod64(r,r,mod);		 
	return c;
}

inline std::uint64_t mred_u128(u128 t,
											const Mont64Ctx&c) noexcept{
	const std::uint64_t m=t.lo*c.nprime;
	const u128 mn=mul_u64(m,c.mod);

	std::uint64_t lo=0;
	const std::uint64_t c1=addc_u64(0,t.lo,mn.lo,&lo);
	MINI_MP_ASSERT(lo==0); 

	std::uint64_t hi=0;
	const std::uint64_t c2=addc_u64(c1,t.hi,mn.hi,&hi);
	u128 u{hi,c2}; 

	while(u.hi!=0||u.lo>=c.mod){
		std::uint64_t lo2=0;
		std::uint64_t b=subb_u64(0,u.lo,c.mod,&lo2);
		std::uint64_t hi2=0;
		b=subb_u64(b,u.hi,0,&hi2);
		MINI_MP_ASSERT(b==0);
		u.lo=lo2;
		u.hi=hi2;
	}
	return u.lo;
}

inline std::uint64_t mmul_u64(std::uint64_t a_bar,std::uint64_t b_bar,
										const Mont64Ctx&c) noexcept{
	return mred_u128(mul_u64(a_bar,b_bar),c);
}

inline std::uint64_t mto_u64(std::uint64_t x,
									   const Mont64Ctx&c) noexcept{
	return mmul_u64(x%c.mod,c.r2,c);
}

inline std::uint64_t mfr_u64(std::uint64_t x_bar,
										 const Mont64Ctx&c) noexcept{
	return mred_u128({x_bar,0},c);
}

inline std::uint64_t powmod64(std::uint64_t a,std::uint64_t e,
								 std::uint64_t mod) noexcept{
	MINI_MP_ASSERT(mod!=0);
	if(mod==1)
		return 0;

	if((mod&1u)!=0u){
		const Mont64Ctx c=mk_m64ctx(mod);
		std::uint64_t base=mto_u64(a,c);
		std::uint64_t res=mto_u64(1,c);
		while(e!=0){
			if((e&1u)!=0u)
				res=mmul_u64(res,base,c);
			e>>=1;
			if(e!=0)
				base=mmul_u64(base,base,c);
		}
		return mfr_u64(res,c);
	}

	std::uint64_t base=a%mod;
	std::uint64_t res=1%mod;
	while(e!=0){
		if((e&1u)!=0u)
			res=mulmod64(res,base,mod);
		e>>=1;
		if(e!=0)
			base=mulmod64(base,base,mod);
	}
	return res;
}

inline std::uint32_t extbit32(const limbs_t&x,std::size_t bit_pos,
									  unsigned width) noexcept{
	MINI_MP_ASSERT(width>0&&width<=32);
	const std::size_t i=bit_pos/64;
	const unsigned off=static_cast<unsigned>(bit_pos%64);
	if(i>=x.size())
		return 0;

	std::uint64_t v=x[i]>>off;
	if(off+width>64&&(i+1)<x.size()){
		v|=(x[i+1]<<(64u-off));
	}
	const std::uint64_t mask=
		(width==32)?0xffffffffULL:((std::uint64_t(1)<<width)-1u);
	return static_cast<std::uint32_t>(v&mask);
}

inline bool prs_p2dl(std::string_view digits,int base,
									   limbs_t*out){
	MINI_MP_ASSERT(out!=nullptr);
	out->clear();
	if(digits.empty())
		return false;
	const std::uint32_t ub=static_cast<std::uint32_t>(base);
	if(!ispow2_32(ub)||ub<2u||ub>32u)
		return false;
	if(base==16){
		const std::size_t limb_count=(digits.size()+15u)/16u;
		out->resize_uninit(limb_count);
		std::size_t limb_index=0;
		for(std::size_t end=digits.size();end>0;){
			const std::size_t begin=(end>16u)?(end-16u):0u;
			std::uint64_t limb=0;
			for(std::size_t i=begin;i<end;++i){
				const int d=hex_to_dig(digits[i]);
				if(d<0)
					return false;
				limb=(limb<<4u)|static_cast<std::uint64_t>(d);
			}
			(*out)[limb_index++]=limb;
			end=begin;
		}
		out->resize(limb_index);
		trim_lz(*out);
		return true;
	}
	const unsigned bit_dig=log2p2_32(ub);
	out->reserve((digits.size()*bit_dig+63u)/64u);

	if((64u%bit_dig)==0u){
		const std::size_t digs_per_limb=64u/bit_dig;
		for(std::size_t end=digits.size();end>0;){
			const std::size_t begin=
				(end>digs_per_limb)?(end-digs_per_limb):0;
			std::uint64_t limb=0;
			for(std::size_t i=begin;i<end;++i){
				const int d=ch_to_dig(digits[i]);
				if(d<0||d>=base)
					return false;
				limb=(limb<<bit_dig)|static_cast<std::uint64_t>(d);
			}
			out->push_back(limb);
			end=begin;
		}
		trim_lz(*out);
		return true;
	}

	std::uint64_t cur=0;
	unsigned cur_bits=0;
	for(std::size_t p=digits.size();p>0;--p){
		const int d=ch_to_dig(digits[p-1]);
		if(d<0||d>=base)
			return false;
		const std::uint64_t v=static_cast<std::uint64_t>(d);

		if(cur_bits+bit_dig<64u){
			cur|=(v<<cur_bits);
			cur_bits+=bit_dig;
		}else if(cur_bits+bit_dig==64u){
			cur|=(v<<cur_bits);
			out->push_back(cur);
			cur=0;
			cur_bits=0;
		}else{
			const unsigned take_low=64u-cur_bits;
			const std::uint64_t low_mask=(std::uint64_t(1)<<take_low)-1u;
			cur|=((v&low_mask)<<cur_bits);
			out->push_back(cur);
			cur=(v>>take_low);
			cur_bits=bit_dig-take_low;
		}
	}
	if(cur_bits!=0u)
		out->push_back(cur);
	trim_lz(*out);
	return true;
}

inline std::string to_str_p2(const limbs_t&x,int sign,int base){
	MINI_MP_ASSERT(base>=2&&base<=32);
	MINI_MP_ASSERT(ispow2_32(static_cast<std::uint32_t>(base)));
	if(x.empty())
		return "0";
	const unsigned bit_dig=
		log2p2_32(static_cast<std::uint32_t>(base));
	if(base==16){
		static constexpr char kHex[]="0123456789abcdef";
		std::string out;
		out.reserve(x.size()*16u+((sign<0)?1u:0u));
		if(sign<0)
			out.push_back('-');

		const std::uint64_t hi=x.back();
		const unsigned hi_bits=64u-std::countl_zero(hi);
		const std::size_t hi_digits=(hi_bits+3u)/4u;
		for(std::size_t k=hi_digits;k>0;--k){
			const unsigned shift=static_cast<unsigned>((k-1u)*4u);
			out.push_back(kHex[(hi>>shift)&0x0fU]);
		}

		for(std::size_t limb_i=x.size()-1;limb_i>0;--limb_i){
			const std::uint64_t limb=x[limb_i-1];
			char buf[16];
			for(std::size_t k=0;k<16u;++k){
				const unsigned shift=static_cast<unsigned>((15u-k)*4u);
				buf[k]=kHex[(limb>>shift)&0x0fU];
			}
			out.append(buf,buf+16);
		}
		return out;
	}
	if((64u%bit_dig)==0u){
		const std::size_t digs_per_limb=64u/bit_dig;
		const std::uint64_t mask=(bit_dig==64u)
			?~std::uint64_t(0)
			:((std::uint64_t(1)<<bit_dig)-1u);
		std::string out;
		out.reserve(x.size()*digs_per_limb+((sign<0)?1u:0u));
		if(sign<0)
			out.push_back('-');

		std::uint64_t hi=x.back();
		bool started=false;
		for(std::size_t p=digs_per_limb;p>0;--p){
			const unsigned shift=
				static_cast<unsigned>((p-1u)*bit_dig);
			const std::uint32_t d=
				static_cast<std::uint32_t>((hi>>shift)&mask);
			if(!started&&d==0u)
				continue;
			started=true;
			out.push_back(dig_to_ch(d));
		}
		if(!started)
			out.push_back('0');

		for(std::size_t limb_i=x.size()-1;limb_i>0;--limb_i){
			const std::uint64_t limb=x[limb_i-1];
			char buf[32];
			MINI_MP_ASSERT(digs_per_limb<=sizeof(buf));
			for(std::size_t k=0;k<digs_per_limb;++k){
				const unsigned shift=static_cast<unsigned>(
					(digs_per_limb-1u-k)*bit_dig);
				const std::uint32_t d=
					static_cast<std::uint32_t>((limb>>shift)&mask);
				buf[k]=dig_to_ch(d);
			}
			out.append(buf,buf+digs_per_limb);
		}
		return out;
	}
	const std::size_t bits=bit_length(x);
	const std::size_t ndigits=(bits+bit_dig-1u)/bit_dig;

	std::string out;
	out.reserve(ndigits+((sign<0)?1u:0u));
	if(sign<0)
		out.push_back('-');

	bool started=false;
	for(std::size_t p=ndigits;p>0;--p){
		const std::size_t bit_pos=(p-1)*bit_dig;
		const std::uint32_t d=extbit32(x,bit_pos,bit_dig);
		if(!started){
			if(d==0u)
				continue;
			started=true;
		}
		out.push_back(dig_to_ch(d));
	}
	if(!started)
		out.push_back('0');
	return out;
}

inline bool is_prp64(std::uint64_t n) noexcept{
	if(n<2u)
		return false;
	if((n&1u)==0u)
		return n==2u;

	static constexpr std::uint32_t kSmPrime[]={3u, 5u, 7u, 11u,13u,17u,
												   19u,23u,29u,31u,37u};
	for(std::uint32_t p : kSmPrime){
		if(n==p)
			return true;
		if((n%p)==0u)
			return false;
	}

	std::uint64_t d=n-1u;
	unsigned s=0;
	while((d&1u)==0u){
		d>>=1;
		++s;
	}

	
	static constexpr std::uint64_t kWitnesses[]={
		2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL};

	for(std::uint64_t a0 : kWitnesses){
		const std::uint64_t a=a0%n;
		if(a==0u)
			continue;
		std::uint64_t x=powmod64(a,d,n);
		if(x==1u||x==(n-1u))
			continue;

		bool witness=true;
		for(unsigned r=1;r<s;++r){
			x=mulmod64(x,x,n);
			if(x==(n-1u)){
				witness=false;
				break;
			}
		}
		if(witness)
			return false;
	}
	return true;
}

inline constexpr std::size_t kCbaMax=4;

inline limbs_t mulcba_ab(const limbs_t&a,const limbs_t&b){
	MINI_MP_ASSERT(!a.empty()&&!b.empty());
	MINI_MP_ASSERT(std::max(a.size(),b.size())<=kCbaMax);

	const std::size_t na=a.size();
	const std::size_t nb=b.size();
	const std::size_t nout=na+nb;
	limbs_t out(nout,0);
	std::uint64_t carry0=0;
	std::uint64_t carry1=0;

	for(std::size_t k=0;k<nout;++k){
		std::uint64_t acc0=carry0;
		std::uint64_t acc1=carry1;
		std::uint64_t acc2=0;
		const std::size_t i_begin=(k>=nb)?(k-nb+1):0;
		const std::size_t i_end=std::min(k,na-1);
		for(std::size_t i=i_begin;i<=i_end;++i){
			const std::size_t j=k-i;
			const u128 p=mul_u64(a[i],b[j]);

			std::uint64_t t0=0;
			const std::uint64_t c0=addc_u64(0,acc0,p.lo,&t0);
			acc0=t0;

			std::uint64_t t1=0;
			const std::uint64_t c1=addc_u64(0,acc1,p.hi,&t1);
			std::uint64_t t2=0;
			const std::uint64_t c2=addc_u64(0,t1,c0,&t2);
			acc1=t2;

			std::uint64_t t3=0;
			const std::uint64_t c3=addc_u64(0,acc2,c1,&t3);
			std::uint64_t t4=0;
			const std::uint64_t c4=addc_u64(0,t3,c2,&t4);
			acc2=t4;
			MINI_MP_ASSERT((c3|c4)==0);
		}
		out[k]=acc0;
		carry0=acc1;
		carry1=acc2;
	}
	if(carry0!=0||carry1!=0){
		out.push_back(carry0);
		if(carry1!=0)
			out.push_back(carry1);
	}
	trim_lz(out);
	return out;
}

inline limbs_t sqrcba_ab(const limbs_t&x){
	MINI_MP_ASSERT(!x.empty());
	MINI_MP_ASSERT(x.size()<=kCbaMax);
	return mulcba_ab(x,x);
}

inline limbs_t mulsbk_ab(const limbs_t&a,const limbs_t&b){
	if(a.empty()||b.empty())
		return {};
	if(std::max(a.size(),b.size())<=kCbaMax)
		return mulcba_ab(a,b);
	const limbs_t*up=&a;
	const limbs_t*vp=&b;
	if(up->size()<vp->size())
		std::swap(up,vp);
	const std::size_t un=up->size();
	const std::size_t vn=vp->size();

	limbs_t out;
	out.resize_uninit(un+vn);
	std::memset(out.data(),0,(un+vn)*sizeof(limb_t));
	out[un]=mul_1n(out.data(),up->data(),un,(*vp)[0]);
	for(std::size_t j=1;j<vn;++j){
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(j),up->data(),un,(*vp)[j]);
		out[j+un]=cy;
	}
	out.resize(lb_nz(out.data(),un+vn));
	return out;
}

inline void mulsbk_in(const limbs_t&a,const limbs_t&b,
									limbs_t&out){
	if(a.empty()||b.empty()){
		out.clear();
		return;
	}
	if(std::max(a.size(),b.size())<=kCbaMax){
		out=mulcba_ab(a,b);
		return;
	}
	const limbs_t*up=&a;
	const limbs_t*vp=&b;
	if(up->size()<vp->size())
		std::swap(up,vp);
	const std::size_t un=up->size();
	const std::size_t vn=vp->size();
	const std::size_t need=un+vn;
	out.resize_uninit(need);
	std::memset(out.data(),0,need*sizeof(limb_t));
	out[un]=mul_1n(out.data(),up->data(),un,(*vp)[0]);
	for(std::size_t j=1;j<vn;++j){
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(j),up->data(),un,(*vp)[j]);
		out[j+un]=cy;
	}
	out.resize(lb_nz(out.data(),need));
}

inline void add_lb_at(limbs_t&acc,std::uint64_t v,std::size_t idx){
	while(v!=0){
		if(idx>=acc.size())
			acc.push_back(0);
		std::uint64_t out=0;
		const std::uint64_t carry=addc_u64(0,acc[idx],v,&out);
		acc[idx]=out;
		v=carry;
		++idx;
	}
}

inline limbs_t sqrsbk_ab(const limbs_t&x){
	if(x.empty())
		return {};
	if(x.size()<=kCbaMax)
		return sqrcba_ab(x);

	const std::size_t n=x.size();
	limbs_t out;
	out.resize_uninit(2u*n+1u);
	std::memset(out.data(),0,(2u*n+1u)*sizeof(limb_t));
	for(std::size_t i=0;i+1u<n;++i){
		const std::size_t len=n-i-1u;
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(2u*i+1u),
			x.data()+static_cast<std::ptrdiff_t>(i+1u),
			len,x[i]);
		add_lb_at(out,cy,n+i);
	}

	std::uint64_t carry=0;
	for(std::size_t i=0;i<out.size();++i){
		const std::uint64_t next=out[i]>>63u;
		out[i]=(out[i]<<1u)|carry;
		carry=next;
	}
	if(carry!=0)
		out.push_back(carry);

	for(std::size_t i=0;i<n;++i){
		const u128 p=mul_u64(x[i],x[i]);
		const std::size_t pos=2u*i;
		std::uint64_t s0=0;
		std::uint64_t cy=addc_u64(0,out[pos],p.lo,&s0);
		out[pos]=s0;
		std::uint64_t s1=0;
		const std::uint64_t cy2=addc_u64(cy,out[pos+1u],p.hi,&s1);
		out[pos+1u]=s1;
		add_lb_at(out,cy2,pos+2u);
	}
	out.resize(lb_nz(out.data(),out.size()));
	return out;
}

struct lbv_t{
	const std::uint64_t*p=nullptr;
	std::size_t n=0;

	bool empty() const noexcept{ return n==0; }
	std::size_t size() const noexcept{ return n; }
	const std::uint64_t*data() const noexcept{ return p; }
	const std::uint64_t&operator[](std::size_t i) const noexcept{
		return p[i];
	}
};

inline lbv_t lbv_trim(lbv_t x) noexcept{
	while(x.n!=0&&x.p[x.n-1u]==0)
		--x.n;
	if(x.n==0)
		x.p=nullptr;
	return x;
}

inline lbv_t lbv_all(const limbs_t&x) noexcept{
	return x.empty()?lbv_t{}:lbv_t{x.data(),x.size()};
}

inline lbv_t lbv_slc(lbv_t x,std::size_t begin,
					 std::size_t end) noexcept{
	if(begin>=end||begin>=x.n)
		return {};
	end=std::min(end,x.n);
	return lbv_trim({x.p+static_cast<std::ptrdiff_t>(begin),end-begin});
}

inline int cmp_abs(lbv_t a,lbv_t b) noexcept{
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.n!=b.n)
		return (a.n<b.n)?-1:1;
	return vecab::cmp_n(a.p,b.p,a.n);
}

inline void add_abs_to(limbs_t&out,lbv_t a,lbv_t b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	const std::size_t na=a.n;
	const std::size_t nb=b.n;
	if(na==0){
		if(nb==0){
			out.clear();
			return;
		}
		out.assign(b.p,b.p+static_cast<std::ptrdiff_t>(nb));
		return;
	}
	if(nb==0){
		out.assign(a.p,a.p+static_cast<std::ptrdiff_t>(na));
		return;
	}
	if(na<=2&&nb<=2){
		const std::uint64_t a0=a[0];
		const std::uint64_t b0=b[0];
		const std::uint64_t a1=(na==2)?a[1]:0;
		const std::uint64_t b1=(nb==2)?b[1]:0;
		std::uint64_t s0=0;
		const std::uint64_t c0=addc_u64(0,a0,b0,&s0);
		std::uint64_t s1=0;
		const std::uint64_t c1=addc_u64(c0,a1,b1,&s1);
		out.clear();
		out.reserve(3);
		out.push_back(s0);
		if(s1!=0||c1!=0)
			out.push_back(s1);
		if(c1!=0)
			out.push_back(c1);
		return;
	}
	lbv_t x=a;
	lbv_t y=b;
	if(y.n>x.n)
		std::swap(x,y);
	out.resize_uninit(x.n+1u);
	const std::uint64_t carry=
		(x.n==y.n)?add_nn(out.data(),x.p,y.p,x.n)
				  :add_nm(out.data(),x.p,x.n,y.p,y.n);
	out[x.n]=carry;
	out.resize(x.n+(carry!=0?1u:0u));
}

inline limbs_t add_abs(lbv_t a,lbv_t b){
	limbs_t out;
	add_abs_to(out,a,b);
	return out;
}

inline void sub_abs_to(limbs_t&out,lbv_t a,lbv_t b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	MINI_MP_ASSERT(cmp_abs(a,b)>=0);
	if(b.empty()){
		if(a.empty())
			out.clear();
		else
			out.assign(a.p,a.p+static_cast<std::ptrdiff_t>(a.n));
		return;
	}
	out.resize_uninit(a.n);
	const std::uint64_t borrow=sub_nm(out.data(),a.p,a.n,b.p,b.n);
	MINI_MP_ASSERT(borrow==0);
	out.resize(lb_nz(out.data(),out.size()));
}

inline limbs_t dif_abs(lbv_t a,lbv_t b,int*sign){
	MINI_MP_ASSERT(sign!=nullptr);
	const int cmp=cmp_abs(a,b);
	if(cmp==0){
		*sign=0;
		return {};
	}
	limbs_t out;
	if(cmp>0){
		*sign=1;
		sub_abs_to(out,a,b);
	}else{
		*sign=-1;
		sub_abs_to(out,b,a);
	}
	return out;
}

inline limbs_t mulcba_v(lbv_t a,lbv_t b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	MINI_MP_ASSERT(!a.empty()&&!b.empty());
	MINI_MP_ASSERT(std::max(a.n,b.n)<=kCbaMax);

	const std::size_t nout=a.n+b.n;
	limbs_t out(nout,0);
	std::uint64_t carry0=0;
	std::uint64_t carry1=0;

	for(std::size_t k=0;k<nout;++k){
		std::uint64_t acc0=carry0;
		std::uint64_t acc1=carry1;
		std::uint64_t acc2=0;
		const std::size_t i_begin=(k>=b.n)?(k-b.n+1u):0;
		const std::size_t i_end=std::min(k,a.n-1u);
		for(std::size_t i=i_begin;i<=i_end;++i){
			const std::size_t j=k-i;
			const u128 p=mul_u64(a[i],b[j]);

			std::uint64_t t0=0;
			const std::uint64_t c0=addc_u64(0,acc0,p.lo,&t0);
			acc0=t0;

			std::uint64_t t1=0;
			const std::uint64_t c1=addc_u64(0,acc1,p.hi,&t1);
			std::uint64_t t2=0;
			const std::uint64_t c2=addc_u64(0,t1,c0,&t2);
			acc1=t2;

			std::uint64_t t3=0;
			const std::uint64_t c3=addc_u64(0,acc2,c1,&t3);
			std::uint64_t t4=0;
			const std::uint64_t c4=addc_u64(0,t3,c2,&t4);
			acc2=t4;
			MINI_MP_ASSERT((c3|c4)==0);
		}
		out[k]=acc0;
		carry0=acc1;
		carry1=acc2;
	}
	if(carry0!=0||carry1!=0){
		out.push_back(carry0);
		if(carry1!=0)
			out.push_back(carry1);
	}
	trim_lz(out);
	return out;
}

inline limbs_t mulsbk_v(lbv_t a,lbv_t b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.empty()||b.empty())
		return {};
	if(std::max(a.n,b.n)<=kCbaMax)
		return mulcba_v(a,b);
	lbv_t u=a;
	lbv_t v=b;
	if(u.n<v.n)
		std::swap(u,v);
	limbs_t out;
	out.resize_uninit(u.n+v.n);
	std::memset(out.data(),0,(u.n+v.n)*sizeof(limb_t));
	out[u.n]=mul_1n(out.data(),u.p,u.n,v[0]);
	for(std::size_t j=1;j<v.n;++j){
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(j),u.p,u.n,v[j]);
		out[j+u.n]=cy;
	}
	out.resize(lb_nz(out.data(),u.n+v.n));
	return out;
}

inline std::size_t mulsbk_to(std::uint64_t*out,lbv_t a,lbv_t b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.empty()||b.empty())
		return 0;
	if(std::max(a.n,b.n)<=kCbaMax){
		const limbs_t tmp=mulcba_v(a,b);
		if(!tmp.empty())
			std::memcpy(out,tmp.data(),tmp.size()*sizeof(limb_t));
		return tmp.size();
	}

	lbv_t u=a;
	lbv_t v=b;
	if(u.n<v.n)
		std::swap(u,v);
	out[u.n]=mul_1n(out,u.p,u.n,v[0]);
	for(std::size_t j=1;j<v.n;++j){
		const std::uint64_t cy=am_1n(
			out+static_cast<std::ptrdiff_t>(j),u.p,u.n,v[j]);
		out[j+u.n]=cy;
	}
	return lb_nz(out,u.n+v.n);
}

inline void add_to_at(std::uint64_t*out,std::size_t n,
					  const limbs_t&x,std::size_t shift){
	if(x.empty())
		return;
	MINI_MP_ASSERT(shift+x.size()<=n);
	const std::uint64_t carry=add_nm(
		out+static_cast<std::ptrdiff_t>(shift),
		out+static_cast<std::ptrdiff_t>(shift),n-shift,
		x.data(),x.size());
	MINI_MP_ASSERT(carry==0);
}

inline void sub_abs_ip(limbs_t&a,lbv_t b){
	b=lbv_trim(b);
	if(b.empty())
		return;
	MINI_MP_ASSERT(cmp_abs(lbv_all(a),b)>=0);
	const std::uint64_t borrow=sub_nm(a.data(),a.data(),a.size(),b.p,b.n);
	MINI_MP_ASSERT(borrow==0);
	a.resize(lb_nz(a.data(),a.size()));
}

inline limbs_t sqrsbk_v(lbv_t x){
	x=lbv_trim(x);
	if(x.empty())
		return {};
	if(x.n<=kCbaMax)
		return mulcba_v(x,x);

	limbs_t out;
	out.resize_uninit(2u*x.n+1u);
	std::memset(out.data(),0,(2u*x.n+1u)*sizeof(limb_t));
	for(std::size_t i=0;i+1u<x.n;++i){
		const std::size_t len=x.n-i-1u;
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(2u*i+1u),
			x.p+static_cast<std::ptrdiff_t>(i+1u),len,x[i]);
		add_lb_at(out,cy,x.n+i);
	}

	std::uint64_t carry=0;
	for(std::size_t i=0;i<out.size();++i){
		const std::uint64_t next=out[i]>>63u;
		out[i]=(out[i]<<1u)|carry;
		carry=next;
	}
	if(carry!=0)
		out.push_back(carry);

	for(std::size_t i=0;i<x.n;++i){
		const u128 p=mul_u64(x[i],x[i]);
		const std::size_t pos=2u*i;
		std::uint64_t s0=0;
		std::uint64_t cy=addc_u64(0,out[pos],p.lo,&s0);
		out[pos]=s0;
		std::uint64_t s1=0;
		const std::uint64_t cy2=addc_u64(cy,out[pos+1u],p.hi,&s1);
		out[pos+1u]=s1;
		add_lb_at(out,cy2,pos+2u);
	}
	out.resize(lb_nz(out.data(),out.size()));
	return out;
}

inline std::size_t sqrsbk_to(std::uint64_t*out,lbv_t x){
	x=lbv_trim(x);
	if(x.empty())
		return 0;
	if(x.n<=kCbaMax){
		const limbs_t tmp=mulcba_v(x,x);
		if(!tmp.empty())
			std::memcpy(out,tmp.data(),tmp.size()*sizeof(limb_t));
		return tmp.size();
	}

	const std::size_t n=x.n;
	const std::size_t out_n=2u*n;
	std::memset(out,0,out_n*sizeof(limb_t));
	for(std::size_t i=0;i+1u<n;++i){
		const std::size_t len=n-i-1u;
		const std::uint64_t cy=am_1n(
			out+static_cast<std::ptrdiff_t>(2u*i+1u),
			x.p+static_cast<std::ptrdiff_t>(i+1u),len,x[i]);
		std::size_t pos=n+i;
		std::uint64_t carry=cy;
		while(carry!=0){
			MINI_MP_ASSERT(pos<out_n);
			std::uint64_t sum=0;
			carry=addc_u64(0,out[pos],carry,&sum);
			out[pos]=sum;
			++pos;
		}
	}

	std::uint64_t carry=0;
	for(std::size_t i=0;i<out_n;++i){
		const std::uint64_t next=out[i]>>63u;
		out[i]=(out[i]<<1u)|carry;
		carry=next;
	}
	MINI_MP_ASSERT(carry==0);

	for(std::size_t i=0;i<n;++i){
		const u128 p=mul_u64(x[i],x[i]);
		const std::size_t pos=2u*i;
		std::uint64_t s0=0;
		std::uint64_t cy=addc_u64(0,out[pos],p.lo,&s0);
		out[pos]=s0;
		std::uint64_t s1=0;
		const std::uint64_t cy2=addc_u64(cy,out[pos+1u],p.hi,&s1);
		out[pos+1u]=s1;
		std::size_t j=pos+2u;
		std::uint64_t extra=cy2;
		while(extra!=0){
			MINI_MP_ASSERT(j<out_n);
			std::uint64_t sum=0;
			extra=addc_u64(0,out[j],extra,&sum);
			out[j]=sum;
			++j;
		}
	}
	return lb_nz(out,out_n);
}

inline limbs_t slc_limb(const limbs_t&x,std::size_t begin,std::size_t end){
	if(begin>=end||begin>=x.size())
		return {};
	end=std::min(end,x.size());
	limbs_t out(x.begin()+static_cast<std::ptrdiff_t>(begin),
				x.begin()+static_cast<std::ptrdiff_t>(end));
	trim_lz(out);
	return out;
}

inline limbs_t shl_limbs(const limbs_t&x,std::size_t limb_shift){
	if(x.empty())
		return {};
	if(limb_shift==0)
		return x;
	limbs_t out;
	out.assign(limb_shift,0);
	out.insert(out.end(),x.begin(),x.end());
	return out;
}

inline void add_sh_ip(limbs_t&acc,const limbs_t&addend,
								std::size_t limb_shift){
	if(addend.empty())
		return;
	const std::size_t need=limb_shift+addend.size();
	if(acc.size()<need)
		acc.resize(need,0);

	std::uint64_t carry=0;
	for(std::size_t i=0;i<addend.size();++i){
		const std::size_t idx=limb_shift+i;
		std::uint64_t t=0;
		const std::uint64_t c1=addc_u64(0,acc[idx],addend[i],&t);
		std::uint64_t out=0;
		const std::uint64_t c2=addc_u64(0,t,carry,&out);
		acc[idx]=out;
		MINI_MP_ASSERT((c1&c2)==0);
		carry=(c1|c2);
	}

	std::size_t idx=need;
	while(carry!=0){
		if(idx>=acc.size())
			acc.push_back(0);
		std::uint64_t out=0;
		carry=addc_u64(0,acc[idx],carry,&out);
		acc[idx]=out;
		++idx;
	}
}

inline void add_shb_ip(limbs_t&acc,const limbs_t&addend,
					   std::size_t shift_bits){
	if(addend.empty())
		return;
	const std::size_t limb_shift=shift_bits/64u;
	const unsigned bit_shift=static_cast<unsigned>(shift_bits%64u);
	if(bit_shift==0){
		add_sh_ip(acc,addend,limb_shift);
		return;
	}
	const std::size_t need=limb_shift+addend.size()+1u;
	if(acc.size()<need)
		acc.resize(need,0);
	std::uint64_t carry=0;
	std::uint64_t upper=0;
	for(std::size_t i=0;i<addend.size();++i){
		const std::uint64_t v=addend[i];
		const std::uint64_t part=(v<<bit_shift)|upper;
		upper=v>>(64u-bit_shift);
		const std::size_t idx=limb_shift+i;
		std::uint64_t t=0;
		const std::uint64_t c1=addc_u64(0,acc[idx],part,&t);
		std::uint64_t out=0;
		const std::uint64_t c2=addc_u64(0,t,carry,&out);
		acc[idx]=out;
		MINI_MP_ASSERT((c1&c2)==0);
		carry=(c1|c2);
	}
	std::size_t idx=limb_shift+addend.size();
	if(upper!=0||carry!=0){
		std::uint64_t t=0;
		const std::uint64_t c1=addc_u64(0,acc[idx],upper,&t);
		std::uint64_t out=0;
		const std::uint64_t c2=addc_u64(0,t,carry,&out);
		acc[idx]=out;
		MINI_MP_ASSERT((c1&c2)==0);
		carry=(c1|c2);
		++idx;
	}
	while(carry!=0){
		if(idx>=acc.size())
			acc.push_back(0);
		std::uint64_t out=0;
		carry=addc_u64(0,acc[idx],carry,&out);
		acc[idx]=out;
		++idx;
	}
	trim_lz(acc);
}

inline constexpr std::size_t kKarRecB=24;
inline constexpr std::size_t kKarDifMin=96;

inline std::size_t tun_krec() noexcept;
inline std::size_t tun_srec() noexcept;
inline std::size_t tun_kar() noexcept;
inline std::size_t tun_kar_imb() noexcept;
inline std::size_t tun_kdif() noexcept;
inline std::size_t tun_ntt_imb() noexcept;
inline std::size_t tun_gcd_qs() noexcept;
inline std::size_t tun_bz_min() noexcept;
inline std::size_t tun_bz_chunk() noexcept;
inline std::size_t tun_prod_leaf() noexcept;
inline std::size_t tun_pow_w5() noexcept;
inline std::size_t tun_pow_w6() noexcept;
inline void ensure_at();
inline limbs_t mulkar_o(lbv_t a,lbv_t b,std::size_t rec_b);
inline limbs_t sqrkar_o(lbv_t x,std::size_t rec_b);

inline limbs_t mulkar_rc(const limbs_t&a,const limbs_t&b,
								 std::size_t rec_b){
	if(a.empty()||b.empty())
		return {};
	if(std::min(a.size(),b.size())<=rec_b){
		return mulsbk_ab(a,b);
	}

	const std::size_t n=std::max(a.size(),b.size());
	const std::size_t m=n/2;
	if(m==0)
		return mulsbk_ab(a,b);

	const limbs_t a0=slc_limb(a,0,m);
	const limbs_t a1=slc_limb(a,m,a.size());
	const limbs_t b0=slc_limb(b,0,m);
	const limbs_t b1=slc_limb(b,m,b.size());

	if(a0.empty()||b0.empty()||a1.empty()||b1.empty()){
		
		return mulsbk_ab(a,b);
	}

	limbs_t z0=mulkar_rc(a0,b0,rec_b);
	const limbs_t z2=mulkar_rc(a1,b1,rec_b);
	const limbs_t a01=add_abs(a0,a1);
	const limbs_t b01=add_abs(b0,b1);

	limbs_t z1=mulkar_rc(a01,b01,rec_b);
	const limbs_t z0z2=add_abs(z0,z2);
	if(cmp_abs(z1,z0z2)<0){
		
		return mulsbk_ab(a,b);
	}
	z1=sub_abs(z1,z0z2);

	limbs_t out=std::move(z0);
	out.reserve(a.size()+b.size());
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2*m);
	trim_lz(out);
	return out;
}

inline std::size_t mulkar_to(std::uint64_t*out,lbv_t a,lbv_t b,
							 std::size_t rec_b,bool clr=true){
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.empty()||b.empty())
		return 0;
	const std::size_t need=a.n+b.n;
	if(clr)
		std::memset(out,0,need*sizeof(limb_t));
	if(std::min(a.n,b.n)<=rec_b)
		return mulsbk_to(out,a,b);

	const std::size_t n=std::max(a.n,b.n);
	const std::size_t m=n/2u;
	if(m==0)
		return mulsbk_to(out,a,b);

	const lbv_t a0=lbv_slc(a,0,m);
	const lbv_t a1=lbv_slc(a,m,a.n);
	const lbv_t b0=lbv_slc(b,0,m);
	const lbv_t b1=lbv_slc(b,m,b.n);
	if(a0.empty()||b0.empty()||a1.empty()||b1.empty())
		return mulsbk_to(out,a,b);

	mulkar_to(out,a0,b0,rec_b,false);
	mulkar_to(out+static_cast<std::ptrdiff_t>(2u*m),a1,b1,rec_b,false);

	const lbv_t z0=lbv_trim({out,std::min(2u*m,need)});
	const lbv_t z2=lbv_trim(
		{out+static_cast<std::ptrdiff_t>(2u*m),need-2u*m});
	limbs_t mid;
	if(n<tun_kdif()){
		const limbs_t a01=add_abs(a0,a1);
		const limbs_t b01=add_abs(b0,b1);
		mid=mulkar_o(lbv_all(a01),lbv_all(b01),rec_b);
		sub_abs_ip(mid,z0);
		sub_abs_ip(mid,z2);
	}else{
		int as=0;
		int bs=0;
		const limbs_t da=dif_abs(a1,a0,&as);
		const limbs_t db=dif_abs(b1,b0,&bs);
		mid=add_abs(z0,z2);
		if(as!=0&&bs!=0){
			const limbs_t dp=mulkar_o(lbv_all(da),lbv_all(db),rec_b);
			if(as==bs){
				if(cmp_abs(lbv_all(mid),lbv_all(dp))<0){
					std::memset(out,0,need*sizeof(limb_t));
					return mulsbk_to(out,a,b);
				}
				sub_abs_ip(mid,lbv_all(dp));
			}else{
				add_abs_ip(mid,dp);
			}
		}
	}
	add_to_at(out,need,mid,m);
	return lb_nz(out,need);
}

inline limbs_t mulkar_o(lbv_t a,lbv_t b,std::size_t rec_b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.empty()||b.empty())
		return {};
	limbs_t out;
	out.resize_uninit(a.n+b.n);
	const std::size_t n=mulkar_to(out.data(),a,b,rec_b);
	out.resize(n);
	return out;
}

inline limbs_t mulkar_v(lbv_t a,lbv_t b,std::size_t rec_b){
	a=lbv_trim(a);
	b=lbv_trim(b);
	if(a.empty()||b.empty())
		return {};
	if(std::min(a.n,b.n)<=rec_b)
		return mulsbk_v(a,b);

	const std::size_t n=std::max(a.n,b.n);
	const std::size_t m=n/2u;
	if(m==0)
		return mulsbk_v(a,b);

	const lbv_t a0=lbv_slc(a,0,m);
	const lbv_t a1=lbv_slc(a,m,a.n);
	const lbv_t b0=lbv_slc(b,0,m);
	const lbv_t b1=lbv_slc(b,m,b.n);
	if(a0.empty()||b0.empty()||a1.empty()||b1.empty())
		return mulsbk_v(a,b);

	limbs_t z0=mulkar_v(a0,b0,rec_b);
	const limbs_t z2=mulkar_v(a1,b1,rec_b);
	limbs_t z1;
	if(n<tun_kdif()){
		const limbs_t a01=add_abs(a0,a1);
		const limbs_t b01=add_abs(b0,b1);
		z1=mulkar_v(lbv_all(a01),lbv_all(b01),rec_b);
		const limbs_t z0z2=add_abs(z0,z2);
		if(cmp_abs(z1,z0z2)<0)
			return mulsbk_v(a,b);
		z1=sub_abs(z1,z0z2);
	}else{
		int as=0;
		int bs=0;
		const limbs_t da=dif_abs(a1,a0,&as);
		const limbs_t db=dif_abs(b1,b0,&bs);
		z1=add_abs(z0,z2);
		if(as!=0&&bs!=0){
			const limbs_t dp=mulkar_v(lbv_all(da),lbv_all(db),rec_b);
			if(as==bs){
				if(cmp_abs(z1,dp)<0)
					return mulsbk_v(a,b);
				z1=sub_abs(z1,dp);
			}else{
				add_abs_ip(z1,dp);
			}
		}
	}

	limbs_t out=std::move(z0);
	out.reserve(a.n+b.n);
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2u*m);
	trim_lz(out);
	return out;
}

inline limbs_t mulkar_ab(const limbs_t&a,const limbs_t&b){
	if(a.empty()||b.empty())
		return {};
	return mulkar_o(lbv_all(a),lbv_all(b),tun_krec());
}

inline std::size_t sqrkar_to(std::uint64_t*out,lbv_t x,
							 std::size_t rec_b,bool clr=true){
	x=lbv_trim(x);
	if(x.empty())
		return 0;
	const std::size_t need=2u*x.n;
	if(clr)
		std::memset(out,0,need*sizeof(limb_t));
	if(x.n<=rec_b)
		return sqrsbk_to(out,x);

	const std::size_t m=x.n/2u;
	if(m==0)
		return sqrsbk_to(out,x);

	const lbv_t x0=lbv_slc(x,0,m);
	const lbv_t x1=lbv_slc(x,m,x.n);
	if(x1.empty())
		return sqrsbk_to(out,x);

	sqrkar_to(out,x0,rec_b,false);
	sqrkar_to(out+static_cast<std::ptrdiff_t>(2u*m),x1,rec_b,false);

	const lbv_t z0=lbv_trim({out,std::min(2u*m,need)});
	const lbv_t z2=lbv_trim(
		{out+static_cast<std::ptrdiff_t>(2u*m),need-2u*m});
	int ds=0;
	const limbs_t d=dif_abs(x1,x0,&ds);
	limbs_t mid=add_abs(z0,z2);
	if(ds!=0){
		const limbs_t dp=sqrkar_o(lbv_all(d),rec_b);
		if(cmp_abs(lbv_all(mid),lbv_all(dp))<0){
			std::memset(out,0,need*sizeof(limb_t));
			return sqrsbk_to(out,x);
		}
		sub_abs_ip(mid,lbv_all(dp));
	}
	add_to_at(out,need,mid,m);
	return lb_nz(out,need);
}

inline limbs_t sqrkar_o(lbv_t x,std::size_t rec_b){
	x=lbv_trim(x);
	if(x.empty())
		return {};
	limbs_t out;
	out.resize_uninit(2u*x.n);
	const std::size_t n=sqrkar_to(out.data(),x,rec_b);
	out.resize(n);
	return out;
}

inline limbs_t sqrkar_rc(const limbs_t&x,std::size_t rec_b){
	if(x.empty())
		return {};
	if(x.size()<=rec_b)
		return sqrsbk_ab(x);

	const std::size_t n=x.size();
	const std::size_t m=n/2;
	if(m==0)
		return sqrsbk_ab(x);

	const limbs_t x0=slc_limb(x,0,m);
	const limbs_t x1=slc_limb(x,m,x.size());
	if(x1.empty())
		return sqrsbk_ab(x);

	limbs_t z0=sqrkar_rc(x0,rec_b);
	const limbs_t z2=sqrkar_rc(x1,rec_b);
	const limbs_t x01=add_abs(x0,x1);

	limbs_t z1=sqrkar_rc(x01,rec_b);
	const limbs_t z0z2=add_abs(z0,z2);
	if(cmp_abs(z1,z0z2)<0)
		return sqrsbk_ab(x);
	z1=sub_abs(z1,z0z2);

	limbs_t out=std::move(z0);
	out.reserve(2u*x.size());
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2*m);
	trim_lz(out);
	return out;
}

inline limbs_t sqrkar_v(lbv_t x,std::size_t rec_b){
	x=lbv_trim(x);
	if(x.empty())
		return {};
	if(x.n<=rec_b)
		return sqrsbk_v(x);

	const std::size_t m=x.n/2u;
	if(m==0)
		return sqrsbk_v(x);

	const lbv_t x0=lbv_slc(x,0,m);
	const lbv_t x1=lbv_slc(x,m,x.n);
	if(x1.empty())
		return sqrsbk_v(x);

	limbs_t z0=sqrkar_v(x0,rec_b);
	const limbs_t z2=sqrkar_v(x1,rec_b);
	int ds=0;
	const limbs_t d=dif_abs(x1,x0,&ds);
	limbs_t z1=add_abs(z0,z2);
	if(ds!=0){
		const limbs_t dp=sqrkar_v(lbv_all(d),rec_b);
		if(cmp_abs(z1,dp)<0)
			return sqrsbk_v(x);
		z1=sub_abs(z1,dp);
	}

	limbs_t out=std::move(z0);
	out.reserve(2u*x.n);
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2u*m);
	trim_lz(out);
	return out;
}

inline limbs_t sqrkar_ab(const limbs_t&x){
	if(x.empty())
		return {};
	return sqrkar_o(lbv_all(x),tun_srec());
}

inline bool mulbig_ok(std::size_t an,std::size_t bn) noexcept{
	const std::size_t nmin=std::min(an,bn);
	const std::size_t nmax=std::max(an,bn);
	return nmin!=0&&nmax<=nmin*tun_kar_imb();
}

inline void mulbig_in(const limbs_t&a,const limbs_t&b,
									limbs_t&out){
	if(a.empty()||b.empty()){
		out.clear();
		return;
	}
	const std::size_t nmax=std::max(a.size(),b.size());
	if(nmax<=kCbaMax){
		out=mulcba_ab(a,b);
		return;
	}
	ensure_at();
	const std::size_t nmin=std::min(a.size(),b.size());
	if(nmin>=tun_kar()&&mulbig_ok(a.size(),b.size())){
		out=mulkar_ab(a,b);
		return;
	}
	mulsbk_in(a,b,out);
}

inline void sqrbig_in(const limbs_t&x,limbs_t&out){
	if(x.empty()){
		out.clear();
		return;
	}
	if(x.size()<=kCbaMax){
		out=sqrcba_ab(x);
		return;
	}
	ensure_at();
	if(x.size()>=tun_kar()){
		out=sqrkar_ab(x);
		return;
	}
	out=sqrsbk_ab(x);
}

inline int cmp_u128(u128 a,u128 b) noexcept{
	if(a.hi!=b.hi)
		return (a.hi<b.hi)?-1:1;
	if(a.lo!=b.lo)
		return (a.lo<b.lo)?-1:1;
	return 0;
}

inline void mulbl_in(const limbs_t&a,std::uint64_t m,limbs_t&out){
	if(a.empty()||m==0){
		out.clear();
		return;
	}
	if(m==1){
		out=a;
		return;
	}

	const std::size_t n=a.size();
	out.resize_uninit(n+1);
	out[n]=mul_1n(out.data(),a.data(),n,m);
	out.resize(lb_nz(out.data(),n+1));
}

inline limbs_t mul_bylb(const limbs_t&a,std::uint64_t m){
	limbs_t out;
	mulbl_in(a,m,out);
	return out;
}



inline std::uint64_t leh_qest(const limbs_t&a,
												 const limbs_t&b){
	MINI_MP_ASSERT(a.size()==b.size());
	MINI_MP_ASSERT(a.size()>=2);
	MINI_MP_ASSERT(cmp_abs(a,b)>=0);

	const std::size_t n=a.size();
	const std::uint64_t a2=a[n-1];
	const std::uint64_t a1=a[n-2];
	const std::uint64_t a0=(n>=3)?a[n-3]:0;
	const std::uint64_t am1=(n>=4)?a[n-4]:0;

	const std::uint64_t b1=b[n-1];
	const std::uint64_t b0=b[n-2];
	const std::uint64_t bm1=(n>=3)?b[n-3]:0;

	MINI_MP_ASSERT(b1!=0);
	const unsigned s=std::countl_zero(b1);

	std::uint64_t un2=0,un1=0,un0=0;
	std::uint64_t vn1=0,vn0=0;
	if(s==0){
		un2=a2;
		un1=a1;
		un0=a0;
		vn1=b1;
		vn0=b0;
	}else{
		un2=(a2<<s)|(a1>>(64u-s));
		un1=(a1<<s)|(a0>>(64u-s));
		un0=(a0<<s)|(am1>>(64u-s));
		vn1=(b1<<s)|(b0>>(64u-s));
		vn0=(b0<<s)|(bm1>>(64u-s));
	}

	std::uint64_t qhat=0;
	std::uint64_t rhat=0;
	std::uint64_t rhat_ovf=0;

	if(un2==vn1){
		qhat=std::numeric_limits<std::uint64_t>::max();
		rhat_ovf=addc_u64(0,un1,vn1,&rhat);
	}else{
		qhat=udiv128(un2,un1,vn1,&rhat);
	}

	while(qhat!=0&&rhat_ovf==0){
		const u128 lhs=mul_u64(qhat,vn0);
		const u128 rhs{un0,rhat};
		if(cmp_u128(lhs,rhs)<=0)
			break;
		--qhat;
		rhat_ovf=addc_u64(0,rhat,vn1,&rhat);
	}

	if(qhat==0)
		qhat=1;
	return qhat;
}

inline std::uint64_t limb_z(const limbs_t&x,std::size_t i) noexcept{
	return (i<x.size())?x[i]:0u;
}

inline u128 sub_u128(u128 a,u128 b) noexcept{
	u128 out{};
	const std::uint64_t br=subb_u64(0,a.lo,b.lo,&out.lo);
	(void)subb_u64(br,a.hi,b.hi,&out.hi);
	return out;
}

inline int cmp_u128_limb(u128 a,std::uint64_t b) noexcept{
	if(a.hi!=0)
		return 1;
	if(a.lo==b)
		return 0;
	return (a.lo<b)?-1:1;
}

inline u128 add_limb2(std::uint64_t a,std::uint64_t b) noexcept{
	u128 out{};
	out.hi=addc_u64(0,a,b,&out.lo);
	return out;
}

inline bool madd_limb_ov(std::uint64_t a,std::uint64_t q,
								  std::uint64_t b,std::uint64_t*out) noexcept{
	const u128 p=mul_u64(q,b);
	std::uint64_t lo=0;
	const std::uint64_t c=addc_u64(0,a,p.lo,&lo);
	if(p.hi!=0||c!=0)
		return true;
	*out=lo;
	return false;
}

inline u128 mul_u128_limb(u128 a,std::uint64_t q) noexcept{
	const u128 lo=mul_u64(a.lo,q);
	const u128 hi=mul_u64(a.hi,q);
	u128 out{};
	out.lo=lo.lo;
	const std::uint64_t c=addc_u64(0,lo.hi,hi.lo,&out.hi);
	MINI_MP_ASSERT(c==0&&hi.hi==0);
	return out;
}

inline std::uint64_t div_u128_limb(u128 u,u128 v,
											  std::uint64_t qs) noexcept{
	MINI_MP_ASSERT(v.hi!=0);
	MINI_MP_ASSERT(cmp_u128(u,v)>=0);

	qs=std::max<std::uint64_t>(1u,qs);
	u128 r=sub_u128(u,v);
	std::uint64_t small=1;
	while(small<qs&&cmp_u128(r,v)>=0){
		r=sub_u128(r,v);
		++small;
	}
	if(cmp_u128(r,v)<0)
		return small;

	std::uint64_t rem=0;
	std::uint64_t q=udiv128(0,u.hi,v.hi,&rem);
	if(q==0)
		q=1;
	while(q!=0){
		const u128 lhs=mul_u64(q,v.lo);
		const u128 rhs{u.lo,rem};
		if(cmp_u128(lhs,rhs)<=0)
			break;
		--q;
		std::uint64_t nr=0;
		const std::uint64_t c=addc_u64(0,rem,v.hi,&nr);
		rem=nr;
		if(c!=0)
			break;
	}
	return q;
}

inline u128 top_u128(const limbs_t&x,std::size_t n,unsigned h) noexcept{
	u128 out{limb_z(x,n-2u),limb_z(x,n-1u)};
	if(h!=0){
		out=shl_u128(out,64u-h);
		out.lo|=(limb_z(x,n-3u)>>h);
	}
	return out;
}

inline std::uint64_t mulcarry_limb(std::uint64_t x,std::uint64_t m,
											std::uint64_t carry,
											std::uint64_t*out) noexcept{
#if MINI_MP_DETAIL_HAS_UINT128
	const unsigned __int128 z=
		static_cast<unsigned __int128>(x)*m+carry;
	*out=static_cast<std::uint64_t>(z);
	return static_cast<std::uint64_t>(z>>64);
#else
	*out=mulad64(x,m,0,&carry);
	return carry;
#endif
}

inline std::uint64_t mulstep_limb(const limbs_t&x,std::size_t i,
										   std::uint64_t m,
										   std::uint64_t carry,
										   std::uint64_t*out) noexcept{
	if(m==0){
		*out=0;
		return 0;
	}
	if(i>=x.size()){
		*out=carry;
		return 0;
	}
	if(m==1){
		return addc_u64(0,x[i],carry,out);
	}
	return mulcarry_limb(x[i],m,carry,out);
}

inline bool leh_apply2(const limbs_t&a,const limbs_t&b,
							  std::uint64_t x0,std::uint64_t y0,
							  std::uint64_t x1,std::uint64_t y1,
							  bool flip0,bool flip1,
							  limbs_t&o0,limbs_t&o1){
	const limbs_t*pa0=flip0?&b:&a;
	const limbs_t*na0=flip0?&a:&b;
	const std::uint64_t pm0=flip0?y0:x0;
	const std::uint64_t nm0=flip0?x0:y0;
	const limbs_t*pa1=flip1?&b:&a;
	const limbs_t*na1=flip1?&a:&b;
	const std::uint64_t pm1=flip1?y1:x1;
	const std::uint64_t nm1=flip1?x1:y1;
	const std::size_t n=std::max(a.size(),b.size())+1u;
	o0.resize_uninit(n);
	o1.resize_uninit(n);
	std::uint64_t pc0=0,nc0=0,br0=0;
	std::uint64_t pc1=0,nc1=0,br1=0;
	for(std::size_t i=0;i<n;++i){
		std::uint64_t pl0=0;
		pc0=mulstep_limb(*pa0,i,pm0,pc0,&pl0);
		std::uint64_t nl0=0;
		nc0=mulstep_limb(*na0,i,nm0,nc0,&nl0);
		br0=subb_u64(br0,pl0,nl0,&o0[i]);

		std::uint64_t pl1=0;
		pc1=mulstep_limb(*pa1,i,pm1,pc1,&pl1);
		std::uint64_t nl1=0;
		nc1=mulstep_limb(*na1,i,nm1,nc1,&nl1);
		br1=subb_u64(br1,pl1,nl1,&o1[i]);
	}
	if(br0!=0||br1!=0)
		return false;
	trim_lz(o0);
	trim_lz(o1);
	return true;
}

struct LehScr{
	limbs_t a0;
	limbs_t b0;
};

inline bool leh_step_ip(limbs_t&a,limbs_t&b,LehScr&sc){
	if(a.size()<3||b.empty())
		return false;
	const std::size_t ab=bit_length(a);
	const std::size_t bb=bit_length(b);
	if(ab<bb||ab-bb>63u)
		return false;

	const std::size_t n=a.size();
	const unsigned h=static_cast<unsigned>(ab&63u);
	u128 u=top_u128(a,n,h);
	u128 v=top_u128(b,n,h);
	if(is0_u128(v)||cmp_u128(u,v)<0||v.hi==0)
		return false;

	const std::uint64_t qsmall=
		static_cast<std::uint64_t>(tun_gcd_qs());
	std::uint64_t x0=1,x1=0,y0=0,y1=1;
	std::size_t cnt=0;
	for(;;){
		if(v.hi==0)
			break;
		const std::uint64_t q=div_u128_limb(u,v,qsmall);
		std::uint64_t x2=0,y2=0;
		if(madd_limb_ov(x0,q,x1,&x2)||
		   madd_limb_ov(y0,q,y1,&y2)){
			++cnt;
			break;
		}

		const u128 old=u;
		u=v;
		v=sub_u128(old,mul_u128_limb(v,q));
		++cnt;
		if(is0_u128(v)||v.hi==0)
			break;

		const u128 gap=sub_u128(u,v);
		if((cnt&1u)==0){
			if(cmp_u128_limb(v,x2)<0||
			   cmp_u128(gap,add_limb2(y2,y1))<0)
				break;
		}else{
			if(cmp_u128_limb(v,y2)<0||
			   cmp_u128(gap,add_limb2(x2,x1))<0)
				break;
		}

		x0=x1;
		x1=x2;
		y0=y1;
		y1=y2;
	}

	if(cnt<=1)
		return false;

	const bool even=(cnt&1u)==0;
	const bool ok=leh_apply2(a,b,x0,y0,x1,y1,
									  even,!even,sc.a0,sc.b0);
	if(!ok)
		return false;
	trim_lz(sc.a0);
	trim_lz(sc.b0);
	if(sc.a0.empty())
		return false;

	a.swap(sc.a0);
	b.swap(sc.b0);
	if(ab<2048u){
		if(!a.empty()&&(a[0]&1u)==0u){
			const std::size_t tz=ctz(a);
			if(tz!=0)
				shr_ip(a,tz);
		}
		if(!b.empty()&&(b[0]&1u)==0u){
			const std::size_t tz=ctz(b);
			if(tz!=0)
				shr_ip(b,tz);
		}
	}
	if(!b.empty()&&cmp_abs(a,b)<0)
		a.swap(b);
	return true;
}

struct ModKScr{
	limbs_t vn;
	limbs_t un;
	limbs_t r;
};

inline std::pair<std::uint64_t,limbs_t> dvm1q_absl(const limbs_t&u,
														 const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	MINI_MP_ASSERT(u.size()==v.size());
	MINI_MP_ASSERT(cmp_abs(u,v)>=0);
	if(u.size()==1){
		const std::uint64_t q=u[0]/v[0];
		const std::uint64_t r=u[0]%v[0];
		limbs_t rem;
		if(r!=0)
			rem.push_back(r);
		return {q,std::move(rem)};
	}

	const std::size_t n=v.size();
	const unsigned s=std::countl_zero(v.back());
	limbs_t vn;
	limbs_t un;
	shl_into(vn,v,s);
	shl_into(un,u,s);
	if(vn.size()<n)
		vn.resize(n,0);
	if(un.size()<n+1)
		un.resize(n+1,0);

	std::uint64_t qhat=0;
	std::uint64_t rhat=0;
	std::uint64_t rhat_ovf=0;
	if(un[n]==vn[n-1]){
		qhat=std::numeric_limits<std::uint64_t>::max();
		rhat_ovf=addc_u64(0,un[n-1],vn[n-1],&rhat);
	}else{
		qhat=udiv128(un[n],un[n-1],vn[n-1],&rhat);
	}

	while(rhat_ovf==0){
		const u128 lhs=mul_u64(qhat,vn[n-2]);
		const u128 rhs{un[n-2],rhat};
		if(cmp_u128(lhs,rhs)<=0)
			break;
		--qhat;
		rhat_ovf=addc_u64(0,rhat,vn[n-1],&rhat);
	}

	const std::uint64_t cy=sm_1n(un.data(),vn.data(),n,qhat);
	std::uint64_t top=0;
	const std::uint64_t borrow=subb_u64(0,un[n],cy,&top);
	un[n]=top;
	if(borrow!=0){
		--qhat;
		const std::uint64_t carry=add_nn(un.data(),un.data(),vn.data(),n);
		std::uint64_t out=0;
		const std::uint64_t ctop=addc_u64(0,un[n],carry,&out);
		MINI_MP_ASSERT(ctop==0);
		un[n]=out;
	}

	limbs_t r(un.begin(),un.begin()+static_cast<std::ptrdiff_t>(n));
	if(s!=0)
		shr_ip(r,s);
	else
		trim_lz(r);
	return {qhat,std::move(r)};
}

inline std::pair<limbs_t,limbs_t> dvmk_absl(const limbs_t&u,
														 const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return {{},{}};
	if(cmp_abs(u,v)<0)
		return {{},u};

	if(v.size()==1){
		limbs_t q=u;
		std::uint64_t rem=0;
		for(std::size_t i=q.size();i>0;--i){
			q[i-1]=udiv128(rem,q[i-1],v[0],&rem);
		}
		trim_lz(q);
		limbs_t r;
		if(rem!=0)
			r.push_back(rem);
		return {std::move(q),std::move(r)};
	}
	if(u.size()==v.size()){
		auto qr=dvm1q_absl(u,v);
		limbs_t q;
		if(qr.first!=0)
			q.push_back(qr.first);
		return {std::move(q),std::move(qr.second)};
	}

	const std::size_t n=v.size();
	const unsigned s=std::countl_zero(v.back());
	limbs_t vn=shl_bits(v,s);
	limbs_t un=shl_bits(u,s);
	un.push_back(0);

	const std::size_t m=un.size()-n-1;
	limbs_t q(m+1,0);

	for(std::size_t jj=m+1;jj>0;--jj){
		const std::size_t j=jj-1;

		std::uint64_t qhat=0;
		std::uint64_t rhat=0;
		std::uint64_t rhat_ovf=0;

		if(un[j+n]==vn[n-1]){
			qhat=std::numeric_limits<std::uint64_t>::max();
			rhat_ovf=addc_u64(0,un[j+n-1],vn[n-1],&rhat);
		}else{
			qhat=udiv128(un[j+n],un[j+n-1],vn[n-1],&rhat);
		}

		if(n>1){
			while(rhat_ovf==0){
				const u128 lhs=mul_u64(qhat,vn[n-2]);
				const u128 rhs{un[j+n-2],rhat};
				if(cmp_u128(lhs,rhs)<=0)
					break;
				--qhat;
				rhat_ovf=addc_u64(0,rhat,vn[n-1],&rhat);
			}
		}

		const std::uint64_t cy=sm_1n(
			un.data()+static_cast<std::ptrdiff_t>(j),vn.data(),n,qhat);
		std::uint64_t outn=0;
		const std::uint64_t borrow_out=subb_u64(0,un[j+n],cy,&outn);
		un[j+n]=outn;

		if(borrow_out!=0){
			--qhat;
			const std::uint64_t carry=add_nn(
				un.data()+static_cast<std::ptrdiff_t>(j),
				un.data()+static_cast<std::ptrdiff_t>(j),vn.data(),n);
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+n],carry,&top);
			un[j+n]=top;
		}

		q[j]=qhat;
	}

	trim_lz(q);
	limbs_t r(un.begin(),un.begin()+static_cast<std::ptrdiff_t>(n));
	r=shr_bits(r,s);
	trim_lz(r);
	return {std::move(q),std::move(r)};
}

inline void modk_liip(limbs_t&u,const limbs_t&v,
										ModKScr&sc){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return;
	const int cmp=cmp_abs(u,v);
	if(cmp<0)
		return;
	if(cmp==0){
		u.clear();
		return;
	}
	if(cmp_abs_dbl(u,v)<0){
		sub_abs_ip(u,v);
		return;
	}

	if(v.size()==1){
		std::uint64_t rem=0;
		for(std::size_t i=u.size();i>0;--i){
			(void)udiv128(rem,u[i-1],v[0],&rem);
		}
		if(rem==0){
			u.clear();
		}else{
			u.resize(1);
			u[0]=rem;
		}
		return;
	}

	const std::size_t n=v.size();
	const unsigned s=std::countl_zero(v.back());

	if(sc.vn.capacity()<v.size()+1){
		sc.vn.reserve(v.size()+1);
	}
	if(sc.un.capacity()<u.size()+2){
		sc.un.reserve(u.size()+2);
	}
	if(sc.r.capacity()<v.size()+1){
		sc.r.reserve(v.size()+1);
	}
	shl_into(sc.vn,v,s);
	shl_into(sc.un,u,s);
	sc.un.resize(sc.un.size()+1,0);

	limbs_t&vn=sc.vn;
	limbs_t&un=sc.un;

	const std::size_t m=un.size()-n-1;
	for(std::size_t jj=m+1;jj>0;--jj){
		const std::size_t j=jj-1;

		std::uint64_t qhat=0;
		std::uint64_t rhat=0;
		std::uint64_t rhat_ovf=0;

		if(un[j+n]==vn[n-1]){
			qhat=std::numeric_limits<std::uint64_t>::max();
			rhat_ovf=addc_u64(0,un[j+n-1],vn[n-1],&rhat);
		}else{
			qhat=udiv128(un[j+n],un[j+n-1],vn[n-1],&rhat);
		}

		if(n>1){
			while(rhat_ovf==0){
				const u128 lhs=mul_u64(qhat,vn[n-2]);
				const u128 rhs{un[j+n-2],rhat};
				if(cmp_u128(lhs,rhs)<=0)
					break;
				--qhat;
				rhat_ovf=addc_u64(0,rhat,vn[n-1],&rhat);
			}
		}

		const std::uint64_t cy=sm_1n(
			un.data()+static_cast<std::ptrdiff_t>(j),vn.data(),n,qhat);
		std::uint64_t outn=0;
		const std::uint64_t borrow_out=subb_u64(0,un[j+n],cy,&outn);
		un[j+n]=outn;

		if(borrow_out!=0){
			const std::uint64_t carry=
				add_nn(un.data()+static_cast<std::ptrdiff_t>(j),
						  un.data()+static_cast<std::ptrdiff_t>(j),vn.data(),n);
			std::uint64_t top=0;
			const std::uint64_t ctop=addc_u64(0,un[j+n],carry,&top);
			MINI_MP_ASSERT(ctop==0);
			un[j+n]=top;
		}
	}

	sc.r.resize(n);
	for(std::size_t i=0;i<n;++i){
		sc.r[i]=un[i];
	}
	if(s!=0){
		shr_ip(sc.r,s);
	}else{
		trim_lz(sc.r);
	}
	u.swap(sc.r);
}

inline void modk_ip(limbs_t&u,const limbs_t&v,ModKScr&sc){
	modk_liip(u,v,sc);
}

inline std::pair<limbs_t,limbs_t> dvm2_absl(const limbs_t&u,
											const limbs_t&v){
	MINI_MP_ASSERT(v.size()==2);
	if(u.empty())
		return {{},{}};
	const int cmp=cmp_abs(u,v);
	if(cmp<0)
		return {{},u};
	if(cmp==0)
		return {{1},{}};
	if(cmp_abs_dbl(u,v)<0){
		limbs_t r=u;
		sub_abs_ip(r,v);
		return {{1},std::move(r)};
	}
	const unsigned s=std::countl_zero(v[1]);
	std::uint64_t vn[2]{};
	if(s==0){
		vn[0]=v[0];
		vn[1]=v[1];
	}else{
		vn[0]=v[0]<<s;
		vn[1]=(v[1]<<s)|(v[0]>>(64u-s));
	}
	limbs_t un;
	shl_into(un,u,s);
	un.push_back(0);
	const std::size_t m=un.size()-3u;
	limbs_t q(m+1u,0);
	for(std::size_t jj=m+1u;jj>0;--jj){
		const std::size_t j=jj-1u;
		std::uint64_t qh=0;
		std::uint64_t rh=0;
		std::uint64_t ro=0;
		if(un[j+2u]==vn[1]){
			qh=std::numeric_limits<std::uint64_t>::max();
			ro=addc_u64(0,un[j+1u],vn[1],&rh);
		}else{
			qh=udiv128(un[j+2u],un[j+1u],vn[1],&rh);
		}
		while(ro==0){
			const u128 lhs=mul_u64(qh,vn[0]);
			const u128 rhs{un[j],rh};
			if(cmp_u128(lhs,rhs)<=0)
				break;
			--qh;
			ro=addc_u64(0,rh,vn[1],&rh);
		}
		const std::uint64_t cy=sm_1n(un.data()+static_cast<std::ptrdiff_t>(j),
									 vn,2u,qh);
		std::uint64_t outn=0;
		const std::uint64_t borrow=subb_u64(0,un[j+2u],cy,&outn);
		un[j+2u]=outn;
		if(borrow!=0){
			--qh;
			const std::uint64_t carry=add_nn(
				un.data()+static_cast<std::ptrdiff_t>(j),
				un.data()+static_cast<std::ptrdiff_t>(j),vn,2u);
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+2u],carry,&top);
			un[j+2u]=top;
		}
		q[j]=qh;
	}
	trim_lz(q);
	limbs_t r(un.begin(),un.begin()+2);
	if(s!=0)
		shr_ip(r,s);
	else
		trim_lz(r);
	return {std::move(q),std::move(r)};
}

inline std::pair<limbs_t,limbs_t> dvm2_shl_absl(const limbs_t&u,
												std::size_t shift_bits,
												const limbs_t&v){
	MINI_MP_ASSERT(v.size()==2);
	if(shift_bits==0)
		return dvm2_absl(u,v);
	if(u.empty())
		return {{},{}};
	const unsigned s=std::countl_zero(v[1]);
	std::uint64_t vn[2]{};
	if(s==0){
		vn[0]=v[0];
		vn[1]=v[1];
	}else{
		vn[0]=v[0]<<s;
		vn[1]=(v[1]<<s)|(v[0]>>(64u-s));
	}
	limbs_t vv{vn[0],vn[1]};
	limbs_t un;
	shl_into(un,u,shift_bits+static_cast<std::size_t>(s));
	const int cmp=cmp_abs(un,vv);
	if(cmp<0){
		if(s!=0)
			shr_ip(un,s);
		return {{},std::move(un)};
	}
	if(cmp==0)
		return {{1},{}};
	if(cmp_abs_dbl(un,vv)<0){
		limbs_t r=std::move(un);
		sub_abs_ip(r,vv);
		if(s!=0)
			shr_ip(r,s);
		return {{1},std::move(r)};
	}
	un.push_back(0);
	const std::size_t m=un.size()-3u;
	limbs_t q(m+1u,0);
	for(std::size_t jj=m+1u;jj>0;--jj){
		const std::size_t j=jj-1u;
		std::uint64_t qh=0;
		std::uint64_t rh=0;
		std::uint64_t ro=0;
		if(un[j+2u]==vn[1]){
			qh=std::numeric_limits<std::uint64_t>::max();
			ro=addc_u64(0,un[j+1u],vn[1],&rh);
		}else{
			qh=udiv128(un[j+2u],un[j+1u],vn[1],&rh);
		}
		while(ro==0){
			const u128 lhs=mul_u64(qh,vn[0]);
			const u128 rhs{un[j],rh};
			if(cmp_u128(lhs,rhs)<=0)
				break;
			--qh;
			ro=addc_u64(0,rh,vn[1],&rh);
		}
		const std::uint64_t cy=sm_1n(un.data()+static_cast<std::ptrdiff_t>(j),
									 vn,2u,qh);
		std::uint64_t outn=0;
		const std::uint64_t borrow=subb_u64(0,un[j+2u],cy,&outn);
		un[j+2u]=outn;
		if(borrow!=0){
			--qh;
			const std::uint64_t carry=add_nn(
				un.data()+static_cast<std::ptrdiff_t>(j),
				un.data()+static_cast<std::ptrdiff_t>(j),vn,2u);
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+2u],carry,&top);
			un[j+2u]=top;
		}
		q[j]=qh;
	}
	trim_lz(q);
	limbs_t r(un.begin(),un.begin()+2);
	if(s!=0)
		shr_ip(r,s);
	else
		trim_lz(r);
	return {std::move(q),std::move(r)};
}

inline limbs_t mod2_absl(const limbs_t&u,const limbs_t&v){
	MINI_MP_ASSERT(v.size()==2);
	if(u.empty())
		return {};
	const int cmp=cmp_abs(u,v);
	if(cmp<0)
		return u;
	if(cmp==0)
		return {};
	if(cmp_abs_dbl(u,v)<0){
		limbs_t r=u;
		sub_abs_ip(r,v);
		return r;
	}
	const unsigned s=std::countl_zero(v[1]);
	std::uint64_t vn[2]{};
	if(s==0){
		vn[0]=v[0];
		vn[1]=v[1];
	}else{
		vn[0]=v[0]<<s;
		vn[1]=(v[1]<<s)|(v[0]>>(64u-s));
	}
	limbs_t un;
	shl_into(un,u,s);
	un.push_back(0);
	const std::size_t m=un.size()-3u;
	for(std::size_t jj=m+1u;jj>0;--jj){
		const std::size_t j=jj-1u;
		std::uint64_t qhat=0;
		std::uint64_t rhat=0;
		std::uint64_t rhat_ovf=0;
		if(un[j+2u]==vn[1]){
			qhat=std::numeric_limits<std::uint64_t>::max();
			rhat_ovf=addc_u64(0,un[j+1u],vn[1],&rhat);
		}else{
			qhat=udiv128(un[j+2u],un[j+1u],vn[1],&rhat);
		}
		while(rhat_ovf==0){
			const u128 lhs=mul_u64(qhat,vn[0]);
			const u128 rhs{un[j],rhat};
			if(cmp_u128(lhs,rhs)<=0)
				break;
			--qhat;
			rhat_ovf=addc_u64(0,rhat,vn[1],&rhat);
		}
		const std::uint64_t cy=sm_1n(un.data()+static_cast<std::ptrdiff_t>(j),
									 vn,2u,qhat);
		std::uint64_t outn=0;
		const std::uint64_t borrow=subb_u64(0,un[j+2u],cy,&outn);
		un[j+2u]=outn;
		if(borrow!=0){
			const std::uint64_t carry=add_nn(
				un.data()+static_cast<std::ptrdiff_t>(j),
				un.data()+static_cast<std::ptrdiff_t>(j),vn,2u);
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+2u],carry,&top);
			un[j+2u]=top;
		}
	}
	limbs_t r(un.begin(),un.begin()+2);
	if(s!=0)
		shr_ip(r,s);
	else
		trim_lz(r);
	return r;
}

inline limbs_t mod2_shl_absl(const limbs_t&u,std::size_t shift_bits,
							 const limbs_t&v){
	MINI_MP_ASSERT(v.size()==2);
	if(shift_bits==0)
		return mod2_absl(u,v);
	if(u.empty())
		return {};
	const unsigned s=std::countl_zero(v[1]);
	std::uint64_t vn[2]{};
	if(s==0){
		vn[0]=v[0];
		vn[1]=v[1];
	}else{
		vn[0]=v[0]<<s;
		vn[1]=(v[1]<<s)|(v[0]>>(64u-s));
	}
	const std::size_t tsh=shift_bits+static_cast<std::size_t>(s);
	const std::size_t lsh=tsh/64u;
	const unsigned bsh=static_cast<unsigned>(tsh%64u);
	const std::size_t raw_n=u.size()+lsh+(bsh!=0?1u:0u);
	if(raw_n+1u<=12u&&raw_n>=3u){
		std::array<std::uint64_t,12> un{};
		if(bsh==0){
			for(std::size_t i=0;i<u.size();++i)
				un[lsh+i]=u[i];
		}else{
			std::uint64_t carry=0;
			for(std::size_t i=0;i<u.size();++i){
				const std::uint64_t cur=u[i];
				un[lsh+i]=(cur<<bsh)|carry;
				carry=cur>>(64u-bsh);
			}
			un[lsh+u.size()]=carry;
		}
		std::size_t un_n=raw_n;
		while(un_n!=0&&un[un_n-1u]==0)
			--un_n;
		if(un_n>=3u){
			un[un_n++]=0;
			const std::size_t m=un_n-3u;
			for(std::size_t jj=m+1u;jj>0;--jj){
				const std::size_t j=jj-1u;
				std::uint64_t qhat=0;
				std::uint64_t rhat=0;
				std::uint64_t rhat_ovf=0;
				if(un[j+2u]==vn[1]){
					qhat=std::numeric_limits<std::uint64_t>::max();
					rhat_ovf=addc_u64(0,un[j+1u],vn[1],&rhat);
				}else{
					qhat=udiv128(un[j+2u],un[j+1u],vn[1],&rhat);
				}
				while(rhat_ovf==0){
					const u128 lhs=mul_u64(qhat,vn[0]);
					const u128 rhs{un[j],rhat};
					if(cmp_u128(lhs,rhs)<=0)
						break;
					--qhat;
					rhat_ovf=addc_u64(0,rhat,vn[1],&rhat);
				}
				const std::uint64_t cy=sm_1n(un.data()+j,vn,2u,qhat);
				std::uint64_t outn=0;
				const std::uint64_t borrow=subb_u64(0,un[j+2u],cy,&outn);
				un[j+2u]=outn;
				if(borrow!=0){
					const std::uint64_t carry=add_nn(un.data()+j,un.data()+j,
													 vn,2u);
					std::uint64_t top=0;
					(void)addc_u64(0,un[j+2u],carry,&top);
					un[j+2u]=top;
				}
			}
			limbs_t r;
			r.reserve(2);
			if(s==0){
				if(un[0]!=0||un[1]!=0){
					r.push_back(un[0]);
					if(un[1]!=0)
						r.push_back(un[1]);
				}
				return r;
			}
			const std::uint64_t lo=(un[0]>>s)|(un[1]<<(64u-s));
			const std::uint64_t hi=un[1]>>s;
			if(lo!=0||hi!=0){
				r.push_back(lo);
				if(hi!=0)
					r.push_back(hi);
			}
			return r;
		}
	}
	limbs_t r;
	limbs_t un;
	shl_into(un,u,shift_bits+static_cast<std::size_t>(s));
	un.push_back(0);
	if(un.size()<3u)
		return {};
	const std::size_t m=un.size()-3u;
	for(std::size_t jj=m+1u;jj>0;--jj){
		const std::size_t j=jj-1u;
		std::uint64_t qhat=0;
		std::uint64_t rhat=0;
		std::uint64_t rhat_ovf=0;
		if(un[j+2u]==vn[1]){
			qhat=std::numeric_limits<std::uint64_t>::max();
			rhat_ovf=addc_u64(0,un[j+1u],vn[1],&rhat);
		}else{
			qhat=udiv128(un[j+2u],un[j+1u],vn[1],&rhat);
		}
		while(rhat_ovf==0){
			const u128 lhs=mul_u64(qhat,vn[0]);
			const u128 rhs{un[j],rhat};
			if(cmp_u128(lhs,rhs)<=0)
				break;
			--qhat;
			rhat_ovf=addc_u64(0,rhat,vn[1],&rhat);
		}
		const std::uint64_t cy=sm_1n(un.data()+static_cast<std::ptrdiff_t>(j),
									 vn,2u,qhat);
		std::uint64_t outn=0;
		const std::uint64_t borrow=subb_u64(0,un[j+2u],cy,&outn);
		un[j+2u]=outn;
		if(borrow!=0){
			const std::uint64_t carry=add_nn(
				un.data()+static_cast<std::ptrdiff_t>(j),
				un.data()+static_cast<std::ptrdiff_t>(j),vn,2u);
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+2u],carry,&top);
			un[j+2u]=top;
		}
	}
	r.assign(un.begin(),un.begin()+2);
	if(s!=0)
		shr_ip(r,s);
	else
		trim_lz(r);
	return r;
}

inline limbs_t modk_absl(const limbs_t&u,const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return {};
	const int cmp=cmp_abs(u,v);
	if(cmp<0)
		return u;
	if(cmp==0)
		return {};
	if(cmp_abs_dbl(u,v)<0){
		limbs_t r=u;
		sub_abs_ip(r,v);
		return r;
	}

	if(v.size()==1){
		std::uint64_t rem=0;
		for(std::size_t i=u.size();i>0;--i){
			(void)udiv128(rem,u[i-1],v[0],&rem);
		}
		limbs_t r;
		if(rem!=0)
			r.push_back(rem);
		return r;
	}
	if(u.size()==v.size()){
		return dvm1q_absl(u,v).second;
	}

	limbs_t r=u;
	ModKScr sc;
	modk_liip(r,v,sc);
	return r;
}

inline std::uint32_t mod_u32_abs(const limbs_t&x,std::uint32_t d){
	return mod_small(x,d);
}

inline void mod_abs_ip(limbs_t&x,const limbs_t&mod){
	if(x.empty())
		return;
	if(mod.size()==1u){
		const std::uint64_t r=mod_limb(x,mod[0]);
		x.clear();
		if(r!=0)
			x.push_back(r);
		return;
	}
	if(mod.size()==2u){
		x=mod2_absl(x,mod);
		return;
	}
	ModKScr sc;
	modk_ip(x,mod,sc);
}

inline void cpylo_in(const limbs_t&src,std::size_t count,
								limbs_t&out){
	const std::size_t n=std::min(count,src.size());
	out.resize_uninit(n);
	if(n!=0)
		std::memcpy(out.data(),src.data(),n*sizeof(limb_t));
	trim_lz(out);
}

inline void cpyhi_in(const limbs_t&src,std::size_t start,
								 limbs_t&out){
	if(start>=src.size()){
		out.clear();
		return;
	}
	const std::size_t n=src.size()-start;
	out.resize_uninit(n);
	std::memcpy(out.data(),
				src.data()+static_cast<std::ptrdiff_t>(start),
				n*sizeof(limb_t));
	trim_lz(out);
}

struct MontCtx{
	limbs_t mod;
	std::uint64_t n0prime=0; 
	std::size_t k=0;
	limbs_t r2;	 
	limbs_t t;	 
	limbs_t tmp; 
};

inline MontCtx mk_mctx(const limbs_t&mod){
	MINI_MP_ASSERT(!mod.empty());
	MINI_MP_ASSERT((mod[0]&1u)!=0u);
	MontCtx ctx{};
	ctx.mod=mod;
	ctx.k=mod.size();
	ctx.n0prime=0u-invodd264(mod[0]);

	ctx.r2.assign(2*ctx.k+1,0);
	ctx.r2[2*ctx.k]=1;
	ModKScr sc;
	modk_liip(ctx.r2,ctx.mod,sc);

	ctx.t.reserve(2*ctx.k+3);
	ctx.tmp.reserve(ctx.k+2);
	return ctx;
}

inline void mont_red(limbs_t&t,MontCtx&ctx,limbs_t&out){
	MINI_MP_ASSERT(ctx.k!=0);
	const std::size_t k=ctx.k;
	if(t.size()<2*k+3)
		t.resize(2*k+3,0);

	for(std::size_t i=0;i<k;++i){
		const std::uint64_t m=t[i]*ctx.n0prime;
		std::uint64_t carry=am_1n(
			t.data()+static_cast<std::ptrdiff_t>(i),
			ctx.mod.data(),k,m);
		std::size_t idx=i+k;
		while(carry!=0){
			std::uint64_t sum=0;
			carry=addc_u64(0,t[idx],carry,&sum);
			t[idx]=sum;
			++idx;
		}
	}

	cpyhi_in(t,k,out);
	if(out.size()>k+1){
		for(std::size_t i=k+1;i<out.size();++i){
			MINI_MP_ASSERT(out[i]==0);
		}
		out.resize(k+1);
		trim_lz(out);
	}
	if(cmp_abs(out,ctx.mod)>=0){
		sub_abs_ip(out,ctx.mod);
	}
}

inline void mont_mul(const limbs_t&a_bar,const limbs_t&b_bar,
						   MontCtx&ctx,limbs_t&out){
	if(&a_bar==&b_bar)
		sqrbig_in(a_bar,ctx.t);
	else
		mulbig_in(a_bar,b_bar,ctx.t);
	if(ctx.t.size()<2*ctx.k+3)
		ctx.t.resize(2*ctx.k+3,0);
	mont_red(ctx.t,ctx,ctx.tmp);
	out.swap(ctx.tmp);
}

inline void mont_to(const limbs_t&x,MontCtx&ctx,limbs_t&out){
	mont_mul(x,ctx.r2,ctx,out);
}

inline void mont_fr(const limbs_t&x_bar,MontCtx&ctx,limbs_t&out){
	ctx.t.assign(2*ctx.k+3,0);
	const std::size_t n=std::min(x_bar.size(),ctx.t.size());
	for(std::size_t i=0;i<n;++i){
		ctx.t[i]=x_bar[i];
	}
	mont_red(ctx.t,ctx,ctx.tmp);
	out.swap(ctx.tmp);
}

struct BarModCtx{
	limbs_t mod;
	std::size_t k=0;
	limbs_t mu; 
	limbs_t q1;
	limbs_t q2;
	limbs_t q3;
	limbs_t r1;
	limbs_t r2;
	limbs_t prod;
};

inline BarModCtx mk_bctx(const limbs_t&mod){
	MINI_MP_ASSERT(!mod.empty());
	BarModCtx ctx{};
	ctx.mod=mod;
	ctx.k=mod.size();

	limbs_t beta_2k(2*ctx.k+1,0);
	beta_2k[2*ctx.k]=1;
	auto qr=dvmk_absl(beta_2k,ctx.mod);
	ctx.mu=std::move(qr.first);

	ctx.q1.reserve(2*ctx.k+2);
	ctx.q2.reserve(3*ctx.k+2);
	ctx.q3.reserve(2*ctx.k+2);
	ctx.r1.reserve(ctx.k+2);
	ctx.r2.reserve(ctx.k+2);
	ctx.prod.reserve(3*ctx.k+2);
	return ctx;
}

inline void bar_redip(limbs_t&x,BarModCtx&ctx){
	if(x.empty())
		return;
	if(cmp_abs(x,ctx.mod)<0)
		return;

	const std::size_t k=ctx.k;
	MINI_MP_ASSERT(k>=2);

	cpyhi_in(x,k-1,ctx.q1);
	mulbig_in(ctx.q1,ctx.mu,ctx.q2);
	cpyhi_in(ctx.q2,k+1,ctx.q3);

	cpylo_in(x,k+1,ctx.r1);
	mulbig_in(ctx.q3,ctx.mod,ctx.prod);
	cpylo_in(ctx.prod,k+1,ctx.r2);

	if(cmp_abs(ctx.r1,ctx.r2)>=0){
		x=sub_abs(ctx.r1,ctx.r2);
	}else{
		x=ctx.r1;
		x.resize(k+2,0);
		x[k+1]=1;
		sub_abs_ip(x,ctx.r2);
	}

	while(cmp_abs(x,ctx.mod)>=0){
		sub_abs_ip(x,ctx.mod);
	}
}

inline void bar_mulm(const limbs_t&a,const limbs_t&b,BarModCtx&ctx,
							limbs_t&out){
	if(&a==&b)
		sqrbig_in(a,out);
	else
		mulbig_in(a,b,out);
	bar_redip(out,ctx);
}

struct MulModScr{
	limbs_t t;
	ModKScr mod_sc;
};

inline void mulmod_ip(limbs_t&acc,const limbs_t&x,const limbs_t&mod,
						   MulModScr&sc){
	if(acc.empty()||x.empty()){
		acc.clear();
		return;
	}
	if(&acc==&x)
		sqrbig_in(acc,sc.t);
	else
		mulbig_in(acc,x,sc.t);
	modk_ip(sc.t,mod,sc.mod_sc);
	acc.swap(sc.t);
}

inline void sqrmod_ip(limbs_t&x,const limbs_t&mod,MulModScr&sc){
	if(x.empty())
		return;
	sqrbig_in(x,sc.t);
	modk_ip(sc.t,mod,sc.mod_sc);
	x.swap(sc.t);
}

inline unsigned mpw_bits(std::size_t exp_bits) noexcept{
	if(exp_bits>=tun_pow_w6())
		return 6;
	if(exp_bits>=tun_pow_w5())
		return 5;
	return 4;
}

inline limbs_t gcd_bin_s(limbs_t a,limbs_t b){
	if(a.empty())
		return b;
	if(b.empty())
		return a;

	const std::size_t shift=std::min(ctz(a),ctz(b));
	if(shift!=0){
		shr_ip(a,shift);
		shr_ip(b,shift);
	}

	if((a[0]&1u)==0){
		shr_ip(a,ctz(a));
	}
	do{
		if((b[0]&1u)==0){
			shr_ip(b,ctz(b));
		}
		if(cmp_abs(a,b)>0){
			std::swap(a,b);
		}
		sub_abs_ip(b,a);
	}while(!b.empty());

	if(shift!=0){
		shl_into(a,a,shift);
	}
	return a;
}

inline limbs_t gcd_2limb_s(const limbs_t&a,const limbs_t&b){
#if MINI_MP_DETAIL_HAS_UINT128
	auto to_u128=[](const limbs_t&x) noexcept{
		unsigned __int128 v=0;
		if(!x.empty())
			v=static_cast<unsigned __int128>(x[0]);
		if(x.size()>1)
			v|=(static_cast<unsigned __int128>(x[1])<<64);
		return v;
	};
	unsigned __int128 x=to_u128(a);
	unsigned __int128 y=to_u128(b);
	while(y!=0){
		const unsigned __int128 r=x%y;
		x=y;
		y=r;
	}
	limbs_t out;
	const std::uint64_t lo=static_cast<std::uint64_t>(x);
	const std::uint64_t hi=static_cast<std::uint64_t>(x>>64);
	if(lo!=0||hi!=0)
		out.push_back(lo);
	if(hi!=0)
		out.push_back(hi);
	return out;
#else
	return gcd_bin_s(a,b);
#endif
}

inline limbs_t divk_absl(const limbs_t&u,const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return {};
	if(cmp_abs(u,v)<0)
		return {};

	if(v.size()==1){
		limbs_t q=u;
		std::uint64_t rem=0;
		for(std::size_t i=q.size();i>0;--i){
			q[i-1]=udiv128(rem,q[i-1],v[0],&rem);
		}
		trim_lz(q);
		return q;
	}
	if(u.size()==v.size()){
		auto qr=dvm1q_absl(u,v);
		limbs_t q;
		if(qr.first!=0)
			q.push_back(qr.first);
		return q;
	}

	const std::size_t n=v.size();
	const unsigned s=std::countl_zero(v.back());
	limbs_t vn=shl_bits(v,s);
	limbs_t un=shl_bits(u,s);
	un.push_back(0);

	const std::size_t m=un.size()-n-1;
	limbs_t q(m+1,0);

	for(std::size_t jj=m+1;jj>0;--jj){
		const std::size_t j=jj-1;

		std::uint64_t qhat=0;
		std::uint64_t rhat=0;
		std::uint64_t rhat_ovf=0;

		if(un[j+n]==vn[n-1]){
			qhat=std::numeric_limits<std::uint64_t>::max();
			rhat_ovf=addc_u64(0,un[j+n-1],vn[n-1],&rhat);
		}else{
			qhat=udiv128(un[j+n],un[j+n-1],vn[n-1],&rhat);
		}

		if(n>1){
			while(rhat_ovf==0){
				const u128 lhs=mul_u64(qhat,vn[n-2]);
				const u128 rhs{un[j+n-2],rhat};
				if(cmp_u128(lhs,rhs)<=0)
					break;
				--qhat;
				rhat_ovf=addc_u64(0,rhat,vn[n-1],&rhat);
			}
		}

		std::uint64_t borrow=0;
		std::uint64_t k=0;
		for(std::size_t i=0;i<n;++i){
			const u128 p=mul_u64(qhat,vn[i]);
			std::uint64_t t=0;
			const std::uint64_t c1=addc_u64(0,p.lo,k,&t);

			std::uint64_t t2=0;
			const std::uint64_t b1=subb_u64(0,un[j+i],t,&t2);
			std::uint64_t out=0;
			const std::uint64_t b2=subb_u64(0,t2,borrow,&out);
			un[j+i]=out;
			borrow=(b1|b2);

			std::uint64_t next_k=0;
			const std::uint64_t kc=addc_u64(0,p.hi,c1,&next_k);
			MINI_MP_ASSERT(kc==0);
			k=next_k;
		}

		std::uint64_t t3=0;
		const std::uint64_t b1=subb_u64(0,un[j+n],k,&t3);
		std::uint64_t outn=0;
		const std::uint64_t b2=subb_u64(0,t3,borrow,&outn);
		un[j+n]=outn;
		const std::uint64_t borrow_out=(b1|b2);

		if(borrow_out!=0){
			--qhat;
			std::uint64_t carry=0;
			for(std::size_t i=0;i<n;++i){
				std::uint64_t t=0;
				const std::uint64_t c1=addc_u64(0,un[j+i],vn[i],&t);
				std::uint64_t out=0;
				const std::uint64_t c2=addc_u64(0,t,carry,&out);
				un[j+i]=out;
				MINI_MP_ASSERT((c1&c2)==0);
				carry=(c1|c2);
			}
			std::uint64_t top=0;
			(void)addc_u64(0,un[j+n],carry,&top);
			un[j+n]=top;
		}

		q[j]=qhat;
	}

	trim_lz(q);
	return q;
}

inline bool use_bzdiv(std::size_t u_limbs,
									   std::size_t v_limbs) noexcept{
	
	
	const std::size_t bz_min=tun_bz_min();
	const std::size_t bz_chunk=tun_bz_chunk();
	if(v_limbs<bz_min)
		return false;
	if(u_limbs<=v_limbs)
		return false;
	const std::size_t q_limbs=u_limbs-v_limbs+1;
	const std::size_t chunks=(q_limbs+v_limbs-1)/v_limbs;
	return chunks>=bz_chunk;
}



inline std::pair<limbs_t,limbs_t> dvmbz_abs(const limbs_t&u,
													  const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return {{},{}};
	if(cmp_abs(u,v)<0)
		return {{},u};

	if(v.size()==1){
		limbs_t q=u;
		std::uint64_t rem=0;
		for(std::size_t i=q.size();i>0;--i){
			q[i-1]=udiv128(rem,q[i-1],v[0],&rem);
		}
		trim_lz(q);
		limbs_t r;
		if(rem!=0)
			r.push_back(rem);
		return {std::move(q),std::move(r)};
	}

	const std::size_t m=v.size();
	const std::size_t num_chunks=(u.size()+m-1)/m;
	std::vector<limbs_t> q_chunks(num_chunks);

	limbs_t r;
	r.reserve(2*m+2);

	for(std::size_t cc=num_chunks;cc>0;--cc){
		const std::size_t chunk_idx=cc-1;
		const std::size_t begin=chunk_idx*m;
		const std::size_t end=std::min(begin+m,u.size());
		const std::size_t chunk_len=end-begin;

		if(!r.empty()){
			r.insert(r.begin(),static_cast<std::ptrdiff_t>(m),0);
			if(chunk_len!=0){
				std::copy(u.begin()+static_cast<std::ptrdiff_t>(begin),
						  u.begin()+static_cast<std::ptrdiff_t>(end),r.begin());
			}
		}else if(chunk_len!=0){
			r.assign(u.begin()+static_cast<std::ptrdiff_t>(begin),
					 u.begin()+static_cast<std::ptrdiff_t>(end));
		}

		trim_lz(r);
		if(r.empty()){
			q_chunks[chunk_idx].clear();
			continue;
		}

		auto qr=dvmk_absl(r,v);
		q_chunks[chunk_idx]=std::move(qr.first);
		r=std::move(qr.second);
	}

	limbs_t q;
	q.reserve(u.size()-v.size()+2);
	for(std::size_t i=0;i<num_chunks;++i){
		if(!q_chunks[i].empty()){
			add_sh_ip(q,q_chunks[i],i*m);
		}
	}

	trim_lz(q);
	trim_lz(r);
	return {std::move(q),std::move(r)};
}

inline int ch_to_dig(char c) noexcept{
	if(c>='0'&&c<='9')
		return c-'0';
	if(c>='a'&&c<='z')
		return c-'a'+10;
	if(c>='A'&&c<='Z')
		return c-'A'+10;
	return -1;
}

inline char dig_to_ch(unsigned d) noexcept{
	MINI_MP_ASSERT(d<36);
	if(d<10)
		return static_cast<char>('0'+d);
	return static_cast<char>('a'+(d-10));
}

inline bool is_space(char ch) noexcept{
	return std::isspace(static_cast<unsigned char>(ch))!=0;
}

inline std::string_view trim_ws(std::string_view sv) noexcept{
	while(!sv.empty()&&is_space(sv.front()))
		sv.remove_prefix(1);
	while(!sv.empty()&&is_space(sv.back()))
		sv.remove_suffix(1);
	return sv;
}

inline constexpr std::size_t kKarTh=384;
inline constexpr std::size_t kNttDef=256;
inline constexpr std::size_t kNttBitsDef=24;
inline constexpr std::size_t kNttOff=
	std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kKarImb=4;
inline constexpr std::size_t kNttImb=2;
inline constexpr std::size_t kHlmDef=3;
inline constexpr std::size_t kHlrDef=0;
inline constexpr std::size_t kBzDivMnDef=128;
inline constexpr std::size_t kBzChunkDef=3;
inline constexpr std::size_t kGcdSmMxDef=2;
inline constexpr std::size_t kGcdLgMxDef=4;
inline constexpr std::size_t kGcdQsDef=16;
inline constexpr std::size_t kProdLeafDef=24;
inline constexpr std::size_t kFacTreeDef=192;
inline constexpr std::size_t kBinomTreeDef=160;
inline constexpr std::size_t kPowW5Def=384;
inline constexpr std::size_t kPowW6Def=1536;
inline constexpr std::size_t kD10DcDef=kNttOff;
inline constexpr std::size_t kD10PrsDef=kNttOff;
inline constexpr std::size_t kT3Def=12288u;

inline std::size_t tun_ntt() noexcept;
inline std::size_t tun_ntt_sq() noexcept;
inline std::size_t tun_ntt_bits() noexcept;
inline std::size_t tun_krec() noexcept;
inline std::size_t tun_srec() noexcept;
inline std::size_t tun_kar() noexcept;
inline std::size_t tun_kar_imb() noexcept;
inline std::size_t tun_ntt_imb() noexcept;
inline std::size_t tun_hmin() noexcept;
inline std::size_t tun_hrnd() noexcept;
inline std::size_t tun_gcd_sm() noexcept;
inline std::size_t tun_gcd_lg() noexcept;
inline std::size_t tun_bz_min() noexcept;
inline std::size_t tun_bz_chunk() noexcept;
inline std::size_t tun_prod_leaf() noexcept;
inline std::size_t tun_fac_tree() noexcept;
inline std::size_t tun_binom_tree() noexcept;
inline std::size_t tun_pow_w5() noexcept;
inline std::size_t tun_pow_w6() noexcept;
inline std::size_t tun_d10_dc() noexcept;
inline std::size_t tun_d10_prs() noexcept;
inline std::size_t tun_t3() noexcept;
inline void ensure_at();

} 

class BigInt;
class BigRat;
class BigFloat;

namespace detail{
inline BigInt factorial_loop(std::uint64_t n);
inline BigInt factorial_tree(std::uint64_t n,std::size_t leaf);
inline BigInt binomial_loop(std::uint64_t n,std::uint64_t k);
inline BigInt binomial_tree(std::uint64_t n,std::uint64_t k,
							std::size_t leaf);
inline std::uint64_t bench_sqr_mul(const BigInt&a,int loops,bool use_ntt);
}

struct ExtGcdRes;
struct CrtRes;

inline BigInt mul_sbk(const BigInt&a,const BigInt&b);
inline BigInt mul_kar(const BigInt&a,const BigInt&b);
inline BigInt mul_t3(const BigInt&a,const BigInt&b);
inline BigInt mul_ntt(const BigInt&a,const BigInt&b);
inline BigInt mul_disp(const BigInt&a,const BigInt&b);
inline BigInt sqr_disp_noat(const BigInt&a);
inline BigInt sqr_disp(const BigInt&a);

inline std::pair<BigInt,BigInt> dvm_simp(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> dvm_p2(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> dvm_knuth(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> divmod(const BigInt&a,const BigInt&b);
inline BigInt divk_q(const BigInt&a,const BigInt&b);
inline BigInt divk_r(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> dvm_ablb(const BigInt&abs_a,limb_t d);

inline BigInt gcd(BigInt a,BigInt b);
inline BigInt gcd_hlprm(BigInt a,BigInt b,
										  std::size_t hl_min,
										  std::size_t hl_rnd);
inline ExtGcdRes extgcd(BigInt a,BigInt b);
inline BigInt lcm(const BigInt&a,const BigInt&b);
inline BigInt pow(BigInt base,std::uint64_t exp);
inline BigInt modpow(BigInt base,BigInt exp,const BigInt&mod);
inline BigInt invmod(const BigInt&a,const BigInt&mod);
inline BigInt gcdext(BigInt*s,BigInt*t,const BigInt&a,const BigInt&b);
inline bool invert(BigInt*rop,const BigInt&a,const BigInt&mod);
inline BigInt invert(const BigInt&a,const BigInt&mod);
inline int jacobi(const BigInt&a,const BigInt&b);
inline int kronecker(const BigInt&a,const BigInt&b);
inline CrtRes crt_solve(const std::vector<std::pair<BigInt,BigInt>>&eqs);
inline bool crt_solve(BigInt*value,BigInt*modulus,
					  const std::vector<std::pair<BigInt,BigInt>>&eqs);
inline std::pair<BigInt,BigInt> crt(
	const std::vector<std::pair<BigInt,BigInt>>&eqs);
inline std::pair<BigInt,BigInt> dvm_trnc(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> dvm_floor(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> dvm_ceil(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> tdiv_qr(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> fdiv_qr(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> cdiv_qr(const BigInt&a,const BigInt&b);
inline BigInt tdiv_q(const BigInt&a,const BigInt&b);
inline BigInt fdiv_q(const BigInt&a,const BigInt&b);
inline BigInt cdiv_q(const BigInt&a,const BigInt&b);
inline BigInt tdiv_r(const BigInt&a,const BigInt&b);
inline BigInt fdiv_r(const BigInt&a,const BigInt&b);
inline BigInt cdiv_r(const BigInt&a,const BigInt&b);
inline BigInt divexact(const BigInt&a,const BigInt&b);
inline std::pair<BigInt,BigInt> sqrtrem_split(const BigInt&n,
											  std::size_t nbits);
inline std::pair<BigInt,BigInt> sqrtrem(const BigInt&n);
inline BigInt isqrt(const BigInt&n);
inline bool sqrtmod(BigInt*root,const BigInt&a,const BigInt&mod);
inline BigInt sqrtmod(const BigInt&a,const BigInt&mod);
inline std::pair<BigInt,BigInt> rootrem(const BigInt&n,std::uint32_t k);
inline BigInt iroot(const BigInt&n,std::uint32_t k);
inline bool is_ppow(const BigInt&n);
inline bool is_square(const BigInt&n);
inline bool is_strong_probabprime(const BigInt&n,const BigInt&base);
inline bool is_lucas_prp(const BigInt&n);
inline bool is_bpsw_prp(const BigInt&n);
inline int pr_prime(const BigInt&n,int rounds=16);
inline BigInt next_prime(const BigInt&n,int rounds=16);
inline BigInt factorial(std::uint64_t n);
inline BigInt binomial(std::uint64_t n,std::uint64_t k);
inline BigInt fibonacci(std::uint64_t n);





class BigInt{
  public:
	BigInt() noexcept=default;

	BigInt(std::int64_t v){
		if(v==0)
			return;
		if(v<0){
			sign_=-1;
			std::uint64_t mag=static_cast<std::uint64_t>(-(v+1));
			++mag;
			limbs_.push_back(mag);
		}else{
			sign_=1;
			limbs_.push_back(static_cast<std::uint64_t>(v));
		}
	}

	explicit BigInt(std::string_view text,int base=10){
		*this=parse(text,base);
	}

	static BigInt from_u64(std::uint64_t v){
		if(v==0)
			return BigInt();
		BigInt out;
		out.sign_=1;
		out.limbs_.push_back(v);
		return out;
	}

	static BigInt from_bytes(const void*data,std::size_t size,
							 bool most_significant_first=true,
							 int sign=1){
		if(data==nullptr&&size!=0){
			detail::throw_inv("BigInt::from_bytes: null data");
		}
		if(size==0||sign==0)
			return BigInt();
		if(size>std::numeric_limits<std::size_t>::max()-7u)
			detail::throw_ovf("BigInt::from_bytes: too many bytes");
		const auto*bytes=static_cast<const unsigned char*>(data);
		const std::size_t nlimbs=(size+7u)/8u;
		detail::limbs_t mag(nlimbs,0);
		if(std::endian::native==std::endian::little){
			if(!most_significant_first){
				std::memcpy(mag.data(),bytes,size);
			}else{
				std::size_t pos=size;
				std::size_t out=0;
				while(pos>=8u){
					pos-=8u;
					limb_t w=0;
					std::memcpy(&w,bytes+pos,8u);
					mag[out++]=detail::bswap64(w);
				}
				if(pos!=0){
					limb_t w=0;
					for(std::size_t i=0;i<pos;++i)
						w=(w<<8u)|static_cast<limb_t>(bytes[i]);
					mag[out]=w;
				}
			}
		}else{
			for(std::size_t i=0;i<size;++i){
				const unsigned char byte=most_significant_first?
					bytes[size-1u-i]:bytes[i];
				mag[i/8u]|=static_cast<limb_t>(byte)<<
					static_cast<unsigned>((i&7u)*8u);
			}
		}
		return from_raw(sign<0?-1:1,std::move(mag));
	}

	static BigInt import_words(const void*data,std::size_t count,
							   std::size_t word_size,
							   int word_order=1,int byte_order=1,
							   std::size_t nail_bits=0,int sign=1){
		if(data==nullptr&&count!=0)
			detail::throw_inv("BigInt::import_words: null data");
		if(sign==0||count==0)
			return BigInt();
		if(word_order!=1&&word_order!=-1)
			detail::throw_inv("BigInt::import_words: bad word order");
		if(byte_order==0){
			byte_order=(std::endian::native==std::endian::big)?1:-1;
		}else if(byte_order!=1&&byte_order!=-1){
			detail::throw_inv("BigInt::import_words: bad byte order");
		}
		if(word_size==0||
		   word_size>std::numeric_limits<std::size_t>::max()/8u)
			detail::throw_inv("BigInt::import_words: bad word size");
		const std::size_t word_bits=word_size*8u;
		if(nail_bits>=word_bits)
			detail::throw_inv("BigInt::import_words: bad nail bits");
		const std::size_t payload_bits=word_bits-nail_bits;
		if(count>(std::numeric_limits<std::size_t>::max()-63u)/
				 payload_bits)
			detail::throw_ovf("BigInt::import_words: too many bits");
		if(nail_bits==0&&word_size==1u)
			return from_bytes(data,count,word_order>0,sign);
		if(nail_bits==0&&word_size==sizeof(limb_t)){
			detail::limbs_t mag;
			mag.resize_uninit(count);
			const auto*bytes=static_cast<const unsigned char*>(data);
			if(word_order<0&&
			   ((byte_order<0&&std::endian::native==std::endian::little)||
				(byte_order>0&&std::endian::native==std::endian::big))){
				std::memcpy(mag.data(),bytes,count*word_size);
				return from_raw(sign<0?-1:1,std::move(mag));
			}
			for(std::size_t oi=0;oi<count;++oi){
				const std::size_t src_word=word_order<0?oi:(count-1u-oi);
				const auto*wp=bytes+src_word*word_size;
				limb_t w=0;
				if((byte_order<0&&std::endian::native==std::endian::little)||
				   (byte_order>0&&std::endian::native==std::endian::big)){
					std::memcpy(&w,wp,sizeof(w));
				}else if(byte_order<0){
					for(std::size_t bi=0;bi<sizeof(limb_t);++bi)
						w|=static_cast<limb_t>(wp[bi])<<
							static_cast<unsigned>(8u*bi);
				}else{
					for(std::size_t bi=0;bi<sizeof(limb_t);++bi)
						w=(w<<8u)|static_cast<limb_t>(wp[bi]);
				}
				mag[oi]=w;
			}
			return from_raw(sign<0?-1:1,std::move(mag));
		}
		const std::size_t total_bits=count*payload_bits;
		detail::limbs_t mag((total_bits+63u)/64u,0);
		const auto*bytes=static_cast<const unsigned char*>(data);
		std::size_t out_limb=0;
		limb_t limb=0;
		unsigned limb_bits=0;

		auto append_bits=[&](limb_t bits,unsigned nbits){
			if(nbits==0)
				return;
			bits&=(limb_t(1)<<nbits)-1u;
			const unsigned room=64u-limb_bits;
			if(nbits<room){
				limb|=bits<<limb_bits;
				limb_bits=static_cast<unsigned>(limb_bits+nbits);
				return;
			}
			limb|=bits<<limb_bits;
			mag[out_limb++]=limb;
			const unsigned spill=static_cast<unsigned>(nbits-room);
			limb=spill==0u?0u:(bits>>room);
			limb_bits=spill;
		};

		const std::size_t whole_bytes=payload_bits/8u;
		const unsigned rem_bits=static_cast<unsigned>(payload_bits&7u);
		for(std::size_t wi=0;wi<count;++wi){
			const std::size_t src_word=word_order<0?wi:(count-1u-wi);
			const auto*wp=bytes+src_word*word_size;
			for(std::size_t bi=0;bi<whole_bytes;++bi){
				const std::size_t src_byte=byte_order<0?
					bi:(word_size-1u-bi);
				append_bits(wp[src_byte],8u);
			}
			if(rem_bits!=0u){
				const std::size_t src_byte=byte_order<0?
					whole_bytes:(word_size-1u-whole_bytes);
				append_bits(wp[src_byte],rem_bits);
			}
		}
		if(limb_bits!=0u){
			MINI_MP_ASSERT(out_limb<mag.size());
			mag[out_limb++]=limb;
		}
		mag.resize(out_limb);
		return from_raw(sign<0?-1:1,std::move(mag));
	}

	static BigInt parse(std::string_view text,int base=10){
		text=detail::trim_ws(text);
		if(text.empty()){
			detail::throw_inv("BigInt::parse: empty input");
		}

		std::size_t i=0;
		int sign=1;
		if(text[i]=='+'||text[i]=='-'){
			sign=(text[i]=='-')?-1:1;
			++i;
		}

		if(i>=text.size()){
			detail::throw_inv(
				"BigInt::parse: sign without digits");
		}

		if(base==0){
			if(i+1<text.size()&&text[i]=='0'&&(text[i+1]=='x'||text[i+1]=='X')){
				base=16;
				i+=2;
			}else{
				base=10;
			}
		}
		if(base<2||base>36){
			detail::throw_inv(
				"BigInt::parse: base out of range [2,36]");
		}
		if(base==16&&i+1<text.size()&&text[i]=='0'&&
		   (text[i+1]=='x'||text[i+1]=='X')){
			i+=2;
		}

		const std::size_t dig_beg=i;
		if(base==10){
			detail::limbs_t mag;
			const std::string_view digits=text.substr(dig_beg);
			if(!digits.empty()){
				if(digits.size()>=512u)
					detail::ensure_at();
				if(detail::prs_d10(digits,&mag,detail::tun_d10_prs()))
					return from_raw(sign,std::move(mag));
			}
		}

		while(i<text.size()&&!detail::is_space(text[i]))
			++i;
		const std::size_t digit_end=i;
		while(i<text.size()&&detail::is_space(text[i]))
			++i;
		if(i!=text.size()){
			detail::throw_inv(
				"BigInt::parse: trailing invalid characters");
		}
		if(dig_beg==digit_end){
			detail::throw_inv("BigInt::parse: no digits");
		}

		detail::limbs_t mag;
		const std::string_view digits=
			text.substr(dig_beg,digit_end-dig_beg);

		
		if(detail::ispow2_32(static_cast<std::uint32_t>(base))&&
		   base<=32){
			if(!detail::prs_p2dl(digits,base,&mag)){
				detail::throw_inv("BigInt::parse: invalid digit");
			}
			return from_raw(sign,std::move(mag));
		}
		if(base==10){
			if(digits.size()>=512u)
				detail::ensure_at();
			if(!detail::prs_d10(digits,&mag,detail::tun_d10_prs())){
				detail::throw_inv("BigInt::parse: invalid digit");
			}
			return from_raw(sign,std::move(mag));
		}

		const auto [chunk_mul,chunk_len]=
			detail::chunk_par_lb(static_cast<std::uint32_t>(base));

		auto prs_chunk=[&](std::size_t pos,std::size_t len) -> std::uint64_t{
			std::uint64_t v=0;
			for(std::size_t j=0;j<len;++j){
				const int d=detail::ch_to_dig(text[pos+j]);
				if(d<0||d>=base){
					detail::throw_inv(
						"BigInt::parse: invalid digit");
				}
				v=v*static_cast<std::uint32_t>(base)+
				  static_cast<std::uint32_t>(d);
			}
			return v;
		};

		std::size_t pos=dig_beg;
		std::size_t first_len=(digit_end-dig_beg)%chunk_len;
		if(first_len==0)
			first_len=chunk_len;

		const std::uint64_t first_v=prs_chunk(pos,first_len);
		if(first_v!=0)
			mag.push_back(first_v);
		pos+=first_len;

		while(pos<digit_end){
			const std::uint64_t v=prs_chunk(pos,chunk_len);
			detail::muladd_lb(mag,chunk_mul,v);
			pos+=chunk_len;
		}

		return from_raw(sign,std::move(mag));
	}

	std::string to_string(int base=10) const{
		if(base<2||base>36){
			detail::throw_inv(
				"BigInt::to_string: base out of range [2,36]");
		}
		if(is_zero())
			return "0";

		
		if(detail::ispow2_32(static_cast<std::uint32_t>(base))&&
		   base<=32){
			return detail::to_str_p2(limbs_,sign_,base);
		}
		if(base==10){
			if(limbs_.size()>=64u)
				detail::ensure_at();
			return detail::to_str_d10(
				limbs_,sign_,detail::tun_d10_dc());
		}

		detail::limbs_t tmp=limbs_;
		const auto [chunk_mul,chunk_len]=
			detail::chunk_par_lb(static_cast<std::uint32_t>(base));

		std::vector<std::uint64_t> chunks;
		chunks.reserve(tmp.size()*3);
		while(!tmp.empty()){
			chunks.push_back(detail::div_limb_ip(tmp,chunk_mul));
		}
		MINI_MP_ASSERT(!chunks.empty());

		std::string out;
		if(sign_<0)
			out.push_back('-');

		auto appch_npd=[&](std::uint64_t v){
			if(v==0){
				out.push_back('0');
				return;
			}
			char buf[64];
			std::size_t n=0;
			while(v!=0){
				const std::uint64_t q=v/static_cast<std::uint32_t>(base);
				const std::uint32_t d=
					static_cast<std::uint32_t>(
						v-q*static_cast<std::uint32_t>(base));
				v=q;
				buf[n++]=detail::dig_to_ch(d);
			}
			while(n>0){
				out.push_back(buf[--n]);
			}
		};

		auto appch_pad=[&](std::uint64_t v){
			char buf[64];
			MINI_MP_ASSERT(chunk_len<sizeof(buf));
			for(std::size_t i=0;i<chunk_len;++i){
				const std::uint64_t q=v/static_cast<std::uint32_t>(base);
				const std::uint32_t d=
					static_cast<std::uint32_t>(
						v-q*static_cast<std::uint32_t>(base));
				v=q;
				buf[chunk_len-1-i]=detail::dig_to_ch(d);
			}
			out.append(buf,buf+chunk_len);
		};

		appch_npd(chunks.back());
		for(std::size_t i=chunks.size();i>1;--i){
			appch_pad(chunks[i-2]);
		}
		return out;
	}

	std::vector<std::uint8_t> to_bytes(
		bool most_significant_first=true) const{
		std::vector<std::uint8_t> out;
		write_bytes(out,most_significant_first);
		return out;
	}

	void write_bytes(std::vector<std::uint8_t>&out,
					 bool most_significant_first=true) const{
		out.clear();
		if(is_zero())
			return;
		const std::size_t n=(bit_length()+7u)/8u;
		out.assign(n,0);
		if(std::endian::native==std::endian::little){
			std::memcpy(out.data(),limbs_.data(),n);
		}else{
			for(std::size_t i=0;i<n;++i){
				out[i]=static_cast<std::uint8_t>(
					(limbs_[i/8u]>>static_cast<unsigned>((i&7u)*8u))&
					0xffu);
			}
		}
		if(most_significant_first)
			std::reverse(out.begin(),out.end());
	}

	std::vector<std::uint8_t> export_words(
		std::size_t word_size=1,int word_order=1,int byte_order=1,
		std::size_t nail_bits=0) const{
		if(word_order!=1&&word_order!=-1)
			detail::throw_inv("BigInt::export_words: bad word order");
		if(byte_order==0){
			byte_order=(std::endian::native==std::endian::big)?1:-1;
		}else if(byte_order!=1&&byte_order!=-1){
			detail::throw_inv("BigInt::export_words: bad byte order");
		}
		if(word_size==0||
		   word_size>std::numeric_limits<std::size_t>::max()/8u)
			detail::throw_inv("BigInt::export_words: bad word size");
		const std::size_t word_bits=word_size*8u;
		if(nail_bits>=word_bits)
			detail::throw_inv("BigInt::export_words: bad nail bits");
		if(is_zero())
			return {};
		const std::size_t payload_bits=word_bits-nail_bits;
		const std::size_t bits=bit_length();
		if(bits>std::numeric_limits<std::size_t>::max()-
				(payload_bits-1u))
			detail::throw_ovf("BigInt::export_words: too many bits");
		if(nail_bits==0&&word_size==1u){
			std::vector<std::uint8_t> out;
			write_bytes(out,word_order>0);
			return out;
		}
		const std::size_t count=(bits+payload_bits-1u)/
			payload_bits;
		if(count>std::numeric_limits<std::size_t>::max()/word_size)
			detail::throw_ovf("BigInt::export_words: too many bytes");
		if(nail_bits==0&&word_size==sizeof(limb_t)){
			std::vector<std::uint8_t> out(count*word_size,0);
			if(word_order<0&&
			   ((byte_order<0&&std::endian::native==std::endian::little)||
				(byte_order>0&&std::endian::native==std::endian::big))){
				std::memcpy(out.data(),limbs_.data(),out.size());
				return out;
			}
			for(std::size_t oi=0;oi<count;++oi){
				const std::size_t dst_word=word_order<0?oi:(count-1u-oi);
				auto*wp=out.data()+dst_word*word_size;
				const limb_t w=oi<limbs_.size()?limbs_[oi]:0u;
				if((byte_order<0&&std::endian::native==std::endian::little)||
				   (byte_order>0&&std::endian::native==std::endian::big)){
					std::memcpy(wp,&w,sizeof(w));
				}else if(byte_order<0){
					for(std::size_t bi=0;bi<sizeof(limb_t);++bi)
						wp[bi]=static_cast<std::uint8_t>(
							w>>static_cast<unsigned>(8u*bi));
				}else{
					for(std::size_t bi=0;bi<sizeof(limb_t);++bi)
						wp[sizeof(limb_t)-1u-bi]=static_cast<std::uint8_t>(
							w>>static_cast<unsigned>(8u*bi));
				}
			}
			return out;
		}
		std::vector<std::uint8_t> out(count*word_size,0);
		std::size_t bit_pos=0;
		auto take_bits=[&](unsigned nbits)->std::uint8_t{
			if(nbits==0)
				return 0;
			const std::size_t li=bit_pos/64u;
			const unsigned off=static_cast<unsigned>(bit_pos&63u);
			limb_t v=(li<limbs_.size())?(limbs_[li]>>off):0u;
			if(off+nbits>64u&&li+1u<limbs_.size())
				v|=limbs_[li+1u]<<(64u-off);
			bit_pos+=nbits;
			return static_cast<std::uint8_t>(
				v&((limb_t(1)<<nbits)-1u));
		};
		const std::size_t whole_bytes=payload_bits/8u;
		const unsigned rem_bits=static_cast<unsigned>(payload_bits&7u);
		for(std::size_t wi=0;wi<count;++wi){
			const std::size_t dst_word=word_order<0?wi:(count-1u-wi);
			auto*wp=out.data()+dst_word*word_size;
			for(std::size_t bi=0;bi<whole_bytes;++bi){
				const std::size_t dst_byte=byte_order<0?
					bi:(word_size-1u-bi);
				wp[dst_byte]=take_bits(8u);
			}
			if(rem_bits!=0u){
				const std::size_t dst_byte=byte_order<0?
					whole_bytes:(word_size-1u-whole_bytes);
				wp[dst_byte]=take_bits(rem_bits);
			}
		}
		return out;
	}

	int sign() const noexcept{ return sign_; }
	bool is_zero() const noexcept{ return sign_==0; }
	bool is_one() const noexcept{
		return sign_>0&&limbs_.size()==1&&limbs_[0]==1;
	}
	bool is_neg() const noexcept{ return sign_<0; }
	bool is_even() const noexcept{ return is_zero()||((limbs_[0]&1u)==0); }
	bool is_odd() const noexcept{ return !is_even(); }
	bool fits_u64() const noexcept{ return sign_>=0&&limbs_.size()<=1; }
	std::uint32_t mod_u32(std::uint32_t d) const{
		if(d==0){
			detail::throw_dom("BigInt::mod_u32 division by zero");
		}
		const std::uint32_t r=detail::mod_small(limbs_,d);
		if(sign_<0&&r!=0){
			return static_cast<std::uint32_t>(d-r);
		}
		return r;
	}
	std::uint64_t to_u64() const{
		if(!fits_u64()){
			detail::throw_ovf("BigInt::to_u64 overflow");
		}
		return limbs_.empty()?0:limbs_[0];
	}

	std::size_t bit_length() const noexcept{
		return detail::bit_length(limbs_);
	}
	std::size_t ctz() const noexcept{ return detail::ctz(limbs_); }
	std::size_t popcount() const noexcept{
		if(sign_<0)
			return npos;
		return detail::popcount_abs(limbs_);
	}
	std::size_t scan1(std::size_t start_bit=0) const noexcept{
		if(sign_>=0)
			return detail::scan1_abs(limbs_,start_bit);
		return detail::scan0_sub1_abs(limbs_,start_bit);
	}
	std::size_t scan0(std::size_t start_bit=0) const noexcept{
		if(sign_>=0)
			return detail::scan0_abs(limbs_,start_bit);
		return detail::scan1_sub1_abs(limbs_,start_bit);
	}
	bool test_bit(std::size_t bit_index) const noexcept{
		return detail::test_bit(limbs_,bit_index);
	}

	static constexpr std::size_t npos=
		std::numeric_limits<std::size_t>::max();

	template<class URBG>
	static BigInt rand_bits(std::size_t bits,URBG&rng);
	static BigInt rand_bits(std::size_t bits);
	template<class URBG>
	static BigInt rand_range(const BigInt&limit,URBG&rng);
	static BigInt rand_range(const BigInt&limit);

	BigInt abs() const{
		BigInt out=*this;
		if(out.sign_<0)
			out.sign_=1;
		return out;
	}

	BigInt neg() const{
		BigInt out=*this;
		if(out.sign_!=0)
			out.sign_=-out.sign_;
		return out;
	}

	BigInt&operator+=(const BigInt&rhs){
		if(rhs.sign_==0)
			return *this;
		if(sign_==0){
			*this=rhs;
			return *this;
		}
		if(limbs_.size()==1&&rhs.limbs_.size()==1){
			const std::uint64_t av=limbs_[0];
			const std::uint64_t bv=rhs.limbs_[0];
			if(sign_==rhs.sign_){
				const std::uint64_t sum=av+bv;
				limbs_[0]=sum;
				if(sum<av){
					limbs_.push_back(1);
				}
				return *this;
			}
			if(av==bv){
				sign_=0;
				limbs_.clear();
				return *this;
			}
			if(av>bv){
				limbs_[0]=av-bv;
				return *this;
			}
			limbs_[0]=bv-av;
			sign_=rhs.sign_;
			return *this;
		}

		if(sign_==rhs.sign_){
			const std::size_t need=
				std::max(limbs_.size(),rhs.limbs_.size())+std::size_t(1);
			limbs_.reserve(need);
			detail::add_abs_ip(limbs_,rhs.limbs_);
			return *this;
		}

		const int cmp=detail::cmp_abs(limbs_,rhs.limbs_);
		if(cmp==0){
			sign_=0;
			limbs_.clear();
			return *this;
		}
		if(cmp>0){
			detail::sub_abs_ip(limbs_,rhs.limbs_);
		}else{
			limbs_=detail::sub_abs(rhs.limbs_,limbs_);
			sign_=rhs.sign_;
		}
		normalize();
		return *this;
	}

	BigInt&operator-=(const BigInt&rhs){
		if(rhs.sign_==0)
			return *this;
		if(sign_==0){
			*this=rhs;
			sign_=-sign_;
			return *this;
		}
		if(sign_!=rhs.sign_){
			const std::size_t need=
				std::max(limbs_.size(),rhs.limbs_.size())+std::size_t(1);
			limbs_.reserve(need);
			detail::add_abs_ip(limbs_,rhs.limbs_);
			return *this;
		}

		const int cmp=detail::cmp_abs(limbs_,rhs.limbs_);
		if(cmp==0){
			sign_=0;
			limbs_.clear();
			return *this;
		}
		if(cmp>0){
			detail::sub_abs_ip(limbs_,rhs.limbs_);
		}else{
			limbs_=detail::sub_abs(rhs.limbs_,limbs_);
			sign_=-sign_;
		}
		normalize();
		return *this;
	}

	BigInt&operator*=(const BigInt&rhs){
		*this=mul_disp(*this,rhs);
		return *this;
	}

	BigInt&operator/=(const BigInt&rhs){
		*this=divk_q(*this,rhs);
		return *this;
	}

	BigInt&operator%=(const BigInt&rhs){
		*this=divk_r(*this,rhs);
		return *this;
	}

	BigInt&operator<<=(std::size_t bits){
		if(sign_==0||bits==0)
			return *this;
		detail::shl_ip(limbs_,bits);
		return *this;
	}

	BigInt&operator>>=(std::size_t bits){
		if(sign_==0||bits==0)
			return *this;
		if(sign_>0){
			detail::shr_ip(limbs_,bits);
			sign_=limbs_.empty()?0:1;
			return *this;
		}

		
		
		const std::size_t limb_shift=bits/64;
		const unsigned bit_shift=static_cast<unsigned>(bits%64);
		bool lost=false;

		if(limb_shift>=limbs_.size()){
			lost=!limbs_.empty();
		}else{
			for(std::size_t i=0;i<limb_shift;++i){
				if(limbs_[i]!=0){
					lost=true;
					break;
				}
			}
			if(!lost&&bit_shift!=0){
				const std::uint64_t mask=(std::uint64_t(1)<<bit_shift)-1u;
				if((limbs_[limb_shift]&mask)!=0){
					lost=true;
				}
			}
		}

		detail::shr_ip(limbs_,bits);
		if(lost){
			static const detail::limbs_t kOne{1};
			limbs_=detail::add_abs(limbs_,kOne);
		}
		sign_=limbs_.empty()?0:-1;
		return *this;
	}

	BigInt&operator&=(const BigInt&rhs){
		if(sign_>=0&&rhs.sign_>=0){
			const std::size_t n=std::min(limbs_.size(),rhs.limbs_.size());
			if(n==0){
				sign_=0;
				limbs_.clear();
				return *this;
			}
			detail::vecab::and_n(limbs_.data(),limbs_.data(),
								  rhs.limbs_.data(),n);
			limbs_.resize(n);
			detail::trim_lz(limbs_);
			sign_=limbs_.empty()?0:1;
			return *this;
		}
		if(sign_>=0&&rhs.sign_<0){
			limbs_=detail::bit_and_pn_abs(limbs_,rhs.limbs_);
			sign_=limbs_.empty()?0:1;
			return *this;
		}
		if(sign_<0&&rhs.sign_>=0){
			limbs_=detail::bit_and_pn_abs(rhs.limbs_,limbs_);
			sign_=limbs_.empty()?0:1;
			return *this;
		}
		{
			detail::limbs_t am1=detail::sub_one(limbs_);
			detail::limbs_t bm1=detail::sub_one(rhs.limbs_);
			limbs_=detail::bit_or_abs(am1,bm1);
			detail::add_one_ip(limbs_);
			sign_=-1;
			return *this;
		}
	}

	BigInt&operator|=(const BigInt&rhs){
		if(sign_>=0&&rhs.sign_>=0){
			if(rhs.sign_==0)
				return *this;
			if(sign_==0){
				*this=rhs;
				return *this;
			}
			const std::size_t old_size=limbs_.size();
			const std::size_t n=std::min(old_size,rhs.limbs_.size());
			const std::size_t out_size=std::max(old_size,rhs.limbs_.size());
			limbs_.resize_uninit(out_size);
			detail::vecab::or_n(limbs_.data(),limbs_.data(),
								 rhs.limbs_.data(),n);
			if(rhs.limbs_.size()>old_size){
				std::memcpy(limbs_.data()+static_cast<std::ptrdiff_t>(old_size),
							rhs.limbs_.data()+static_cast<std::ptrdiff_t>(old_size),
							(rhs.limbs_.size()-old_size)*sizeof(limb_t));
			}
			sign_=1;
			return *this;
		}
		if(sign_>=0&&rhs.sign_<0){
			limbs_=detail::bit_or_pn_mag(limbs_,rhs.limbs_);
			sign_=-1;
			return *this;
		}
		if(sign_<0&&rhs.sign_>=0){
			limbs_=detail::bit_or_pn_mag(rhs.limbs_,limbs_);
			sign_=-1;
			return *this;
		}
		{
			detail::limbs_t am1=detail::sub_one(limbs_);
			detail::limbs_t bm1=detail::sub_one(rhs.limbs_);
			limbs_=detail::bit_and_abs(am1,bm1);
			detail::add_one_ip(limbs_);
			sign_=-1;
			return *this;
		}
	}

	BigInt&operator^=(const BigInt&rhs){
		if(sign_>=0&&rhs.sign_>=0){
			if(rhs.sign_==0)
				return *this;
			if(sign_==0){
				*this=rhs;
				return *this;
			}
			const std::size_t old_size=limbs_.size();
			const std::size_t n=std::min(old_size,rhs.limbs_.size());
			const std::size_t out_size=std::max(old_size,rhs.limbs_.size());
			limbs_.resize_uninit(out_size);
			detail::vecab::xor_n(limbs_.data(),limbs_.data(),
								  rhs.limbs_.data(),n);
			if(rhs.limbs_.size()>old_size){
				std::memcpy(limbs_.data()+static_cast<std::ptrdiff_t>(old_size),
							rhs.limbs_.data()+static_cast<std::ptrdiff_t>(old_size),
							(rhs.limbs_.size()-old_size)*sizeof(limb_t));
			}
			detail::trim_lz(limbs_);
			sign_=limbs_.empty()?0:1;
			return *this;
		}
		if(sign_>=0&&rhs.sign_<0){
			limbs_=detail::bit_xor_pn_mag(limbs_,rhs.limbs_);
			sign_=-1;
			return *this;
		}
		if(sign_<0&&rhs.sign_>=0){
			limbs_=detail::bit_xor_pn_mag(rhs.limbs_,limbs_);
			sign_=-1;
			return *this;
		}
		{
			detail::limbs_t am1=detail::sub_one(limbs_);
			detail::limbs_t bm1=detail::sub_one(rhs.limbs_);
			limbs_=detail::bit_xor_abs(am1,bm1);
			sign_=limbs_.empty()?0:1;
			return *this;
		}
	}

	friend int compare(const BigInt&a,const BigInt&b) noexcept{
		if(a.sign_!=b.sign_)
			return (a.sign_<b.sign_)?-1:1;
		if(a.sign_==0)
			return 0;
		const int c=detail::cmp_abs(a.limbs_,b.limbs_);
		return (a.sign_>0)?c:-c;
	}

	friend bool operator==(const BigInt&a,const BigInt&b) noexcept{
		return a.sign_==b.sign_&&a.limbs_==b.limbs_;
	}
	friend bool operator!=(const BigInt&a,const BigInt&b) noexcept{
		return !(a==b);
	}
	friend bool operator<(const BigInt&a,const BigInt&b) noexcept{
		return compare(a,b)<0;
	}
	friend bool operator>(const BigInt&a,const BigInt&b) noexcept{
		return compare(a,b)>0;
	}
	friend bool operator<=(const BigInt&a,const BigInt&b) noexcept{
		return compare(a,b)<=0;
	}
	friend bool operator>=(const BigInt&a,const BigInt&b) noexcept{
		return compare(a,b)>=0;
	}

	friend BigInt mul_sbk(const BigInt&a,const BigInt&b);
	friend BigInt mul_kar(const BigInt&a,const BigInt&b);
	friend BigInt mul_t3(const BigInt&a,const BigInt&b);
	friend BigInt mul_ntt(const BigInt&a,const BigInt&b);
	friend BigInt mul_disp(const BigInt&a,const BigInt&b);
	friend BigInt sqr_disp_noat(const BigInt&a);
	friend BigInt sqr_disp(const BigInt&a);
	friend std::pair<BigInt,BigInt> dvm_simp(const BigInt&a,
												  const BigInt&b);
	friend std::pair<BigInt,BigInt> dvm_p2(const BigInt&a,const BigInt&b);
	friend std::pair<BigInt,BigInt> dvm_knuth(const BigInt&a,const BigInt&b);
	friend std::pair<BigInt,BigInt> divmod(const BigInt&a,const BigInt&b);
	friend BigInt divk_q(const BigInt&a,const BigInt&b);
	friend BigInt divk_r(const BigInt&a,const BigInt&b);
	friend std::pair<BigInt,BigInt> dvm_ablb(const BigInt&abs_a,
													   limb_t d);
	friend BigInt gcd(BigInt a,BigInt b);
	friend BigInt gcd_hlprm(BigInt a,BigInt b,
											  std::size_t hl_min,
											  std::size_t hl_rnd);
	friend BigInt lcm(const BigInt&a,const BigInt&b);
	friend void mkodd_ip(BigInt&x);
	friend BigInt gcd_sm_bin(BigInt a,BigInt b);
	friend BigInt modpow(BigInt base,BigInt exp,const BigInt&mod);
	friend bool invert(BigInt*rop,const BigInt&a,const BigInt&mod);
	friend int kronecker(const BigInt&a,const BigInt&b);
	friend int jacobi(const BigInt&a,const BigInt&b);
	friend bool sqrtmod(BigInt*root,const BigInt&a,const BigInt&mod);
	friend BigInt sqrtmod(const BigInt&a,const BigInt&mod);
	friend BigInt divexact(const BigInt&a,const BigInt&b);
	friend std::pair<BigInt,BigInt> sqrtrem_split(const BigInt&n,
												  std::size_t nbits);
	friend std::pair<BigInt,BigInt> sqrtrem_u128(const BigInt&n);
	friend std::pair<BigInt,BigInt> sqrtrem_u256(const BigInt&n);
	friend std::pair<BigInt,BigInt> sqrtrem(const BigInt&n);
	friend std::pair<BigInt,BigInt> rootrem(const BigInt&n,std::uint32_t k);
	friend bool is_square(const BigInt&n);
	friend BigInt factorial(std::uint64_t n);
	friend BigInt binomial(std::uint64_t n,std::uint64_t k);
	friend BigInt detail::factorial_loop(std::uint64_t n);
	friend BigInt detail::factorial_tree(std::uint64_t n,std::size_t leaf);
	friend BigInt detail::binomial_loop(std::uint64_t n,std::uint64_t k);
	friend BigInt detail::binomial_tree(std::uint64_t n,std::uint64_t k,
										 std::size_t leaf);
	friend std::uint64_t detail::bench_sqr_mul(const BigInt&a,int loops,
											   bool use_ntt);
	friend BigInt operator+(const BigInt&a,const BigInt&b);
	friend BigInt operator-(const BigInt&a,const BigInt&b);
	friend BigInt operator<<(const BigInt&v,std::size_t bits);
	friend BigInt operator>>(const BigInt&v,std::size_t bits);
	friend BigInt operator&(const BigInt&a,const BigInt&b);
	friend BigInt operator|(const BigInt&a,const BigInt&b);
	friend BigInt operator^(const BigInt&a,const BigInt&b);
	friend class BigFloat;
	friend class BigRat;

  private:
	static BigInt from_raw(int sign,detail::limbs_t mag){
		BigInt out;
		out.sign_=sign;
		out.limbs_=std::move(mag);
		out.normalize();
		return out;
	}

	void normalize(){ detail::norm_sign(sign_,limbs_); }

	void setb_abs(std::size_t bit_index){
		detail::set_bit(limbs_,bit_index);
		if(!limbs_.empty()&&sign_==0)
			sign_=1;
	}

  private:
	int sign_=0;			
	detail::limbs_t limbs_; 
};

template<class URBG>
inline BigInt BigInt::rand_bits(std::size_t bits,URBG&rng){
	if(bits==0)
		return BigInt();
	return from_raw(1,detail::rand_limbs_bits(bits,rng));
}

inline BigInt BigInt::rand_bits(std::size_t bits){
	return rand_bits(bits,detail::default_rng());
}

template<class URBG>
inline BigInt BigInt::rand_range(const BigInt&limit,URBG&rng){
	if(limit.sign_<=0){
		detail::throw_dom("BigInt::rand_range: limit must be positive");
	}
	if(limit.is_one())
		return BigInt();
	if(limit.fits_u64())
		return BigInt::from_u64(detail::rand_u64_below(limit.limbs_[0],rng));
	std::size_t bits=limit.bit_length();
	if(detail::pow2_abs(limit.limbs_))
		return BigInt::rand_bits(bits-1u,rng);
	for(;;){
		detail::limbs_t mag=detail::rand_limbs_bits(bits,rng);
		if(detail::cmp_abs(mag,limit.limbs_)<0)
			return from_raw(mag.empty()?0:1,std::move(mag));
	}
}

inline BigInt BigInt::rand_range(const BigInt&limit){
	return rand_range(limit,detail::default_rng());
}

template<class URBG>
inline BigInt rand_bits(std::size_t bits,URBG&rng){
	return BigInt::rand_bits(bits,rng);
}

inline BigInt rand_bits(std::size_t bits){
	return BigInt::rand_bits(bits);
}

template<class URBG>
inline BigInt rand_range(const BigInt&limit,URBG&rng){
	return BigInt::rand_range(limit,rng);
}

inline BigInt rand_range(const BigInt&limit){
	return BigInt::rand_range(limit);
}

inline BigInt operator+(const BigInt&a,const BigInt&b){
	if(a.sign_==0)
		return b;
	if(b.sign_==0)
		return a;
	if(a.limbs_.size()==1&&b.limbs_.size()==1){
		const std::uint64_t av=a.limbs_[0];
		const std::uint64_t bv=b.limbs_[0];
		if(a.sign_==b.sign_){
			const std::uint64_t sum=av+bv;
			BigInt out;
			out.sign_=a.sign_;
			out.limbs_.push_back(sum);
			if(sum<av){
				out.limbs_.push_back(1);
			}
			return out;
		}
		if(av==bv)
			return BigInt();
		BigInt out;
		if(av>bv){
			out.sign_=a.sign_;
			out.limbs_.push_back(av-bv);
		}else{
			out.sign_=b.sign_;
			out.limbs_.push_back(bv-av);
		}
		return out;
	}
	if(a.sign_==b.sign_&&a.limbs_.size()<=2&&b.limbs_.size()<=2){
		const std::uint64_t a0=a.limbs_[0];
		const std::uint64_t b0=b.limbs_[0];
		const std::uint64_t a1=(a.limbs_.size()==2)?a.limbs_[1]:0;
		const std::uint64_t b1=(b.limbs_.size()==2)?b.limbs_[1]:0;
		std::uint64_t s0=0;
		const std::uint64_t c0=detail::addc_u64(0,a0,b0,&s0);
		std::uint64_t s1=0;
		const std::uint64_t c1=detail::addc_u64(c0,a1,b1,&s1);
		BigInt out;
		out.sign_=a.sign_;
		out.limbs_.reserve(3);
		out.limbs_.push_back(s0);
		if(s1!=0||c1!=0){
			out.limbs_.push_back(s1);
		}
		if(c1!=0){
			out.limbs_.push_back(c1);
		}
		return out;
	}

	BigInt out;
	if(a.sign_==b.sign_){
		out.sign_=a.sign_;
		detail::add_abs_to(out.limbs_,a.limbs_,b.limbs_);
		return out;
	}

	const int cmp=detail::cmp_abs(a.limbs_,b.limbs_);
	if(cmp==0)
		return BigInt();
	if(cmp>0){
		out.sign_=a.sign_;
		detail::sub_abs_to(out.limbs_,a.limbs_,b.limbs_);
	}else{
		out.sign_=b.sign_;
		detail::sub_abs_to(out.limbs_,b.limbs_,a.limbs_);
	}
	return out;
}
inline BigInt operator-(const BigInt&a,const BigInt&b){
	if(std::max(a.limbs_.size(),b.limbs_.size())<=MINI_MP_INLINE_LIMBS){
		BigInt out=a;
		out-=b;
		return out;
	}
	if(b.sign_==0)
		return a;
	if(a.sign_==0)
		return b.neg();
	BigInt out;
	if(a.sign_!=b.sign_){
		detail::add_abs_to(out.limbs_,a.limbs_,b.limbs_);
		out.sign_=a.sign_;
		return out;
	}
	const int cmp=detail::cmp_abs(a.limbs_,b.limbs_);
	if(cmp==0)
		return BigInt();
	if(cmp>0){
		detail::sub_abs_to(out.limbs_,a.limbs_,b.limbs_);
		out.sign_=a.sign_;
	}else{
		detail::sub_abs_to(out.limbs_,b.limbs_,a.limbs_);
		out.sign_=-a.sign_;
	}
	return out;
}
inline BigInt operator*(const BigInt&lhs,const BigInt&rhs){
	return mul_disp(lhs,rhs);
}
inline BigInt operator/(BigInt lhs,const BigInt&rhs){
	lhs/=rhs;
	return lhs;
}
inline BigInt operator%(BigInt lhs,const BigInt&rhs){
	lhs%=rhs;
	return lhs;
}
inline BigInt operator-(const BigInt&v){ return v.neg(); }
inline BigInt operator+(const BigInt&v){ return v; }

inline BigInt operator<<(const BigInt&v,std::size_t bits){
	if(v.sign_==0||bits==0)
		return v;
	BigInt out;
	out.sign_=v.sign_;
	detail::shl_into(out.limbs_,v.limbs_,bits);
	return out;
}
inline BigInt operator>>(const BigInt&v,std::size_t bits){
	if(v.sign_==0||bits==0)
		return v;
	if(v.sign_>0){
		BigInt out;
		detail::shr_into(out.limbs_,v.limbs_,bits);
		out.sign_=out.limbs_.empty()?0:1;
		return out;
	}
	if(v.limbs_.size()<=MINI_MP_INLINE_LIMBS){
		BigInt out=v;
		out>>=bits;
		return out;
	}
	BigInt out=v;
	out>>=bits;
	return out;
}
inline BigInt operator&(const BigInt&a,const BigInt&b){
	if(a.sign_>=0&&b.sign_>=0){
		detail::limbs_t mag=detail::bit_and_abs(a.limbs_,b.limbs_);
		return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
	}
	if(a.sign_>=0){
		detail::limbs_t mag=detail::bit_and_pn_abs(a.limbs_,b.limbs_);
		return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
	}
	if(b.sign_>=0){
		detail::limbs_t mag=detail::bit_and_pn_abs(b.limbs_,a.limbs_);
		return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
	}
	detail::limbs_t am1=detail::sub_one(a.limbs_);
	detail::limbs_t bm1=detail::sub_one(b.limbs_);
	detail::limbs_t mag=detail::bit_or_abs(am1,bm1);
	detail::add_one_ip(mag);
	return BigInt::from_raw(-1,std::move(mag));
}
inline BigInt operator|(const BigInt&a,const BigInt&b){
	if(a.sign_>=0&&b.sign_>=0){
		detail::limbs_t mag=detail::bit_or_abs(a.limbs_,b.limbs_);
		return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
	}
	if(a.sign_>=0){
		detail::limbs_t mag=detail::bit_or_pn_mag(a.limbs_,b.limbs_);
		return BigInt::from_raw(-1,std::move(mag));
	}
	if(b.sign_>=0){
		detail::limbs_t mag=detail::bit_or_pn_mag(b.limbs_,a.limbs_);
		return BigInt::from_raw(-1,std::move(mag));
	}
	detail::limbs_t am1=detail::sub_one(a.limbs_);
	detail::limbs_t bm1=detail::sub_one(b.limbs_);
	detail::limbs_t mag=detail::bit_and_abs(am1,bm1);
	detail::add_one_ip(mag);
	return BigInt::from_raw(-1,std::move(mag));
}
inline BigInt operator^(const BigInt&a,const BigInt&b){
	if(a.sign_>=0&&b.sign_>=0){
		detail::limbs_t mag=detail::bit_xor_abs(a.limbs_,b.limbs_);
		return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
	}
	if(a.sign_>=0){
		detail::limbs_t mag=detail::bit_xor_pn_mag(a.limbs_,b.limbs_);
		return BigInt::from_raw(-1,std::move(mag));
	}
	if(b.sign_>=0){
		detail::limbs_t mag=detail::bit_xor_pn_mag(b.limbs_,a.limbs_);
		return BigInt::from_raw(-1,std::move(mag));
	}
	detail::limbs_t am1=detail::sub_one(a.limbs_);
	detail::limbs_t bm1=detail::sub_one(b.limbs_);
	detail::limbs_t mag=detail::bit_xor_abs(am1,bm1);
	return BigInt::from_raw(mag.empty()?0:1,std::move(mag));
}

inline BigInt operator~(const BigInt&v){
	
	return -(v+BigInt(1));
}





inline BigInt mul_sbk(const BigInt&a,const BigInt&b){
	if(a.is_zero()||b.is_zero())
		return BigInt();
	detail::limbs_t mag;
	if(a.limbs_.size()==b.limbs_.size()&&a.limbs_==b.limbs_){
		mag=detail::sqrsbk_ab(a.limbs_);
	}else{
		mag=detail::mulsbk_ab(a.limbs_,b.limbs_);
	}
	const int sign=(a.sign_==b.sign_)?1:-1;
	return BigInt::from_raw(sign,mag);
}

inline BigInt mul_kar(const BigInt&a,const BigInt&b){
	if(a.is_zero()||b.is_zero())
		return BigInt();
	const std::size_t amin=std::min(a.limbs_.size(),b.limbs_.size());
	const std::size_t amax=std::max(a.limbs_.size(),b.limbs_.size());
	if(amin<detail::tun_krec()||
	   (amin>0&&amax/amin>detail::tun_kar_imb())){
		return mul_sbk(a,b);
	}
	const detail::limbs_t mag=detail::mulkar_ab(a.limbs_,b.limbs_);
	const int sign=(a.sign_==b.sign_)?1:-1;
	return BigInt::from_raw(sign,mag);
}

inline BigInt mul_t3(const BigInt&a,const BigInt&b){
	if(a.is_zero()||b.is_zero())
		return BigInt();
	const std::size_t an=a.limbs_.size();
	const std::size_t bn=b.limbs_.size();
	const std::size_t n=std::max(an,bn);
	const std::size_t m=(n+2u)/3u;
	if(m==0||std::min(an,bn)<=2u*m)
		return mul_kar(a,b);

	auto part=[](const detail::limbs_t&x,std::size_t lo,
				 std::size_t hi)->BigInt{
		detail::limbs_t p=detail::slc_limb(x,lo,hi);
		return BigInt::from_raw(p.empty()?0:1,std::move(p));
	};

	const BigInt a0=part(a.limbs_,0,m);
	const BigInt a1=part(a.limbs_,m,2u*m);
	const BigInt a2=part(a.limbs_,2u*m,an);
	const BigInt b0=part(b.limbs_,0,m);
	const BigInt b1=part(b.limbs_,m,2u*m);
	const BigInt b2=part(b.limbs_,2u*m,bn);
	if(a2.is_zero()||b2.is_zero())
		return mul_kar(a,b);

	const BigInt ax1=a0+a1+a2;
	const BigInt bx1=b0+b1+b2;
	const BigInt axm1=a0-a1+a2;
	const BigInt bxm1=b0-b1+b2;
	const BigInt axm2=a0-(a1<<1)+(a2<<2);
	const BigInt bxm2=b0-(b1<<1)+(b2<<2);

	auto tprod=[](const BigInt&x,const BigInt&y)->BigInt{
		const std::size_t xn=x.limbs_.size();
		const std::size_t yn=y.limbs_.size();
		const std::size_t mn=std::min(xn,yn);
		const std::size_t mx=std::max(xn,yn);
		const std::size_t th=detail::tun_t3();
		if(th!=detail::kNttOff&&mn>=th&&mx<=mn+mn/2u)
			return mul_t3(x,y);
		return mul_kar(x,y);
	};

	const BigInt w0=tprod(a0,b0);
	const BigInt w1=tprod(ax1,bx1);
	const BigInt wm1=tprod(axm1,bxm1);
	const BigInt wm2=tprod(axm2,bxm2);
	const BigInt wi=tprod(a2,b2);

	const BigInt two(2);
	const BigInt three(3);
	const BigInt c0=w0;
	const BigInt c4=wi;
	BigInt c1=divexact(w1-wm1,two);
	BigInt c2=wm1-c0;
	BigInt c3=divexact(wm2-w1,three);
	c3=divexact(c2-c3,two)+(c4<<1);
	c2=c2+c1-c4;
	c1-=c3;

	const std::size_t sh=m*64u;
	BigInt out=c0;
	out+=c1<<sh;
	out+=c2<<(2u*sh);
	out+=c3<<(3u*sh);
	out+=c4<<(4u*sh);
	if(a.sign_!=b.sign_)
		out=-out;
	return out;
}

#if MINI_MP_ENABLE_NTT
namespace detail::ntt{

struct NTTPrime{
	std::uint32_t mod;
	std::uint32_t prim_root;
};


inline constexpr std::array<NTTPrime,3> kPrimes={{
	{998244353u,3u},  
	{1004535809u,3u}, 
	{469762049u,3u},  
}};

inline constexpr unsigned kDigitBits=static_cast<unsigned>(kNttBitsDef);

inline std::size_t next_pow2(std::size_t n){
	std::size_t p=1;
	while(p<n){
		if(p>(std::numeric_limits<std::size_t>::max()>>1)){
			throw_ovf("NTT size overflow");
		}
		p<<=1;
	}
	return p;
}

struct Barrett32{
	std::uint32_t mod;
	std::uint64_t im; 
};

inline Barrett32 mk_bar32(std::uint32_t mod){
	MINI_MP_ASSERT(mod>1);
	std::uint64_t rem=0;
	const std::uint64_t im=udiv128(1u,0u,mod,&rem);
	return {mod,im};
}

inline std::uint32_t barred64(const Barrett32&br,
										std::uint64_t x) noexcept{
	const u128 prod=mul_u64(x,br.im);
	const std::uint64_t q=prod.hi;
	std::uint64_t r=x-q*br.mod;
	if(r>=br.mod)
		r-=br.mod;
	if(r>=br.mod)
		r-=br.mod;
	return static_cast<std::uint32_t>(r);
}

inline std::uint32_t barmul32(const Barrett32&br,std::uint32_t a,
									 std::uint32_t b) noexcept{
#if defined(__clang__)&&defined(_MSC_VER)
	
	
	return static_cast<std::uint32_t>((static_cast<std::uint64_t>(a)*b)%br.mod);
#else
	return barred64(br,static_cast<std::uint64_t>(a)*b);
#endif
}

inline std::uint32_t mod_pow(std::uint32_t a,std::uint64_t e,std::uint32_t mod){
	const Barrett32 br=mk_bar32(mod);
	std::uint32_t base=static_cast<std::uint32_t>(a%mod);
	std::uint32_t res=1u%mod;
	while(e!=0){
		if(e&1u)
			res=barmul32(br,res,base);
		base=barmul32(br,base,base);
		e>>=1;
	}
	return res;
}

inline std::uint32_t modinv_p(std::uint32_t a,std::uint32_t mod){
	if(a==0){
		throw_dom("modinv_p: inverse of zero");
	}
	return mod_pow(a,mod-2u,mod);
}

struct RootEnt{
	std::uint32_t mod;
	std::size_t n;
	std::vector<std::uint32_t> roots_fwd;
	std::vector<std::uint32_t> roots_inv;
	std::uint32_t inv_n;
};

inline std::shared_ptr<const std::vector<std::size_t>>
bitrev_tb(std::size_t n){
	static std::mutex mu;
	static std::vector<
		std::pair<std::size_t,std::shared_ptr<const std::vector<std::size_t>>>>
		cache;
	static thread_local std::size_t last_n=0;
	static thread_local std::shared_ptr<const std::vector<std::size_t>>
		last_rev;

	if(last_rev&&last_n==n){
		return last_rev;
	}

	{
		std::lock_guard<std::mutex> lock(mu);
		for(const auto&kv : cache){
			if(kv.first==n){
				last_n=n;
				last_rev=kv.second;
				return kv.second;
			}
		}
	}

	auto rev=std::make_shared<std::vector<std::size_t>>(n,0);
	if(n>1){
		const unsigned lg_n=std::countr_zero(n);
		for(std::size_t i=1;i<n;++i){
			(*rev)[i]=((*rev)[i>>1]>>1)|((i&1u)<<(lg_n-1u));
		}
	}

	{
		std::lock_guard<std::mutex> lock(mu);
		for(const auto&kv : cache){
			if(kv.first==n){
				last_n=n;
				last_rev=kv.second;
				return kv.second;
			}
		}
		cache.emplace_back(n,rev);
	}
	last_n=n;
	last_rev=rev;
	return rev;
}

inline std::shared_ptr<const RootEnt> root_table(const NTTPrime&p,
														std::size_t n){
	static std::mutex mu;
	static std::vector<std::shared_ptr<const RootEnt>> cache;
	static thread_local std::uint32_t last_mod=0;
	static thread_local std::size_t last_n=0;
	static thread_local std::shared_ptr<const RootEnt> last_plan;

	if(last_plan&&last_mod==p.mod&&last_n==n){
		return last_plan;
	}

	{
		std::lock_guard<std::mutex> lock(mu);
		for(const auto&e : cache){
			if(e->mod==p.mod&&e->n==n){
				last_mod=p.mod;
				last_n=n;
				last_plan=e;
				return e;
			}
		}
	}

	auto e=std::make_shared<RootEnt>();
	e->mod=p.mod;
	e->n=n;
	e->roots_fwd.assign(n,1u);
	e->roots_inv.assign(n,1u);
	const Barrett32 br=mk_bar32(p.mod);

	const std::uint32_t w1=
		mod_pow(p.prim_root,static_cast<std::uint64_t>(p.mod-1u)/n,p.mod);
	const std::uint32_t w1_inv=modinv_p(w1,p.mod);
	for(std::size_t i=1;i<n;++i){
		e->roots_fwd[i]=barmul32(br,e->roots_fwd[i-1],w1);
		e->roots_inv[i]=barmul32(br,e->roots_inv[i-1],w1_inv);
	}
	e->inv_n=modinv_p(static_cast<std::uint32_t>(n%p.mod),p.mod);

	{
		std::lock_guard<std::mutex> lock(mu);
		for(const auto&it : cache){
			if(it->mod==p.mod&&it->n==n){
				last_mod=p.mod;
				last_n=n;
				last_plan=it;
				return it;
			}
		}
		cache.push_back(e);
	}
	last_mod=p.mod;
	last_n=n;
	last_plan=e;
	return e;
}

inline void ntt_ip(std::vector<std::uint32_t>&a,bool invert,
						const NTTPrime&p){
	const std::size_t n=a.size();
	if(n==0)
		return;
	if((n&(n-1))!=0){
		throw_inv("ntt_ip: size must be power of two");
	}
	if(((p.mod-1u)%n)!=0u){
		throw_inv("ntt_ip: size not supported by prime");
	}

	const auto rev=bitrev_tb(n);
	for(std::size_t i=0;i<n;++i){
		const std::size_t j=(*rev)[i];
		if(i<j)
			std::swap(a[i],a[j]);
	}

	const auto plan=root_table(p,n);
	const std::vector<std::uint32_t>&roots=
		invert?plan->roots_inv:plan->roots_fwd;
	const Barrett32 br=mk_bar32(p.mod);

	for(std::size_t len=2;len<=n;len<<=1){
		const std::size_t half=len>>1;
		const std::size_t step=n/len;
		for(std::size_t i=0;i<n;i+=len){
			std::uint32_t*const block=a.data()+static_cast<std::ptrdiff_t>(i);

			
			{
				const std::uint32_t u=block[0];
				const std::uint32_t v=block[half];
				std::uint32_t x=u+v;
				if(x>=p.mod)
					x-=p.mod;
				const std::uint32_t y=(u>=v)?(u-v):(u+p.mod-v);
				block[0]=x;
				block[half]=y;
			}

			std::size_t w_idx=step;
			for(std::size_t j=1;j<half;++j,w_idx+=step){
				const std::uint32_t w=roots[w_idx];
				const std::uint32_t u=block[j];
				const std::uint32_t v=barmul32(br,block[j+half],w);

				std::uint32_t x=u+v;
				if(x>=p.mod)
					x-=p.mod;
				const std::uint32_t y=(u>=v)?(u-v):(u+p.mod-v);

				block[j]=x;
				block[j+half]=y;
			}
		}
	}

	if(invert){
		const std::uint32_t inv_n=plan->inv_n;
		for(auto&x : a){
			x=barmul32(br,x,inv_n);
		}
	}
}

inline std::vector<std::uint32_t>
conv_mod(const std::vector<std::uint32_t>&a,
				const std::vector<std::uint32_t>&b,const NTTPrime&p){
	if(a.empty()||b.empty())
		return {};
	const std::size_t need=a.size()+b.size()-1;
	const std::size_t n=next_pow2(need);

	
	
	static thread_local std::vector<std::uint32_t> fa;
	static thread_local std::vector<std::uint32_t> fb;
	const bool same_input=(&a==&b);
	fa.assign(n,0);
	if(!same_input){
		fb.assign(n,0);
	}

	for(std::size_t i=0;i<a.size();++i){
		MINI_MP_ASSERT(a[i]<p.mod);
		fa[i]=a[i];
	}
	if(!same_input){
		for(std::size_t i=0;i<b.size();++i){
			MINI_MP_ASSERT(b[i]<p.mod);
			fb[i]=b[i];
		}
	}

	ntt_ip(fa,false,p);
	if(same_input){
		const Barrett32 br=mk_bar32(p.mod);
		for(std::size_t i=0;i<n;++i){
			fa[i]=barmul32(br,fa[i],fa[i]);
		}
	}else{
		ntt_ip(fb,false,p);
		const Barrett32 br=mk_bar32(p.mod);
		for(std::size_t i=0;i<n;++i){
			fa[i]=barmul32(br,fa[i],fb[i]);
		}
	}
	ntt_ip(fa,true,p);
	return std::vector<std::uint32_t>(
		fa.begin(),fa.begin()+static_cast<std::ptrdiff_t>(need));
}

inline std::uint64_t invm_u64(std::uint64_t a,std::uint64_t mod){
	if(mod==0)
		throw_dom("invm_u64: mod == 0");
	a%=mod;
	if(a==0)
		throw_dom("invm_u64: inverse of zero");

	std::int64_t t=0;
	std::int64_t new_t=1;
	std::int64_t r=static_cast<std::int64_t>(mod);
	std::int64_t new_r=static_cast<std::int64_t>(a);

	while(new_r!=0){
		const std::int64_t q=r/new_r;
		const std::int64_t next_t=t-q*new_t;
		t=new_t;
		new_t=next_t;
		const std::int64_t next_r=r-q*new_r;
		r=new_r;
		new_r=next_r;
	}
	if(r!=1){
		throw_dom("invm_u64: inverse does not exist");
	}
	if(t<0)
		t+=static_cast<std::int64_t>(mod);
	return static_cast<std::uint64_t>(t);
}

struct CRT2Ctx{
	std::uint64_t m1;
	std::uint64_t m2;
	std::uint64_t m1m2;
	std::uint64_t inv_m1m2;
};

inline const CRT2Ctx&crt2_ctx(){
	static const CRT2Ctx ctx=[](){
		CRT2Ctx c{};
		c.m1=kPrimes[0].mod;
		c.m2=kPrimes[1].mod;
		c.m1m2=c.m1*c.m2;
		c.inv_m1m2=invm_u64(c.m1%c.m2,c.m2);
		return c;
	}();
	return ctx;
}

inline u128 crt2_u128(std::uint32_t r1,std::uint32_t r2){
	const CRT2Ctx&c=crt2_ctx();
	const std::uint64_t x1=r1;
	std::uint64_t t=(r2+c.m2-static_cast<std::uint32_t>(x1%c.m2))%c.m2;
	t=(t*c.inv_m1m2)%c.m2;
	const std::uint64_t x=x1+t*c.m1; 
	return {x,0};
}

struct CRT3Ctx{
	std::uint64_t m1;
	std::uint64_t m2;
	std::uint64_t m3;
	std::uint64_t m1m2;
	std::uint64_t inv_m1m2;
	std::uint64_t inv_m12m3;
};

inline const CRT3Ctx&crt3_ctx(){
	static const CRT3Ctx ctx=[](){
		CRT3Ctx c{};
		c.m1=kPrimes[0].mod;
		c.m2=kPrimes[1].mod;
		c.m3=kPrimes[2].mod;
		c.m1m2=c.m1*c.m2;
		c.inv_m1m2=invm_u64(c.m1%c.m2,c.m2);
		c.inv_m12m3=invm_u64(c.m1m2%c.m3,c.m3);
		return c;
	}();
	return ctx;
}

inline u128 crt3_u128(std::uint32_t r1,std::uint32_t r2,std::uint32_t r3){
	const CRT3Ctx&c=crt3_ctx();
	const std::uint64_t x1=r1;

	std::uint64_t t=(r2+c.m2-static_cast<std::uint32_t>(x1%c.m2))%c.m2;
	t=(t*c.inv_m1m2)%c.m2;
	const std::uint64_t x12=x1+t*c.m1; 

	std::uint64_t t2=(r3+c.m3-static_cast<std::uint32_t>(x12%c.m3))%c.m3;
	t2=(t2*c.inv_m12m3)%c.m3;

	const u128 term=mul_u64(t2,c.m1m2);
	return add128_64(term,x12);
}

inline void trim_dz(std::vector<std::uint32_t>&digits){
	while(!digits.empty()&&digits.back()==0)
		digits.pop_back();
}

inline std::vector<std::uint32_t>
limbs2d2k(const detail::limbs_t&limbs,unsigned k_bits){
	if(limbs.empty())
		return {};
	if(k_bits==0||k_bits>=31){
		throw_inv("limbs2d2k: invalid k_bits");
	}
	const std::size_t bits=detail::bit_length(limbs);
	std::vector<std::uint32_t> digits;
	digits.reserve((bits+k_bits-1u)/k_bits);
	for(std::size_t bit_pos=0;bit_pos<bits;bit_pos+=k_bits){
		digits.push_back(detail::extbit32(limbs,bit_pos,k_bits));
	}
	trim_dz(digits);
	return digits;
}

inline detail::limbs_t
digs2l2k(const std::vector<std::uint32_t>&digits,unsigned k_bits){
	if(digits.empty())
		return {};
	if(k_bits==0||k_bits>=31){
		throw_inv("digs2l2k: invalid k_bits");
	}
	const std::uint32_t base=(1u<<k_bits);
	const std::uint32_t mask=base-1u;

	detail::limbs_t out;
	out.reserve((digits.size()*k_bits+63u)/64u+1u);

	std::uint64_t cur=0;
	unsigned cur_bits=0;
	for(std::uint32_t d : digits){
		if(d>mask){
			throw_inv(
				"digs2l2k: digit out of range");
		}
		const std::uint64_t v=d;

		if(cur_bits+k_bits<64u){
			cur|=(v<<cur_bits);
			cur_bits+=k_bits;
		}else if(cur_bits+k_bits==64u){
			cur|=(v<<cur_bits);
			out.push_back(cur);
			cur=0;
			cur_bits=0;
		}else{
			const unsigned take_low=64u-cur_bits;
			const std::uint64_t low_mask=(std::uint64_t(1)<<take_low)-1u;
			cur|=((v&low_mask)<<cur_bits);
			out.push_back(cur);
			cur=(v>>take_low);
			cur_bits=k_bits-take_low;
		}
	}
	if(cur_bits!=0u)
		out.push_back(cur);

	detail::trim_lz(out);
	return out;
}

inline std::vector<std::uint32_t>
carry_p2k(const std::vector<u128>&coeffs,unsigned k_bits){
	if(k_bits==0||k_bits>=31){
		throw_inv("carry_p2k: invalid k_bits");
	}
	const std::uint64_t mask=(std::uint64_t(1)<<k_bits)-1u;

	std::vector<std::uint32_t> digits;
	digits.reserve(coeffs.size()+8);
	u128 carry{0,0};

	for(const u128 c : coeffs){
		const u128 sum=add_u128(c,carry);
		digits.push_back(static_cast<std::uint32_t>(sum.lo&mask));
		carry=shr_u128(sum,k_bits);
	}

	while(!is0_u128(carry)){
		digits.push_back(static_cast<std::uint32_t>(carry.lo&mask));
		carry=shr_u128(carry,k_bits);
	}
	trim_dz(digits);
	return digits;
}

inline std::vector<std::uint32_t>
carry_c2k(const std::vector<std::uint32_t>&c0,
							const std::vector<std::uint32_t>&c1,
							unsigned k_bits){
	MINI_MP_ASSERT(c0.size()==c1.size());
	if(k_bits==0||k_bits>=31){
		throw_inv("carry_c2k: invalid k_bits");
	}
	const std::uint64_t mask=(std::uint64_t(1)<<k_bits)-1u;

	std::vector<std::uint32_t> digits;
	digits.reserve(c0.size()+8);
	u128 carry{0,0};

	for(std::size_t i=0;i<c0.size();++i){
		const u128 coeff=crt2_u128(c0[i],c1[i]);
		const u128 sum=add_u128(coeff,carry);
		digits.push_back(static_cast<std::uint32_t>(sum.lo&mask));
		carry=shr_u128(sum,k_bits);
	}

	while(!is0_u128(carry)){
		digits.push_back(static_cast<std::uint32_t>(carry.lo&mask));
		carry=shr_u128(carry,k_bits);
	}
	trim_dz(digits);
	return digits;
}

inline std::vector<std::uint32_t> carry_c3k(
	const std::vector<std::uint32_t>&c0,const std::vector<std::uint32_t>&c1,
	const std::vector<std::uint32_t>&c2,unsigned k_bits){
	MINI_MP_ASSERT(c0.size()==c1.size()&&c1.size()==c2.size());
	if(k_bits==0||k_bits>=31){
		throw_inv("carry_c3k: invalid k_bits");
	}
	const std::uint64_t mask=(std::uint64_t(1)<<k_bits)-1u;

	std::vector<std::uint32_t> digits;
	digits.reserve(c0.size()+8);
	u128 carry{0,0};

	for(std::size_t i=0;i<c0.size();++i){
		const u128 coeff=crt3_u128(c0[i],c1[i],c2[i]);
		const u128 sum=add_u128(coeff,carry);
		digits.push_back(static_cast<std::uint32_t>(sum.lo&mask));
		carry=shr_u128(sum,k_bits);
	}

	while(!is0_u128(carry)){
		digits.push_back(static_cast<std::uint32_t>(carry.lo&mask));
		carry=shr_u128(carry,k_bits);
	}
	trim_dz(digits);
	return digits;
}

} 
#endif 

inline BigInt mul_ntt(const BigInt&a,const BigInt&b){
#if MINI_MP_ENABLE_NTT
	if(a.is_zero()||b.is_zero())
		return BigInt();

	const unsigned kBits=static_cast<unsigned>(detail::tun_ntt_bits());
	const std::uint64_t kDigitMax=(std::uint64_t(1)<<kBits)-1u;
	const std::uint64_t kDigMaxS=kDigitMax*kDigitMax;
	
	
	
	const std::vector<std::uint32_t> da=
		detail::ntt::limbs2d2k(a.limbs_,kBits);
	std::vector<std::uint32_t> db_local;
	const std::vector<std::uint32_t>*db_ptr=&da;
	if(&a!=&b){
		db_local=detail::ntt::limbs2d2k(b.limbs_,kBits);
		db_ptr=&db_local;
	}
	const std::vector<std::uint32_t>&db=*db_ptr;
	if(da.empty()||db.empty())
		return BigInt();

	const auto&P=detail::ntt::kPrimes;
	const std::vector<std::uint32_t> c0=
		detail::ntt::conv_mod(da,db,P[0]);
	const std::vector<std::uint32_t> c1=
		detail::ntt::conv_mod(da,db,P[1]);

	bool use2_mod=false;
	{
		const std::size_t dmin=std::min(da.size(),db.size());
		const detail::u128 coeff_bd=
			detail::mul_u64(static_cast<std::uint64_t>(dmin),kDigMaxS);
		const auto&c2ctx=detail::ntt::crt2_ctx();
		use2_mod=(coeff_bd.hi==0&&coeff_bd.lo<c2ctx.m1m2);
	}

	std::vector<std::uint32_t> digits;
	if(use2_mod){
		digits=detail::ntt::carry_c2k(c0,c1,kBits);
	}else{
		const std::vector<std::uint32_t> c2=
			detail::ntt::conv_mod(da,db,P[2]);
		digits=detail::ntt::carry_c3k(c0,c1,c2,kBits);
	}
	const detail::limbs_t mag=detail::ntt::digs2l2k(digits,kBits);
	const int sign=(a.sign_==b.sign_)?1:-1;
	return BigInt::from_raw(sign,mag);
#else
	
	return mul_sbk(a,b);
#endif
}

inline BigInt mul_disp(const BigInt&a,const BigInt&b){
	if(a.is_zero()||b.is_zero())
		return BigInt();
	const std::size_t an=a.limbs_.size();
	const std::size_t bn=b.limbs_.size();
	const int sign=(a.sign_==b.sign_)?1:-1;
	if(an==1&&bn==1){
		const detail::u128 p=detail::mul_u64(a.limbs_[0],b.limbs_[0]);
		detail::limbs_t mag;
		mag.reserve((p.hi!=0)?2u:1u);
		mag.push_back(p.lo);
		if(p.hi!=0)
			mag.push_back(p.hi);
		return BigInt::from_raw(sign,std::move(mag));
	}
	if(an==1){
		detail::limbs_t mag=detail::mul_bylb(b.limbs_,a.limbs_[0]);
		return BigInt::from_raw(sign,std::move(mag));
	}
	if(bn==1){
		detail::limbs_t mag=detail::mul_bylb(a.limbs_,b.limbs_[0]);
		return BigInt::from_raw(sign,std::move(mag));
	}
	const std::size_t nmax=std::max(an,bn);
	if(nmax<=detail::kCbaMax){
		detail::limbs_t mag=detail::mulcba_ab(a.limbs_,b.limbs_);
		return BigInt::from_raw(sign,std::move(mag));
	}
	if(&a==&b)
		return sqr_disp(a);
	detail::ensure_at();
	const std::size_t nmin=std::min(an,bn);
	const std::size_t kar_thr=detail::tun_kar();

#if MINI_MP_ENABLE_NTT
	const std::size_t ntt_thr=detail::tun_ntt();
	if(ntt_thr!=detail::kNttOff&&nmin>=ntt_thr&&
	   nmax<=nmin*detail::tun_ntt_imb()){
		return mul_ntt(a,b);
	}
#endif
	const std::size_t t3_thr=detail::tun_t3();
	if(t3_thr!=detail::kNttOff&&nmin>=t3_thr&&
	   nmax<=nmin+nmin/2u){
		return mul_t3(a,b);
	}
	if(nmin>=kar_thr&&nmax<=nmin*detail::tun_kar_imb()){
		return mul_kar(a,b);
	}
	return mul_sbk(a,b);
}

inline BigInt sqr_disp(const BigInt&a){
	if(a.is_zero())
		return BigInt();
	detail::ensure_at();
	const std::size_t n=a.limbs_.size();
	const std::size_t kar_thr=detail::tun_kar();

#if MINI_MP_ENABLE_NTT
	const std::size_t ntt_thr=detail::tun_ntt_sq();
	if(ntt_thr!=detail::kNttOff&&n>=ntt_thr){
		return mul_ntt(a,a);
	}
#endif
	if(n>=kar_thr){
		detail::limbs_t mag=detail::sqrkar_ab(a.limbs_);
		return BigInt::from_raw(1,std::move(mag));
	}
	const detail::limbs_t mag=detail::sqrsbk_ab(a.limbs_);
	return BigInt::from_raw(1,std::move(mag));
}

inline BigInt sqr_disp_noat(const BigInt&a){
	if(a.is_zero())
		return BigInt();
	const std::size_t n=a.limbs_.size();
	if(n==1u){
		const detail::u128 p=detail::mul_u64(a.limbs_[0],a.limbs_[0]);
		detail::limbs_t mag;
		mag.reserve((p.hi!=0)?2u:1u);
		mag.push_back(p.lo);
		if(p.hi!=0)
			mag.push_back(p.hi);
		return BigInt::from_raw(1,std::move(mag));
	}
	detail::limbs_t mag=(n>=detail::kKarTh)
		?detail::sqrkar_ab(a.limbs_)
		:detail::sqrsbk_ab(a.limbs_);
	return BigInt::from_raw(1,std::move(mag));
}





inline std::pair<BigInt,BigInt> dvm_ablb(const BigInt&abs_a,limb_t d){
	MINI_MP_ASSERT(abs_a.sign_>=0);
	MINI_MP_ASSERT(d!=0);

	std::uint64_t rem=0;
	detail::limbs_t q=detail::div_limb(abs_a.limbs_,d,&rem);
	BigInt quot=BigInt::from_raw(q.empty()?0:1,std::move(q));
	BigInt r=(rem==0)?BigInt():BigInt::from_u64(rem);
	return {std::move(quot),std::move(r)};
}

inline std::pair<BigInt,BigInt> dvm_p2(const BigInt&a,const BigInt&b){
	MINI_MP_ASSERT(!b.is_zero());
	MINI_MP_ASSERT(detail::pow2_abs(b.limbs_));
	if(a.is_zero())
		return {BigInt(),BigInt()};

	const std::size_t sh=detail::pow2_exp_abs(b.limbs_);
	detail::limbs_t qmag=detail::shr_bits(a.limbs_,sh);
	detail::limbs_t rmag=detail::low_bits_abs(a.limbs_,sh);

	const int qsign=qmag.empty()?0:((a.sign_==b.sign_)?1:-1);
	BigInt q=BigInt::from_raw(qsign,std::move(qmag));
	BigInt r=BigInt::from_raw(rmag.empty()?0:a.sign_,std::move(rmag));
	r.sign_=r.is_zero()?0:a.sign_;
	return {std::move(q),std::move(r)};
}

inline std::pair<BigInt,BigInt> dvm_simp(const BigInt&a,const BigInt&b){
	if(b.is_zero()){
		detail::throw_dom("dvm_simp: division by zero");
	}
	if(a.is_zero()){
		return {BigInt(),BigInt()};
	}
	if(detail::pow2_abs(b.limbs_))
		return dvm_p2(a,b);

	const BigInt abs_a=a.abs();
	const BigInt abs_b=b.abs();

	if(detail::cmp_abs(abs_a.limbs_,abs_b.limbs_)<0){
		return {BigInt(),a};
	}

	BigInt q_abs;
	BigInt r_abs;

	if(abs_b.limbs_.size()==1){
		auto qr=dvm_ablb(abs_a,abs_b.limbs_[0]);
		q_abs=std::move(qr.first);
		r_abs=std::move(qr.second);
	}else{
		
		
		std::size_t shift=abs_a.bit_length()-abs_b.bit_length();
		BigInt d=abs_b<<shift;
		q_abs=BigInt();
		r_abs=abs_a;

		for(std::size_t i=shift+1;i>0;--i){
			const std::size_t bit=i-1;
			if(detail::cmp_abs(r_abs.limbs_,d.limbs_)>=0){
				r_abs.limbs_=detail::sub_abs(r_abs.limbs_,d.limbs_);
				r_abs.normalize();
				q_abs.setb_abs(bit);
			}
			if(bit!=0){
				d>>=1;
			}
		}
	}

	q_abs.sign_=q_abs.is_zero()?0:((a.sign_==b.sign_)?1:-1);
	r_abs.sign_=r_abs.is_zero()?0:a.sign_;
	return {std::move(q_abs),std::move(r_abs)};
}

inline std::pair<BigInt,BigInt> dvm_knuth(const BigInt&a,const BigInt&b){
	if(b.is_zero()){
		detail::throw_dom("dvm_knuth: division by zero");
	}
	if(a.is_zero()){
		return {BigInt(),BigInt()};
	}

	const detail::limbs_t&ua=a.limbs_;
	const detail::limbs_t&vb=b.limbs_;
	if(vb.size()==1&&vb[0]==1){
		BigInt q=a;
		if(b.sign_<0&&q.sign_!=0)
			q.sign_=-q.sign_;
		return {std::move(q),BigInt()};
	}
	if(detail::pow2_abs(vb))
		return dvm_p2(a,b);
	if(ua.size()==1&&vb.size()==1){
		const std::uint64_t qv=ua[0]/vb[0];
		const std::uint64_t rv=ua[0]%vb[0];
		BigInt q=(qv==0)?BigInt():BigInt::from_u64(qv);
		BigInt r=(rv==0)?BigInt():BigInt::from_u64(rv);
		q.sign_=q.is_zero()?0:((a.sign_==b.sign_)?1:-1);
		r.sign_=r.is_zero()?0:a.sign_;
		return {std::move(q),std::move(r)};
	}
	if(ua.size()<=2&&vb.size()<=2){
		const int cmp=detail::cmp_abs(ua,vb);
		if(cmp<0){
			return {BigInt(),a};
		}
		if(cmp==0){
			BigInt q=BigInt::from_u64(1);
			q.sign_=(a.sign_==b.sign_)?1:-1;
			return {std::move(q),BigInt()};
		}

		BigInt q_abs;
		BigInt r_abs;
		if(vb.size()==1){
			std::uint64_t rem=0;
			detail::limbs_t q=detail::div_limb(ua,vb[0],&rem);
			q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
			r_abs=(rem==0)?BigInt():BigInt::from_u64(rem);
		}else{
			auto qr=detail::dvmk_absl(ua,vb);
			q_abs=BigInt::from_raw(qr.first.empty()?0:1,std::move(qr.first));
			r_abs=BigInt::from_raw(qr.second.empty()?0:1,std::move(qr.second));
		}
		q_abs.sign_=q_abs.is_zero()?0:((a.sign_==b.sign_)?1:-1);
		r_abs.sign_=r_abs.is_zero()?0:a.sign_;
		return {std::move(q_abs),std::move(r_abs)};
	}

	const int cmp_big=detail::cmp_abs(ua,vb);
	if(cmp_big<0){
		return {BigInt(),a};
	}
	if(cmp_big==0){
		BigInt q=BigInt::from_u64(1);
		q.sign_=(a.sign_==b.sign_)?1:-1;
		return {std::move(q),BigInt()};
	}
	if(detail::cmp_abs_dbl(ua,vb)<0){
		detail::limbs_t r;
		detail::sub_abs_to(r,ua,vb);
		BigInt q=BigInt::from_u64(1);
		q.sign_=(a.sign_==b.sign_)?1:-1;
		BigInt rr=BigInt::from_raw(r.empty()?0:1,std::move(r));
		rr.sign_=rr.is_zero()?0:a.sign_;
		return {std::move(q),std::move(rr)};
	}

	BigInt q_abs;
	BigInt r_abs;
	if(vb.size()==1){
		std::uint64_t rem=0;
		detail::limbs_t q=detail::div_limb(ua,vb[0],&rem);
		q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
		r_abs=(rem==0)?BigInt():BigInt::from_u64(rem);
	}else{
		auto qr=detail::use_bzdiv(ua.size(),vb.size())
					?detail::dvmbz_abs(ua,vb)
					:detail::dvmk_absl(ua,vb);
		q_abs=BigInt::from_raw(qr.first.empty()?0:1,std::move(qr.first));
		r_abs=BigInt::from_raw(qr.second.empty()?0:1,std::move(qr.second));
	}

	q_abs.sign_=q_abs.is_zero()?0:((a.sign_==b.sign_)?1:-1);
	r_abs.sign_=r_abs.is_zero()?0:a.sign_;
	return {std::move(q_abs),std::move(r_abs)};
}

inline std::pair<BigInt,BigInt> divmod(const BigInt&a,const BigInt&b){
	detail::ensure_at();
	return dvm_knuth(a,b);
}

inline BigInt divk_q(const BigInt&a,const BigInt&b){
	detail::ensure_at();
	if(b.is_zero()){
		detail::throw_dom("divk_q: division by zero");
	}
	if(a.is_zero())
		return BigInt();

	const detail::limbs_t&ua=a.limbs_;
	const detail::limbs_t&vb=b.limbs_;
	if(vb.size()==1&&vb[0]==1){
		BigInt q=a;
		if(b.sign_<0&&q.sign_!=0)
			q.sign_=-q.sign_;
		return q;
	}
	if(detail::pow2_abs(vb))
		return dvm_p2(a,b).first;
	if(ua.size()==1&&vb.size()==1){
		const std::uint64_t qv=ua[0]/vb[0];
		if(qv==0)
			return BigInt();
		BigInt q=BigInt::from_u64(qv);
		q.sign_=(a.sign_==b.sign_)?1:-1;
		return q;
	}
	if(ua.size()<=2&&vb.size()<=2){
		const int cmp=detail::cmp_abs(ua,vb);
		if(cmp<0){
			return BigInt();
		}
		if(cmp==0){
			BigInt q=BigInt::from_u64(1);
			q.sign_=(a.sign_==b.sign_)?1:-1;
			return q;
		}

		BigInt q_abs;
		if(vb.size()==1){
			std::uint64_t rem=0;
			detail::limbs_t q=detail::div_limb(ua,vb[0],&rem);
			q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
		}else{
			detail::limbs_t q=detail::divk_absl(ua,vb);
			q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
		}
		q_abs.sign_=q_abs.is_zero()?0:((a.sign_==b.sign_)?1:-1);
		return q_abs;
	}

	const int cmp_big=detail::cmp_abs(ua,vb);
	if(cmp_big<0){
		return BigInt();
	}
	if(cmp_big==0||detail::cmp_abs_dbl(ua,vb)<0){
		BigInt q=BigInt::from_u64(1);
		q.sign_=(a.sign_==b.sign_)?1:-1;
		return q;
	}

	BigInt q_abs;
	if(vb.size()==1){
		std::uint64_t rem=0;
		detail::limbs_t q=detail::div_limb(ua,vb[0],&rem);
		q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
	}else{
		detail::limbs_t q;
		if(detail::use_bzdiv(ua.size(),vb.size())){
			auto qr=detail::dvmbz_abs(ua,vb);
			q=std::move(qr.first);
		}else{
			q=detail::divk_absl(ua,vb);
		}
		q_abs=BigInt::from_raw(q.empty()?0:1,std::move(q));
	}
	q_abs.sign_=q_abs.is_zero()?0:((a.sign_==b.sign_)?1:-1);
	return q_abs;
}

inline BigInt divk_r(const BigInt&a,const BigInt&b){
	detail::ensure_at();
	if(b.is_zero()){
		detail::throw_dom("divk_r: division by zero");
	}
	if(a.is_zero())
		return BigInt();

	const detail::limbs_t&ua=a.limbs_;
	const detail::limbs_t&vb=b.limbs_;
	if(vb.size()==1&&vb[0]==1)
		return BigInt();
	if(detail::pow2_abs(vb))
		return dvm_p2(a,b).second;
	if(ua.size()==1&&vb.size()==1){
		const std::uint64_t rem=ua[0]%vb[0];
		if(rem==0)
			return BigInt();
		BigInt r=BigInt::from_u64(rem);
		r.sign_=a.sign_;
		return r;
	}
	if(ua.size()<=2&&vb.size()<=2){
		const int cmp=detail::cmp_abs(ua,vb);
		if(cmp<0){
			return a;
		}
		if(cmp==0)
			return BigInt();

		BigInt r_abs;
		if(vb.size()==1){
			const std::uint64_t rem=detail::mod_limb(ua,vb[0]);
			r_abs=(rem==0)?BigInt():BigInt::from_u64(rem);
		}else{
			detail::limbs_t r=detail::modk_absl(ua,vb);
			r_abs=BigInt::from_raw(r.empty()?0:1,std::move(r));
		}
		r_abs.sign_=r_abs.is_zero()?0:a.sign_;
		return r_abs;
	}

	const int cmp_big=detail::cmp_abs(ua,vb);
	if(cmp_big<0){
		return a;
	}
	if(cmp_big==0)
		return BigInt();
	if(detail::cmp_abs_dbl(ua,vb)<0){
		detail::limbs_t r;
		detail::sub_abs_to(r,ua,vb);
		BigInt rr=BigInt::from_raw(r.empty()?0:1,std::move(r));
		rr.sign_=rr.is_zero()?0:a.sign_;
		return rr;
	}

	BigInt r_abs;
	if(vb.size()==1){
		const std::uint64_t rem=detail::mod_limb(ua,vb[0]);
		r_abs=(rem==0)?BigInt():BigInt::from_u64(rem);
	}else{
		detail::limbs_t r;
		if(detail::use_bzdiv(ua.size(),vb.size())){
			auto qr=detail::dvmbz_abs(ua,vb);
			r=std::move(qr.second);
		}else{
			r=detail::modk_absl(ua,vb);
		}
		r_abs=BigInt::from_raw(r.empty()?0:1,std::move(r));
	}
	r_abs.sign_=r_abs.is_zero()?0:a.sign_;
	return r_abs;
}

namespace detail{

struct ATState{
	std::size_t ntt_th=kNttDef;
	std::size_t ntt_sq_th=kNttDef;
	std::size_t ntt_bits=kNttBitsDef;
	std::size_t kar_rec=kKarRecB;
	std::size_t sqr_rec=kKarRecB;
	std::size_t kar_th=kKarTh;
	std::size_t kar_imb=kKarImb;
	std::size_t kar_dif=kKarDifMin;
	std::size_t ntt_imb=kNttImb;
	std::size_t hl_min=kHlmDef;
	std::size_t hl_rnd=kHlrDef;
	std::size_t gcd_sm=kGcdSmMxDef;
	std::size_t gcd_lg=kGcdLgMxDef;
	std::size_t gcd_qs=kGcdQsDef;
	std::size_t bz_min=kBzDivMnDef;
	std::size_t bz_chunk=kBzChunkDef;
	std::size_t prod_leaf=kProdLeafDef;
	std::size_t fac_tree=kFacTreeDef;
	std::size_t binom_tree=kBinomTreeDef;
	std::size_t pow_w5=kPowW5Def;
	std::size_t pow_w6=kPowW6Def;
	std::size_t d10_dc=kD10DcDef;
	std::size_t d10_prs=kD10PrsDef;
	std::size_t t3_th=kT3Def;
};

inline ATState&at_state() noexcept{
	static ATState state{};
	return state;
}

inline std::atomic<unsigned>&at_done() noexcept{
	static std::atomic<unsigned> done{
#if MINI_MP_ENABLE_AUTOTUNE
		0u
#else
		2u
#endif
	};
	return done;
}

inline bool&at_busy() noexcept{
	static thread_local bool busy=false;
	return busy;
}

inline ATState at_base(){
	ATState tuned{};
	tuned.ntt_th=kNttOff;
	tuned.ntt_sq_th=kNttOff;
	tuned.ntt_bits=kNttBitsDef;
	tuned.kar_rec=kKarRecB;
	tuned.sqr_rec=kKarRecB;
	tuned.kar_th=kKarTh;
	tuned.kar_imb=kKarImb;
	tuned.kar_dif=kKarDifMin;
	tuned.ntt_imb=kNttImb;
	tuned.hl_min=kHlmDef;
	tuned.hl_rnd=kHlrDef;
	tuned.gcd_sm=kGcdSmMxDef;
	tuned.gcd_lg=kGcdLgMxDef;
	tuned.gcd_qs=kGcdQsDef;
	tuned.bz_min=kBzDivMnDef;
	tuned.bz_chunk=kBzChunkDef;
	tuned.prod_leaf=kProdLeafDef;
	tuned.fac_tree=kFacTreeDef;
	tuned.binom_tree=kBinomTreeDef;
	tuned.pow_w5=kPowW5Def;
	tuned.pow_w6=kPowW6Def;
	tuned.d10_dc=kD10DcDef;
	tuned.d10_prs=kD10PrsDef;
	tuned.t3_th=kT3Def;
	return tuned;
}

inline char at_lc(char c) noexcept{
	return (c>='A'&&c<='Z')?static_cast<char>(c-'A'+'a'):c;
}

inline bool at_eq(const char*s,const char*lit) noexcept{
	while(*s!=0&&*lit!=0){
		if(at_lc(*s)!=*lit)
			return false;
		++s;
		++lit;
	}
	return *s==0&&*lit==0;
}

inline unsigned at_env_level() noexcept{
	const char*const v=std::getenv("MINI_MP_AUTOTUNE");
	if(v==nullptr||*v==0)
		return 1u;
	if(at_eq(v,"0")||at_eq(v,"off")||at_eq(v,"none"))
		return 0u;
	if(at_eq(v,"2")||at_eq(v,"full")||at_eq(v,"all"))
		return 2u;
	return 1u;
}

inline unsigned at_default_level() noexcept{
	static const unsigned level=at_env_level();
	return level;
}

inline std::uint64_t at_rng(std::uint64_t&s) noexcept{
	s^=(s<<7);
	s^=(s>>9);
	s^=(s<<8);
	return s;
}

inline BigInt mk_bench(std::uint64_t&seed,std::size_t limbs){
	if(limbs==0)
		return BigInt(1);
	BigInt out(0);
	for(std::size_t i=0;i<limbs;++i){
		out<<=64;
		std::uint64_t v=at_rng(seed);
		if(i==0&&v==0)
			v=1;
		out+=BigInt::from_u64(v);
	}
	if(out.is_zero())
		out=BigInt(1);
	return out;
}

inline BigInt gcd_hl_i(BigInt a,BigInt b,
								   std::size_t hl_min,
								   std::size_t hl_rnd){
	return gcd_hlprm(
		std::move(a),std::move(b),hl_min,hl_rnd);
}

inline std::uint64_t
bench_gcd(const std::vector<std::pair<BigInt,BigInt>>&samples,
					 std::size_t min_limbs,std::size_t rounds){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(const auto&ab : samples){
		BigInt g=gcd_hl_i(ab.first,ab.second,min_limbs,rounds);
		sink^=g.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_sbk(const BigInt&a,
													  const BigInt&b,int loops){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt c=mul_sbk(a,b);
		sink^=c.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_mul(const BigInt&a,const BigInt&b,
										   int loops,bool use_ntt){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt c=use_ntt?mul_ntt(a,b):mul_kar(a,b);
		sink^=c.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_t3(const BigInt&a,const BigInt&b,int loops){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt c=mul_t3(a,b);
		sink^=c.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_sqr_mul(const BigInt&a,int loops,bool use_ntt){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		if(use_ntt){
			BigInt c=mul_ntt(a,a);
			sink^=c.bit_length();
		}else{
			limbs_t mag=sqrkar_ab(a.limbs_);
			sink^=bit_length(mag);
		}
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_fmt_d10(const limbs_t&x,int loops,
								   std::size_t leaf){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		std::string s=to_str_d10(x,1,leaf);
		sink^=s.size();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_prs_d10(std::string_view s,int loops,
								   std::size_t leaf){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		limbs_t x;
		const bool ok=prs_d10(s,&x,leaf);
		MINI_MP_ASSERT(ok);
		sink^=x.size();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline limbs_t mk_limb_bench(std::uint64_t&seed,std::size_t limbs){
	limbs_t out(limbs);
	for(std::size_t i=0;i<limbs;++i)
		out[i]=at_rng(seed);
	if(!out.empty()&&out.back()==0)
		out.back()=1;
	return out;
}

inline std::uint64_t bench_div_abs(const limbs_t&u,const limbs_t&v,
											  int loops,bool use_bz){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		auto qr=use_bz?dvmbz_abs(u,v):dvmk_absl(u,v);
		sink^=qr.first.size();
		sink^=qr.second.size();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline BigInt prod_u64_range_leaf(std::uint64_t lo,std::uint64_t hi,
								  std::size_t leaf){
	if(lo>hi)
		return BigInt(1);
	if(hi-lo<=leaf){
		BigInt out(1);
		for(std::uint64_t i=lo;i<=hi;++i)
			out*=BigInt::from_u64(i);
		return out;
	}
	const std::uint64_t mid=lo+(hi-lo)/2u;
	return prod_u64_range_leaf(lo,mid,leaf)*
		   prod_u64_range_leaf(mid+1u,hi,leaf);
}

inline BigInt prod_u64_range(std::uint64_t lo,std::uint64_t hi){
	return prod_u64_range_leaf(lo,hi,tun_prod_leaf());
}

inline BigInt prod_odd_leaf(std::uint64_t lo,std::uint64_t hi,
							std::size_t leaf){
	if(lo>hi)
		return BigInt(1);
	if((lo&1u)==0u)
		++lo;
	if(lo>hi)
		return BigInt(1);
	if(hi-lo<=leaf*2u){
		BigInt out(1);
		for(std::uint64_t i=lo;i<=hi;i+=2u)
			out*=BigInt::from_u64(i);
		return out;
	}
	const std::uint64_t mid=lo+(hi-lo)/2u;
	return prod_odd_leaf(lo,mid,leaf)*
		   prod_odd_leaf(mid+1u,hi,leaf);
}

inline BigInt factorial_loop(std::uint64_t n){
	BigInt out(1);
	limbs_t tmp;
	for(std::uint64_t i=2;i<=n;++i){
		mulbl_in(out.limbs_,i,tmp);
		out.limbs_.swap(tmp);
		out.sign_=1;
	}
	return out;
}

inline BigInt factorial_tree(std::uint64_t n,std::size_t leaf){
	BigInt odd(1);
	for(std::uint64_t m=n;m>1u;m>>=1u)
		odd*=prod_odd_leaf(3u,m,leaf);
	const std::uint64_t twos=n-std::popcount(n);
	if(twos!=0u)
		odd<<=static_cast<std::size_t>(twos);
	return odd;
}

inline BigInt binomial_loop(std::uint64_t n,std::uint64_t k){
	k=std::min(k,n-k);
	BigInt out(1);
	limbs_t tmp;
	for(std::uint64_t i=1;i<=k;++i){
		mulbl_in(out.limbs_,n-k+i,tmp);
		out.limbs_.swap(tmp);
		const std::uint64_t rem=div_limb_ip(out.limbs_,i);
		MINI_MP_ASSERT(rem==0);
		out.sign_=out.limbs_.empty()?0:1;
	}
	return out;
}

inline BigInt binomial_tree(std::uint64_t n,std::uint64_t k,
							std::size_t leaf){
	k=std::min(k,n-k);
	BigInt num=prod_u64_range_leaf(n-k+1u,n,leaf);
	BigInt den=prod_u64_range_leaf(2,k,leaf);
	return divexact(num,den);
}

inline std::uint64_t bench_fact(std::uint64_t n,int loops,
									   bool tree,std::size_t leaf){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt x=tree?factorial_tree(n,leaf):factorial_loop(n);
		sink^=x.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline std::uint64_t bench_binom(std::uint64_t n,std::uint64_t k,int loops,
										bool tree,std::size_t leaf){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt x=tree?binomial_tree(n,k,leaf):binomial_loop(n,k);
		sink^=x.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

inline void at_impl_fast(){
	ATState tuned=at_base();
	tuned.kar_th=48u;
	tuned.kar_rec=48u;
	tuned.sqr_rec=64u;
	tuned.kar_dif=96u;
	tuned.hl_min=kHlmDef;
	tuned.hl_rnd=kHlrDef;
	tuned.gcd_sm=kGcdSmMxDef;
	tuned.gcd_lg=kGcdLgMxDef;
	tuned.gcd_qs=kGcdQsDef;
	tuned.d10_dc=256u;
	tuned.d10_prs=256u;
	at_state()=tuned;
	auto per_op=[](std::uint64_t ns,std::uint64_t ops)->std::uint64_t{
		return (ns+ops/2u)/ops;
	};

	{
		std::uint64_t seed=0x4f1bbcdc8a5a3f21ULL;
		constexpr std::array<std::size_t,2> kRecCand={40u,48u};
		constexpr std::array<std::size_t,2> kDifCand={64u,96u};
		struct Smp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<Smp,3> kSmp={{
			{64u,4,1u},{128u,2,2u},{256u,1,3u}}};
		auto cost_of=[&](std::size_t rec,std::size_t dif)->std::uint64_t{
			at_state().kar_rec=rec;
			at_state().kar_dif=dif;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(const auto&sm : kSmp){
				const BigInt a=mk_bench(s,sm.n);
				const BigInt b=mk_bench(s,sm.n);
				cost+=per_op(bench_mul(a,b,sm.loops,false),
							 static_cast<std::uint64_t>(sm.loops))*sm.weight;
			}
			return cost;
		};
		std::size_t best_rec=tuned.kar_rec;
		std::size_t best_dif=tuned.kar_dif;
		std::uint64_t best=cost_of(best_rec,best_dif);
		for(std::size_t rec : kRecCand){
			for(std::size_t dif : kDifCand){
				const std::uint64_t cost=cost_of(rec,dif);
				if(cost*std::uint64_t(100)<best*std::uint64_t(90)){
					best=cost;
					best_rec=rec;
					best_dif=dif;
				}
			}
		}
		tuned.kar_rec=best_rec;
		tuned.kar_dif=best_dif;
		at_state().kar_rec=best_rec;
		at_state().kar_dif=best_dif;
	}

	{
		std::uint64_t seed=0x736f6d6570736575ULL;
		constexpr std::array<std::size_t,3> kThCand={40u,48u,64u};
		std::array<std::uint64_t,kThCand.size()> school_ns{};
		std::array<std::uint64_t,kThCand.size()> kar_ns{};
		for(std::size_t i=0;i<kThCand.size();++i){
			const std::size_t n=kThCand[i];
			const int loops=(n<=64u)?4:(n<=128u)?2:1;
			const BigInt a=mk_bench(seed,n);
			const BigInt b=mk_bench(seed,n);
			school_ns[i]=per_op(bench_sbk(a,b,loops),
								static_cast<std::uint64_t>(loops));
			kar_ns[i]=per_op(bench_mul(a,b,loops,false),
							  static_cast<std::uint64_t>(loops));
		}
		auto th_cost=[&](std::size_t th)->std::uint64_t{
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kThCand.size();++i){
				const std::uint64_t w=(kThCand[i]>=128u)?3u:2u;
				const std::uint64_t ns=
					(kThCand[i]>=th)?kar_ns[i]:school_ns[i];
				cost+=ns*w;
			}
			return cost;
		};
		std::size_t best_th=tuned.kar_th;
		std::uint64_t best=th_cost(best_th);
		for(std::size_t th : kThCand){
			const std::uint64_t cost=th_cost(th);
			if(cost*std::uint64_t(100)<best*std::uint64_t(95)){
				best=cost;
				best_th=th;
			}
		}
		tuned.kar_th=best_th;
		at_state().kar_th=best_th;
	}

	{
		std::uint64_t seed=0xe7037ed1a0b428dbULL;
		constexpr std::array<std::size_t,3> kRecCand={56u,64u,80u};
		struct Smp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<Smp,3> kSmp={{
			{64u,4,1u},{128u,2,2u},{256u,1,3u}}};
		auto cost_of=[&](std::size_t rec)->std::uint64_t{
			at_state().sqr_rec=rec;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(const auto&sm : kSmp){
				const BigInt a=mk_bench(s,sm.n);
				cost+=per_op(bench_sqr_mul(a,sm.loops,false),
							 static_cast<std::uint64_t>(sm.loops))*sm.weight;
			}
			return cost;
		};
		std::size_t best_rec=tuned.sqr_rec;
		std::uint64_t best=cost_of(best_rec);
		for(std::size_t rec : kRecCand){
			const std::uint64_t cost=cost_of(rec);
			if(cost*std::uint64_t(100)<best*std::uint64_t(98)){
				best=cost;
				best_rec=rec;
			}
		}
		tuned.sqr_rec=best_rec;
		at_state().sqr_rec=best_rec;
	}

	{
		std::uint64_t seed=0xc6bc279692b5cc83ULL;
		std::vector<std::pair<BigInt,BigInt>> samples;
		for(std::size_t n : {64u,96u,256u}){
			BigInt a=mk_bench(seed,n);
			BigInt b=mk_bench(seed,n);
			if(a<b)
				std::swap(a,b);
			samples.emplace_back(std::move(a),std::move(b));
		}
		constexpr std::array<std::size_t,4> kMinCand={4u,6u,8u,16u};
		constexpr std::array<std::size_t,4> kRndCand={0u,1u,4u,8u};
		constexpr std::array<std::size_t,6> kSmCand={
			2u,16u,64u,128u,256u,512u};
		constexpr std::array<std::size_t,8> kLgCand={
			2u,4u,8u,16u,32u,64u,128u,192u};
		constexpr std::array<std::size_t,3> kQsCand={8u,16u,32u};
		std::size_t best_min=tuned.hl_min;
		std::size_t best_rnd=tuned.hl_rnd;
		std::size_t best_sm=tuned.gcd_sm;
		std::size_t best_lg=tuned.gcd_lg;
		std::size_t best_qs=tuned.gcd_qs;
		std::uint64_t best=bench_gcd(samples,best_min,best_rnd);
		for(std::size_t mn : kMinCand){
			for(std::size_t rd : kRndCand){
				for(std::size_t sm : kSmCand){
					for(std::size_t lg : kLgCand){
						for(std::size_t qs : kQsCand){
							at_state().gcd_sm=sm;
							at_state().gcd_lg=lg;
							at_state().gcd_qs=qs;
							const std::uint64_t cost=bench_gcd(samples,mn,rd);
							if(cost*std::uint64_t(100)<
							   best*std::uint64_t(98)){
								best=cost;
								best_min=mn;
								best_rnd=rd;
								best_sm=sm;
								best_lg=lg;
								best_qs=qs;
							}
						}
					}
				}
			}
		}
		tuned.hl_min=best_min;
		tuned.hl_rnd=best_rnd;
		tuned.gcd_sm=best_sm;
		tuned.gcd_lg=best_lg;
		tuned.gcd_qs=best_qs;
		at_state().hl_min=best_min;
		at_state().hl_rnd=best_rnd;
		at_state().gcd_sm=best_sm;
		at_state().gcd_lg=best_lg;
		at_state().gcd_qs=best_qs;
	}

	{
		std::uint64_t seed=0x9fb21c651e98df25ULL;
		constexpr std::array<std::size_t,3> kCand={192u,256u,384u};
		constexpr std::size_t kCandN=3u;
		struct Smp{
			limbs_t x;
			std::string s;
			std::size_t n=0;
			std::uint64_t base_ns=0;
			std::array<std::uint64_t,kCandN> fmt_ns{};
			std::uint64_t prs_base_ns=0;
			std::array<std::uint64_t,kCandN> prs_ns{};
		};
		std::vector<Smp> samples;
		for(std::size_t n : {256u,512u}){
			Smp sm{};
			sm.n=n;
			sm.x=mk_limb_bench(seed,n);
			sm.base_ns=bench_fmt_d10(sm.x,1,kNttOff);
			sm.s=to_str_d10(sm.x,1,kNttOff);
			sm.prs_base_ns=bench_prs_d10(sm.s,1,kNttOff);
			for(std::size_t i=0;i<kCand.size();++i){
				sm.fmt_ns[i]=bench_fmt_d10(sm.x,1,kCand[i]);
				const std::string chk=to_str_d10(sm.x,1,kCand[i]);
				MINI_MP_ASSERT(chk==sm.s);
				sm.prs_ns[i]=bench_prs_d10(sm.s,1,kCand[i]);
				limbs_t v;
				const bool ok=prs_d10(sm.s,&v,kCand[i]);
				MINI_MP_ASSERT(ok&&v==sm.x);
			}
			samples.push_back(std::move(sm));
		}
		auto pick=[&](bool parse)->std::size_t{
			std::uint64_t best=0;
			for(const auto&sm : samples)
				best+=parse?sm.prs_base_ns:sm.base_ns;
			std::size_t best_th=kNttOff;
			for(std::size_t i=0;i<kCand.size();++i){
				std::uint64_t cost=0;
				bool used=false;
				for(const auto&sm : samples){
					const std::size_t k=parse?
						((sm.s.size()+kD10Len-1u)/kD10Len):sm.n;
					if(k>=kCand[i]){
						used=true;
						cost+=parse?sm.prs_ns[i]:sm.fmt_ns[i];
					}else{
						cost+=parse?sm.prs_base_ns:sm.base_ns;
					}
				}
				if(used&&cost*std::uint64_t(100)<best*std::uint64_t(99)){
					best=cost;
					best_th=kCand[i];
				}
			}
			return best_th;
		};
		tuned.d10_dc=pick(false);
		tuned.d10_prs=pick(true);
		at_state().d10_dc=tuned.d10_dc;
		at_state().d10_prs=tuned.d10_prs;
	}

	at_state()=tuned;
}

inline void at_impl_full(){
	ATState tuned=at_base();
	at_state()=tuned;
	auto per_op=[](std::uint64_t ns,std::uint64_t ops)->std::uint64_t{
		return (ns+ops/2u)/ops;
	};

	{
		std::uint64_t seed=0x27bb2ee687b0b0fdULL;
		constexpr std::array<std::size_t,5> kRecCand={
			32u,40u,48u,56u,64u};
		struct RecSmp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<RecSmp,4> kRecSmp={{
			{64u,6,2u},{128u,4,2u},{256u,2,3u},{512u,1,4u}}};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_rec=tuned.kar_rec;
		for(std::size_t rec : kRecCand){
			at_state().kar_rec=rec;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(const auto&sm : kRecSmp){
				const BigInt a=mk_bench(s,sm.n);
				const BigInt b=mk_bench(s,sm.n);
				cost+=per_op(bench_mul(a,b,sm.loops,false),
							 static_cast<std::uint64_t>(sm.loops))*sm.weight;
			}
			if(cost<best_cost){
				best_cost=cost;
				best_rec=rec;
			}
		}
		tuned.kar_rec=best_rec;
		at_state().kar_rec=best_rec;
	}

	{
		std::uint64_t seed=0xa24baed4963ee407ULL;
		constexpr std::array<std::size_t,15> kKarCand={
			40u,48u,56u,64u,80u,96u,128u,160u,192u,
			224u,256u,320u,384u,512u,640u};
		std::array<std::uint64_t,kKarCand.size()> school_ns{};
		std::array<std::uint64_t,kKarCand.size()> kar_ns{};

		for(std::size_t idx=0;idx<kKarCand.size();++idx){
			const std::size_t n=kKarCand[idx];
			const int loops=
				(n<=64u)?8:(n<=128u)?5:(n<=256u)?3:(n<=512u)?2:1;
			std::uint64_t t_schsum=0;
			std::uint64_t t_kar_sum=0;
			for(int sample=0;sample<2;++sample){
				const BigInt a=mk_bench(seed,n);
				const BigInt b=mk_bench(seed,n);
				t_schsum+=bench_sbk(a,b,loops);
				t_kar_sum+=bench_mul(a,b,loops,false);
			}
			const std::uint64_t ops=
				static_cast<std::uint64_t>(loops)*std::uint64_t(2);
			school_ns[idx]=per_op(t_schsum,ops);
			kar_ns[idx]=per_op(t_kar_sum,ops);
		}

		auto kar_weight=[](std::size_t n)->std::uint64_t{
			return (n>=512u)?4u:(n>=224u)?3u:2u;
		};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_th=tuned.kar_th;
		for(std::size_t threshold : kKarCand){
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kKarCand.size();++i){
				const std::uint64_t ns=
					(kKarCand[i]>=threshold)?kar_ns[i]:school_ns[i];
				cost+=ns*kar_weight(kKarCand[i]);
			}
			if(cost<best_cost){
				best_cost=cost;
				best_th=threshold;
			}
		}
		tuned.kar_th=best_th;
		at_state().kar_th=best_th;
	}

	{
		std::uint64_t seed=0x165667b19e3779f9ULL;
		struct MulImbSample{
			BigInt a;
			BigInt b;
			std::size_t nmin=0;
			std::size_t nmax=0;
			std::uint64_t school_ns=0;
			std::uint64_t kar_ns=0;
		};
		std::vector<MulImbSample> samples;
		for(const auto ab : {std::pair<std::size_t,std::size_t>{96u,192u},
							 std::pair<std::size_t,std::size_t>{128u,384u},
							 std::pair<std::size_t,std::size_t>{160u,640u}}){
			MulImbSample s{};
			s.a=mk_bench(seed,ab.first);
			s.b=mk_bench(seed,ab.second);
			s.nmin=std::min(ab.first,ab.second);
			s.nmax=std::max(ab.first,ab.second);
			const int loops=(s.nmax<=192u)?3:1;
			s.school_ns=bench_sbk(s.a,s.b,loops);
			s.kar_ns=bench_mul(s.a,s.b,loops,false);
			samples.push_back(std::move(s));
		}
		constexpr std::array<std::size_t,5> kImbCand={2u,3u,4u,6u,8u};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_imb=tuned.kar_imb;
		for(std::size_t imb : kImbCand){
			std::uint64_t cost=0;
			for(const auto&s : samples){
				const bool use_kar=s.nmin>=tuned.kar_th&&s.nmax<=s.nmin*imb;
				cost+=use_kar?s.kar_ns:s.school_ns;
			}
			if(cost<best_cost){
				best_cost=cost;
				best_imb=imb;
			}
		}
		tuned.kar_imb=best_imb;
		at_state().kar_imb=best_imb;
	}

	{
		std::uint64_t seed=0x7d2b4f8a93c1e51dULL;
		constexpr std::array<std::size_t,5> kDifCand={
			64u,96u,128u,192u,kNttOff};
		struct DifSmp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<DifSmp,4> kDifSmp={{
			{64u,6,2u},{128u,4,2u},{256u,2,3u},{512u,1,4u}}};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_dif=tuned.kar_dif;
		for(std::size_t dif : kDifCand){
			at_state().kar_dif=dif;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(const auto&sm : kDifSmp){
				const BigInt a=mk_bench(s,sm.n);
				const BigInt b=mk_bench(s,sm.n);
				cost+=per_op(bench_mul(a,b,sm.loops,false),
							 static_cast<std::uint64_t>(sm.loops))*sm.weight;
			}
			if(cost<best_cost){
				best_cost=cost;
				best_dif=dif;
			}
		}
		tuned.kar_dif=best_dif;
		at_state().kar_dif=best_dif;
	}

	{
		std::uint64_t seed=0x92d4c37a56a5f39bULL;
		constexpr std::array<std::size_t,5> kRecCand={
			32u,40u,48u,56u,64u};
		constexpr std::array<std::size_t,5> kDifCand={
			64u,96u,128u,192u,kNttOff};
		struct PairSmp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<PairSmp,5> kPairSmp={{
			{64u,8,1u},{128u,5,2u},{256u,3,3u},
			{512u,1,4u},{1024u,1,4u}}};
		auto pair_cost=[&](std::size_t rec,std::size_t dif)->std::uint64_t{
			at_state().kar_rec=rec;
			at_state().kar_dif=dif;
			std::uint64_t cost=0;
			for(int pass=0;pass<2;++pass){
				std::uint64_t s=seed+std::uint64_t(pass)*0x9e3779b97f4a7c15ULL;
				for(const auto&sm : kPairSmp){
					const BigInt a=mk_bench(s,sm.n);
					const BigInt b=mk_bench(s,sm.n);
					const std::uint64_t ops=static_cast<std::uint64_t>(sm.loops);
					cost+=per_op(bench_mul(a,b,sm.loops,false),ops)*sm.weight;
				}
			}
			return cost;
		};
		std::size_t best_rec=tuned.kar_rec;
		std::size_t best_dif=tuned.kar_dif;
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		for(std::size_t rec : kRecCand){
			for(std::size_t dif : kDifCand){
				const std::uint64_t cost=pair_cost(rec,dif);
				if(cost<best_cost){
					best_cost=cost;
					best_rec=rec;
					best_dif=dif;
				}
			}
		}
		tuned.kar_rec=best_rec;
		tuned.kar_dif=best_dif;
		at_state().kar_rec=best_rec;
		at_state().kar_dif=best_dif;
	}

	{
		std::uint64_t seed=0x8c03fef146b1d2a3ULL;
		constexpr std::array<std::size_t,6> kSqrRecCand={
			40u,48u,56u,64u,80u,96u};
		struct SqrSmp{
			std::size_t n;
			int loops;
			std::uint64_t weight;
		};
		constexpr std::array<SqrSmp,4> kSqrSmp={{
			{64u,8,1u},{128u,5,2u},{256u,3,3u},{512u,1,4u}}};
		auto sqr_cost=[&](std::size_t rec)->std::uint64_t{
			at_state().sqr_rec=rec;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(const auto&sm : kSqrSmp){
				const BigInt a=mk_bench(s,sm.n);
				const std::uint64_t ops=static_cast<std::uint64_t>(sm.loops);
				cost+=per_op(bench_sqr_mul(a,sm.loops,false),ops)*sm.weight;
			}
			return cost;
		};
		std::size_t best_rec=tuned.sqr_rec;
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		for(std::size_t rec : kSqrRecCand){
			const std::uint64_t cost=sqr_cost(rec);
			if(cost<best_cost){
				best_cost=cost;
				best_rec=rec;
			}
		}
		tuned.sqr_rec=best_rec;
		at_state().sqr_rec=best_rec;
	}

	{
		std::uint64_t seed=0x3bd39e10cb0ef593ULL;
		constexpr std::array<std::size_t,15> kKarCand={
			40u,48u,56u,64u,80u,96u,128u,160u,192u,
			224u,256u,320u,384u,512u,640u};
		std::array<std::uint64_t,kKarCand.size()> school_ns{};
		std::array<std::uint64_t,kKarCand.size()> kar_ns{};
		for(std::size_t idx=0;idx<kKarCand.size();++idx){
			const std::size_t n=kKarCand[idx];
			const int loops=
				(n<=64u)?8:(n<=128u)?5:(n<=256u)?3:(n<=512u)?2:1;
			std::uint64_t t_sch=0;
			std::uint64_t t_kar=0;
			for(int sample=0;sample<2;++sample){
				const BigInt a=mk_bench(seed,n);
				const BigInt b=mk_bench(seed,n);
				t_sch+=bench_sbk(a,b,loops);
				t_kar+=bench_mul(a,b,loops,false);
			}
			const std::uint64_t ops=
				static_cast<std::uint64_t>(loops)*std::uint64_t(2);
			school_ns[idx]=per_op(t_sch,ops);
			kar_ns[idx]=per_op(t_kar,ops);
		}
		auto weight=[](std::size_t n)->std::uint64_t{
			return (n>=512u)?4u:(n>=224u)?3u:2u;
		};
		auto th_cost=[&](std::size_t threshold)->std::uint64_t{
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kKarCand.size();++i){
				const std::uint64_t ns=
					(kKarCand[i]>=threshold)?kar_ns[i]:school_ns[i];
				cost+=ns*weight(kKarCand[i]);
			}
			return cost;
		};
		std::size_t best_th=tuned.kar_th;
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		for(std::size_t threshold : kKarCand){
			const std::uint64_t cost=th_cost(threshold);
			if(cost<best_cost){
				best_cost=cost;
				best_th=threshold;
			}
		}
		tuned.kar_th=best_th;
		at_state().kar_th=best_th;
	}

	{
		std::uint64_t seed=0xf1357aea5151d00dULL;
		constexpr std::array<std::size_t,5> kT3Cand={
			384u,512u,640u,768u,1024u};
		struct T3Smp{
			BigInt a;
			BigInt b;
			std::size_t n=0;
			std::uint64_t kar_ns=0;
			std::uint64_t t3_ns=0;
		};
		std::vector<T3Smp> samples;
		samples.reserve(kT3Cand.size());
		for(std::size_t n : kT3Cand){
			if(n<tuned.kar_th)
				continue;
			T3Smp s{};
			s.n=n;
			s.a=mk_bench(seed,n);
			s.b=mk_bench(seed,n);
			const int loops=(n<=512u)?2:1;
			const std::uint64_t loop_n=static_cast<std::uint64_t>(loops);
			s.kar_ns=(bench_mul(s.a,s.b,loops,false)+loop_n/2u)/loop_n;
			s.t3_ns=(bench_t3(s.a,s.b,loops)+loop_n/2u)/loop_n;
			MINI_MP_ASSERT(mul_kar(s.a,s.b)==mul_t3(s.a,s.b));
			samples.push_back(std::move(s));
		}

		std::uint64_t off_cost=0;
		for(const auto&s : samples)
			off_cost+=s.kar_ns;
		std::uint64_t best_cost=off_cost;
		std::size_t best_th=tuned.t3_th;
		for(std::size_t th : kT3Cand){
			bool used=false;
			bool ok=true;
			std::uint64_t cost=0;
			for(const auto&s : samples){
				if(s.n>=th){
					used=true;
					if(s.t3_ns>s.kar_ns){
						ok=false;
						break;
					}
					cost+=s.t3_ns;
				}else{
					cost+=s.kar_ns;
				}
			}
			if(ok&&used&&cost*std::uint64_t(100)<best_cost*std::uint64_t(99)){
				best_cost=cost;
				best_th=th;
			}
		}
		tuned.t3_th=best_th;
		at_state().t3_th=best_th;
	}

#if MINI_MP_ENABLE_NTT
	{
		std::uint64_t seed=0x9e3779b97f4a7c15ULL;
		constexpr std::array<std::size_t,7> kBitsCand={
			15u,18u,21u,23u,24u,25u,26u};
		constexpr std::array<std::size_t,10> kNttCand={
			192u,256u,320u,384u,512u,640u,768u,1024u,1280u,1536u};
		struct NttSmp{
			BigInt a;
			BigInt b;
			std::size_t n=0;
			std::uint64_t mul_base_ns=0;
			std::uint64_t sqr_base_ns=0;
			std::array<std::uint64_t,kBitsCand.size()> mul_ntt_ns{};
			std::array<std::uint64_t,kBitsCand.size()> sqr_ntt_ns{};
		};
		std::vector<NttSmp> samples;
		samples.reserve(kNttCand.size());
		for(std::size_t n : kNttCand){
			if(n<tuned.kar_th)
				continue;
			NttSmp s{};
			s.n=n;
			s.a=mk_bench(seed,n);
			s.b=mk_bench(seed,n);
			const int loops=(n<=384u)?2:1;
			const std::uint64_t loop_n=static_cast<std::uint64_t>(loops);
			s.mul_base_ns=(bench_mul(s.a,s.b,loops,false)+loop_n/2u)/loop_n;
			s.sqr_base_ns=(bench_sqr_mul(s.a,loops,false)+loop_n/2u)/loop_n;
			for(std::size_t bi=0;bi<kBitsCand.size();++bi){
				at_state().ntt_bits=kBitsCand[bi];
				s.mul_ntt_ns[bi]=
					(bench_mul(s.a,s.b,loops,true)+loop_n/2u)/loop_n;
				s.sqr_ntt_ns[bi]=
					(bench_sqr_mul(s.a,loops,true)+loop_n/2u)/loop_n;
			}
			samples.push_back(std::move(s));
		}

		std::uint64_t mul_off=0;
		std::uint64_t sqr_off=0;
		for(const auto&s : samples){
			mul_off+=s.mul_base_ns;
			sqr_off+=s.sqr_base_ns;
		}
		auto pick_nth=[&](std::size_t bi,bool square,
						 std::uint64_t off_cost,
						 std::uint64_t&best_cost)->std::size_t{
			best_cost=off_cost;
			std::size_t best_th=kNttOff;
			for(std::size_t threshold : kNttCand){
				bool used=false;
				bool ok=true;
				std::uint64_t cost=0;
				for(const auto&s : samples){
					const std::uint64_t base=
						square?s.sqr_base_ns:s.mul_base_ns;
					const std::uint64_t ntt=
						square?s.sqr_ntt_ns[bi]:s.mul_ntt_ns[bi];
					if(s.n>=threshold){
						used=true;
						if(ntt>base){
							ok=false;
							break;
						}
						cost+=ntt;
					}else{
						cost+=base;
					}
				}
				if(ok&&used&&cost*std::uint64_t(100)<
				   best_cost*std::uint64_t(99)){
					best_cost=cost;
					best_th=threshold;
				}
			}
			return best_th;
		};

		std::uint64_t best_direct=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_direct_bits=tuned.ntt_bits;
		for(std::size_t bi=0;bi<kBitsCand.size();++bi){
			std::uint64_t cost=0;
			for(const auto&s : samples){
				cost+=s.mul_ntt_ns[bi];
				cost+=s.sqr_ntt_ns[bi];
			}
			if(cost<best_direct){
				best_direct=cost;
				best_direct_bits=kBitsCand[bi];
			}
		}

		std::uint64_t best_all=mul_off+sqr_off;
		std::size_t best_bits=best_direct_bits;
		std::size_t best_mul_th=tuned.ntt_th;
		std::size_t best_sqr_th=tuned.ntt_sq_th;
		for(std::size_t bi=0;bi<kBitsCand.size();++bi){
			std::uint64_t mul_cost=0;
			std::uint64_t sqr_cost=0;
			const std::size_t mul_th=pick_nth(bi,false,mul_off,mul_cost);
			const std::size_t sqr_th=pick_nth(bi,true,sqr_off,sqr_cost);
			const std::uint64_t total=mul_cost+sqr_cost;
			if(total<best_all){
				best_all=total;
				best_bits=kBitsCand[bi];
				best_mul_th=mul_th;
				best_sqr_th=sqr_th;
			}
		}
		tuned.ntt_bits=best_bits;
		tuned.ntt_th=best_mul_th;
		tuned.ntt_sq_th=best_sqr_th;
		at_state().ntt_bits=tuned.ntt_bits;
		at_state().ntt_th=tuned.ntt_th;
		at_state().ntt_sq_th=tuned.ntt_sq_th;
		if(tuned.ntt_th!=kNttOff){
			std::uint64_t seed2=0x2f2f2218beefcafeULL;
			std::size_t best_imb=1u;
			for(std::size_t ratio : {2u,3u}){
				const BigInt a=mk_bench(seed2,tuned.ntt_th);
				const BigInt b=mk_bench(seed2,tuned.ntt_th*ratio);
				const std::uint64_t t_kar=bench_mul(a,b,1,false);
				const std::uint64_t t_ntt=bench_mul(a,b,1,true);
				if(t_ntt<=t_kar)
					best_imb=ratio;
				else
					break;
			}
			tuned.ntt_imb=best_imb;
			at_state().ntt_imb=best_imb;
		}
	}
#endif

	{
		std::uint64_t seed=0xd1b54a32d192ed03ULL;
		std::vector<std::pair<BigInt,BigInt>> samples;
		constexpr std::array<std::size_t,8> kSizes={
			16u,24u,40u,64u,96u,160u,256u,512u};
		samples.reserve(kSizes.size()*2);
		for(std::size_t limbs : kSizes){
			const int count=(limbs>=160u)?1:2;
			for(int i=0;i<count;++i){
				BigInt a=mk_bench(seed,limbs);
				BigInt b=mk_bench(seed,limbs);
				if(a==b)
					b+=BigInt(1);
				if(a<b)
					std::swap(a,b);
				samples.emplace_back(std::move(a),std::move(b));
			}
		}

		constexpr std::array<std::size_t,5> kMinCand={3u,4u,6u,8u,16u};
		constexpr std::array<std::size_t,7> kRndCand={
			0u,1u,2u,4u,6u,8u,10u};

		std::size_t best_min=tuned.hl_min;
		std::size_t best_rnd=tuned.hl_rnd;
		std::uint64_t best_t=bench_gcd(samples,best_min,best_rnd);

		for(std::size_t min_limbs : kMinCand){
			for(std::size_t rounds : kRndCand){
				const std::uint64_t t=
					bench_gcd(samples,min_limbs,rounds);
				if(t*std::uint64_t(100)<best_t*std::uint64_t(95)){
					best_t=t;
					best_min=min_limbs;
					best_rnd=rounds;
				}
			}
		}
		tuned.hl_min=best_min;
		tuned.hl_rnd=best_rnd;
		at_state().hl_min=best_min;
		at_state().hl_rnd=best_rnd;

		constexpr std::array<std::size_t,13> kGcdSmCand={
			2u,4u,8u,16u,32u,64u,96u,128u,192u,256u,384u,512u,768u};
		constexpr std::array<std::size_t,10> kGcdLgCand={
			2u,4u,8u,16u,32u,64u,96u,128u,192u,256u};
		constexpr std::array<std::size_t,4> kGcdQsCand={
			8u,16u,32u,48u};
		std::size_t best_sm=tuned.gcd_sm;
		std::size_t best_lg=tuned.gcd_lg;
		std::size_t best_qs=tuned.gcd_qs;
		best_t=bench_gcd(samples,tuned.hl_min,tuned.hl_rnd);
		for(std::size_t sm : kGcdSmCand){
			for(std::size_t lg : kGcdLgCand){
				for(std::size_t qs : kGcdQsCand){
					at_state().gcd_sm=sm;
					at_state().gcd_lg=lg;
					at_state().gcd_qs=qs;
					const std::uint64_t t=
						bench_gcd(samples,tuned.hl_min,tuned.hl_rnd);
					if(t*std::uint64_t(100)<best_t*std::uint64_t(95)){
						best_t=t;
						best_sm=sm;
						best_lg=lg;
						best_qs=qs;
					}
				}
			}
		}
		tuned.gcd_sm=best_sm;
		tuned.gcd_lg=best_lg;
		tuned.gcd_qs=best_qs;
		at_state().gcd_sm=best_sm;
		at_state().gcd_lg=best_lg;
		at_state().gcd_qs=best_qs;
	}

	{
		std::uint64_t seed=0x9e0b5ad15eed1234ULL;
		struct DivSample{
			limbs_t u;
			limbs_t v;
			std::size_t v_limbs=0;
			std::size_t chunks=0;
			std::uint64_t knuth_ns=0;
			std::uint64_t bz_ns=0;
		};
		std::vector<DivSample> samples;
		for(const auto vc : {std::pair<std::size_t,std::size_t>{64u,2u},
							 std::pair<std::size_t,std::size_t>{96u,3u},
							 std::pair<std::size_t,std::size_t>{128u,4u}}){
			DivSample s{};
			s.v_limbs=vc.first;
			s.chunks=vc.second;
			s.v=mk_limb_bench(seed,s.v_limbs);
			s.u=mk_limb_bench(seed,s.v_limbs*s.chunks);
			if(cmp_abs(s.u,s.v)<0)
				s.u.push_back(1);
			s.knuth_ns=bench_div_abs(s.u,s.v,1,false);
			s.bz_ns=bench_div_abs(s.u,s.v,1,true);
			samples.push_back(std::move(s));
		}
		constexpr std::array<std::size_t,4> kBzMinCand={64u,96u,128u,192u};
		constexpr std::array<std::size_t,3> kBzChunkCand={2u,3u,4u};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_min=tuned.bz_min;
		std::size_t best_chunk=tuned.bz_chunk;
		for(std::size_t mn : kBzMinCand){
			for(std::size_t ch : kBzChunkCand){
				std::uint64_t cost=0;
				for(const auto&s : samples){
					const bool use_bz=s.v_limbs>=mn&&s.chunks>=ch;
					cost+=use_bz?s.bz_ns:s.knuth_ns;
				}
				if(cost<best_cost){
					best_cost=cost;
					best_min=mn;
					best_chunk=ch;
				}
			}
		}
		tuned.bz_min=best_min;
		tuned.bz_chunk=best_chunk;
		at_state().bz_min=best_min;
		at_state().bz_chunk=best_chunk;
	}

	{
		std::uint64_t seed=0x91e10da5d10c0de1ULL;
		constexpr std::array<std::size_t,5> kD10Cand={
			96u,128u,192u,256u,384u};
		constexpr std::size_t kD10CandN=5u;
		struct D10Smp{
			limbs_t x;
			std::string s;
			std::size_t n=0;
			std::uint64_t base_ns=0;
			std::array<std::uint64_t,kD10CandN> dc_ns{};
			std::uint64_t prs_base_ns=0;
			std::array<std::uint64_t,kD10CandN> prs_dc_ns{};
		};
		constexpr std::array<std::size_t,4> kD10Sizes={
			64u,128u,256u,512u};
		std::vector<D10Smp> samples;
		samples.reserve(kD10Sizes.size());
		for(std::size_t n : kD10Sizes){
			D10Smp s{};
			s.n=n;
			s.x=mk_limb_bench(seed,n);
			const int loops=(n<=64u)?2:1;
			const std::uint64_t loop_n=static_cast<std::uint64_t>(loops);
			s.base_ns=(bench_fmt_d10(s.x,loops,kNttOff)+loop_n/2u)/loop_n;
			const std::string ref=to_str_d10(s.x,1,kNttOff);
			s.s=ref;
			s.prs_base_ns=
				(bench_prs_d10(s.s,loops,kNttOff)+loop_n/2u)/loop_n;
			for(std::size_t i=0;i<kD10Cand.size();++i){
				s.dc_ns[i]=
					(bench_fmt_d10(s.x,loops,kD10Cand[i])+loop_n/2u)/loop_n;
				const std::string chk=to_str_d10(s.x,1,kD10Cand[i]);
				MINI_MP_ASSERT(ref==chk);
				s.prs_dc_ns[i]=
					(bench_prs_d10(s.s,loops,kD10Cand[i])+loop_n/2u)/loop_n;
				limbs_t chk_x;
				const bool ok=prs_d10(s.s,&chk_x,kD10Cand[i]);
				MINI_MP_ASSERT(ok&&chk_x==s.x);
			}
			samples.push_back(std::move(s));
		}

		std::uint64_t off_cost=0;
		for(const auto&s : samples)
			off_cost+=s.base_ns;
		std::uint64_t best_cost=off_cost;
		std::size_t best_th=kNttOff;
		for(std::size_t i=0;i<kD10Cand.size();++i){
			const std::size_t th=kD10Cand[i];
			bool used=false;
			std::uint64_t cost=0;
			for(const auto&s : samples){
				if(s.n>=th){
					used=true;
					cost+=s.dc_ns[i];
				}else{
					cost+=s.base_ns;
				}
			}
			if(used&&cost*std::uint64_t(100)<best_cost*std::uint64_t(99)){
				best_cost=cost;
				best_th=th;
			}
		}
		tuned.d10_dc=best_th;
		at_state().d10_dc=best_th;

		off_cost=0;
		for(const auto&s : samples)
			off_cost+=s.prs_base_ns;
		best_cost=off_cost;
		best_th=kNttOff;
		for(std::size_t i=0;i<kD10Cand.size();++i){
			const std::size_t th=kD10Cand[i];
			bool used=false;
			std::uint64_t cost=0;
			for(const auto&s : samples){
				const std::size_t chunks=
					(s.s.size()+kD10Len-1u)/kD10Len;
				if(chunks>=th){
					used=true;
					cost+=s.prs_dc_ns[i];
				}else{
					cost+=s.prs_base_ns;
				}
			}
			if(used&&cost*std::uint64_t(100)<best_cost*std::uint64_t(99)){
				best_cost=cost;
				best_th=th;
			}
		}
		tuned.d10_prs=best_th;
		at_state().d10_prs=best_th;
	}

	{
		constexpr std::array<std::size_t,5> kLeafCand={
			12u,18u,24u,36u,48u};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_leaf=tuned.prod_leaf;
		for(std::size_t leaf : kLeafCand){
			const std::uint64_t t=bench_fact(384,2,true,leaf);
			if(t<best_cost){
				best_cost=t;
				best_leaf=leaf;
			}
		}
		tuned.prod_leaf=best_leaf;
		at_state().prod_leaf=best_leaf;

		constexpr std::array<std::size_t,5> kFacCand={
			96u,128u,160u,192u,256u};
		std::array<std::uint64_t,3> loop_ns{};
		std::array<std::uint64_t,3> tree_ns{};
		constexpr std::array<std::uint64_t,3> kFacSizes={96u,192u,384u};
		for(std::size_t i=0;i<kFacSizes.size();++i){
			const int loops=(kFacSizes[i]<=96u)?3:1;
			loop_ns[i]=bench_fact(kFacSizes[i],loops,false,best_leaf);
			tree_ns[i]=bench_fact(kFacSizes[i],loops,true,best_leaf);
		}
		best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_fac=tuned.fac_tree;
		for(std::size_t th : kFacCand){
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kFacSizes.size();++i)
				cost+=(kFacSizes[i]>=th)?tree_ns[i]:loop_ns[i];
			if(cost<best_cost){
				best_cost=cost;
				best_fac=th;
			}
		}
		tuned.fac_tree=best_fac;
		at_state().fac_tree=best_fac;

		constexpr std::array<std::size_t,5> kBinomCand={
			64u,96u,128u,160u,224u};
		std::array<std::uint64_t,3> bloop_ns{};
		std::array<std::uint64_t,3> btree_ns{};
		constexpr std::array<std::uint64_t,3> kBinomK={64u,128u,256u};
		for(std::size_t i=0;i<kBinomK.size();++i){
			const int loops=(kBinomK[i]<=64u)?3:1;
			bloop_ns[i]=bench_binom(kBinomK[i]*2u,kBinomK[i],
									loops,false,best_leaf);
			btree_ns[i]=bench_binom(kBinomK[i]*2u,kBinomK[i],
									loops,true,best_leaf);
		}
		best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_binom=tuned.binom_tree;
		for(std::size_t th : kBinomCand){
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kBinomK.size();++i)
				cost+=(kBinomK[i]>=th)?btree_ns[i]:bloop_ns[i];
			if(cost<best_cost){
				best_cost=cost;
				best_binom=th;
			}
		}
		tuned.binom_tree=best_binom;
		at_state().binom_tree=best_binom;
	}

	{
		std::uint64_t seed=0x7f4a7c159e3779b9ULL;
		const BigInt base=mk_bench(seed,16);
		BigInt mod=mk_bench(seed,16);
		if(mod.is_even())
			mod+=BigInt(1);
		const BigInt exp_mid=mk_bench(seed,8);
		const BigInt exp_big=mk_bench(seed,32);
		constexpr std::array<std::size_t,3> kW5Cand={256u,384u,512u};
		constexpr std::array<std::size_t,3> kW6Cand={1024u,1536u,2048u};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_w5=tuned.pow_w5;
		std::size_t best_w6=tuned.pow_w6;
		for(std::size_t w5 : kW5Cand){
			for(std::size_t w6 : kW6Cand){
				if(w6<=w5)
					continue;
				at_state().pow_w5=w5;
				at_state().pow_w6=w6;
				volatile std::size_t sink=0;
				const auto t0=std::chrono::steady_clock::now();
				sink^=modpow(base,exp_mid,mod).bit_length();
				sink^=modpow(base,exp_big,mod).bit_length();
				const auto t1=std::chrono::steady_clock::now();
				(void)sink;
				const std::uint64_t cost=static_cast<std::uint64_t>(
					std::chrono::duration_cast<std::chrono::nanoseconds>(
						t1-t0).count());
				if(cost<best_cost){
					best_cost=cost;
					best_w5=w5;
					best_w6=w6;
				}
			}
		}
		tuned.pow_w5=best_w5;
		tuned.pow_w6=best_w6;
		at_state().pow_w5=best_w5;
		at_state().pow_w6=best_w6;
	}

	at_state()=tuned;
}

inline void ensure_at_level(unsigned need){
#if MINI_MP_ENABLE_AUTOTUNE
	if(need==0u)
		return;
	if(at_done().load(std::memory_order_acquire)>=need)
		return;
	if(at_busy())
		return;
	static std::mutex mtx;
	std::lock_guard<std::mutex> lock(mtx);
	if(at_done().load(std::memory_order_acquire)>=need)
		return;
	at_busy()=true;
	if(need>=2u)
		at_impl_full();
	else
		at_impl_fast();
	at_busy()=false;
	at_done().store(need,std::memory_order_release);
#else
	(void)need;
#endif
}

inline void ensure_at(){
	ensure_at_level(at_default_level());
}

inline std::size_t tun_ntt() noexcept{
	return at_state().ntt_th;
}

inline std::size_t tun_ntt_sq() noexcept{
	return at_state().ntt_sq_th;
}

inline std::size_t tun_ntt_bits() noexcept{
	return at_state().ntt_bits;
}

inline std::size_t tun_krec() noexcept{
	return at_state().kar_rec;
}

inline std::size_t tun_srec() noexcept{
	return at_state().sqr_rec;
}

inline std::size_t tun_kar() noexcept{
	return at_state().kar_th;
}

inline std::size_t tun_kar_imb() noexcept{
	return at_state().kar_imb;
}

inline std::size_t tun_kdif() noexcept{
	return at_state().kar_dif;
}

inline std::size_t tun_ntt_imb() noexcept{
	return at_state().ntt_imb;
}

inline std::size_t tun_hmin() noexcept{
	return at_state().hl_min;
}

inline std::size_t tun_hrnd() noexcept{
	return at_state().hl_rnd;
}

inline std::size_t tun_gcd_sm() noexcept{
	return at_state().gcd_sm;
}

inline std::size_t tun_gcd_lg() noexcept{
	return at_state().gcd_lg;
}

inline std::size_t tun_gcd_qs() noexcept{
	return at_state().gcd_qs;
}

inline std::size_t tun_bz_min() noexcept{
	return at_state().bz_min;
}

inline std::size_t tun_bz_chunk() noexcept{
	return at_state().bz_chunk;
}

inline std::size_t tun_prod_leaf() noexcept{
	return at_state().prod_leaf;
}

inline std::size_t tun_fac_tree() noexcept{
	return at_state().fac_tree;
}

inline std::size_t tun_binom_tree() noexcept{
	return at_state().binom_tree;
}

inline std::size_t tun_pow_w5() noexcept{
	return at_state().pow_w5;
}

inline std::size_t tun_pow_w6() noexcept{
	return at_state().pow_w6;
}

inline std::size_t tun_d10_dc() noexcept{
	return at_state().d10_dc;
}

inline std::size_t tun_d10_prs() noexcept{
	return at_state().d10_prs;
}

inline std::size_t tun_t3() noexcept{
	return at_state().t3_th;
}

} 

inline void autotune_fast(){
	detail::ensure_at_level(1u);
}

inline void autotune_full(){
	detail::ensure_at_level(2u);
}





inline void mkodd_ip(BigInt&x){
	if(x.is_zero())
		return;
	const std::size_t k=x.ctz();
	if(k!=0)
		x>>=k;
}

inline BigInt gcd_sm_bin(BigInt a,BigInt b){
	if(a.sign_<0)
		a.sign_=1;
	if(b.sign_<0)
		b.sign_=1;
	if(a.is_zero())
		return b;
	if(b.is_zero())
		return a;
	if(a.limbs_.size()<=2&&b.limbs_.size()<=2){
		detail::limbs_t g=detail::gcd_2limb_s(a.limbs_,b.limbs_);
		return BigInt::from_raw(g.empty()?0:1,std::move(g));
	}
	detail::limbs_t g=detail::gcd_bin_s(std::move(a.limbs_),
														 std::move(b.limbs_));
	return BigInt::from_raw(g.empty()?0:1,std::move(g));
}

inline BigInt gcd_hlprm(BigInt a,BigInt b,
										  std::size_t hl_min,
										  std::size_t hl_rnd){
	if(a.sign_<0)
		a.sign_=1;
	if(b.sign_<0)
		b.sign_=1;
	if(a.is_zero())
		return b;
	if(b.is_zero())
		return a;
	if(a.limbs_.size()==1&&b.limbs_.size()==1){
		return BigInt::from_u64(std::gcd(a.limbs_[0],b.limbs_[0]));
	}
	if(a.limbs_.size()==1){
		const std::uint64_t r=detail::mod_limb(b.limbs_,a.limbs_[0]);
		return BigInt::from_u64(std::gcd(a.limbs_[0],r));
	}
	if(b.limbs_.size()==1){
		const std::uint64_t r=detail::mod_limb(a.limbs_,b.limbs_[0]);
		return BigInt::from_u64(std::gcd(b.limbs_[0],r));
	}
	const std::size_t init_mx=
		std::max(a.limbs_.size(),b.limbs_.size());
	const std::size_t smbin_mx=
		(init_mx<=detail::tun_gcd_sm())
			?detail::tun_gcd_sm()
			:detail::tun_gcd_lg();
	if(std::max(a.limbs_.size(),b.limbs_.size())<=smbin_mx){
		return gcd_sm_bin(std::move(a),std::move(b));
	}

	const std::size_t shift=std::min(a.ctz(),b.ctz());
	if(shift!=0){
		a>>=shift;
		b>>=shift;
	}
	mkodd_ip(a);
	mkodd_ip(b);
	if(a<b)
		std::swap(a,b);
	detail::ModKScr sc;
	detail::LehScr hsc;

	while(!b.is_zero()){
		if(std::max(a.limbs_.size(),b.limbs_.size())<=smbin_mx){
			a=gcd_sm_bin(std::move(a),std::move(b));
			if(shift!=0)
				a<<=shift;
			return a;
		}
		if(b.limbs_.size()==1){
			const std::uint64_t bl=b.limbs_[0];
			const std::uint64_t r=detail::mod_limb(a.limbs_,bl);
			BigInt g=BigInt::from_u64(std::gcd(bl,r));
			if(shift!=0)
				g<<=shift;
			return g;
		}

		bool did_leh=false;
		if(hl_rnd!=0&&
		   std::max(a.limbs_.size(),b.limbs_.size())>=hl_min){
			for(std::size_t step=0;step<hl_rnd;++step){
				if(b.is_zero()||
				   std::max(a.limbs_.size(),b.limbs_.size())<hl_min){
					break;
				}
				if(!detail::leh_step_ip(a.limbs_,b.limbs_,hsc))
					break;
				a.sign_=a.limbs_.empty()?0:1;
				b.sign_=b.limbs_.empty()?0:1;
				did_leh=true;
			}
			if(did_leh)
				continue;
		}

		detail::modk_ip(a.limbs_,b.limbs_,sc);
		a.limbs_.swap(b.limbs_);
		a.sign_=a.limbs_.empty()?0:1;
		b.sign_=b.limbs_.empty()?0:1;
		if(!b.is_zero()){
			const std::size_t tz=detail::ctz(b.limbs_);
			if(tz!=0)
				detail::shr_ip(b.limbs_,tz);
		}
	}

	if(shift!=0)
		a<<=shift;
	return a;
}

inline BigInt gcd(BigInt a,BigInt b){
	detail::ensure_at();
	return gcd_hlprm(std::move(a),std::move(b),
									   detail::tun_hmin(),
									   detail::tun_hrnd());
}

struct ExtGcdRes{
	BigInt g;
	BigInt x;
	BigInt y;
};

inline ExtGcdRes extgcd(BigInt a,BigInt b){
	if(a.is_zero()&&b.is_zero()){
		return {BigInt(),BigInt(),BigInt()};
	}

	BigInt old_r=a;
	BigInt r=b;
	BigInt old_s(1),s(0);
	BigInt old_t(0),t(1);

	while(!r.is_zero()){
		auto qr=divmod(old_r,r);
		const BigInt q=qr.first;

		BigInt next_r=old_r-q*r;
		old_r=std::move(r);
		r=std::move(next_r);

		BigInt next_s=old_s-q*s;
		old_s=std::move(s);
		s=std::move(next_s);

		BigInt next_t=old_t-q*t;
		old_t=std::move(t);
		t=std::move(next_t);
	}

	if(old_r.sign()<0){
		old_r=-old_r;
		old_s=-old_s;
		old_t=-old_t;
	}
	return {std::move(old_r),std::move(old_s),std::move(old_t)};
}

inline BigInt lcm(const BigInt&a,const BigInt&b){
	if(a.is_zero()||b.is_zero())
		return BigInt();
	const BigInt g=gcd(a,b);
	BigInt out=(detail::cmp_abs(a.limbs_,b.limbs_)<=0)
		?(divexact(a,g)*b)
		:(divexact(b,g)*a);
	return out.abs();
}

inline BigInt pow(BigInt base,std::uint64_t exp){
	BigInt result(1);
	while(exp!=0){
		if(exp&1u)
			result*=base;
		exp>>=1;
		if(exp)
			base=sqr_disp(base);
	}
	return result;
}

namespace detail{
inline BigInt old_modpow(BigInt base,const BigInt&exp,const BigInt&mod){
	if(mod.sign()<=0){
		detail::throw_dom("old_modpow: modulus must be positive");
	}
	base%=mod;
	if(base.sign()<0)
		base+=mod;
	if(mod.fits_u64()&&mod.to_u64()==1u)
		return BigInt();
	BigInt result(1);
	result%=mod;
	if(result.sign()<0)
		result+=mod;
	const std::size_t bits=exp.bit_length();
	for(std::size_t bit=0;bit<bits;++bit){
		if(exp.test_bit(bit))
			result=(result*base)%mod;
		if(bit+1<bits)
			base=(base*base)%mod;
	}
	return result;
}
} 

inline BigInt modpow(BigInt base,BigInt exp,const BigInt&mod){
	detail::ensure_at();
	if(mod.sign()<=0){
		detail::throw_dom("modpow: modulus must be positive");
	}
	if(exp.sign()<0){
		detail::throw_dom("modpow: exponent must be non-negative");
	}
	if(!(base.sign_>=0&&detail::cmp_abs(base.limbs_,mod.limbs_)<0)){
		base%=mod;
		if(base.sign()<0)
			base+=mod;
	}
	if(mod.limbs_.size()==1&&mod.limbs_[0]==1u)
		return BigInt();

	BigInt one_mod(1);
	one_mod%=mod;
	if(one_mod.sign()<0)
		one_mod+=mod;
	if(exp.is_zero())
		return one_mod;
	if(base.is_zero())
		return BigInt();

	if(exp.fits_u64()){
		const std::uint64_t e_u64=exp.to_u64();

		if(mod.limbs_.size()==1){
			const std::uint64_t m=mod.limbs_[0];
			std::uint64_t base_u=0;
			if(!base.limbs_.empty())
				base_u=base.limbs_[0];

			if((m&1u)!=0u){
				const detail::Mont64Ctx c=
					detail::mk_m64ctx(m);
				std::uint64_t b=detail::mto_u64(base_u,c);
				std::uint64_t r=detail::mto_u64(1,c);
				std::uint64_t e=e_u64;
				if(e==65537u){
					std::uint64_t t=b;
					for(unsigned i=0;i<16;++i)
						t=detail::mmul_u64(t,t,c);
					r=detail::mmul_u64(t,b,c);
				}else{
					while(e!=0){
						if((e&1u)!=0u)
							r=detail::mmul_u64(r,b,c);
						e>>=1;
						if(e!=0)
							b=detail::mmul_u64(b,b,c);
					}
				}
				return BigInt::from_u64(detail::mfr_u64(r,c));
			}

			std::uint64_t b=base_u%m;
			std::uint64_t r=1u%m;
			std::uint64_t e=e_u64;
			if(e==65537u){
				std::uint64_t t=b;
				for(unsigned i=0;i<16;++i)
					t=detail::mulmod64(t,t,m);
				r=detail::mulmod64(t,b,m);
			}else{
				while(e!=0){
					if((e&1u)!=0u)
						r=detail::mulmod64(r,b,m);
					e>>=1;
					if(e!=0)
						b=detail::mulmod64(b,b,m);
				}
			}
			return BigInt::from_u64(r);
		}

		if((mod.limbs_[0]&1u)!=0u){
			detail::MontCtx ctx=detail::mk_mctx(mod.limbs_);
			detail::limbs_t base_bar;
			detail::mont_to(base.limbs_,ctx,base_bar);

			detail::limbs_t res_bar;
			detail::limbs_t tmp;
			std::uint64_t e=e_u64;
			if(e==65537u){
				res_bar=base_bar;
				for(unsigned i=0;i<16;++i){
					detail::mont_mul(res_bar,res_bar,ctx,tmp);
					res_bar.swap(tmp);
				}
				detail::mont_mul(res_bar,base_bar,ctx,tmp);
				res_bar.swap(tmp);
			}else{
				detail::limbs_t one{1};
				detail::limbs_t one_bar;
				detail::mont_to(one,ctx,one_bar);
				res_bar=one_bar;
				while(e!=0){
					if((e&1u)!=0u){
						detail::mont_mul(res_bar,base_bar,ctx,tmp);
						res_bar.swap(tmp);
					}
					e>>=1;
					if(e!=0){
						detail::mont_mul(base_bar,base_bar,ctx,tmp);
						base_bar.swap(tmp);
					}
				}
			}

			detail::limbs_t out;
			detail::mont_fr(res_bar,ctx,out);
			return BigInt::from_raw(out.empty()?0:1,std::move(out));
		}

		BigInt result=one_mod;
		BigInt b=base;
		detail::MulModScr sc;
		std::uint64_t e=e_u64;
		if(e==65537u){
			BigInt t=b;
			for(unsigned i=0;i<16;++i){
				detail::sqrmod_ip(t.limbs_,mod.limbs_,sc);
			}
			detail::mulmod_ip(t.limbs_,b.limbs_,mod.limbs_,sc);
			t.sign_=t.limbs_.empty()?0:1;
			return t;
		}

		while(e!=0){
			if((e&1u)!=0u){
				detail::mulmod_ip(result.limbs_,b.limbs_,mod.limbs_,sc);
				result.sign_=result.limbs_.empty()?0:1;
			}
			e>>=1;
			if(e!=0){
				detail::sqrmod_ip(b.limbs_,mod.limbs_,sc);
				b.sign_=b.limbs_.empty()?0:1;
			}
		}
		return result;
	}

	const std::size_t exp_bits=detail::bit_length(exp.limbs_);

	
	if(mod.limbs_.size()==1){
		const std::uint64_t m=mod.limbs_[0];
		std::uint64_t base_u=0;
		if(!base.limbs_.empty())
			base_u=base.limbs_[0];

		if((m&1u)!=0u){
			const detail::Mont64Ctx c=detail::mk_m64ctx(m);
			std::uint64_t b=detail::mto_u64(base_u,c);
			std::uint64_t r=detail::mto_u64(1,c);
			for(std::size_t bit=0;bit<exp_bits;++bit){
				if(detail::test_bit(exp.limbs_,bit)){
					r=detail::mmul_u64(r,b,c);
				}
				if(bit+1<exp_bits){
					b=detail::mmul_u64(b,b,c);
				}
			}
			return BigInt::from_u64(detail::mfr_u64(r,c));
		}

		std::uint64_t b=base_u%m;
		std::uint64_t r=1u%m;
		for(std::size_t bit=0;bit<exp_bits;++bit){
			if(detail::test_bit(exp.limbs_,bit)){
				r=detail::mulmod64(r,b,m);
			}
			if(bit+1<exp_bits){
				b=detail::mulmod64(b,b,m);
			}
		}
		return BigInt::from_u64(r);
	}

	if((mod.limbs_[0]&1u)!=0u){
		detail::MontCtx ctx=detail::mk_mctx(mod.limbs_);
		detail::limbs_t one{1};
		detail::limbs_t base_bar;
		detail::limbs_t one_bar;
		detail::mont_to(base.limbs_,ctx,base_bar);
		detail::mont_to(one,ctx,one_bar);
		detail::limbs_t res_bar=one_bar;

		if(exp_bits!=0){
			const unsigned w=detail::mpw_bits(exp_bits);
			const std::size_t table_size=std::size_t(1)<<(w-1);
			std::vector<detail::limbs_t> odd_pows(table_size);
			odd_pows[0]=base_bar;

			detail::limbs_t base_sq;
			detail::mont_mul(base_bar,base_bar,ctx,base_sq);
			for(std::size_t i=1;i<table_size;++i){
				detail::mont_mul(odd_pows[i-1],base_sq,ctx,odd_pows[i]);
			}

			detail::limbs_t tmp;
			std::size_t pos=exp_bits;
			while(pos!=0){
				if(!detail::test_bit(exp.limbs_,pos-1)){
					detail::mont_mul(res_bar,res_bar,ctx,tmp);
					res_bar.swap(tmp);
					--pos;
					continue;
				}

				unsigned win_len=
					static_cast<unsigned>(std::min<std::size_t>(w,pos));
				std::size_t start=pos-win_len;
				std::uint32_t win=
					detail::extbit32(exp.limbs_,start,win_len);
				while((win&1u)==0u){
					win>>=1;
					++start;
					--win_len;
				}

				for(unsigned i=0;i<win_len;++i){
					detail::mont_mul(res_bar,res_bar,ctx,tmp);
					res_bar.swap(tmp);
				}
				const std::size_t idx=(win-1u)>>1;
				detail::mont_mul(res_bar,odd_pows[idx],ctx,tmp);
				res_bar.swap(tmp);
				pos=start;
			}
		}

		detail::limbs_t out;
		detail::mont_fr(res_bar,ctx,out);
		return BigInt::from_raw(out.empty()?0:1,std::move(out));
	}

	detail::BarModCtx bctx=detail::mk_bctx(mod.limbs_);
	detail::limbs_t res{1};
	detail::bar_redip(res,bctx);

	if(exp_bits!=0){
		const unsigned w=detail::mpw_bits(exp_bits);
		const std::size_t table_size=std::size_t(1)<<(w-1);
		std::vector<detail::limbs_t> odd_pows(table_size);
		odd_pows[0]=base.limbs_;

		detail::limbs_t base_sq;
		detail::bar_mulm(base.limbs_,base.limbs_,bctx,base_sq);
		for(std::size_t i=1;i<table_size;++i){
			detail::bar_mulm(odd_pows[i-1],base_sq,bctx,odd_pows[i]);
		}

		detail::limbs_t tmp;
		std::size_t pos=exp_bits;
		while(pos!=0){
			if(!detail::test_bit(exp.limbs_,pos-1)){
				detail::bar_mulm(res,res,bctx,tmp);
				res.swap(tmp);
				--pos;
				continue;
			}

			unsigned win_len=
				static_cast<unsigned>(std::min<std::size_t>(w,pos));
			std::size_t start=pos-win_len;
			std::uint32_t win=
				detail::extbit32(exp.limbs_,start,win_len);
			while((win&1u)==0u){
				win>>=1;
				++start;
				--win_len;
			}

			for(unsigned i=0;i<win_len;++i){
				detail::bar_mulm(res,res,bctx,tmp);
				res.swap(tmp);
			}
			const std::size_t idx=(win-1u)>>1;
			detail::bar_mulm(res,odd_pows[idx],bctx,tmp);
			res.swap(tmp);
			pos=start;
		}
	}
	return BigInt::from_raw(res.empty()?0:1,std::move(res));
}

inline BigInt invmod(const BigInt&a,const BigInt&mod){ return invert(a,mod); }

inline BigInt gcdext(BigInt*s,BigInt*t,const BigInt&a,const BigInt&b){
	const ExtGcdRes eg=extgcd(a,b);
	if(s!=nullptr)
		*s=eg.x;
	if(t!=nullptr)
		*t=eg.y;
	return eg.g;
}

inline bool invert(BigInt*rop,const BigInt&a,const BigInt&mod){
	if(mod.sign()<=0){
		detail::throw_dom("invert: modulus must be positive");
	}
	BigInt old_r=mod;
	BigInt r;
	if(a.sign_>=0&&detail::cmp_abs(a.limbs_,mod.limbs_)<0){
		r=a;
	}else{
		r=a%mod;
		if(r.sign()<0)
			r+=mod;
	}
	BigInt old_t(0);
	BigInt t(1);
	detail::limbs_t scaled;

	auto submul_limb=[&](const BigInt&lhs,const BigInt&rhs,
						 std::uint64_t mul) -> BigInt{
		if(mul==0||rhs.sign_==0)
			return lhs;
		if(mul==1){
			scaled=rhs.limbs_;
		}else{
			detail::mulbl_in(rhs.limbs_,mul,scaled);
		}

		BigInt out;
		if(lhs.sign_==0){
			out.sign_=-rhs.sign_;
			out.limbs_=std::move(scaled);
			return out;
		}
		if(lhs.sign_!=rhs.sign_){
			out.sign_=lhs.sign_;
			detail::add_abs_to(out.limbs_,lhs.limbs_,scaled);
			return out;
		}

		const int cmp=detail::cmp_abs(lhs.limbs_,scaled);
		if(cmp==0)
			return BigInt();
		if(cmp>0){
			out.sign_=lhs.sign_;
			detail::sub_abs_to(out.limbs_,lhs.limbs_,scaled);
		}else{
			out.sign_=-lhs.sign_;
			detail::sub_abs_to(out.limbs_,scaled,lhs.limbs_);
		}
		return out;
	};

	while(!r.is_zero()){
		auto qr=divmod(old_r,r);
		const BigInt&q=qr.first;

		old_r=std::move(r);
		r=std::move(qr.second);

		BigInt next_t;
		if(q.limbs_.size()==1){
			const std::uint64_t ql=q.limbs_[0];
			if(ql==1){
				next_t=old_t-t;
			}else{
				next_t=submul_limb(old_t,t,ql);
			}
		}else{
			next_t=old_t-q*t;
		}
		old_t=std::move(t);
		t=std::move(next_t);
	}

	if(old_r!=BigInt(1)){
		if(rop!=nullptr)
			*rop=BigInt();
		return false;
	}
	BigInt x;
	if(detail::cmp_abs(old_t.limbs_,mod.limbs_)<0){
		x=std::move(old_t);
	}else{
		x=old_t%mod;
	}
	if(x.sign()<0)
		x+=mod;
	if(rop!=nullptr)
		*rop=std::move(x);
	return true;
}

inline BigInt invert(const BigInt&a,const BigInt&mod){
	BigInt out;
	if(!invert(&out,a,mod)){
		detail::throw_dom("invert: inverse does not exist");
	}
	return out;
}

namespace detail{

inline BigInt mod_pos(BigInt x,const BigInt&mod){
	x%=mod;
	if(x.sign()<0)
		x+=mod;
	return x;
}

inline int kron_u64(int as,std::uint64_t amag,
					int bs,std::uint64_t bmag) noexcept{
	if(bmag==0)
		return amag==1u?1:0;
	if(amag==0)
		return bmag==1u?1:0;

	int res=1;
	if(bs<0&&as<0)
		res=-res;

	const unsigned b_twos=std::countr_zero(bmag);
	if(b_twos!=0u){
		if((amag&1u)==0u)
			return 0;
		if((b_twos&1u)!=0u){
			const std::uint64_t r8=amag&7u;
			if(r8==3u||r8==5u)
				res=-res;
		}
		bmag>>=b_twos;
	}

	if(bmag==1u)
		return res;
	if(as<0&&(bmag&3u)==3u)
		res=-res;

	std::uint64_t a=amag%bmag;
	while(a!=0u){
		const unsigned twos=std::countr_zero(a);
		if(twos!=0u){
			a>>=twos;
			if((twos&1u)!=0u){
				const std::uint64_t r8=bmag&7u;
				if(r8==3u||r8==5u)
					res=-res;
			}
		}
		if((a&3u)==3u&&(bmag&3u)==3u)
			res=-res;
		const std::uint64_t t=a;
		a=bmag%t;
		bmag=t;
	}

	return bmag==1u?res:0;
}

inline bool sqrtmod_u64(std::uint64_t*root,
						std::uint64_t n,std::uint64_t p){
	MINI_MP_ASSERT(root!=nullptr);
	MINI_MP_ASSERT(p>2u&&(p&1u)!=0u);
	n%=p;
	if(n==0u){
		*root=0;
		return true;
	}
	auto set_root=[&](std::uint64_t r)->bool{
		r%=p;
		if(mulmod64(r,r,p)!=n)
			return false;
		const std::uint64_t alt=p-r;
		*root=(alt<r)?alt:r;
		return true;
	};
	if(p<600u){
		std::uint64_t sq=0;
		for(std::uint64_t r=0;r<=p/2u;++r){
			if(sq==n)
				return set_root(r);
			sq+=2u*r+1u;
			sq%=p;
		}
		return false;
	}
	if(kron_u64(1,n,1,p)!=1)
		return false;

	if((p&3u)==3u)
		return set_root(powmod64(n,p/4u+1u,p));
	if((p&7u)==5u){
		std::uint64_t r=powmod64(n,p/8u+1u,p);
		if(mulmod64(r,r,p)!=n){
			const std::uint64_t f=powmod64(2u,(p-1u)/4u,p);
			r=mulmod64(r,f,p);
		}
		return set_root(r);
	}

	std::uint64_t q=p-1u;
	unsigned s=std::countr_zero(q);
	q>>=s;
	std::uint64_t z=(p&7u)==1u?3u:2u;
	while(z<p&&kron_u64(1,z,1,p)!=-1)
		++z;
	if(z>=p)
		return false;

	std::uint64_t c=powmod64(z,q,p);
	std::uint64_t x=powmod64(n,(q+1u)/2u,p);
	std::uint64_t t=powmod64(n,q,p);
	unsigned m=s;

	while(t!=1u){
		std::uint64_t tt=t;
		unsigned i=0;
		for(i=1;i<m;++i){
			tt=mulmod64(tt,tt,p);
			if(tt==1u)
				break;
		}
		if(i==m)
			return false;

		std::uint64_t b=c;
		for(unsigned j=0;j<m-i-1u;++j)
			b=mulmod64(b,b,p);
		x=mulmod64(x,b,p);
		c=mulmod64(b,b,p);
		t=mulmod64(t,c,p);
		m=i;
	}

	return set_root(x);
}

inline int kronecker_limbs(int as,limbs_t a,int bs,limbs_t b){
	if(b.empty())
		return one_abs(a)?1:0;
	if(a.empty())
		return one_abs(b)?1:0;
	if(((a[0]|b[0])&1u)==0u)
		return 0;

	int res=1;
	if(bs<0&&as<0)
		res=-res;

	const std::size_t b_twos=ctz(b);
	if(b_twos!=0u){
		if((a[0]&1u)==0u)
			return 0;
		if((b_twos&1u)!=0u){
			const std::uint32_t r8=mod_u32_abs(a,8u);
			if(r8==3u||r8==5u)
				res=-res;
		}
		shr_ip(b,b_twos);
	}

	if(one_abs(b))
		return res;
	if(as<0&&mod_u32_abs(b,4u)==3u)
		res=-res;

	if(cmp_abs(a,b)>=0)
		mod_abs_ip(a,b);
	while(!a.empty()){
		const std::size_t a_twos=ctz(a);
		if(a_twos!=0u){
			shr_ip(a,a_twos);
			if((a_twos&1u)!=0u){
				const std::uint32_t r8=mod_u32_abs(b,8u);
				if(r8==3u||r8==5u)
					res=-res;
			}
		}

		if(mod_u32_abs(a,4u)==3u&&mod_u32_abs(b,4u)==3u)
			res=-res;
		a.swap(b);
		if(cmp_abs(a,b)>=0)
			mod_abs_ip(a,b);
	}
	return one_abs(b)?res:0;
}

}

inline int kronecker(const BigInt&a_in,const BigInt&b_in){
	if(a_in.limbs_.size()<=1u&&b_in.limbs_.size()<=1u){
		const std::uint64_t av=a_in.limbs_.empty()?0u:a_in.limbs_[0];
		const std::uint64_t bv=b_in.limbs_.empty()?0u:b_in.limbs_[0];
		return detail::kron_u64(a_in.sign(),av,b_in.sign(),bv);
	}
	if(b_in.limbs_.size()==1u){
		const std::uint64_t bm=b_in.limbs_[0];
		if(bm==1u)
			return (b_in.sign()<0&&a_in.sign()<0)?-1:1;
		const std::uint64_t ar=detail::mod_limb(a_in.limbs_,bm);
		return detail::kron_u64(a_in.sign(),ar,b_in.sign(),bm);
	}
	return detail::kronecker_limbs(a_in.sign(),a_in.limbs_,
								   b_in.sign(),b_in.limbs_);
}

inline int jacobi(const BigInt&a,const BigInt&b){
	if(b.sign()<=0||b.is_even()){
		detail::throw_dom(
			"jacobi: denominator must be positive and odd");
	}
	return kronecker(a,b);
}

struct CrtRes{
	bool ok=false;
	BigInt value;
	BigInt modulus;
};

inline CrtRes crt_solve(
	const std::vector<std::pair<BigInt,BigInt>>&eqs){
	auto inv_mod_u64=[](std::uint64_t a,std::uint64_t m)->std::uint64_t{
		return detail::invmod64_coprime(a,m);
	};

	bool small=true;
	std::uint64_t sx=0;
	std::uint64_t smod=1;
	for(const auto&eq : eqs){
		const BigInt&ai=eq.first;
		const BigInt&mi=eq.second;
		if(mi.sign()<=0){
			detail::throw_dom(
				"crt_solve: moduli must be positive");
		}
		if(!mi.fits_u64()||
		   mi.to_u64()>static_cast<std::uint64_t>(
			   std::numeric_limits<std::int64_t>::max())){
			small=false;
			break;
		}
		const std::uint64_t m=mi.to_u64();
		const std::uint64_t r=(ai.sign()>=0&&ai.fits_u64())?
			(ai.to_u64()%m):detail::mod_pos(ai,mi).to_u64();
		const std::uint64_t g=std::gcd(smod,m);
		const std::uint64_t diff=(r+m-(sx%m))%m;
		if(diff%g!=0)
			return {false,BigInt(),BigInt()};
		const std::uint64_t mg=m/g;
		if(mg==1u)
			continue;
		if(smod>std::numeric_limits<std::uint64_t>::max()/mg){
			small=false;
			break;
		}
		const std::uint64_t rhs=diff/g;
		const std::uint64_t a=(smod/g)%mg;
		const std::uint64_t inv=inv_mod_u64(a,mg);
		const std::uint64_t t=detail::mulmod64(rhs,inv,mg);
		const std::uint64_t new_mod=smod*mg;
		sx=detail::addmod64_red(detail::mulmod64(smod,t,new_mod),
								 sx,new_mod);
		smod=new_mod;
	}
	if(small)
		return {true,BigInt::from_u64(sx),BigInt::from_u64(smod)};

	BigInt x(0);
	BigInt mod(1);

	for(const auto&eq : eqs){
		const BigInt&ai=eq.first;
		const BigInt&mi=eq.second;
		if(mi.sign()<=0){
			detail::throw_dom(
				"crt_solve: moduli must be positive");
		}

		const BigInt r=(ai.sign()>=0&&ai<mi)?ai:detail::mod_pos(ai,mi);
		const BigInt g=gcd(mod,mi);
		const BigInt diff=r-x;
		BigInt mi_g;
		BigInt mod_g;
		BigInt rhs;
		if(g.is_one()){
			mi_g=mi;
			mod_g=mod;
			rhs=diff;
		}else{
			if(!detail::mod_pos(diff,g).is_zero()){
				return {false,BigInt(),BigInt()};
			}
			mi_g=divexact(mi,g);
			if(mi_g.is_one()){
				x=detail::mod_pos(std::move(x),mod);
				continue;
			}
			mod_g=divexact(mod,g);
			rhs=divexact(diff,g);
		}
		const BigInt inv=invert(mod_g,mi_g);
		const BigInt t=detail::mod_pos(rhs*inv,mi_g);
		x+=mod*t;
		mod*=mi_g;
		if(x.sign()<0||x>=mod)
			x=detail::mod_pos(std::move(x),mod);
	}

	return {true,std::move(x),std::move(mod)};
}

inline bool crt_solve(BigInt*value,BigInt*modulus,
					  const std::vector<std::pair<BigInt,BigInt>>&eqs){
	CrtRes res=crt_solve(eqs);
	if(!res.ok)
		return false;
	if(value!=nullptr)
		*value=std::move(res.value);
	if(modulus!=nullptr)
		*modulus=std::move(res.modulus);
	return true;
}

inline std::pair<BigInt,BigInt> crt(
	const std::vector<std::pair<BigInt,BigInt>>&eqs){
	CrtRes res=crt_solve(eqs);
	if(!res.ok)
		detail::throw_dom("crt: inconsistent congruences");
	return {std::move(res.value),std::move(res.modulus)};
}

inline std::pair<BigInt,BigInt> dvm_trnc(const BigInt&a,const BigInt&b){
	return divmod(a,b);
}

inline std::pair<BigInt,BigInt> dvm_floor(const BigInt&a,const BigInt&b){
	auto qr=divmod(a,b);
	if(!qr.second.is_zero()&&qr.second.sign()!=b.sign()){
		qr.first-=BigInt(1);
		qr.second+=b;
	}
	return qr;
}

inline std::pair<BigInt,BigInt> dvm_ceil(const BigInt&a,const BigInt&b){
	auto qr=divmod(a,b);
	if(!qr.second.is_zero()&&qr.second.sign()==b.sign()){
		qr.first+=BigInt(1);
		qr.second-=b;
	}
	return qr;
}

inline std::pair<BigInt,BigInt> tdiv_qr(const BigInt&a,const BigInt&b){
	return dvm_trnc(a,b);
}

inline std::pair<BigInt,BigInt> fdiv_qr(const BigInt&a,const BigInt&b){
	return dvm_floor(a,b);
}

inline std::pair<BigInt,BigInt> cdiv_qr(const BigInt&a,const BigInt&b){
	return dvm_ceil(a,b);
}

inline BigInt tdiv_q(const BigInt&a,const BigInt&b){ return divk_q(a,b); }
inline BigInt fdiv_q(const BigInt&a,const BigInt&b){
	return dvm_floor(a,b).first;
}
inline BigInt cdiv_q(const BigInt&a,const BigInt&b){
	return dvm_ceil(a,b).first;
}
inline BigInt tdiv_r(const BigInt&a,const BigInt&b){ return divk_r(a,b); }
inline BigInt fdiv_r(const BigInt&a,const BigInt&b){
	return dvm_floor(a,b).second;
}
inline BigInt cdiv_r(const BigInt&a,const BigInt&b){
	return dvm_ceil(a,b).second;
}

inline BigInt divexact(const BigInt&a,const BigInt&b){
	if(b.is_zero()){
		detail::throw_dom("divexact: division by zero");
	}
	if(a.is_zero())
		return BigInt();
	if(detail::pow2_abs(b.limbs_)){
		const std::size_t sh=detail::pow2_exp_abs(b.limbs_);
		if(!detail::low_bits_abs(a.limbs_,sh).empty()){
			detail::throw_dom(
				"divexact: dividend not divisible by divisor");
		}
		detail::limbs_t q=detail::shr_bits(a.limbs_,sh);
		const int sign=q.empty()?0:((a.sign_==b.sign_)?1:-1);
		return BigInt::from_raw(sign,std::move(q));
	}
	if(b.limbs_.size()==1){
		detail::limbs_t q=a.limbs_;
		if(!detail::divex_lb(q,b.limbs_[0])){
			detail::throw_dom(
				"divexact: dividend not divisible by divisor");
		}
		const int sign=q.empty()?0:((a.sign_==b.sign_)?1:-1);
		return BigInt::from_raw(sign,std::move(q));
	}
	const std::size_t btz=detail::ctz(b.limbs_);
	if(btz!=0){
		if(detail::ctz(a.limbs_)<btz){
			detail::throw_dom(
				"divexact: dividend not divisible by divisor");
		}
		detail::limbs_t amag=detail::shr_bits(a.limbs_,btz);
		detail::limbs_t bmag=detail::shr_bits(b.limbs_,btz);
		BigInt aa=BigInt::from_raw(a.sign_,std::move(amag));
		BigInt bb=BigInt::from_raw(b.sign_,std::move(bmag));
		return divexact(aa,bb);
	}
	{
		detail::limbs_t q;
		if(!detail::divex_odd(a.limbs_,b.limbs_,q)){
			detail::throw_dom(
				"divexact: dividend not divisible by divisor");
		}
		const int sign=q.empty()?0:((a.sign_==b.sign_)?1:-1);
		return BigInt::from_raw(sign,std::move(q));
	}
}

inline std::pair<BigInt,BigInt> sqrtrem_split(const BigInt&n,
											  std::size_t nbits){
	const std::size_t k=(nbits+3u)/4u;
	const BigInt hi=n>>(2u*k);
	const BigInt mid=BigInt::from_raw(1,
		detail::bit_slice_abs(n.limbs_,k,k));
	const BigInt lo=BigInt::from_raw(1,
		detail::low_bits_abs(n.limbs_,k));
	auto sr=sqrtrem(hi);
	BigInt s=std::move(sr.first);
	BigInt r=std::move(sr.second);
	const BigInt den=s<<1;
	BigInt num=(r<<k)+mid;
	std::pair<detail::limbs_t,detail::limbs_t> qr;
	if(den.limbs_.size()==1){
		qr=detail::dvmk_absl(num.limbs_,den.limbs_);
	}else{
		qr=detail::use_bzdiv(num.limbs_.size(),den.limbs_.size())
			?detail::dvmbz_abs(num.limbs_,den.limbs_)
			:detail::dvmk_absl(num.limbs_,den.limbs_);
	}
	BigInt q=BigInt::from_raw(qr.first.empty()?0:1,std::move(qr.first));
	BigInt rh=BigInt::from_raw(qr.second.empty()?0:1,std::move(qr.second));
	if(q.bit_length()>k){
		const BigInt base=BigInt(1)<<k;
		q=base-BigInt(1);
		rh=num-q*den;
	}
	BigInt root=(s<<k)+q;
	BigInt rem=(rh<<k)+lo-sqr_disp_noat(q);
	while(rem.sign()<0){
		root-=BigInt(1);
		rem+=(root<<1)+BigInt(1);
	}
	return {std::move(root),std::move(rem)};
}

inline std::uint64_t sqrt_u128_floor(detail::u128 n) noexcept{
	long double v=std::ldexp(static_cast<long double>(n.hi),64)+
				  static_cast<long double>(n.lo);
	long double s=std::sqrt(v);
	std::uint64_t r=0;
	if(s>=std::ldexp(1.0L,64)){
		r=std::numeric_limits<std::uint64_t>::max();
	}else{
		r=static_cast<std::uint64_t>(s);
	}
	detail::u128 sq=detail::mul_u64(r,r);
	while(detail::cmp_u128(sq,n)>0){
		--r;
		sq=detail::mul_u64(r,r);
	}
	while(r!=std::numeric_limits<std::uint64_t>::max()){
		const std::uint64_t nr=r+1u;
		const detail::u128 nsq=detail::mul_u64(nr,nr);
		if(detail::cmp_u128(nsq,n)>0)
			break;
		r=nr;
	}
	return r;
}

inline std::pair<BigInt,BigInt> sqrtrem_u128(const BigInt&n){
	detail::u128 v{n.limbs_.empty()?0u:n.limbs_[0],
				   n.limbs_.size()>1u?n.limbs_[1]:0u};
	const std::uint64_t root=sqrt_u128_floor(v);
	const detail::u128 sq=detail::mul_u64(root,root);
	const detail::u128 rem=detail::sub_u128(v,sq);
	detail::limbs_t rmag;
	if(root!=0)
		rmag.push_back(root);
	detail::limbs_t mmag;
	detail::set_u128(mmag,rem);
	return {BigInt::from_raw(rmag.empty()?0:1,std::move(rmag)),
			BigInt::from_raw(mmag.empty()?0:1,std::move(mmag))};
}

inline std::pair<BigInt,BigInt> sqrtrem_u256(const BigInt&n){
	detail::limbs_t hi;
	if(n.limbs_.size()>2u){
		hi.assign(n.limbs_.begin()+2,n.limbs_.end());
		detail::trim_lz(hi);
	}
	const BigInt hv=BigInt::from_raw(hi.empty()?0:1,std::move(hi));
	auto sr=sqrtrem_u128(hv);
	BigInt s=std::move(sr.first);
	BigInt r=std::move(sr.second);
	detail::limbs_t den=s.limbs_;
	detail::shl_ip(den,std::size_t(1));
	detail::limbs_t num=r.limbs_;
	detail::shl_ip(num,std::size_t(64));
	if(n.limbs_.size()>1u)
		detail::add_lb_at(num,n.limbs_[1],0);
	std::pair<detail::limbs_t,detail::limbs_t> qr;
	if(den.size()==1)
		qr=detail::dvmk_absl(num,den);
	else
		qr=detail::dvmk_absl(num,den);
	BigInt q=BigInt::from_raw(qr.first.empty()?0:1,std::move(qr.first));
	BigInt rh=BigInt::from_raw(qr.second.empty()?0:1,std::move(qr.second));
	if(q.bit_length()>64u){
		return sqrtrem_split(n,n.bit_length());
	}
	BigInt root=(s<<64)+q;
	detail::limbs_t lo;
	if(!n.limbs_.empty())
		lo.push_back(n.limbs_[0]);
	BigInt rem=(rh<<64)+BigInt::from_raw(lo.empty()?0:1,std::move(lo))-
		sqr_disp_noat(q);
	while(rem.sign()<0){
		root-=BigInt(1);
		rem+=(root<<1)+BigInt(1);
	}
	return {std::move(root),std::move(rem)};
}

inline std::pair<BigInt,BigInt> sqrtrem(const BigInt&n){
	if(n.sign()<0){
		detail::throw_dom("sqrtrem: negative input");
	}
	if(n.is_zero())
		return {BigInt(),BigInt()};
	if(n==BigInt(1))
		return {BigInt(1),BigInt()};

	const std::size_t nbits=n.bit_length();
	if(nbits<=128u)
		return sqrtrem_u128(n);
	if(nbits<=256u)
		return sqrtrem_u256(n);
	return sqrtrem_split(n,nbits);
}

inline BigInt isqrt(const BigInt&n){ return sqrtrem(n).first; }

inline bool pow_wlim(const BigInt&base,std::uint64_t exp,
						   const BigInt&limit,BigInt*out){
	if(exp==0){
		if(out!=nullptr)
			*out=BigInt(1);
		return true;
	}

	BigInt result(1);
	BigInt b=base;
	const BigInt lim1=limit+BigInt(1);
	const std::size_t limit_bits=limit.bit_length();
	const std::size_t lim1_bit=limit_bits+1u;
	std::uint64_t e=exp;

	auto gt_lim_bt=[&](const BigInt&x,const BigInt&y) -> bool{
		if(x.is_zero()||y.is_zero())
			return false;
		const std::size_t xb=x.bit_length();
		const std::size_t yb=y.bit_length();
		if(xb>lim1_bit||yb>lim1_bit)
			return true;
		return xb>(lim1_bit-yb);
	};

	while(e!=0){
		if((e&1u)!=0u){
			if(gt_lim_bt(result,b)){
				if(out!=nullptr)
					*out=lim1;
				return false;
			}
			result*=b;
			if(result>limit){
				if(out!=nullptr)
					*out=lim1;
				return false;
			}
		}
		e>>=1;
		if(e!=0){
			if(b>limit){
				b=lim1;
			}else{
				if(gt_lim_bt(b,b)){
					b=lim1;
				}else{
					b=sqr_disp(b);
					if(b>limit)
						b=lim1;
				}
			}
		}
	}

	if(out!=nullptr)
		*out=std::move(result);
	return true;
}

inline std::pair<BigInt,BigInt> rootrem(const BigInt&n,std::uint32_t k){
	if(k==0){
		detail::throw_dom("rootrem: zeroth root is undefined");
	}
	if(n.sign()<0){
		detail::throw_dom("rootrem: negative input");
	}
	if(n.is_zero())
		return {BigInt(),BigInt()};
	if(k==1)
		return {n,BigInt()};
	if(k==2)
		return sqrtrem(n);

	if(n==BigInt(1))
		return {BigInt(1),BigInt()};
	if(k>n.bit_length())
		return {BigInt(1),n-BigInt(1)};

	const std::uint64_t ku=static_cast<std::uint64_t>(k);
	const BigInt k_big=BigInt::from_u64(ku);
	const BigInt km1_big=BigInt::from_u64(ku-1u);

	const std::size_t init_bits=(n.bit_length()+k-1u)/k;
	BigInt x=BigInt(1)<<init_bits; 

	
	
	for(;;){
		BigInt x_pow_km1;
		BigInt term(0);
		if(pow_wlim(x,ku-1u,n,&x_pow_km1)){
			if(!x_pow_km1.is_zero()){
				term=n/x_pow_km1;
			}
		}
		const BigInt y=(km1_big*x+term)/k_big;
		if(y>=x)
			break;
		x=y;
	}

	BigInt xk;
	(void)pow_wlim(x,ku,n,&xk);
	while(xk>n){
		x-=BigInt(1);
		(void)pow_wlim(x,ku,n,&xk);
	}
	for(;;){
		const BigInt xp1=x+BigInt(1);
		BigInt xp1k;
		(void)pow_wlim(xp1,ku,n,&xp1k);
		if(xp1k>n)
			break;
		x=xp1;
		xk=xp1k;
	}

	return {x,n-xk};
}

inline BigInt iroot(const BigInt&n,std::uint32_t k){
	return rootrem(n,k).first;
}

inline bool is_ppow(const BigInt&n){
	if(n.is_zero()||n==BigInt(1)||n==BigInt(-1))
		return true;
	const BigInt abs_n=n.abs();
	const std::uint32_t max_k=static_cast<std::uint32_t>(std::min<std::size_t>(
		abs_n.bit_length(),std::numeric_limits<std::uint32_t>::max()));
	auto is_p_exp=[](std::uint32_t k) -> bool{
		if(k<2)
			return false;
		if((k&1u)==0u)
			return k==2;
		for(std::uint32_t d=3;static_cast<std::uint64_t>(d)*d<=k;d+=2){
			if((k%d)==0u)
				return false;
		}
		return true;
	};
	for(std::uint32_t k=2;k<=max_k;++k){
		if(!is_p_exp(k))
			continue;
		if(n.sign()<0&&(k%2u)==0u)
			continue;
		if(rootrem(abs_n,k).second.is_zero())
			return true;
	}
	return false;
}

inline bool is_square(const BigInt&n){
	if(n.sign()<0)
		return false;
	const std::uint32_t r256=n.limbs_.empty()
		?0u
		:static_cast<std::uint32_t>(n.limbs_[0]&255u);
	const std::uint32_t r4095=n.mod_u32(4095u);
	if(!detail::sqr_res_ok<256u>(r256)||
	   !detail::sqr_res_ok<63u>(r4095%63u)||
	   !detail::sqr_res_ok<65u>(r4095%65u)){
		return false;
	}
	const BigInt r=isqrt(n);
	return sqr_disp_noat(r)==n;
}

inline bool is_strong_probabprime(const BigInt&n,const BigInt&base){
	if(n.sign()<=0)
		return false;
	if(n==BigInt(2)||n==BigInt(3))
		return true;
	if(n<BigInt(2)||n.is_even())
		return false;
	BigInt a=detail::mod_pos(base,n);
	if(a.is_zero())
		return true;
	if(!gcd(a,n).is_one())
		return false;
	BigInt d=n-BigInt(1);
	const std::size_t s=d.ctz();
	d>>=s;
	BigInt x=modpow(a,d,n);
	const BigInt n_minus_1=n-BigInt(1);
	if(x==BigInt(1)||x==n_minus_1)
		return true;
	for(std::size_t r=1;r<s;++r){
		x=sqr_disp(x)%n;
		if(x==n_minus_1)
			return true;
	}
	return false;
}

namespace detail{

struct LucasSelfridgeParams{
	BigInt D;
	BigInt Q;
};

inline BigInt half_mod_odd(BigInt x,const BigInt&mod){
	x=mod_pos(std::move(x),mod);
	if(x.is_odd())
		x+=mod;
	x>>=1u;
	return x;
}

inline bool select_lucas_selfridge_params(LucasSelfridgeParams&out,
										  const BigInt&n){
	BigInt absD(5);
	bool positive=true;
	for(;;){
		BigInt D=positive?absD:-absD;
		const int j=jacobi(D,n);
		if(j==-1){
			BigInt Q=(BigInt(1)-D)>>2u;
			const BigInt g=gcd(Q.abs(),n);
			if(!g.is_one())
				return false;
			out.D=std::move(D);
			out.Q=std::move(Q);
			return true;
		}
		if(j==0){
			const BigInt g=gcd(absD,n);
			if(g>BigInt(1)&&g<n)
				return false;
		}
		absD+=BigInt(2);
		positive=!positive;
	}
}

inline void lucas_uv_p1(BigInt&U,BigInt&V,BigInt&Qk,
						const BigInt&k,const BigInt&Q,
						const BigInt&D,const BigInt&mod){
	if(k.is_zero()){
		U=BigInt();
		V=BigInt(2);
		Qk=BigInt(1);
		return;
	}
	U=BigInt(1);
	V=BigInt(1);
	Qk=mod_pos(Q,mod);
	const std::size_t bits=k.bit_length();
	for(std::size_t ii=bits-1u;ii>0;--ii){
		const std::size_t i=ii-1u;
		BigInt U2=mod_pos(U*V,mod);
		BigInt V2=mod_pos(V*V-BigInt(2)*Qk,mod);
		BigInt Q2=mod_pos(Qk*Qk,mod);
		if(k.test_bit(i)){
			const BigInt oldU=std::move(U2);
			const BigInt oldV=std::move(V2);
			U=half_mod_odd(oldU+oldV,mod);
			V=half_mod_odd(D*oldU+oldV,mod);
			Qk=mod_pos(Q2*Q,mod);
		}else{
			U=std::move(U2);
			V=std::move(V2);
			Qk=std::move(Q2);
		}
	}
}

} 

inline bool is_lucas_prp(const BigInt&n){
	if(n.sign()<=0)
		return false;
	if(n==BigInt(2))
		return true;
	if(n<BigInt(2)||n.is_even())
		return false;
	if(is_square(n))
		return false;
	detail::LucasSelfridgeParams p;
	if(!detail::select_lucas_selfridge_params(p,n))
		return false;
	BigInt d=n+BigInt(1);
	const std::size_t s=d.ctz();
	d>>=s;
	BigInt U,V,Qk;
	detail::lucas_uv_p1(U,V,Qk,d,p.Q,p.D,n);
	if(U.is_zero()||V.is_zero())
		return true;
	for(std::size_t r=1;r<s;++r){
		V=detail::mod_pos(V*V-BigInt(2)*Qk,n);
		if(V.is_zero())
			return true;
		Qk=detail::mod_pos(Qk*Qk,n);
	}
	return false;
}

inline bool is_bpsw_prp(const BigInt&n){
	if(n.sign()<=0)
		return false;
	if(n==BigInt(2)||n==BigInt(3))
		return true;
	if(n<BigInt(2)||n.is_even())
		return false;
	if(n.fits_u64())
		return detail::is_prp64(n.to_u64());
	auto hit_small=[&](std::uint32_t prod,
					   const std::uint32_t*ps,
					   std::size_t count)->bool{
		const std::uint32_t r=n.mod_u32(prod);
		for(std::size_t i=0;i<count;++i){
			if((r%ps[i])==0u)
				return true;
		}
		return false;
	};
	static constexpr std::uint32_t kG0[]={
		3u,5u,7u,11u,13u,17u,19u,23u,29u};
	static constexpr std::uint32_t kG1[]={
		31u,37u,41u,43u,47u};
	static constexpr std::uint32_t kG2[]={
		53u,59u,61u,67u,71u};
	static constexpr std::uint32_t kG3[]={
		73u,79u,83u,89u,97u};
	if(hit_small(3234846615u,kG0,sizeof(kG0)/sizeof(kG0[0]))||
	   hit_small(95041567u,kG1,sizeof(kG1)/sizeof(kG1[0]))||
	   hit_small(907383479u,kG2,sizeof(kG2)/sizeof(kG2[0]))||
	   hit_small(4132280413u,kG3,sizeof(kG3)/sizeof(kG3[0]))){
		return false;
	}
	return is_strong_probabprime(n,BigInt(2))&&is_lucas_prp(n);
}

inline int pr_prime(const BigInt&n,int rounds){
	if(n.sign()<=0)
		return 0;
	if(n==BigInt(2)||n==BigInt(3))
		return 2;
	if(n<BigInt(2)||n.is_even())
		return 0;

	if(n.fits_u64()){
		return detail::is_prp64(n.to_u64())?2:0;
	}else{
		auto hit_small=[&](std::uint32_t prod,
						   const std::uint32_t*ps,
						   std::size_t count)->bool{
			const std::uint32_t r=n.mod_u32(prod);
			for(std::size_t i=0;i<count;++i){
				if((r%ps[i])==0u)
					return true;
			}
			return false;
		};
		static constexpr std::uint32_t kG0[]={
			3u,5u,7u,11u,13u,17u,19u,23u,29u};
		static constexpr std::uint32_t kG1[]={
			31u,37u,41u,43u,47u};
		static constexpr std::uint32_t kG2[]={
			53u,59u,61u,67u,71u};
		static constexpr std::uint32_t kG3[]={
			73u,79u,83u,89u,97u};
		if(hit_small(3234846615u,kG0,sizeof(kG0)/sizeof(kG0[0]))||
		   hit_small(95041567u,kG1,sizeof(kG1)/sizeof(kG1[0]))||
		   hit_small(907383479u,kG2,sizeof(kG2)/sizeof(kG2[0]))||
		   hit_small(4132280413u,kG3,sizeof(kG3)/sizeof(kG3[0]))){
			return 0;
		}
	}

	if(!is_bpsw_prp(n))
		return 0;

	static constexpr std::uint64_t kBases[]={
		2ULL, 3ULL, 5ULL, 7ULL, 11ULL,13ULL,17ULL,19ULL,
		23ULL,29ULL,31ULL,37ULL,41ULL,43ULL,47ULL,53ULL};
	if(rounds<=24)
		return 1;
	const std::size_t max_rounds=sizeof(kBases)/sizeof(kBases[0]);
	const std::size_t extra=static_cast<std::size_t>(
		std::min<int>(rounds-24,static_cast<int>(max_rounds-1u)));
	for(std::size_t i=1;i<=extra;++i){
		BigInt a=BigInt::from_u64(kBases[i]);
		if(a>=n)
			continue;
		if(!is_strong_probabprime(n,a))
			return 0;
	}
	return 1;
}

inline BigInt next_prime(const BigInt&n,int rounds){
	if(n<BigInt(2))
		return BigInt(2);
	BigInt x=n+BigInt(1);
	if(x<=BigInt(2))
		return BigInt(2);
	if(x.is_even())
		x+=BigInt(1);
	while(pr_prime(x,rounds)==0){
		x+=BigInt(2);
	}
	return x;
}

inline bool sqrtmod(BigInt*root,const BigInt&a,const BigInt&mod){
	if(mod.sign()<=0){
		detail::throw_dom("sqrtmod: modulus must be positive");
	}
	if(mod.is_one()){
		if(root!=nullptr)
			*root=BigInt();
		return true;
	}

	BigInt n=detail::mod_pos(a,mod);
	if(n.is_zero()){
		if(root!=nullptr)
			*root=BigInt();
		return true;
	}
	if(mod==BigInt(2)){
		if(root!=nullptr)
			*root=n;
		return true;
	}
	if(mod.is_even()){
		detail::throw_dom(
			"sqrtmod: even moduli greater than 2 are unsupported");
	}
	if(mod.fits_u64()){
		std::uint64_t r=0;
		if(!detail::sqrtmod_u64(&r,n.to_u64(),mod.to_u64()))
			return false;
		if(root!=nullptr)
			*root=BigInt::from_u64(r);
		return true;
	}

	if(jacobi(n,mod)!=1)
		return false;

	auto checked=[&](BigInt r)->bool{
		r=detail::mod_pos(std::move(r),mod);
		BigInt chk=sqr_disp(r)%mod;
		if(chk.sign()<0)
			chk+=mod;
		if(chk!=n)
			return false;
		if(root!=nullptr){
			const BigInt other=mod-r;
			*root=(other<r)?other:r;
		}
		return true;
	};

	const std::uint32_t mod8=mod.mod_u32(8u);
	if((mod8&3u)==3u){
		return checked(modpow(n,(mod+BigInt(1))>>2,mod));
	}
	if(mod8==5u){
		BigInt r=modpow(n,(mod+BigInt(3))>>3,mod);
		BigInt chk=sqr_disp(r)%mod;
		if(chk.sign()<0)
			chk+=mod;
		if(chk!=n){
			const BigInt factor=modpow(BigInt(2),(mod-BigInt(1))>>2,mod);
			r=(r*factor)%mod;
		}
		return checked(std::move(r));
	}

	BigInt q=mod-BigInt(1);
	std::size_t s=q.ctz();
	q>>=s;
	BigInt z((mod.mod_u32(8u)==1u)?3:2);
	while(z<mod&&jacobi(z,mod)!=-1)
		z+=BigInt(1);
	if(z>=mod)
		return false;

	BigInt c=modpow(z,q,mod);
	BigInt x=modpow(n,(q+BigInt(1))>>1,mod);
	BigInt t=modpow(n,q,mod);
	std::size_t m=s;
	detail::MontCtx mctx=detail::mk_mctx(mod.limbs_);
	detail::limbs_t one_l{1};
	detail::limbs_t one_bar;
	detail::limbs_t c_bar;
	detail::limbs_t x_bar;
	detail::limbs_t t_bar;
	detail::limbs_t tmp;
	detail::mont_to(one_l,mctx,one_bar);
	detail::mont_to(c.limbs_,mctx,c_bar);
	detail::mont_to(x.limbs_,mctx,x_bar);
	detail::mont_to(t.limbs_,mctx,t_bar);

	while(detail::cmp_abs(t_bar,one_bar)!=0){
		detail::limbs_t tt=t_bar;
		std::size_t i=0;
		for(i=1;i<m;++i){
			detail::mont_mul(tt,tt,mctx,tmp);
			tt.swap(tmp);
			if(detail::cmp_abs(tt,one_bar)==0)
				break;
		}
		if(i==m)
			return false;

		detail::limbs_t b=c_bar;
		for(std::size_t j=0;j<m-i-1u;++j){
			detail::mont_mul(b,b,mctx,tmp);
			b.swap(tmp);
		}
		detail::mont_mul(x_bar,b,mctx,tmp);
		x_bar.swap(tmp);
		detail::mont_mul(b,b,mctx,tmp);
		c_bar.swap(tmp);
		detail::mont_mul(t_bar,c_bar,mctx,tmp);
		t_bar.swap(tmp);
		m=i;
	}

	detail::limbs_t out;
	detail::mont_fr(x_bar,mctx,out);
	return checked(BigInt::from_raw(out.empty()?0:1,std::move(out)));
}

inline BigInt sqrtmod(const BigInt&a,const BigInt&mod){
	BigInt out;
	if(!sqrtmod(&out,a,mod)){
		detail::throw_dom("sqrtmod: square root does not exist");
	}
	return out;
}

inline BigInt factorial(std::uint64_t n){
	detail::ensure_at();
	if(n>=detail::tun_fac_tree())
		return detail::factorial_tree(n,detail::tun_prod_leaf());
	return detail::factorial_loop(n);
}

inline BigInt binomial(std::uint64_t n,std::uint64_t k){
	if(k>n)
		return BigInt();
	k=std::min(k,n-k);
	if(k==0)
		return BigInt(1);
	detail::ensure_at();
	if(k>=detail::tun_binom_tree())
		return detail::binomial_tree(n,k,detail::tun_prod_leaf());
	return detail::binomial_loop(n,k);
}

inline std::pair<BigInt,BigInt> fib_pair(std::uint64_t n){
	if(n==0)
		return {BigInt(0),BigInt(1)};
	auto ab=fib_pair(n>>1);
	const BigInt&a=ab.first;
	const BigInt&b=ab.second;
	const BigInt c=a*((b<<1)-a);
	const BigInt d=sqr_disp(a)+sqr_disp(b);
	if((n&1u)==0u)
		return {c,d};
	return {d,c+d};
}

inline BigInt fibonacci(std::uint64_t n){ return fib_pair(n).first; }





enum class FloatRnd{
	nearest,
	zero,
	down,
	up,
	away,
};

class BigRat{
  public:
	BigRat() : num_(0),den_(1){}
	BigRat(std::int64_t v) : num_(v),den_(1){}
	explicit BigRat(BigInt n) : num_(std::move(n)),den_(1){}
	template<class Float,
			 class=std::enable_if_t<std::is_floating_point_v<Float>>>
	explicit BigRat(Float v){
		*this=from_floating(v);
	}

	BigRat(BigInt n,BigInt d) : num_(std::move(n)),den_(std::move(d)){
		canon();
	}

	static BigRat from_float(float v){ return from_floating(v); }
	static BigRat from_double(double v){ return from_floating(v); }

	template<class Float,
			 class=std::enable_if_t<std::is_floating_point_v<Float>>>
	static BigRat from_floating(Float v){
		static_assert(std::numeric_limits<Float>::radix==2,
					  "BigRat::from_floating requires binary floating point");
		static_assert(std::numeric_limits<Float>::digits<=64,
					  "BigRat::from_floating requires at most 64 bits");
		if(std::isnan(v)||std::isinf(v)){
			detail::throw_dom(
				"BigRat::from_floating: non-finite value");
		}
		if(v==Float(0))
			return BigRat();
		const int s=std::signbit(v)?-1:1;
		constexpr int fb=std::numeric_limits<Float>::digits;
		int ex=0;
		const Float fr=std::frexp(std::fabs(v),&ex);
		const Float sc=std::ldexp(fr,fb);
		std::uint64_t mag=static_cast<std::uint64_t>(sc);
		std::int64_t be=static_cast<std::int64_t>(ex)-fb;
		if(be<0&&mag!=0){
			const std::uint64_t drop=std::min<std::uint64_t>(
				static_cast<std::uint64_t>(std::countr_zero(mag)),
				abs_i64(be));
			mag>>=static_cast<unsigned>(drop);
			be+=static_cast<std::int64_t>(drop);
		}
		BigInt n=BigInt::from_u64(mag);
		if(s<0)
			n=-n;
		if(be>=0){
			n<<=checked_size(static_cast<std::uint64_t>(be),
				"BigRat::from_floating: exponent too large");
			return from_canon(std::move(n),BigInt(1));
		}
		BigInt d(1);
		d<<=checked_size(abs_i64(be),
			"BigRat::from_floating: exponent too small");
		return from_canon(std::move(n),std::move(d));
	}

	static BigRat parse(std::string_view text,int base=10){
		text=detail::trim_ws(text);
		if(text.empty()){
			detail::throw_inv("BigRat::parse: empty input");
		}
		const std::size_t slash=text.find('/');
		if(slash==std::string_view::npos){
			if(text.find('.')!=std::string_view::npos){
				if(base!=10){
					detail::throw_inv(
						"BigRat::parse: decimal form requires base 10");
				}
				std::string_view sv=text;
				int sign=1;
				if(!sv.empty()&&(sv.front()=='+'||sv.front()=='-')){
					sign=(sv.front()=='-')?-1:1;
					sv.remove_prefix(1);
				}
				const std::size_t dot=sv.find('.');
				if(dot==std::string_view::npos){
					detail::throw_inv(
						"BigRat::parse: malformed decimal");
				}
				const std::string_view left=sv.substr(0,dot);
				const std::string_view right=sv.substr(dot+1);
				if(left.empty()&&right.empty()){
					detail::throw_inv(
						"BigRat::parse: malformed decimal");
				}
				for(char ch : left){
					if(ch<'0'||ch>'9'){
						detail::throw_inv(
							"BigRat::parse: malformed decimal");
					}
				}
				for(char ch : right){
					if(ch<'0'||ch>'9'){
						detail::throw_inv(
							"BigRat::parse: malformed decimal");
					}
				}

				std::string digits;
				digits.reserve(left.size()+right.size()+1);
				if(sign<0)
					digits.push_back('-');
				if(left.empty()){
					digits.push_back('0');
				}else{
					digits.append(left.begin(),left.end());
				}
				digits.append(right.begin(),right.end());

				const BigInt num=BigInt::parse(digits,10);
				const BigInt den=
					pow(BigInt(10),static_cast<std::uint64_t>(right.size()));
				return BigRat(num,den);
			}
			return BigRat(BigInt::parse(text,base));
		}
		const std::string_view left=detail::trim_ws(text.substr(0,slash));
		const std::string_view right=detail::trim_ws(text.substr(slash+1));
		if(left.empty()||right.empty()){
			detail::throw_inv("BigRat::parse: malformed rational");
		}
		return BigRat(BigInt::parse(left,base),BigInt::parse(right,base));
	}

	std::string to_string(int base=10) const{
		if(den_.is_one())
			return num_.to_string(base);
		return num_.to_string(base)+"/"+den_.to_string(base);
	}

	const BigInt&num() const noexcept{ return num_; }
	const BigInt&den() const noexcept{ return den_; }
	int sign() const noexcept{ return num_.sign(); }
	bool is_zero() const noexcept{ return num_.is_zero(); }
	bool is_one() const noexcept{ return num_.is_one()&&den_.is_one(); }
	bool is_integer() const noexcept{ return den_.is_one(); }

	void canon(){
		if(den_.is_zero()){
			detail::throw_dom("BigRat: denominator is zero");
		}
		if(num_.is_zero()){
			den_=BigInt(1);
			return;
		}
		if(den_.sign()<0){
			den_=-den_;
			num_=-num_;
		}
		const BigInt g=gcd(num_.abs(),den_);
		if(!g.is_one()){
			num_/=g;
			den_/=g;
		}
	}

	BigRat abs() const{
		BigRat out=*this;
		if(out.num_.sign()<0)
			out.num_=-out.num_;
		return out;
	}

	BigRat neg() const{
		BigRat out=*this;
		if(!out.num_.is_zero())
			out.num_=-out.num_;
		return out;
	}

	BigRat recip() const{
		if(num_.is_zero()){
			detail::throw_dom("BigRat::recip: division by zero");
		}
		BigRat out;
		if(num_.sign()<0){
			out.num_=-den_;
			out.den_=-num_;
		}else{
			out.num_=den_;
			out.den_=num_;
		}
		return out;
	}

	BigInt trunc() const{
		if(den_.is_one())
			return num_;
		if(detail::pow2_abs(den_.limbs_)){
			const std::size_t shift=detail::pow2_exp_abs(den_.limbs_);
			return BigInt::from_raw(num_.sign()<0?-1:1,
				detail::shr_bits(num_.limbs_,shift));
		}
		return tdiv_q(num_,den_);
	}
	BigInt floor() const{
		if(den_.is_one())
			return num_;
		if(detail::pow2_abs(den_.limbs_)){
			const std::size_t shift=detail::pow2_exp_abs(den_.limbs_);
			BigInt q=BigInt::from_raw(num_.sign()<0?-1:1,
				detail::shr_bits(num_.limbs_,shift));
			if(num_.sign()<0&&!detail::low_bits_abs(num_.limbs_,shift).empty())
				q-=BigInt(1);
			return q;
		}
		return fdiv_q(num_,den_);
	}
	BigInt ceil() const{
		if(den_.is_one())
			return num_;
		if(detail::pow2_abs(den_.limbs_)){
			const std::size_t shift=detail::pow2_exp_abs(den_.limbs_);
			BigInt q=BigInt::from_raw(num_.sign()<0?-1:1,
				detail::shr_bits(num_.limbs_,shift));
			if(num_.sign()>0&&!detail::low_bits_abs(num_.limbs_,shift).empty())
				q+=BigInt(1);
			return q;
		}
		return cdiv_q(num_,den_);
	}

	BigInt to_bigint(FloatRnd rnd=FloatRnd::zero) const{
		if(den_.is_one())
			return num_;
		if(detail::pow2_abs(den_.limbs_)){
			const std::size_t shift=detail::pow2_exp_abs(den_.limbs_);
			detail::limbs_t rem=detail::low_bits_abs(num_.limbs_,shift);
			BigInt q=BigInt::from_raw(num_.sign()<0?-1:1,
				detail::shr_bits(num_.limbs_,shift));
			const bool has_rem=!rem.empty();
			switch(rnd){
			case FloatRnd::zero:
				return q;
			case FloatRnd::down:
				if(num_.sign()<0&&has_rem)
					q-=BigInt(1);
				return q;
			case FloatRnd::up:
				if(num_.sign()>0&&has_rem)
					q+=BigInt(1);
				return q;
			case FloatRnd::away:
				if(has_rem)
					q+=BigInt(num_.sign());
				return q;
			case FloatRnd::nearest:
				if(has_rem){
					detail::limbs_t twice=detail::shl_bits(rem,1u);
					const int cmp=detail::cmp_abs(twice,den_.limbs_);
					if(cmp>0||(cmp==0&&q.is_odd()))
						q+=BigInt(num_.sign());
				}
				return q;
			}
		}
		switch(rnd){
		case FloatRnd::zero:
			return trunc();
		case FloatRnd::down:
			return floor();
		case FloatRnd::up:
			return ceil();
		case FloatRnd::away:{
			auto qr=divmod(num_,den_);
			if(!qr.second.is_zero())
				qr.first+=BigInt(num_.sign());
			return qr.first;
		}
		case FloatRnd::nearest:
			break;
		}

		auto qr=divmod(num_,den_);
		if(qr.second.is_zero())
			return qr.first;
		const BigInt twice=qr.second.abs()<<1u;
		const int cmp=compare(twice,den_);
		if(cmp>0||(cmp==0&&qr.first.is_odd()))
			qr.first+=BigInt(num_.sign());
		return qr.first;
	}

	BigInt rint(FloatRnd rnd=FloatRnd::nearest) const{
		return to_bigint(rnd);
	}

	bool fits_u64(FloatRnd rnd=FloatRnd::zero) const{
		return to_bigint(rnd).fits_u64();
	}

	bool fits_i64(FloatRnd rnd=FloatRnd::zero) const{
		BigInt v=to_bigint(rnd);
		if(v.sign()>=0){
			return v.bit_length()<63u||
				(v.bit_length()==63u&&v.to_u64()<=
				 static_cast<std::uint64_t>(
					 std::numeric_limits<std::int64_t>::max()));
		}
		v=-v;
		return v.bit_length()<64u||
			(v.bit_length()==64u&&v.test_bit(63u)&&
			 (v-(BigInt(1)<<63)).is_zero());
	}

	std::uint64_t to_u64(FloatRnd rnd=FloatRnd::zero) const{
		return to_bigint(rnd).to_u64();
	}

	std::int64_t to_i64(FloatRnd rnd=FloatRnd::zero) const{
		BigInt v=to_bigint(rnd);
		if(v.sign()>=0){
			const std::uint64_t u=v.to_u64();
			if(u>static_cast<std::uint64_t>(
				   std::numeric_limits<std::int64_t>::max()))
				detail::throw_ovf("BigRat::to_i64 overflow");
			return static_cast<std::int64_t>(u);
		}
		v=-v;
		if(v==(BigInt(1)<<63))
			return std::numeric_limits<std::int64_t>::min();
		const std::uint64_t u=v.to_u64();
		if(u>static_cast<std::uint64_t>(
			   std::numeric_limits<std::int64_t>::max()))
			detail::throw_ovf("BigRat::to_i64 overflow");
		return -static_cast<std::int64_t>(u);
	}

	float to_float(FloatRnd rnd=FloatRnd::zero) const;
	double to_double(FloatRnd rnd=FloatRnd::zero) const;
	template<class Float>
	Float to_floating(FloatRnd rnd=FloatRnd::zero) const;

	BigRat pow_uint(std::uint64_t exp) const{
		if(exp==0)
			return BigRat(1);
		if(exp==1)
			return *this;
		if(num_.is_zero())
			return BigRat();
		BigRat out;
		out.num_=pow_bigint_uint(num_,exp);
		out.den_=den_.is_one()?BigInt(1):pow_bigint_uint(den_,exp);
		return out;
	}

	BigRat pow_int(std::int64_t exp) const{
		if(exp>=0)
			return pow_uint(static_cast<std::uint64_t>(exp));
		if(num_.is_zero()){
			detail::throw_dom("BigRat::pow_int: division by zero");
		}
		const std::uint64_t e=abs_i64(exp);
		BigRat out;
		out.num_=den_.is_one()?BigInt(1):pow_bigint_uint(den_,e);
		const BigInt an=num_.abs();
		out.den_=an.is_one()?BigInt(1):pow_bigint_uint(an,e);
		if(num_.sign()<0&&(e&1u)!=0u)
			out.num_=-out.num_;
		return out;
	}

	BigRat&operator+=(const BigRat&rhs){
		if(rhs.num_.is_zero())
			return *this;
		if(num_.is_zero()){
			*this=rhs;
			return *this;
		}
		if(den_==rhs.den_){
			num_+=rhs.num_;
			reduce_num_den();
			return *this;
		}
		if(den_.is_one()){
			num_=num_*rhs.den_+rhs.num_;
			den_=rhs.den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}
		if(rhs.den_.is_one()){
			num_+=rhs.num_*den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}

		
		
		
		
		
		const BigInt g=gcd(den_,rhs.den_);
		if(g.is_one()){
			num_=num_*rhs.den_+rhs.num_*den_;
			den_*=rhs.den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}
		const BigInt d1p=den_/g;
		const BigInt d2p=rhs.den_/g;
		const BigInt n=num_*d2p+rhs.num_*d1p;
		if(n.is_zero()){
			num_=BigInt();
			den_=BigInt(1);
			return *this;
		}
		const BigInt h=gcd(n.abs(),g);
		num_=n/h;
		den_=(d1p*rhs.den_)/h;
		if(den_.sign()<0){
			den_=-den_;
			num_=-num_;
		}
		return *this;
	}

	BigRat&operator-=(const BigRat&rhs){
		if(rhs.num_.is_zero())
			return *this;
		if(num_.is_zero()){
			*this=rhs;
			num_=-num_;
			return *this;
		}
		if(den_==rhs.den_){
			num_-=rhs.num_;
			reduce_num_den();
			return *this;
		}
		if(den_.is_one()){
			num_=num_*rhs.den_-rhs.num_;
			den_=rhs.den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}
		if(rhs.den_.is_one()){
			num_-=rhs.num_*den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}

		const BigInt g=gcd(den_,rhs.den_);
		if(g.is_one()){
			num_=num_*rhs.den_-rhs.num_*den_;
			den_*=rhs.den_;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}
		const BigInt d1p=den_/g;
		const BigInt d2p=rhs.den_/g;
		const BigInt n=num_*d2p-rhs.num_*d1p;
		if(n.is_zero()){
			num_=BigInt();
			den_=BigInt(1);
			return *this;
		}
		const BigInt h=gcd(n.abs(),g);
		num_=n/h;
		den_=(d1p*rhs.den_)/h;
		if(den_.sign()<0){
			den_=-den_;
			num_=-num_;
		}
		return *this;
	}

	BigRat&operator*=(const BigRat&rhs){
		if(num_.is_zero()||rhs.num_.is_zero()){
			num_=BigInt();
			den_=BigInt(1);
			return *this;
		}
		if(den_.is_one()&&rhs.den_.is_one()){
			num_*=rhs.num_;
			return *this;
		}
		if(den_==rhs.den_){
			num_*=rhs.num_;
			den_*=rhs.den_;
			return *this;
		}
		if(den_.is_one()){
			BigInt a=num_;
			BigInt d=rhs.den_;
			const BigInt g=gcd(a.abs(),d);
			if(!g.is_one()){
				a/=g;
				d/=g;
			}
			num_=a*rhs.num_;
			den_=std::move(d);
			return *this;
		}
		if(rhs.den_.is_one()){
			BigInt c=rhs.num_;
			BigInt b=den_;
			const BigInt g=gcd(c.abs(),b);
			if(!g.is_one()){
				c/=g;
				b/=g;
			}
			num_=num_*c;
			den_=std::move(b);
			return *this;
		}
		BigInt a=num_;
		BigInt b=den_;
		BigInt c=rhs.num_;
		BigInt d=rhs.den_;
		mul_xred(a,b,c,d);
		num_=a*c;
		den_=b*d;
		if(num_.is_zero())
			den_=BigInt(1);
		if(den_.sign()<0){
			den_=-den_;
			num_=-num_;
		}
		return *this;
	}

	BigRat&operator/=(const BigRat&rhs){
		if(rhs.num_.is_zero()){
			detail::throw_dom("BigRat division by zero");
		}
		if(num_.is_zero())
			return *this;
		if(den_==rhs.den_){
			*this=BigRat(num_,rhs.num_);
			return *this;
		}
		if(rhs.den_.is_one()){
			BigInt c=rhs.num_;
			if(c.sign()<0){
				c=-c;
				num_=-num_;
			}
			const BigInt g=gcd(num_.abs(),c);
			if(!g.is_one()){
				num_/=g;
				c/=g;
			}
			den_*=c;
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}
		if(den_.is_one()){
			BigInt a=num_;
			BigInt c=rhs.num_;
			if(c.sign()<0){
				c=-c;
				a=-a;
			}
			const BigInt g=gcd(a.abs(),c);
			if(!g.is_one()){
				a/=g;
				c/=g;
			}
			num_=a*rhs.den_;
			den_=std::move(c);
			if(num_.is_zero())
				den_=BigInt(1);
			return *this;
		}

		BigInt a=num_;
		BigInt b=den_;
		BigInt c=rhs.num_;
		BigInt d=rhs.den_;
		if(c.sign()<0){
			c=-c;
			a=-a;
		}

		BigInt g1=gcd(a.abs(),c.abs());
		if(!g1.is_one()){
			a/=g1;
			c/=g1;
		}
		BigInt g2=gcd(d,b);
		if(!g2.is_one()){
			d/=g2;
			b/=g2;
		}

		num_=a*d;
		den_=b*c;
		if(num_.is_zero())
			den_=BigInt(1);
		if(den_.sign()<0){
			den_=-den_;
			num_=-num_;
		}
		return *this;
	}

	BigRat&operator++(){
		num_+=den_;
		return *this;
	}

	BigRat operator++(int){
		BigRat old=*this;
		++(*this);
		return old;
	}

	BigRat&operator--(){
		num_-=den_;
		return *this;
	}

	BigRat operator--(int){
		BigRat old=*this;
		--(*this);
		return old;
	}

	friend int compare(const BigRat&a,const BigRat&b){
		if(a.num_.sign()!=b.num_.sign())
			return a.num_.sign()<b.num_.sign()?-1:1;
		const int s=a.num_.sign();
		if(s==0)
			return 0;
		if(a.den_==b.den_)
			return compare(a.num_,b.num_);
		const std::size_t lb=a.num_.bit_length()+b.den_.bit_length();
		const std::size_t rb=b.num_.bit_length()+a.den_.bit_length();
		if(lb+1u<rb)
			return -s;
		if(lb>rb+1u)
			return s;
		if(a.den_.is_one())
			return compare(a.num_*b.den_,b.num_);
		if(b.den_.is_one())
			return compare(a.num_,b.num_*a.den_);
		return compare(a.num_*b.den_,b.num_*a.den_);
	}

	friend bool operator==(const BigRat&a,const BigRat&b){
		return a.num_==b.num_&&a.den_==b.den_;
	}
	friend bool operator!=(const BigRat&a,const BigRat&b){ return !(a==b); }
	friend bool operator<(const BigRat&a,const BigRat&b){
		return compare(a,b)<0;
	}
	friend bool operator>(const BigRat&a,const BigRat&b){ return b<a; }
	friend bool operator<=(const BigRat&a,const BigRat&b){ return !(b<a); }
	friend bool operator>=(const BigRat&a,const BigRat&b){ return !(a<b); }

  private:
	static BigInt pow_bigint_uint(const BigInt&v,std::uint64_t exp){
		if(exp==0)
			return BigInt(1);
		if(exp==1)
			return v;
		if(exp==2)
			return sqr_disp(v);
		return mini_mp::pow(v,exp);
	}

	static BigRat from_canon(BigInt n,BigInt d){
		BigRat out;
		out.num_=std::move(n);
		out.den_=std::move(d);
		MINI_MP_ASSERT(!out.den_.is_zero());
		MINI_MP_ASSERT(out.den_.sign()>0);
		if(out.num_.is_zero())
			out.den_=BigInt(1);
		return out;
	}

	static std::uint64_t abs_i64(std::int64_t v) noexcept{
		return v<0?
			static_cast<std::uint64_t>(-(v+1))+std::uint64_t(1):
			static_cast<std::uint64_t>(v);
	}

	static std::size_t checked_size(std::uint64_t v,const char*msg){
		if(v>std::numeric_limits<std::size_t>::max())
			detail::throw_ovf(msg);
		return static_cast<std::size_t>(v);
	}

	void reduce_num_den(){
		if(num_.is_zero()){
			den_=BigInt(1);
			return;
		}
		if(den_.is_one())
			return;
		const BigInt g=gcd(num_.abs(),den_);
		if(!g.is_one()){
			num_/=g;
			den_/=g;
		}
	}

	static void mul_xred(BigInt&a,BigInt&b,BigInt&c,BigInt&d){
		BigInt g1=gcd(a.abs(),d);
		if(!g1.is_one()){
			a/=g1;
			d/=g1;
		}
		BigInt g2=gcd(c.abs(),b);
		if(!g2.is_one()){
			c/=g2;
			b/=g2;
		}
	}

  private:
	BigInt num_; 
	BigInt den_; 
};

inline BigRat operator+(BigRat lhs,const BigRat&rhs){
	lhs+=rhs;
	return lhs;
}
inline BigRat operator-(BigRat lhs,const BigRat&rhs){
	lhs-=rhs;
	return lhs;
}
inline BigRat operator*(BigRat lhs,const BigRat&rhs){
	lhs*=rhs;
	return lhs;
}
inline BigRat operator/(BigRat lhs,const BigRat&rhs){
	lhs/=rhs;
	return lhs;
}
inline BigRat operator-(const BigRat&v){ return v.neg(); }
inline BigRat operator+(const BigRat&v){ return v; }
inline BigRat abs(const BigRat&x){ return x.abs(); }
inline BigRat recip(const BigRat&x){ return x.recip(); }
inline BigInt floor(const BigRat&x){ return x.floor(); }
inline BigInt ceil(const BigRat&x){ return x.ceil(); }
inline BigInt trunc(const BigRat&x){ return x.trunc(); }
inline BigInt rint(const BigRat&x,FloatRnd rnd=FloatRnd::nearest){
	return x.rint(rnd);
}
template<class Int,
		 class=std::enable_if_t<std::is_integral_v<Int>>>
inline BigRat pow(const BigRat&base,Int exp){
	if constexpr(std::is_signed_v<Int>)
		return base.pow_int(static_cast<std::int64_t>(exp));
	else
		return base.pow_uint(static_cast<std::uint64_t>(exp));
}

enum class FloatKind{
	finite,
	inf,
	nan,
};

class BigFloat{
  public:
	static constexpr std::size_t default_prec=128;

	BigFloat() noexcept=default;

	template<class Int,
			 class=std::enable_if_t<std::is_integral_v<Int>&&
									!std::is_same_v<std::remove_cv_t<Int>,bool>>>
	BigFloat(Int v,std::size_t prec=default_prec,
			 FloatRnd rnd=FloatRnd::nearest){
		prec_=chk_prec(prec);
		if constexpr(std::is_signed_v<Int>){
			set_int(BigInt(static_cast<std::int64_t>(v)),rnd);
		}else{
			set_int(BigInt::from_u64(static_cast<std::uint64_t>(v)),rnd);
		}
	}

	explicit BigFloat(const BigInt&v,std::size_t prec=default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
		prec_=chk_prec(prec);
		set_int(v,rnd);
	}

	explicit BigFloat(const BigRat&v,std::size_t prec=default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
		*this=from_ratio(v.num().sign(),v.num().abs(),v.den(),
						 chk_prec(prec),rnd);
	}

	explicit BigFloat(double v,std::size_t prec=default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
		*this=from_double(v,prec,rnd);
	}

	explicit BigFloat(std::string_view text,std::size_t prec=default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
		*this=parse(text,prec,rnd);
	}

	static BigFloat zero(std::size_t prec=default_prec){
		BigFloat out;
		out.prec_=chk_prec(prec);
		return out;
	}

	static BigFloat one(std::size_t prec=default_prec){
		return BigFloat(1,prec);
	}

	static BigFloat epsilon(std::size_t prec=default_prec){
		prec=chk_prec(prec);
		return from_limbs(1,detail::limbs_t{1},sub_exp_sz(1,prec,
			"BigFloat::epsilon: exponent overflow"),prec,FloatRnd::zero);
	}

	static BigFloat inf(int sign=1,std::size_t prec=default_prec){
		BigFloat out;
		out.prec_=chk_prec(prec);
		out.kind_=FloatKind::inf;
		out.sign_=(sign<0)?-1:1;
		return out;
	}

	static BigFloat nan(std::size_t prec=default_prec){
		BigFloat out;
		out.prec_=chk_prec(prec);
		out.kind_=FloatKind::nan;
		return out;
	}

	static BigFloat from_parts(int sign,BigInt mag,std::int64_t exp,
							   std::size_t prec=default_prec,
							   FloatRnd rnd=FloatRnd::nearest){
		return from_parts_more(sign,std::move(mag),exp,prec,rnd,false);
	}

	static BigFloat from_double(double v,std::size_t prec=default_prec,
								FloatRnd rnd=FloatRnd::nearest){
		if(std::isnan(v))
			return nan(prec);
		if(std::isinf(v))
			return inf(std::signbit(v)?-1:1,prec);
		if(v==0.0)
			return zero(prec);
		const int s=std::signbit(v)?-1:1;
		double av=std::fabs(v);
		int ex=0;
		const double fr=std::frexp(av,&ex);
		constexpr int db=std::numeric_limits<double>::digits;
		const double sc=std::ldexp(fr,db);
		const auto im=static_cast<std::uint64_t>(sc);
		return from_parts(s,BigInt::from_u64(im),
						  static_cast<std::int64_t>(ex-db),prec,rnd);
	}

	static BigFloat parse(std::string_view text,
						  std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		text=detail::trim_ws(text);
		if(text.empty())
			detail::throw_inv("BigFloat::parse: empty input");

		int s=1;
		if(text.front()=='+'||text.front()=='-'){
			s=(text.front()=='-')?-1:1;
			text.remove_prefix(1);
			text=detail::trim_ws(text);
			if(text.empty())
				detail::throw_inv("BigFloat::parse: sign without value");
		}

		if(eq_ci(text,"nan"))
			return nan(prec);
		if(eq_ci(text,"inf")||eq_ci(text,"infinity"))
			return inf(s,prec);

		if(text.size()>=2&&text[0]=='0'&&(text[1]=='b'||text[1]=='B'))
			return parse_bin(s,text.substr(2),prec,rnd);
		return parse_dec(s,text,prec,rnd);
	}

	FloatKind kind() const noexcept{ return kind_; }
	bool is_finite() const noexcept{ return kind_==FloatKind::finite; }
	bool is_inf() const noexcept{ return kind_==FloatKind::inf; }
	bool is_nan() const noexcept{ return kind_==FloatKind::nan; }
	bool is_zero() const noexcept{ return kind_==FloatKind::finite&&sign_==0; }
	bool is_neg() const noexcept{ return sign_<0; }
	bool is_integer() const noexcept{
		if(kind_!=FloatKind::finite)
			return false;
		if(sign_==0||exp_>=0)
			return true;
		return !low_nz(mag_,neg_exp_size(exp_,
			"BigFloat::is_integer: exponent too small"));
	}
	int sign() const noexcept{ return sign_; }
	std::size_t precision() const noexcept{ return prec_; }
	std::int64_t exponent() const noexcept{ return exp_; }
	int inexact() const noexcept{ return inex_; }
	const BigInt&mantissa() const noexcept{ return mag_; }
	std::int64_t ilogb() const{
		if(kind_==FloatKind::nan)
			detail::throw_dom("BigFloat::ilogb: nan");
		if(kind_==FloatKind::inf)
			detail::throw_dom("BigFloat::ilogb: infinity");
		if(sign_==0)
			detail::throw_dom("BigFloat::ilogb: zero");
		return add_exp(top_exp(*this),-1,"BigFloat::ilogb: exponent overflow");
	}
	std::size_t min_prec() const noexcept{
		return (kind_==FloatKind::finite&&!mag_.is_zero())?
			mag_.bit_length():1u;
	}

	BigFloat ulp() const{
		if(kind_==FloatKind::nan)
			return nan(prec_);
		if(kind_==FloatKind::inf)
			return inf(1,prec_);
		if(sign_==0)
			return epsilon(prec_);
		const std::int64_t e=sub_exp_sz(top_exp(*this),prec_,
			"BigFloat::ulp: exponent overflow");
		return from_limbs(1,detail::limbs_t{1},e,prec_,FloatRnd::zero);
	}

	void set_precision(std::size_t prec,FloatRnd rnd=FloatRnd::nearest){
		prec_=chk_prec(prec);
		round_self(rnd);
	}

	BigFloat rounded(std::size_t prec,
					 FloatRnd rnd=FloatRnd::nearest) const{
		BigFloat out=*this;
		prec=chk_prec(prec);
		if(out.prec_==prec){
			out.inex_=0;
			return out;
		}
		out.set_precision(prec,rnd);
		return out;
	}

	BigFloat abs() const{
		BigFloat out=*this;
		if(out.sign_<0)
			out.sign_=1;
		return out;
	}

	BigFloat neg() const{
		BigFloat out=*this;
		if(out.sign_!=0)
			out.sign_=-out.sign_;
		return out;
	}

	BigFloat with_sign(int sign,FloatRnd rnd=FloatRnd::nearest) const{
		BigFloat out=*this;
		if(out.kind_==FloatKind::nan)
			return out;
		if(out.sign_!=0)
			out.sign_=(sign<0)?-1:1;
		out.round_self(rnd);
		return out;
	}

	BigFloat copy_sign(const BigFloat&src,
					   FloatRnd rnd=FloatRnd::nearest) const{
		return with_sign(src.sign_,rnd);
	}

	BigFloat ldexp(std::int64_t bits) const{
		BigFloat out=*this;
		if(out.kind_==FloatKind::finite&&out.sign_!=0)
			out.exp_=add_exp(out.exp_,bits,"BigFloat::ldexp: exponent overflow");
		return out;
	}

	BigFloat scalbn(std::int64_t bits) const{
		return ldexp(bits);
	}

	std::pair<BigFloat,std::int64_t> frexp(
		std::size_t prec=default_prec,
		FloatRnd rnd=FloatRnd::nearest) const{
		prec=chk_prec(prec);
		if(kind_!=FloatKind::finite||sign_==0)
			return {rounded(prec,rnd),0};
		const std::int64_t e=top_exp(*this);
		return {ldexp(-e).rounded(prec,rnd),e};
	}

	BigFloat trunc() const{
		if(kind_!=FloatKind::finite)
			return *this;
		return int_part(FloatRnd::zero);
	}

	BigFloat floor() const{
		if(kind_!=FloatKind::finite)
			return *this;
		return int_part(FloatRnd::down);
	}

	BigFloat ceil() const{
		if(kind_!=FloatKind::finite)
			return *this;
		return int_part(FloatRnd::up);
	}

	BigFloat rint(FloatRnd rnd=FloatRnd::nearest) const{
		if(kind_!=FloatKind::finite)
			return *this;
		return int_part(rnd);
	}

	BigFloat frac(std::size_t prec=default_prec,
				  FloatRnd rnd=FloatRnd::nearest) const{
		if(kind_==FloatKind::nan||kind_==FloatKind::inf)
			return nan(chk_prec(prec));
		prec=chk_prec(prec);
		if(sign_==0||exp_>=0)
			return zero(prec);
		const std::size_t sh=neg_exp_size(exp_,
			"BigFloat::frac: exponent too small");
		if(sh>=mag_.bit_length())
			return rounded(prec,rnd);
		detail::limbs_t r=detail::low_bits_abs(mag_.limbs_,sh);
		if(r.empty())
			return zero(prec);
		if(detail::bit_length(r)<=prec)
			return from_exact_limbs(sign_,std::move(r),exp_,prec);
		return from_limbs(sign_,std::move(r),exp_,prec,rnd);
	}

	BigInt to_bigint(FloatRnd rnd=FloatRnd::zero) const{
		if(kind_==FloatKind::nan)
			detail::throw_dom("BigFloat::to_bigint: nan");
		if(kind_==FloatKind::inf)
			detail::throw_dom("BigFloat::to_bigint: infinity");
		if(sign_==0)
			return BigInt();
		BigInt q=mag_;
		if(exp_>=0){
			q<<=to_size_u64(static_cast<std::uint64_t>(exp_),
							"BigFloat::to_bigint: exponent too large");
		}else{
			const std::size_t sh=neg_exp_size(exp_,
				"BigFloat::to_bigint: exponent too small");
			const bool lost=low_nz(q,sh);
			detail::shr_ip(q.limbs_,sh);
			q.sign_=q.limbs_.empty()?0:1;
			bool inc=false;
			if(lost){
				if(rnd==FloatRnd::up&&sign_>0)
					inc=true;
				else if(rnd==FloatRnd::down&&sign_<0)
					inc=true;
				else if(rnd==FloatRnd::away)
					inc=true;
				else if(rnd==FloatRnd::nearest&&sh!=0){
					const bool guard=detail::test_bit(mag_.limbs_,sh-1u);
					const bool sticky=low_nz(mag_,sh-1u);
					inc=guard&&(sticky||q.is_odd());
				}
			}
			if(inc){
				detail::add_one_ip(q.limbs_);
				q.sign_=1;
			}
		}
		if(q.is_zero())
			return q;
		if(sign_<0)
			q=-q;
		return q;
	}

	BigRat to_bigrat() const{
		if(kind_==FloatKind::nan)
			detail::throw_dom("BigFloat::to_bigrat: nan");
		if(kind_==FloatKind::inf)
			detail::throw_dom("BigFloat::to_bigrat: infinity");
		if(sign_==0)
			return BigRat();
		BigInt num=mag_;
		if(sign_<0)
			num=-num;
		if(exp_>=0){
			num<<=to_size_u64(static_cast<std::uint64_t>(exp_),
				"BigFloat::to_bigrat: exponent too large");
			return BigRat(std::move(num));
		}
		BigInt den=BigInt(1)<<neg_exp_size(exp_,
			"BigFloat::to_bigrat: exponent too small");
		return BigRat(std::move(num),std::move(den));
	}

	bool fits_u64(FloatRnd rnd=FloatRnd::zero) const{
		if(kind_!=FloatKind::finite)
			return false;
		BigInt v=to_bigint(rnd);
		return v.fits_u64();
	}

	bool fits_i64(FloatRnd rnd=FloatRnd::zero) const{
		if(kind_!=FloatKind::finite)
			return false;
		BigInt v=to_bigint(rnd);
		if(v.sign()>=0){
			return v.bit_length()<63u||
				(v.bit_length()==63u&&v.to_u64()<=
				 static_cast<std::uint64_t>(
					 std::numeric_limits<std::int64_t>::max()));
		}
		v=-v;
		return v.bit_length()<64u||
			(v.bit_length()==64u&&v.test_bit(63u)&&
			 (v-(BigInt(1)<<63)).is_zero());
	}

	std::uint64_t to_u64(FloatRnd rnd=FloatRnd::zero) const{
		return to_bigint(rnd).to_u64();
	}

	std::int64_t to_i64(FloatRnd rnd=FloatRnd::zero) const{
		BigInt v=to_bigint(rnd);
		if(v.sign()>=0){
			const std::uint64_t u=v.to_u64();
			if(u>static_cast<std::uint64_t>(
				   std::numeric_limits<std::int64_t>::max()))
				detail::throw_ovf("BigFloat::to_i64 overflow");
			return static_cast<std::int64_t>(u);
		}
		v=-v;
		if(v==(BigInt(1)<<63))
			return std::numeric_limits<std::int64_t>::min();
		const std::uint64_t u=v.to_u64();
		if(u>static_cast<std::uint64_t>(
			   std::numeric_limits<std::int64_t>::max()))
			detail::throw_ovf("BigFloat::to_i64 overflow");
		return -static_cast<std::int64_t>(u);
	}

	std::string to_string(int base=10,std::size_t digits=0) const{
		if(kind_==FloatKind::nan)
			return "nan";
		if(kind_==FloatKind::inf)
			return sign_<0?"-inf":"inf";
		if(sign_==0)
			return "0";
		if(base==2)
			return to_bin_string();
		if(base!=10)
			detail::throw_inv("BigFloat::to_string: base must be 2 or 10");
		const std::size_t digs=(digits==0)?dec_digits():digits;
		std::string out;
		if(sign_<0)
			out.push_back('-');
		if(exp_>=0){
			BigInt v=mag_;
			v<<=to_size_u64(static_cast<std::uint64_t>(exp_),
							"BigFloat::to_string: exponent too large");
			out+=v.to_string(10);
			return out;
		}
		const std::size_t sh=neg_exp_size(exp_,
			"BigFloat::to_string: exponent too small");
		BigInt sc=mag_*mini_mp::pow(BigInt(10),to_u64_sz(digs,
			"BigFloat::to_string: too many digits"));
		BigInt q=sc;
		if(sh!=0){
			const bool guard=detail::test_bit(sc.limbs_,sh-1u);
			const bool sticky=low_nz(sc,sh-1u);
			detail::shr_ip(q.limbs_,sh);
			q.sign_=q.limbs_.empty()?0:1;
			if(guard&&(sticky||q.is_odd())){
				detail::add_one_ip(q.limbs_);
				q.sign_=1;
			}
		}
		std::string raw=q.to_string(10);
		if(digs==0){
			out+=raw;
			return out;
		}
		if(raw.size()<=digs){
			out+="0.";
			out.append(digs-raw.size(),'0');
			out+=raw;
		}else{
			const std::size_t cut=raw.size()-digs;
			out.append(raw.data(),raw.data()+static_cast<std::ptrdiff_t>(cut));
			out.push_back('.');
			out.append(raw.data()+static_cast<std::ptrdiff_t>(cut),
					   raw.data()+static_cast<std::ptrdiff_t>(raw.size()));
		}
		return out;
	}

	double to_double() const{
		if(kind_==FloatKind::nan)
			return std::numeric_limits<double>::quiet_NaN();
		if(kind_==FloatKind::inf)
			return sign_<0?-std::numeric_limits<double>::infinity():
				std::numeric_limits<double>::infinity();
		if(sign_==0)
			return sign_<0?-0.0:0.0;
		const std::size_t bits=mag_.bit_length();
		const std::int64_t he=top_exp_sat(exp_,bits);
		if(he>std::numeric_limits<double>::max_exponent-1)
			return sign_<0?-std::numeric_limits<double>::infinity():
				std::numeric_limits<double>::infinity();
		if(he<std::numeric_limits<double>::min_exponent-
			  std::numeric_limits<double>::digits-1)
			return sign_<0?-0.0:0.0;
		const std::size_t take=std::min<std::size_t>(bits,64u);
		const std::size_t sh=bits-take;
		std::uint64_t top=top_bits(mag_.limbs_,bits,take);
		if(sh!=0&&low_nz(mag_,sh))
			top|=1u;
		std::int64_t e=add_exp_sz_sat(exp_,sh);
		e=std::max<std::int64_t>(
			std::min<std::int64_t>(e,
				static_cast<std::int64_t>(std::numeric_limits<int>::max())),
			static_cast<std::int64_t>(std::numeric_limits<int>::min()));
		double out=std::ldexp(static_cast<double>(top),static_cast<int>(e));
		if(sign_<0)
			out=-out;
		return out;
	}

	static BigFloat add(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.kind_==FloatKind::inf&&b.kind_==FloatKind::inf&&
			   a.sign_!=b.sign_)
				return nan(prec);
			return a.kind_==FloatKind::inf?inf(a.sign_,prec):inf(b.sign_,prec);
		}
		if(a.sign_==0)
			return b.rounded(prec,rnd);
		if(b.sign_==0)
			return a.rounded(prec,rnd);
		if(a.sign_==b.sign_)
			return add_same(a,b,prec,rnd);
		return add_mag_exp(a.sign_,a.mag_,a.exp_,b.sign_,b.mag_,b.exp_,
						   prec,rnd);
	}

	static BigFloat sub(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.kind_==FloatKind::inf&&b.kind_==FloatKind::inf&&
			   a.sign_==b.sign_)
				return nan(prec);
			return a.kind_==FloatKind::inf?inf(a.sign_,prec):inf(-b.sign_,prec);
		}
		if(a.sign_==0)
			return b.neg().rounded(prec,rnd);
		if(b.sign_==0)
			return a.rounded(prec,rnd);
		return add_mag_exp(a.sign_,a.mag_,a.exp_,-b.sign_,b.mag_,b.exp_,
						   prec,rnd);
	}

	static BigFloat mul(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.sign_==0||b.sign_==0)
				return nan(prec);
			return inf(a.sign_*b.sign_,prec);
		}
		if(a.sign_==0||b.sign_==0)
			return zero(prec);
		const std::int64_t e=add_exp(a.exp_,b.exp_,
			"BigFloat::mul: exponent overflow");
		if(&a==&b)
			return from_limbs(1,sqr_limbs(a.mag_),e,prec,rnd);
		if(a.mag_.is_one())
			return from_limbs(a.sign_*b.sign_,b.mag_.limbs_,e,prec,rnd);
		if(b.mag_.is_one())
			return from_limbs(a.sign_*b.sign_,a.mag_.limbs_,e,prec,rnd);
		return from_limbs(a.sign_*b.sign_,mul_limbs(a.mag_,b.mag_),
						  e,prec,rnd);
	}

	static BigFloat div(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(b.sign_==0&&b.kind_==FloatKind::finite){
			if(a.sign_==0)
				return nan(prec);
			return inf(a.sign_,prec);
		}
		if(a.kind_==FloatKind::inf&&b.kind_==FloatKind::inf)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf(a.sign_*b.sign_,prec);
		if(b.kind_==FloatKind::inf)
			return zero(prec);
		if(a.sign_==0)
			return zero(prec);
		if(b.mag_.is_one()){
			BigFloat q;
			q.prec_=prec;
			q.kind_=FloatKind::finite;
			q.sign_=a.sign_*b.sign_;
			q.mag_=a.mag_;
			q.exp_=add_exp(a.exp_,-b.exp_,
				"BigFloat::div: exponent overflow");
			q.round_self(rnd);
			return q;
		}
		BigFloat q=from_ratio(a.sign_*b.sign_,a.mag_,b.mag_,prec,rnd);
		const int qix=q.inex_;
		q.exp_=add_exp(q.exp_,a.exp_,
			"BigFloat::div: exponent overflow");
		q.exp_=add_exp(q.exp_,-b.exp_,
			"BigFloat::div: exponent overflow");
		q.inex_=qix;
		return q;
	}

	static BigFloat sqr(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf(1,prec);
		if(a.sign_==0)
			return zero(prec);
		const std::int64_t e=add_exp(a.exp_,a.exp_,
			"BigFloat::sqr: exponent overflow");
		return from_limbs(1,sqr_limbs(a.mag_),e,prec,rnd);
	}

	static BigFloat fma(const BigFloat&a,const BigFloat&b,const BigFloat&c,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan||
		   c.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.sign_==0||b.sign_==0)
				return nan(prec);
			const int ps=a.sign_*b.sign_;
			if(c.kind_==FloatKind::inf&&c.sign_!=ps)
				return nan(prec);
			return inf(ps,prec);
		}
		if(c.kind_==FloatKind::inf)
			return inf(c.sign_,prec);
		if(a.sign_==0||b.sign_==0)
			return c.rounded(prec,rnd);
		detail::limbs_t pm=mul_limbs(a.mag_,b.mag_);
		const int ps=a.sign_*b.sign_;
		const std::int64_t pe=add_exp(a.exp_,b.exp_,
			"BigFloat::fma: exponent overflow");
		if(c.sign_!=0&&ps==c.sign_){
			const std::int64_t ph=add_exp_sz(pe,detail::bit_length(pm),
				"BigFloat::fma: exponent overflow");
			const std::int64_t ch=top_exp(c);
			if(ph!=ch){
				const std::int64_t hh=std::max(ph,ch);
				const std::int64_t lh=std::min(ph,ch);
				const std::size_t gap=exp_gap(hh,lh,
					"BigFloat::fma: exponent gap too large");
				if(gap>prec_add(prec,8,
					   "BigFloat::fma: precision too large")){
					if(ph>ch){
						return from_limbs(ps,std::move(pm),pe,
											   prec,rnd,true);
					}
					return from_parts_more(c.sign_,c.mag_,c.exp_,
										   prec,rnd,true);
				}
			}
		}
		if(c.sign_==0)
			return from_limbs(ps,std::move(pm),pe,prec,rnd);
		return add_mag_exp(ps,BigInt::from_raw(1,std::move(pm)),pe,
						   c.sign_,c.mag_,c.exp_,
						   prec,rnd);
	}

	static BigFloat sqrt(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return a.sign_<0?nan(prec):inf(1,prec);
		if(a.sign_<0)
			return nan(prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,24,
			"BigFloat::sqrt: precision too large");
		auto fr=a.frexp(work,FloatRnd::nearest);
		std::int64_t e=fr.second;
		std::int64_t k=e/2;
		if(e<0&&(e&1))
			--k;
		const int rem=static_cast<int>(e-2*k);
		const double m=std::ldexp(fr.first.to_double(),rem);
		BigFloat y=from_double(std::sqrt(m),work,FloatRnd::nearest).
			ldexp(k);
		const std::size_t target=prec_add(work,8,
			"BigFloat::sqrt: precision too large");
		unsigned it=2;
		std::size_t bits=std::numeric_limits<double>::digits;
		while(bits<target){
			bits*=2u;
			++it;
		}
		for(unsigned i=0;i<it;++i){
			y=add(y,div(a,y,work,FloatRnd::nearest),
				  work,FloatRnd::nearest).ldexp(-1);
		}
		return y.rounded(prec,rnd);
	}

	static BigFloat rootn(const BigFloat&a,std::uint32_t k,
						  std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(k==0)
			return nan(prec);
		if(k==1)
			return a.rounded(prec,rnd);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf){
			if(a.sign_<0&&(k&1u)==0u)
				return nan(prec);
			return inf(a.sign_,prec);
		}
		if(a.sign_==0)
			return zero(prec);
		if(a.sign_<0&&(k&1u)==0u)
			return nan(prec);
		if(k==2)
			return sqrt(a,prec,rnd);
		BigInt n=a.mag_;
		std::int64_t e=a.exp_;
		const std::int64_t kk=static_cast<std::int64_t>(k);
		std::int64_t rem=e%kk;
		if(rem<0)
			rem+=kk;
		if(rem!=0){
			const std::size_t r=to_size_u64(
				static_cast<std::uint64_t>(rem),
				"BigFloat::rootn: exponent overflow");
			n<<=r;
			e=add_exp(e,-rem,"BigFloat::rootn: exponent overflow");
		}
		const std::size_t want=prec_add(prec,8,
			"BigFloat::rootn: precision too large");
		const std::size_t nb=n.bit_length();
		const std::size_t rb=(nb+k-1u)/k;
		std::size_t sh=0;
		if(rb<want){
			const std::size_t d=want-rb;
			sh=mul_sz_u32(d,k,"BigFloat::rootn: precision too large");
			n<<=sh;
			e=sub_exp_sz(e,sh,"BigFloat::rootn: exponent overflow");
		}
		auto rr=rootrem(n,k);
		BigFloat out;
		out.prec_=prec;
		out.kind_=FloatKind::finite;
		out.sign_=a.sign_<0?-1:1;
		out.mag_=std::move(rr.first);
		out.exp_=e/kk;
		round_mag(out.mag_,out.sign_,out.exp_,out.prec_,rnd,out.inex_,
				  !rr.second.is_zero());
		out.pack();
		return out;
	}

	static BigFloat cbrt(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		return rootn(a,3,prec,rnd);
	}

	static BigFloat recip(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		return div(one(prec),a,prec,rnd);
	}

	static BigFloat pow_ui(const BigFloat&a,std::uint64_t n,
						   std::size_t prec=default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(n==0)
			return one(prec).rounded(prec,rnd);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf((a.sign_<0&&(n&1u)!=0u)?-1:1,prec);
		if(a.sign_==0)
			return zero(prec);
		if(n==1)
			return a.rounded(prec,rnd);
		const std::size_t mb=a.mag_.bit_length();
		const bool exact_ok=n<=2u||
			(n<=std::numeric_limits<std::size_t>::max()/std::max<std::size_t>(mb,1u)&&
			 mb*static_cast<std::size_t>(n)<=
				 prec_add(prec,32,"BigFloat::pow_ui: precision too large"));
		if(!exact_ok){
			const std::size_t work=prec_add(prec,
				static_cast<std::size_t>(std::bit_width(n))+16u,
				"BigFloat::pow_ui: precision too large");
			BigFloat out=one(work);
			BigFloat base=a.rounded(work,FloatRnd::nearest);
			std::uint64_t e=n;
			while(e!=0){
				if((e&1u)!=0u)
					out=mul(out,base,work,FloatRnd::nearest);
				e>>=1;
				if(e!=0)
					base=sqr(base,work,FloatRnd::nearest);
			}
			return out.rounded(prec,rnd);
		}
		const int s=(a.sign_<0&&(n&1u)!=0u)?-1:1;
		BigInt m=mini_mp::pow(a.mag_,n);
		const std::int64_t e=mul_exp_u64(a.exp_,n,
			"BigFloat::pow_ui: exponent overflow");
		return from_parts(s,std::move(m),e,prec,rnd);
	}

	static BigFloat pow_si(const BigFloat&a,std::int64_t n,
						   std::size_t prec=default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		if(n>=0)
			return pow_ui(a,static_cast<std::uint64_t>(n),prec,rnd);
		const std::uint64_t e=abs_i64(n);
		prec=chk_prec(prec);
		if(e==0)
			return one(prec).rounded(prec,rnd);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return zero(prec);
		if(a.sign_==0)
			return inf(1,prec);
		const std::size_t work=prec_add(prec,
			static_cast<std::size_t>(std::bit_width(e))+16u,
			"BigFloat::pow_si: precision too large");
		return recip(pow_ui(a,e,work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat const_pi(std::size_t prec=default_prec,
							 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		return cache_pi(prec,rnd);
	}

	static BigFloat const_log2(std::size_t prec=default_prec,
							   FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		return cache_log2(prec,rnd);
	}

	static BigFloat exp(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return a.sign_<0?zero(prec):inf(1,prec);
		if(a.sign_==0)
			return one(prec);
		if(a.exp_==0&&a.mag_.is_one())
			return a.sign_>0?cache_e(prec,rnd):cache_inv_e(prec,rnd);
		const std::size_t small_work=prec_add(prec,48,
			"BigFloat::exp: precision too large");
		if(cmpabs(a,BigFloat(2,small_work))<=0)
			return exp_kernel(a.rounded(small_work,FloatRnd::nearest),
							  small_work).rounded(prec,rnd);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::exp: precision too large");
		const BigFloat l2=const_log2(work,FloatRnd::nearest);
		BigFloat qf=mul(a,cache_inv_log2(work,FloatRnd::nearest),
						 work,FloatRnd::nearest);
		if(qf.sign_!=0&&qf.ilogb()>62)
			return qf.sign()<0?zero(prec):inf(1,prec);
		BigInt q=qf.to_bigint(FloatRnd::nearest);
		std::int64_t qb=0;
		if(!to_i64_checked(qb,q))
			return q.sign()<0?zero(prec):inf(1,prec);
		const BigFloat qbf(q,work,FloatRnd::nearest);
		const BigFloat r=sub(a,mul(qbf,l2,work,FloatRnd::nearest),
							 work,FloatRnd::nearest);
		BigFloat y=exp_kernel(r,work);
		y=y.ldexp(qb);
		return y.rounded(prec,rnd);
	}

	static BigFloat exp2(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::finite&&a.is_integer()&&
		   a.fits_i64(FloatRnd::zero)){
			return one(prec).ldexp(a.to_i64(FloatRnd::zero)).
				rounded(prec,rnd);
		}
		const std::size_t work=prec_add(prec,64,
			"BigFloat::exp2: precision too large");
		return exp(mul(a,const_log2(work,FloatRnd::nearest),
					   work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat exp10(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::finite&&a.is_integer()&&
		   a.fits_i64(FloatRnd::zero)){
			const std::int64_t n=a.to_i64(FloatRnd::zero);
			const std::uint64_t e=abs_i64(n);
			if(e<=10000u){
				const std::size_t work=prec_add(prec,24,
					"BigFloat::exp10: precision too large");
				BigFloat y=pow_ui(BigFloat(10,work),e,work,
								   FloatRnd::nearest);
				if(n<0)
					y=recip(y,work,FloatRnd::nearest);
				return y.rounded(prec,rnd);
			}
		}
		const std::size_t work=prec_add(prec,80,
			"BigFloat::exp10: precision too large");
		return exp(mul(a,cache_log10(work,FloatRnd::nearest),
					   work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat expm1(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_!=FloatKind::finite)
			return exp(a,prec,rnd);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::expm1: precision too large");
		if(cmpabs(a,BigFloat(1,work).ldexp(-1))<0)
			return expm1_kernel(a.rounded(work,FloatRnd::nearest),
								prec,rnd);
		return sub(exp(a,work,FloatRnd::nearest),one(work),prec,rnd);
	}

	static BigFloat log(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return a.sign_<0?nan(prec):inf(1,prec);
		if(a.sign_<0)
			return nan(prec);
		if(a.sign_==0)
			return inf(-1,prec);
		std::int64_t p2e=0;
		if(exact_pow2_exp(p2e,a)){
			if(p2e==0)
				return zero(prec);
			if(p2e==1)
				return const_log2(prec,rnd);
			const std::size_t cwork=prec_add(prec,32,
				"BigFloat::log: precision too large");
			return mul(BigFloat(p2e,cwork),
					   const_log2(cwork,FloatRnd::nearest),prec,rnd);
		}
		const std::size_t work=prec_add(prec,80,
			"BigFloat::log: precision too large");
		auto fr=a.frexp(work,FloatRnd::nearest);
		BigFloat y=fr.first;
		std::int64_t e=fr.second;
		const BigFloat three_q=div(BigFloat(3,work),BigFloat(4,work),
								   work,FloatRnd::nearest);
		if(compare(y,three_q)<0){
			y=y.ldexp(1);
			--e;
		}
		BigFloat out=log_unit(y,work);
		if(e!=0){
			const BigFloat eb(e,work);
			out=add(out,mul(eb,const_log2(work,FloatRnd::nearest),
							work,FloatRnd::nearest),work,FloatRnd::nearest);
		}
		return out.rounded(prec,rnd);
	}

	static BigFloat log2(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		std::int64_t p2e=0;
		if(exact_pow2_exp(p2e,a))
			return BigFloat(p2e,prec,rnd);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::log2: precision too large");
		return mul(log(a,work,FloatRnd::nearest),
				   cache_inv_log2(work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat log10(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::finite&&a.sign_>0){
			if(compare(a,one(prec))==0)
				return zero(prec);
			if(compare(a,BigFloat(10,prec))==0)
				return one(prec).rounded(prec,rnd);
		}
		const std::size_t work=prec_add(prec,80,
			"BigFloat::log10: precision too large");
		return mul(log(a,work,FloatRnd::nearest),
				   cache_inv_log10(work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat log1p(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_!=FloatKind::finite)
			return log(add(one(prec),a,prec,FloatRnd::nearest),prec,rnd);
		const BigFloat neg_one(-1,prec);
		if(compare(a,neg_one)<0)
			return nan(prec);
		if(compare(a,neg_one)==0)
			return inf(-1,prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::log1p: precision too large");
		if(cmpabs(a,BigFloat(1,work).ldexp(-1))<0){
			const BigFloat z=div(a,add(BigFloat(2,work),a,work,
									   FloatRnd::nearest),
								 work,FloatRnd::nearest);
			return atanh_series(z,work).ldexp(1).rounded(prec,rnd);
		}
		return log(add(one(work),a,work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat pow(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(b.sign_==0)
			return one(prec);
		if(a.sign_==0){
			if(b.sign_<0)
				return inf(1,prec);
			return zero(prec);
		}
		if(a.kind_==FloatKind::inf)
			return b.sign_<0?zero(prec):inf(1,prec);
		if(compare(b,one(prec))==0)
			return a.rounded(prec,rnd);
		if(compare(b,one(prec).neg())==0)
			return recip(a,prec,rnd);
		const BigFloat half=one(prec).ldexp(-1);
		if(compare(b,half)==0)
			return sqrt(a,prec,rnd);
		if(compare(b,half.neg())==0)
			return recip(sqrt(a,prec_add(prec,24,
				"BigFloat::pow: precision too large"),FloatRnd::nearest),
				prec,rnd);
		if(a.sign_>0){
			if(compare(a,one(prec))==0)
				return one(prec).rounded(prec,rnd);
			if(compare(a,BigFloat(2,prec))==0)
				return exp2(b,prec,rnd);
			if(compare(a,BigFloat(10,prec))==0)
				return exp10(b,prec,rnd);
		}
		if(b.is_integer()&&b.fits_i64(FloatRnd::zero))
			return pow_si(a,b.to_i64(FloatRnd::zero),prec,rnd);
		const std::size_t work=prec_add(prec,96,
			"BigFloat::pow: precision too large");
		if(a.sign_<0){
			if(!b.is_integer())
				return nan(prec);
			const int s=integer_is_odd(b)?-1:1;
			BigFloat mag=pow(a.abs(),b,work,FloatRnd::nearest);
			return mag.with_sign(s,rnd).rounded(prec,rnd);
		}
		const BigFloat lx=log(a,work,FloatRnd::nearest);
		return exp(mul(b,lx,work,FloatRnd::nearest),prec,rnd);
	}

	static std::pair<BigFloat,BigFloat> sin_cos(
		const BigFloat&a,std::size_t prec=default_prec,
		FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.kind_==FloatKind::inf){
			return {nan(prec),nan(prec)};
		}
		if(a.sign_==0)
			return {zero(prec),one(prec)};
		const std::size_t quick=prec_add(prec,32,
			"BigFloat::sin_cos: precision too large");
		if(cmpabs(a,BigFloat(2,quick))<=0){
			auto sc=sin_cos_kernel(a.rounded(quick,FloatRnd::nearest),
								   quick);
			return {sc.first.rounded(prec,rnd),
					sc.second.rounded(prec,rnd)};
		}
		const std::size_t work=trig_work_prec(a,prec);
		const BigFloat pi=const_pi(work,FloatRnd::nearest);
		const BigFloat hpi=pi.ldexp(-1);
		BigFloat qf=div(a,hpi,work,FloatRnd::nearest);
		BigInt q=qf.to_bigint(FloatRnd::nearest);
		BigFloat r=sub(a,mul(BigFloat(q,work,FloatRnd::nearest),
							 hpi,work,FloatRnd::nearest),
					   work,FloatRnd::nearest);
		auto sc=sin_cos_kernel(r,work);
		BigFloat s=std::move(sc.first);
		BigFloat c=std::move(sc.second);
		switch(q.mod_u32(4u)){
		case 0:
			break;
		case 1:
			std::swap(s,c);
			c=c.neg();
			break;
		case 2:
			s=s.neg();
			c=c.neg();
			break;
		default:
			std::swap(s,c);
			s=s.neg();
			break;
		}
		return {s.rounded(prec,rnd),c.rounded(prec,rnd)};
	}

	static BigFloat sin(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.kind_==FloatKind::inf)
			return nan(prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t quick=prec_add(prec,32,
			"BigFloat::sin: precision too large");
		if(cmpabs(a,BigFloat(2,quick))<=0)
			return sin_kernel(a.rounded(quick,FloatRnd::nearest),
							  quick).rounded(prec,rnd);
		const std::size_t work=trig_work_prec(a,prec);
		const BigFloat hpi=const_pi(work,FloatRnd::nearest).ldexp(-1);
		BigFloat qf=div(a,hpi,work,FloatRnd::nearest);
		BigInt q=qf.to_bigint(FloatRnd::nearest);
		BigFloat r=sub(a,mul(BigFloat(q,work,FloatRnd::nearest),
							 hpi,work,FloatRnd::nearest),
					   work,FloatRnd::nearest);
		BigFloat y;
		switch(q.mod_u32(4u)){
		case 0:
			y=sin_kernel(r,work);
			break;
		case 1:
			y=cos_kernel(r,work);
			break;
		case 2:
			y=sin_kernel(r,work).neg();
			break;
		default:
			y=cos_kernel(r,work).neg();
			break;
		}
		return y.rounded(prec,rnd);
	}

	static BigFloat cos(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.kind_==FloatKind::inf)
			return nan(prec);
		if(a.sign_==0)
			return one(prec);
		const std::size_t quick=prec_add(prec,32,
			"BigFloat::cos: precision too large");
		if(cmpabs(a,BigFloat(2,quick))<=0)
			return cos_kernel(a.rounded(quick,FloatRnd::nearest),
							  quick).rounded(prec,rnd);
		const std::size_t work=trig_work_prec(a,prec);
		const BigFloat hpi=const_pi(work,FloatRnd::nearest).ldexp(-1);
		BigFloat qf=div(a,hpi,work,FloatRnd::nearest);
		BigInt q=qf.to_bigint(FloatRnd::nearest);
		BigFloat r=sub(a,mul(BigFloat(q,work,FloatRnd::nearest),
							 hpi,work,FloatRnd::nearest),
					   work,FloatRnd::nearest);
		BigFloat y;
		switch(q.mod_u32(4u)){
		case 0:
			y=cos_kernel(r,work);
			break;
		case 1:
			y=sin_kernel(r,work).neg();
			break;
		case 2:
			y=cos_kernel(r,work).neg();
			break;
		default:
			y=sin_kernel(r,work);
			break;
		}
		return y.rounded(prec,rnd);
	}

	static BigFloat tan(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=prec_add(chk_prec(prec),64,
			"BigFloat::tan: precision too large");
		auto sc=sin_cos(a,work,FloatRnd::nearest);
		return div(sc.first,sc.second,prec,rnd);
	}

	static BigFloat atan(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf){
			BigFloat hpi=const_pi(prec_add(prec,16,
				"BigFloat::atan: precision too large"),FloatRnd::nearest).
				ldexp(-1);
			return hpi.with_sign(a.sign_,rnd).rounded(prec,rnd);
		}
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::atan: precision too large");
		BigFloat y=atan_pos(a.abs().rounded(work,FloatRnd::nearest),
							work);
		if(a.sign_<0)
			y=y.neg();
		return y.rounded(prec,rnd);
	}

	static BigFloat asin(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||cmpabs(a,one(prec))>0)
			return nan(prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::asin: precision too large");
		const BigFloat halfv=one(work).ldexp(-1);
		if(compare(a,halfv)==0)
			return div(const_pi(work,FloatRnd::nearest),BigFloat(6,work),
					   prec,rnd);
		if(compare(a,halfv.neg())==0)
			return div(const_pi(work,FloatRnd::nearest),BigFloat(-6,work),
					   prec,rnd);
		if(cmpabs(a,halfv)<=0)
			return asin_series(a.rounded(work,FloatRnd::nearest),
							   work).rounded(prec,rnd);
		if(cmpabs(a,one(work))==0){
			BigFloat hpi=const_pi(work,FloatRnd::nearest).ldexp(-1);
			return hpi.with_sign(a.sign_,rnd).rounded(prec,rnd);
		}
		const BigFloat aa=a.rounded(work,FloatRnd::nearest);
		BigFloat t=sqrt(sub(one(work),sqr(aa,work,FloatRnd::nearest),
							work,FloatRnd::nearest),work,FloatRnd::nearest);
		BigFloat z=div(aa,add(one(work),t,work,FloatRnd::nearest),
					   work,FloatRnd::nearest);
		return atan(z,work,FloatRnd::nearest).ldexp(1).rounded(prec,rnd);
	}

	static BigFloat acos(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||
		   a.kind_==FloatKind::inf||cmpabs(a,one(prec))>0)
			return nan(prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::acos: precision too large");
		const BigFloat halfv=one(work).ldexp(-1);
		if(compare(a,halfv)==0)
			return div(const_pi(work,FloatRnd::nearest),BigFloat(3,work),
					   prec,rnd);
		if(compare(a,halfv.neg())==0)
			return div(mul(BigFloat(2,work),
						   const_pi(work,FloatRnd::nearest),
						   work,FloatRnd::nearest),
					   BigFloat(3,work),prec,rnd);
		BigFloat hpi=const_pi(work,FloatRnd::nearest).ldexp(-1);
		return sub(hpi,asin(a,work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat atan2(const BigFloat&y,const BigFloat&x,
						  std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(x.kind_==FloatKind::nan||y.kind_==FloatKind::nan)
			return nan(prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::atan2: precision too large");
		if(x.sign_>0)
			return atan(div(y,x,work,FloatRnd::nearest),prec,rnd);
		if(x.sign_<0){
			BigFloat a=atan(div(y,x,work,FloatRnd::nearest),
							work,FloatRnd::nearest);
			BigFloat p=const_pi(work,FloatRnd::nearest);
			a=(y.sign_>=0)?add(a,p,work,FloatRnd::nearest):
						 sub(a,p,work,FloatRnd::nearest);
			return a.rounded(prec,rnd);
		}
		if(y.sign_>0)
			return const_pi(work,FloatRnd::nearest).ldexp(-1).
				rounded(prec,rnd);
		if(y.sign_<0)
			return const_pi(work,FloatRnd::nearest).ldexp(-1).neg().
				rounded(prec,rnd);
		return nan(prec);
	}

	static BigFloat sinh(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf(a.sign_,prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::sinh: precision too large");
		if(cmpabs(a,BigFloat(1,work).ldexp(-1))<0){
			BigFloat t=expm1(a,work,FloatRnd::nearest);
			BigFloat num=mul(t,add(t,BigFloat(2,work),work,
								   FloatRnd::nearest),work,
							 FloatRnd::nearest);
			BigFloat den=mul(BigFloat(2,work),
							 add(t,one(work),work,FloatRnd::nearest),
							 work,FloatRnd::nearest);
			return div(num,den,prec,rnd);
		}
		BigFloat e=exp(a,work,FloatRnd::nearest);
		return sub(e,recip(e,work,FloatRnd::nearest),
				   work,FloatRnd::nearest).ldexp(-1).rounded(prec,rnd);
	}

	static BigFloat cosh(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf(1,prec);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::cosh: precision too large");
		BigFloat e=exp(a.abs(),work,FloatRnd::nearest);
		return add(e,recip(e,work,FloatRnd::nearest),
				   work,FloatRnd::nearest).ldexp(-1).rounded(prec,rnd);
	}

	static std::pair<BigFloat,BigFloat> sinh_cosh(
		const BigFloat&a,std::size_t prec=default_prec,
		FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return {nan(prec),nan(prec)};
		if(a.kind_==FloatKind::inf)
			return {inf(a.sign_,prec),inf(1,prec)};
		if(a.sign_==0)
			return {zero(prec),one(prec)};
		const std::size_t work=prec_add(prec,64,
			"BigFloat::sinh_cosh: precision too large");
		BigFloat sh;
		BigFloat ch;
		if(cmpabs(a,one(work).ldexp(-1))<0){
			BigFloat t=expm1(a,work,FloatRnd::nearest);
			BigFloat ep=add(one(work),t,work,FloatRnd::nearest);
			BigFloat den=mul(BigFloat(2,work),ep,work,FloatRnd::nearest);
			sh=div(mul(t,add(t,BigFloat(2,work),work,
							 FloatRnd::nearest),work,FloatRnd::nearest),
				   den,work,FloatRnd::nearest);
			BigFloat num=add(add(sqr(t,work,FloatRnd::nearest),
								 t.ldexp(1),work,FloatRnd::nearest),
							 BigFloat(2,work),work,FloatRnd::nearest);
			ch=div(num,den,work,FloatRnd::nearest);
		}else{
			BigFloat e=exp(a.abs(),work,FloatRnd::nearest);
			BigFloat inv=recip(e,work,FloatRnd::nearest);
			sh=sub(e,inv,work,FloatRnd::nearest).ldexp(-1);
			if(a.sign_<0)
				sh=sh.neg();
			ch=add(e,inv,work,FloatRnd::nearest).ldexp(-1);
		}
		return {sh.rounded(prec,rnd),ch.rounded(prec,rnd)};
	}

	static BigFloat tanh(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return BigFloat(a.sign_,prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,64,
			"BigFloat::tanh: precision too large");
		if(cmpabs(a,one(work).ldexp(-1))<0){
			BigFloat t=expm1(a.ldexp(1),work,FloatRnd::nearest);
			return div(t,add(t,BigFloat(2,work),work,
					   FloatRnd::nearest),prec,rnd);
		}
		BigFloat e=exp(a.abs(),work,FloatRnd::nearest);
		if(e.kind_==FloatKind::inf)
			return BigFloat(a.sign_,prec);
		BigFloat inv=recip(e,work,FloatRnd::nearest);
		BigFloat y=div(sub(e,inv,work,FloatRnd::nearest),
					   add(e,inv,work,FloatRnd::nearest),work,
					   FloatRnd::nearest);
		if(a.sign_<0)
			y=y.neg();
		return y.rounded(prec,rnd);
	}

	static BigFloat asinh(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_!=FloatKind::finite)
			return a.rounded(prec,rnd);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::asinh: precision too large");
		if(cmpabs(a,one(work).ldexp(-4))<=0)
			return asinh_series(a.rounded(work,FloatRnd::nearest),
								work).rounded(prec,rnd);
		BigFloat ax=a.abs().rounded(work,FloatRnd::nearest);
		if(compare(ax,one(work).ldexp(-1))<=0){
			BigFloat x2=sqr(ax,work,FloatRnd::nearest);
			BigFloat t=sqrt(add(one(work),x2,work,FloatRnd::nearest),
							work,FloatRnd::nearest);
			BigFloat u=add(ax,div(x2,add(one(work),t,work,
										 FloatRnd::nearest),
								 work,FloatRnd::nearest),
						   work,FloatRnd::nearest);
			const std::size_t lprec=prec_add(prec,16,
				"BigFloat::asinh: precision too large");
			BigFloat y=log1p(u,lprec,FloatRnd::nearest);
			if(a.sign_<0)
				y=y.neg();
			return y.rounded(prec,rnd);
		}
		BigFloat y=log(add(ax,sqrt(add(sqr(ax,work,FloatRnd::nearest),
										 one(work),work,FloatRnd::nearest),
								  work,FloatRnd::nearest),
							 work,FloatRnd::nearest),work,FloatRnd::nearest);
		if(a.sign_<0)
			y=y.neg();
		return y.rounded(prec,rnd);
	}

	static BigFloat acosh(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.sign_<0||
		   (a.kind_==FloatKind::finite&&compare(a,one(prec))<0))
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return inf(1,prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::acosh: precision too large");
		BigFloat am1=sub(a,one(work),work,FloatRnd::nearest);
		BigFloat ap1=add(a,one(work),work,FloatRnd::nearest);
		return log(add(a,mul(sqrt(am1,work,FloatRnd::nearest),
							 sqrt(ap1,work,FloatRnd::nearest),
							 work,FloatRnd::nearest),
					   work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat atanh(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.kind_==FloatKind::inf)
			return nan(prec);
		const int ca=cmpabs(a,one(prec));
		if(ca>0)
			return nan(prec);
		if(ca==0)
			return inf(a.sign_,prec);
		if(a.sign_==0)
			return zero(prec);
		const std::size_t work=prec_add(prec,72,
			"BigFloat::atanh: precision too large");
		if(cmpabs(a,one(work).ldexp(-2))<=0)
			return atanh_series(a.rounded(work,FloatRnd::nearest),
								work).rounded(prec,rnd);
		return log(div(add(one(work),a,work,FloatRnd::nearest),
					   sub(one(work),a,work,FloatRnd::nearest),
					   work,FloatRnd::nearest),
				   work,FloatRnd::nearest).ldexp(-1).rounded(prec,rnd);
	}

	static BigFloat gamma(const BigFloat&a,std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return a.sign_<0?nan(prec):inf(1,prec);
		if(a.sign_==0)
			return inf(1,prec);
		if(a.sign_<0&&a.is_integer())
			return nan(prec);
		const std::size_t work=prec_add(prec,112,
			"BigFloat::gamma: precision too large");
		if(a.sign_<0){
			const BigFloat pi=const_pi(work,FloatRnd::nearest);
			BigFloat s=sin(mul(pi,a,work,FloatRnd::nearest),
						   work,FloatRnd::nearest);
			BigFloat g=gamma(sub(one(work),a,work,FloatRnd::nearest),
							 work,FloatRnd::nearest);
			return div(pi,mul(s,g,work,FloatRnd::nearest),prec,rnd);
		}
		if(a.is_integer()&&a.fits_u64(FloatRnd::zero)){
			const std::uint64_t n=a.to_u64(FloatRnd::zero);
			if(n>0&&n<=10000u)
				return BigFloat(factorial(n-1u),prec,rnd);
		}
		std::uint64_t half_idx=0;
		if(positive_half_index(half_idx,a)&&half_idx<=10000u)
			return gamma_half_integer(half_idx,work).rounded(prec,rnd);
		if(prec<=192u)
			return gamma_lanczos(a.rounded(work,FloatRnd::nearest),
								 prec,rnd);
		return gamma_spouge(a.rounded(work,FloatRnd::nearest),
							prec,rnd);
	}

	static BigFloat erf(const BigFloat&a,std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return BigFloat(a.sign_,prec);
		if(a.sign_==0)
			return zero(prec);
		if(a.sign_<0)
			return erf(a.neg(),prec,rnd).neg();
		const double xd=a.to_double();
		if(std::isfinite(xd)&&
		   xd*xd>static_cast<double>(prec+4u)*0.69314718055994530942)
			return one(prec).rounded(prec,rnd);
		if(!std::isfinite(xd))
			return one(prec).rounded(prec,rnd);
		const std::size_t work=erf_work_prec(prec,xd);
		if(std::isfinite(xd)&&xd>4.0){
			BigFloat tail=erfc_asymp(a.rounded(work,FloatRnd::nearest),
									 work);
			return sub(one(work),tail,prec,rnd);
		}
		return erf_series(a.rounded(work,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat erfc(const BigFloat&a,std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf)
			return a.sign_>0?zero(prec):BigFloat(2,prec);
		if(a.sign_==0)
			return one(prec);
		const double xd=a.to_double();
		if(!std::isfinite(xd))
			return a.sign_>0?zero(prec):BigFloat(2,prec);
		const std::size_t work=erf_work_prec(prec,std::fabs(xd));
		if(a.sign_>0&&std::isfinite(xd)&&xd>4.0)
			return erfc_asymp(a.rounded(work,FloatRnd::nearest),
							  work).rounded(prec,rnd);
		if(a.sign_<0){
			BigFloat tail=erfc(a.neg(),work,FloatRnd::nearest);
			return sub(BigFloat(2,work),tail,prec,rnd);
		}
		const std::size_t ework=prec_add(prec,48,
			"BigFloat::erfc: precision too large");
		return sub(one(ework),erf(a,ework,FloatRnd::nearest),prec,rnd);
	}

	static BigFloat dim(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(chk_prec(prec));
		return compare(a,b)>0?sub(a,b,prec,rnd):zero(prec);
	}

	static BigFloat min(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		return compare(a,b)<=0?a.rounded(prec,rnd):b.rounded(prec,rnd);
	}

	static BigFloat max(const BigFloat&a,const BigFloat&b,
						std::size_t prec=default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		return compare(a,b)>=0?a.rounded(prec,rnd):b.rounded(prec,rnd);
	}

	static BigFloat fmod(const BigFloat&a,const BigFloat&b,
						 std::size_t prec=default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan||
		   a.kind_==FloatKind::inf||
		   (b.kind_==FloatKind::finite&&b.sign_==0))
			return nan(prec);
		if(a.sign_==0)
			return zero(prec);
		if(b.kind_==FloatKind::inf)
			return a.rounded(prec,rnd);
		const int ac=cmpabs(a,b);
		if(ac<0)
			return a.rounded(prec,rnd);
		if(ac==0)
			return zero(prec);
		const std::int64_t e=std::min(a.exp_,b.exp_);
		const std::size_t ash=exp_gap(a.exp_,e,
			"BigFloat::fmod: exponent gap too large");
		const std::size_t bsh=exp_gap(b.exp_,e,
			"BigFloat::fmod: exponent gap too large");
		if(bsh==0&&b.mag_.limbs_.size()==2&&
		   !detail::pow2_abs(b.mag_.limbs_)){
			detail::limbs_t r=detail::mod2_shl_absl(
				a.mag_.limbs_,ash,b.mag_.limbs_);
			if(r.empty())
				return zero(prec);
			if(detail::bit_length(r)<=prec)
				return from_exact_limbs(a.sign_,std::move(r),e,prec);
			return from_limbs(a.sign_,std::move(r),e,prec,rnd);
		}
		BigInt am=a.mag_;
		BigInt bm=b.mag_;
		am<<=ash;
		bm<<=bsh;
		detail::limbs_t r;
		if(detail::pow2_abs(bm.limbs_)){
			r=detail::low_bits_abs(am.limbs_,detail::pow2_exp_abs(bm.limbs_));
		}else if(bm.limbs_.size()==1){
			const std::uint64_t rem=detail::mod_limb(am.limbs_,bm.limbs_[0]);
			if(rem!=0)
				r.push_back(rem);
		}else{
			detail::ensure_at();
			if(detail::use_bzdiv(am.limbs_.size(),bm.limbs_.size())){
				auto qr=detail::dvmbz_abs(am.limbs_,bm.limbs_);
				r=std::move(qr.second);
			}else{
				r=detail::modk_absl(am.limbs_,bm.limbs_);
			}
		}
		if(r.empty())
			return zero(prec);
		if(detail::bit_length(r)<=prec)
			return from_exact_limbs(a.sign_,std::move(r),e,prec);
		return from_limbs(a.sign_,std::move(r),e,prec,rnd);
	}

	static BigFloat remainder(const BigFloat&a,const BigFloat&b,
							  std::size_t prec=default_prec,
							  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan||
		   a.kind_==FloatKind::inf||
		   (b.kind_==FloatKind::finite&&b.sign_==0))
			return nan(prec);
		if(a.sign_==0)
			return zero(prec);
		if(b.kind_==FloatKind::inf)
			return a.rounded(prec,rnd);
		const int ac=cmpabs(a,b);
		if(ac==0)
			return zero(prec);
		const std::int64_t e=std::min(a.exp_,b.exp_);
		BigInt am=a.mag_;
		BigInt bm=b.mag_;
		am<<=exp_gap(a.exp_,e,
			"BigFloat::remainder: exponent gap too large");
		bm<<=exp_gap(b.exp_,e,
			"BigFloat::remainder: exponent gap too large");
		detail::limbs_t q;
		detail::limbs_t r;
		if(detail::pow2_abs(bm.limbs_)){
			const std::size_t sh=detail::pow2_exp_abs(bm.limbs_);
			r=detail::low_bits_abs(am.limbs_,sh);
			if(sh<am.bit_length()&&detail::test_bit(am.limbs_,sh))
				q.push_back(1);
		}else{
			std::pair<detail::limbs_t,detail::limbs_t> qr;
			if(bm.limbs_.size()==1){
				qr=detail::dvmk_absl(am.limbs_,bm.limbs_);
			}else if(bm.limbs_.size()==2){
				qr=detail::dvm2_absl(am.limbs_,bm.limbs_);
			}else{
				detail::ensure_at();
				qr=detail::use_bzdiv(am.limbs_.size(),bm.limbs_.size())
					?detail::dvmbz_abs(am.limbs_,bm.limbs_)
					:detail::dvmk_absl(am.limbs_,bm.limbs_);
			}
			q=std::move(qr.first);
			r=std::move(qr.second);
		}
		if(r.empty())
			return zero(prec);
		const int cmp=detail::cmp_abs_dbl(bm.limbs_,r);
		int rs=a.sign_;
		if(cmp<0||(cmp==0&&!q.empty()&&((q[0]&1u)!=0u))){
			detail::limbs_t rm;
			detail::sub_abs_to(rm,bm.limbs_,r);
			r=std::move(rm);
			rs=-rs;
		}
		if(detail::bit_length(r)<=prec)
			return from_exact_limbs(rs,std::move(r),e,prec);
		return from_limbs(rs,std::move(r),e,prec,rnd);
	}

	static std::pair<BigFloat,BigFloat> modf(
		const BigFloat&a,std::size_t prec=default_prec,
		FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||a.kind_==FloatKind::inf)
			return {a.rounded(prec,rnd),nan(prec)};
		if(a.sign_==0)
			return {zero(prec),zero(prec)};
		if(a.exp_>=0)
			return {a.rounded(prec,rnd),zero(prec)};
		const std::size_t sh=neg_exp_size(a.exp_,
			"BigFloat::modf: exponent too small");
		if(sh>=a.mag_.bit_length())
			return {zero(prec),a.rounded(prec,rnd)};
		detail::limbs_t q=a.mag_.limbs_;
		detail::shr_ip(q,sh);
		detail::limbs_t r=detail::low_bits_abs(a.mag_.limbs_,sh);
		BigFloat ip=from_limbs(a.sign_,std::move(q),0,prec,rnd);
		BigFloat fp=detail::bit_length(r)<=prec?
			from_exact_limbs(a.sign_,std::move(r),a.exp_,prec):
			from_limbs(a.sign_,std::move(r),a.exp_,prec,rnd);
		return {std::move(ip),std::move(fp)};
	}

	static BigFloat hypot(const BigFloat&a,const BigFloat&b,
						  std::size_t prec=default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		prec=chk_prec(prec);
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return nan(prec);
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf)
			return inf(1,prec);
		if(a.sign_==0)
			return b.abs().rounded(prec,rnd);
		if(b.sign_==0)
			return a.abs().rounded(prec,rnd);
		const std::size_t work=prec_add(prec,16,
			"BigFloat::hypot: precision too large");
		const BigFloat&hi=(cmpabs(a,b)>=0)?a:b;
		const BigFloat&lo=(&hi==&a)?b:a;
		const std::int64_t hh=top_exp(hi);
		const std::int64_t lh=top_exp(lo);
		const std::size_t gap=exp_gap(hh,lh,
			"BigFloat::hypot: exponent gap too large");
		if(gap>prec_add(prec,8,
			  "BigFloat::hypot: precision too large")/2u)
			return from_parts_more(1,hi.mag_,hi.exp_,prec,rnd,true);
		if(gap>512u){
			const std::size_t sw=prec_add(prec,32,
				"BigFloat::hypot: precision too large");
			BigFloat r=div(lo.abs(),hi.abs(),sw,FloatRnd::nearest);
			BigFloat s=sqr(r,sw,FloatRnd::nearest);
			BigFloat u=add(one(sw),s,sw,FloatRnd::nearest);
			BigFloat w=sqrt(u,sw,FloatRnd::nearest);
			return mul(hi.abs(),w,prec,rnd);
		}
		BigFloat sqh=sqr(hi,work,FloatRnd::nearest);
		BigFloat sum=fma(lo,lo,sqh,work,FloatRnd::nearest);
		return sqrt(sum,prec,rnd);
	}

	static int cmpabs(const BigFloat&a,const BigFloat&b){
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			detail::throw_dom("BigFloat::cmpabs: nan");
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.kind_==b.kind_)
				return 0;
			return a.kind_==FloatKind::inf?1:-1;
		}
		if(a.sign_==0||b.sign_==0){
			if(a.sign_==b.sign_)
				return 0;
			return a.sign_==0?-1:1;
		}
		const std::int64_t ah=top_exp(a);
		const std::int64_t bh=top_exp(b);
		if(ah!=bh)
			return ah<bh?-1:1;
		if(a.exp_==b.exp_)
			return detail::cmp_abs(a.mag_.limbs_,b.mag_.limbs_);
		BigInt am=a.mag_;
		BigInt bm=b.mag_;
		const std::int64_t e=std::min(a.exp_,b.exp_);
		am<<=exp_gap(a.exp_,e,"BigFloat::cmpabs: exponent gap too large");
		bm<<=exp_gap(b.exp_,e,"BigFloat::cmpabs: exponent gap too large");
		return compare(am,bm);
	}

	BigFloat next_up() const{
		BigFloat out=*this;
		if(out.kind_==FloatKind::nan)
			return out;
		if(out.kind_==FloatKind::inf)
			return out;
		if(out.sign_<0)
			out.step_mag(false);
		else
			out.step_mag(true);
		return out;
	}

	BigFloat next_down() const{
		BigFloat out=*this;
		if(out.kind_==FloatKind::nan)
			return out;
		if(out.kind_==FloatKind::inf)
			return out;
		if(out.sign_<0)
			out.step_mag(true);
		else
			out.step_mag(false);
		return out;
	}

	BigFloat next_toward(const BigFloat&to) const{
		if(kind_==FloatKind::nan||to.kind_==FloatKind::nan)
			return nan(prec_);
		const int c=compare(*this,to);
		if(c==0)
			return *this;
		return c<0?next_up():next_down();
	}

	BigFloat&operator+=(const BigFloat&rhs){
		*this=add(*this,rhs,prec_);
		return *this;
	}

	BigFloat&operator-=(const BigFloat&rhs){
		*this=sub(*this,rhs,prec_);
		return *this;
	}

	BigFloat&operator*=(const BigFloat&rhs){
		*this=mul(*this,rhs,prec_);
		return *this;
	}

	BigFloat&operator/=(const BigFloat&rhs){
		*this=div(*this,rhs,prec_);
		return *this;
	}

	friend BigFloat operator+(const BigFloat&a,const BigFloat&b){
		return add(a,b,std::max(a.prec_,b.prec_));
	}
	friend BigFloat operator-(const BigFloat&a,const BigFloat&b){
		return sub(a,b,std::max(a.prec_,b.prec_));
	}
	friend BigFloat operator*(const BigFloat&a,const BigFloat&b){
		return mul(a,b,std::max(a.prec_,b.prec_));
	}
	friend BigFloat operator/(const BigFloat&a,const BigFloat&b){
		return div(a,b,std::max(a.prec_,b.prec_));
	}
	friend BigFloat operator-(const BigFloat&a){ return a.neg(); }

	friend bool operator==(const BigFloat&a,const BigFloat&b){
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			return false;
		if(a.kind_!=b.kind_)
			return false;
		if(a.kind_==FloatKind::inf)
			return a.sign_==b.sign_;
		return a.sign_==b.sign_&&a.exp_==b.exp_&&a.mag_==b.mag_;
	}
	friend bool operator!=(const BigFloat&a,const BigFloat&b){
		return !(a==b);
	}
	friend bool operator<(const BigFloat&a,const BigFloat&b){
		return compare(a,b)<0;
	}
	friend bool operator>(const BigFloat&a,const BigFloat&b){
		return compare(a,b)>0;
	}
	friend bool operator<=(const BigFloat&a,const BigFloat&b){
		return compare(a,b)<=0;
	}
	friend bool operator>=(const BigFloat&a,const BigFloat&b){
		return compare(a,b)>=0;
	}

	friend int compare(const BigFloat&a,const BigFloat&b){
		if(a.kind_==FloatKind::nan||b.kind_==FloatKind::nan)
			detail::throw_dom("BigFloat compare: nan");
		if(a.kind_==FloatKind::inf||b.kind_==FloatKind::inf){
			if(a.kind_==b.kind_)
				return (a.sign_==b.sign_)?0:((a.sign_<b.sign_)?-1:1);
			return a.kind_==FloatKind::inf?
				(a.sign_<0?-1:1):(b.sign_<0?1:-1);
		}
		if(a.sign_!=b.sign_)
			return (a.sign_<b.sign_)?-1:1;
		if(a.sign_==0)
			return 0;
		const int c=cmpabs(a,b);
		return a.sign_>0?c:-c;
	}

  private:
	static std::size_t chk_prec(std::size_t p){
		if(p==0)
			detail::throw_inv("BigFloat: precision must be positive");
		return p;
	}

	static std::size_t prec_add(std::size_t a,std::size_t b,
								const char*msg){
		if(a>std::numeric_limits<std::size_t>::max()-b)
			detail::throw_ovf(msg);
		return a+b;
	}

	static std::size_t mul_sz_u32(std::size_t a,std::uint32_t b,
								  const char*msg){
		if(b!=0&&a>std::numeric_limits<std::size_t>::max()/b)
			detail::throw_ovf(msg);
		return a*static_cast<std::size_t>(b);
	}

	static std::uint64_t to_u64_sz(std::size_t v,const char*msg){
		if constexpr(sizeof(std::size_t)>sizeof(std::uint64_t)){
			if(v>std::numeric_limits<std::uint64_t>::max())
				detail::throw_ovf(msg);
		}
		return static_cast<std::uint64_t>(v);
	}

	static std::size_t to_size_u64(std::uint64_t v,const char*msg){
		if(v>std::numeric_limits<std::size_t>::max())
			detail::throw_ovf(msg);
		return static_cast<std::size_t>(v);
	}

	static std::uint64_t abs_i64(std::int64_t v) noexcept{
		return v<0?
			static_cast<std::uint64_t>(-(v+1))+std::uint64_t(1):
			static_cast<std::uint64_t>(v);
	}

	static std::int64_t add_exp(std::int64_t a,std::int64_t b,
								const char*msg){
		if(b>0&&a>std::numeric_limits<std::int64_t>::max()-b)
			detail::throw_ovf(msg);
		if(b<0&&a<std::numeric_limits<std::int64_t>::min()-b)
			detail::throw_ovf(msg);
		return static_cast<std::int64_t>(a+b);
	}

	static std::int64_t mul_exp_u64(std::int64_t a,std::uint64_t b,
									const char*msg){
		if(b==0||a==0)
			return 0;
		const std::uint64_t av=abs_i64(a);
		const std::uint64_t lim=a<0?
			(static_cast<std::uint64_t>(
				std::numeric_limits<std::int64_t>::max())+1u):
			static_cast<std::uint64_t>(
				std::numeric_limits<std::int64_t>::max());
		if(av>lim/b)
			detail::throw_ovf(msg);
		const std::uint64_t p=av*b;
		if(a>=0)
			return static_cast<std::int64_t>(p);
		if(p==lim)
			return std::numeric_limits<std::int64_t>::min();
		return -static_cast<std::int64_t>(p);
	}

	static std::int64_t add_exp_sz(std::int64_t e,std::size_t bits,
								   const char*msg){
		if(bits>static_cast<std::size_t>(
			   std::numeric_limits<std::int64_t>::max()))
			detail::throw_ovf(msg);
		return add_exp(e,static_cast<std::int64_t>(bits),msg);
	}

	static std::int64_t sub_exp_sz(std::int64_t e,std::size_t bits,
								   const char*msg){
		if(bits>static_cast<std::size_t>(
			   std::numeric_limits<std::int64_t>::max()))
			detail::throw_ovf(msg);
		return add_exp(e,-static_cast<std::int64_t>(bits),msg);
	}

	static std::size_t exp_gap(std::int64_t hi,std::int64_t lo,
							   const char*msg){
		MINI_MP_ASSERT(hi>=lo);
		std::uint64_t d=0;
		if(hi>=0&&lo<0){
			const std::uint64_t a=static_cast<std::uint64_t>(hi);
			const std::uint64_t b=abs_i64(lo);
			if(a>std::numeric_limits<std::uint64_t>::max()-b)
				detail::throw_ovf(msg);
			d=a+b;
		}else{
			d=static_cast<std::uint64_t>(hi-lo);
		}
		return to_size_u64(d,msg);
	}

	static std::size_t neg_exp_size(std::int64_t e,const char*msg){
		MINI_MP_ASSERT(e<0);
		return to_size_u64(abs_i64(e),msg);
	}

	static std::int64_t top_exp(const BigFloat&x){
		return add_exp_sz(x.exp_,x.mag_.bit_length(),
						  "BigFloat: exponent overflow");
	}

	static std::int64_t add_exp_sz_sat(std::int64_t e,
									   std::size_t bits) noexcept{
		if(bits>static_cast<std::size_t>(
			   std::numeric_limits<std::int64_t>::max())){
			return std::numeric_limits<std::int64_t>::max();
		}
		const std::int64_t b=static_cast<std::int64_t>(bits);
		if(e>std::numeric_limits<std::int64_t>::max()-b)
			return std::numeric_limits<std::int64_t>::max();
		return e+b;
	}

	static std::int64_t top_exp_sat(std::int64_t e,
									std::size_t bits) noexcept{
		if(bits==0)
			return e;
		return add_exp_sz_sat(e,bits-1u);
	}

	static std::uint64_t top_bits(const detail::limbs_t&x,
								  std::size_t bits,
								  std::size_t take) noexcept{
		const std::size_t sh=bits-take;
		const std::size_t idx=sh/64u;
		const unsigned off=static_cast<unsigned>(sh%64u);
		std::uint64_t out=(idx<x.size())?(x[idx]>>off):0;
		if(off!=0&&idx+1u<x.size())
			out|=x[idx+1u]<<(64u-off);
		if(take<64u)
			out&=(std::uint64_t(1)<<take)-1u;
		return out;
	}

	static bool low_nz(const BigInt&x,std::size_t bits) noexcept{
		if(bits==0||x.limbs_.empty())
			return false;
		const std::size_t full=bits/64u;
		const unsigned rem=static_cast<unsigned>(bits%64u);
		const std::size_t n=std::min(full,x.limbs_.size());
		for(std::size_t i=0;i<n;++i){
			if(x.limbs_[i]!=0)
				return true;
		}
		if(rem!=0&&full<x.limbs_.size()){
			const limb_t mask=(limb_t(1)<<rem)-1u;
			return (x.limbs_[full]&mask)!=0;
		}
		return full>=x.limbs_.size()?detail::lb_nz(x.limbs_.data(),
			x.limbs_.size())!=0:false;
	}

	static void set_pos(BigInt&x){
		x.sign_=x.limbs_.empty()?0:1;
	}

	static int cmp_mag_exp(const BigInt&a,std::int64_t ae,
						   const BigInt&b,std::int64_t be){
		if(a.is_zero()||b.is_zero()){
			if(a.is_zero()&&b.is_zero())
				return 0;
			return a.is_zero()?-1:1;
		}
		const std::int64_t ah=add_exp_sz(ae,a.bit_length(),
			"BigFloat: exponent overflow");
		const std::int64_t bh=add_exp_sz(be,b.bit_length(),
			"BigFloat: exponent overflow");
		if(ah!=bh)
			return ah<bh?-1:1;
		if(ae==be)
			return detail::cmp_abs(a.limbs_,b.limbs_);
		const std::int64_t e=std::min(ae,be);
		detail::limbs_t am=a.limbs_;
		if(ae!=e)
			detail::shl_ip(am,exp_gap(ae,e,
				"BigFloat: exponent gap too large"));
		detail::limbs_t bm=b.limbs_;
		if(be!=e)
			detail::shl_ip(bm,exp_gap(be,e,
				"BigFloat: exponent gap too large"));
		return detail::cmp_abs(am,bm);
	}

	void pack(){
		if(kind_!=FloatKind::finite)
			return;
		if(mag_.sign_<0){
			mag_=-mag_;
			sign_=-sign_;
		}
		if(sign_==0||mag_.is_zero()){
			sign_=0;
			exp_=0;
			mag_=BigInt();
			return;
		}
		sign_=(sign_<0)?-1:1;
		set_pos(mag_);
		const std::size_t z=detail::ctz(mag_.limbs_);
		if(z!=0){
			detail::shr_ip(mag_.limbs_,z);
			set_pos(mag_);
			exp_=add_exp_sz(exp_,z,"BigFloat: exponent overflow");
		}
	}

	void fill_prec(){
		if(kind_!=FloatKind::finite||sign_==0)
			return;
		const std::size_t bits=mag_.bit_length();
		if(bits<prec_){
			const std::size_t sh=prec_-bits;
			mag_<<=sh;
			exp_=sub_exp_sz(exp_,sh,"BigFloat: exponent overflow");
		}
	}

	void step_mag(bool away){
		if(kind_!=FloatKind::finite)
			return;
		if(sign_==0){
			sign_=away?1:-1;
			mag_=BigInt(1);
			exp_=sub_exp_sz(0,prec_-1u,"BigFloat: exponent overflow");
			return;
		}
		fill_prec();
		if(away){
			detail::add_one_ip(mag_.limbs_);
			set_pos(mag_);
			if(mag_.bit_length()>prec_){
				detail::shr_ip(mag_.limbs_,std::size_t(1));
				set_pos(mag_);
				exp_=add_exp(exp_,1,"BigFloat: exponent overflow");
			}
		}else{
			if(mag_.is_one()){
				sign_=0;
				exp_=0;
				mag_=BigInt();
				return;
			}
			detail::sub_abs_ip(mag_.limbs_,detail::limbs_t{1});
			set_pos(mag_);
		}
		pack();
	}

	void round_self(FloatRnd rnd){
		inex_=0;
		if(kind_!=FloatKind::finite)
			return;
		pack();
		if(sign_==0)
			return;
		round_mag(mag_,sign_,exp_,prec_,rnd,inex_);
		if(!mag_.limbs_.empty()&&((mag_.limbs_[0]&1u)!=0u))
			return;
		pack();
	}

	static void round_mag(BigInt&mag,int sign,std::int64_t&exp,
						  std::size_t prec,FloatRnd rnd,int&inex,
						  bool more=false){
		const std::size_t bits=mag.bit_length();
		if(bits<=prec){
			if(!more){
				inex=0;
				return;
			}
			bool inc=false;
			if((rnd==FloatRnd::up&&sign>0)||
			   (rnd==FloatRnd::down&&sign<0)||
			   rnd==FloatRnd::away)
				inc=true;
			if(inc){
				detail::add_one_ip(mag.limbs_);
				set_pos(mag);
				if(mag.bit_length()>prec){
					detail::shr_ip(mag.limbs_,std::size_t(1));
					set_pos(mag);
					exp=add_exp(exp,1,"BigFloat: exponent overflow");
				}
			}
			inex=sign>0?(inc?1:-1):(inc?-1:1);
			return;
		}
		const std::size_t drop=bits-prec;
		const bool guard=detail::test_bit(mag.limbs_,drop-1u);
		const bool sticky=low_nz(mag,drop-1u)||more;
		const bool lost=guard||sticky;
		bool inc=false;
		detail::shr_ip(mag.limbs_,drop);
		set_pos(mag);
		if(lost){
			switch(rnd){
			case FloatRnd::nearest:
				inc=guard&&(sticky||mag.is_odd());
				break;
			case FloatRnd::zero:
				inc=false;
				break;
			case FloatRnd::up:
				inc=sign>0;
				break;
			case FloatRnd::down:
				inc=sign<0;
				break;
			case FloatRnd::away:
				inc=true;
				break;
			}
		}
		if(inc){
			detail::add_one_ip(mag.limbs_);
			set_pos(mag);
		}
		exp=add_exp_sz(exp,drop,"BigFloat: exponent overflow");
		if(mag.bit_length()>prec){
			detail::shr_ip(mag.limbs_,std::size_t(1));
			set_pos(mag);
			exp=add_exp(exp,1,"BigFloat: exponent overflow");
		}
		if(!lost){
			inex=0;
		}else if(sign>0){
			inex=inc?1:-1;
		}else{
			inex=inc?-1:1;
		}
	}

	static BigFloat from_limbs(int sign,detail::limbs_t mag,
							   std::int64_t exp,std::size_t prec,
							   FloatRnd rnd,bool more=false){
		BigFloat out;
		out.prec_=chk_prec(prec);
		out.kind_=FloatKind::finite;
		out.sign_=(sign<0)?-1:((sign>0)?1:0);
		detail::trim_lz(mag);
		out.mag_.limbs_=std::move(mag);
		out.mag_.sign_=out.mag_.limbs_.empty()?0:1;
		out.exp_=exp;
		if(out.sign_==0||out.mag_.sign_==0){
			out.sign_=0;
			out.exp_=0;
			out.mag_=BigInt();
			out.inex_=0;
			return out;
		}
		round_mag(out.mag_,out.sign_,out.exp_,out.prec_,rnd,out.inex_,more);
		if(!out.mag_.limbs_.empty()&&((out.mag_.limbs_[0]&1u)!=0u))
			return out;
		out.pack();
		return out;
	}

	static BigFloat from_exact_limbs(int sign,detail::limbs_t mag,
									 std::int64_t exp,std::size_t prec){
		BigFloat out;
		out.prec_=chk_prec(prec);
		out.kind_=FloatKind::finite;
		out.sign_=(sign<0)?-1:((sign>0)?1:0);
		detail::trim_lz(mag);
		out.mag_.limbs_=std::move(mag);
		out.mag_.sign_=out.mag_.limbs_.empty()?0:1;
		out.exp_=exp;
		out.inex_=0;
		if(out.sign_==0||out.mag_.sign_==0){
			out.sign_=0;
			out.exp_=0;
			out.mag_=BigInt();
			return out;
		}
		MINI_MP_ASSERT(out.mag_.bit_length()<=out.prec_);
		if((out.mag_.limbs_[0]&1u)!=0u)
			return out;
		out.pack();
		return out;
	}

	static detail::limbs_t sqr_limbs(const BigInt&a){
		const detail::limbs_t&x=a.limbs_;
		if(x.empty())
			return {};
		if(x.size()==1){
			const detail::u128 p=detail::mul_u64(x[0],x[0]);
			detail::limbs_t out;
			out.reserve((p.hi!=0)?2u:1u);
			out.push_back(p.lo);
			if(p.hi!=0)
				out.push_back(p.hi);
			return out;
		}
		if(x.size()==2)
			return sqr2_limbs(x);
		if(x.size()<=detail::kCbaMax)
			return detail::sqrcba_ab(x);
		detail::ensure_at();
#if MINI_MP_ENABLE_NTT
		const std::size_t ntt_thr=detail::tun_ntt_sq();
		if(ntt_thr!=detail::kNttOff&&x.size()>=ntt_thr){
			BigInt r=mul_ntt(a,a);
			return std::move(r.limbs_);
		}
#endif
		if(x.size()>=detail::tun_kar())
			return detail::sqrkar_ab(x);
		return detail::sqrsbk_ab(x);
	}

	static detail::limbs_t mul_limbs(const BigInt&a,const BigInt&b){
		const detail::limbs_t&am=a.limbs_;
		const detail::limbs_t&bm=b.limbs_;
		if(am.empty()||bm.empty())
			return {};
		if(&a==&b)
			return sqr_limbs(a);
		const std::size_t an=am.size();
		const std::size_t bn=bm.size();
		if(an==1&&bn==1){
			const detail::u128 p=detail::mul_u64(am[0],bm[0]);
			detail::limbs_t out;
			out.reserve((p.hi!=0)?2u:1u);
			out.push_back(p.lo);
			if(p.hi!=0)
				out.push_back(p.hi);
			return out;
		}
		if(an==1)
			return detail::mul_bylb(bm,am[0]);
		if(bn==1)
			return detail::mul_bylb(am,bm[0]);
		if(an==2&&bn==2)
			return mul2_limbs(am,bm);
		const std::size_t nmax=std::max(an,bn);
		if(nmax<=detail::kCbaMax)
			return detail::mulcba_ab(am,bm);
		detail::ensure_at();
		const std::size_t nmin=std::min(an,bn);
#if MINI_MP_ENABLE_NTT
		const std::size_t ntt_thr=detail::tun_ntt();
		if(ntt_thr!=detail::kNttOff&&nmin>=ntt_thr&&
		   nmax<=nmin*detail::tun_ntt_imb()){
			BigInt r=mul_ntt(a,b);
			return std::move(r.limbs_);
		}
#endif
		const std::size_t t3_thr=detail::tun_t3();
		if(t3_thr!=detail::kNttOff&&nmin>=t3_thr&&
		   nmax<=nmin+nmin/2u){
			BigInt r=mul_t3(a,b);
			return std::move(r.limbs_);
		}
		if(nmin>=detail::tun_kar()&&nmax<=nmin*detail::tun_kar_imb())
			return detail::mulkar_ab(am,bm);
		return detail::mulsbk_ab(am,bm);
	}

	static detail::limbs_t mul2_limbs(const detail::limbs_t&a,
									  const detail::limbs_t&b){
		MINI_MP_ASSERT(a.size()==2&&b.size()==2);
		const detail::u128 p00=detail::mul_u64(a[0],b[0]);
		const detail::u128 p01=detail::mul_u64(a[0],b[1]);
		const detail::u128 p10=detail::mul_u64(a[1],b[0]);
		const detail::u128 p11=detail::mul_u64(a[1],b[1]);
		detail::limbs_t out(4u,0);
		out[0]=p00.lo;
		std::uint64_t t=0;
		const std::uint64_t c0=detail::addc_u64(0,p00.hi,p01.lo,&t);
		std::uint64_t u=0;
		const std::uint64_t c1=detail::addc_u64(0,t,p10.lo,&u);
		out[1]=u;
		const std::uint64_t k=c0+c1;
		std::uint64_t v=0;
		const std::uint64_t c2=detail::addc_u64(0,p01.hi,p10.hi,&v);
		std::uint64_t w=0;
		const std::uint64_t c3=detail::addc_u64(0,v,p11.lo,&w);
		std::uint64_t x=0;
		const std::uint64_t c4=detail::addc_u64(0,w,k,&x);
		out[2]=x;
		const std::uint64_t k2=c2+c3+c4;
		std::uint64_t y=0;
		const std::uint64_t c5=detail::addc_u64(0,p11.hi,k2,&y);
		MINI_MP_ASSERT(c5==0);
		out[3]=y;
		detail::trim_lz(out);
		return out;
	}

	static detail::limbs_t sqr2_limbs(const detail::limbs_t&x){
		MINI_MP_ASSERT(x.size()==2);
		const detail::u128 p00=detail::mul_u64(x[0],x[0]);
		const detail::u128 p01=detail::mul_u64(x[0],x[1]);
		const detail::u128 p11=detail::mul_u64(x[1],x[1]);
		const std::uint64_t cross0=p01.lo<<1u;
		const std::uint64_t cross1=(p01.hi<<1u)|(p01.lo>>63u);
		const std::uint64_t cross2=p01.hi>>63u;
		detail::limbs_t out(4u,0);
		out[0]=p00.lo;
		std::uint64_t a=0;
		const std::uint64_t c0=detail::addc_u64(0,p00.hi,cross0,&a);
		out[1]=a;
		std::uint64_t b=0;
		const std::uint64_t c1=detail::addc_u64(0,p11.lo,cross1,&b);
		std::uint64_t c=0;
		const std::uint64_t c2=detail::addc_u64(0,b,c0,&c);
		out[2]=c;
		std::uint64_t d=0;
		const std::uint64_t c3=detail::addc_u64(0,p11.hi,cross2,&d);
		std::uint64_t e=0;
		const std::uint64_t c4=detail::addc_u64(0,d,c1+c2,&e);
		MINI_MP_ASSERT((c3|c4)==0);
		out[3]=e;
		detail::trim_lz(out);
		return out;
	}

	static BigFloat keep_more(const BigFloat&x,std::size_t prec,
							  FloatRnd rnd){
		if(x.kind_==FloatKind::finite&&x.sign_!=0&&x.prec_==prec&&
		   x.mag_.bit_length()<=prec){
			const bool inc=rnd==FloatRnd::away||
				(rnd==FloatRnd::up&&x.sign_>0)||
				(rnd==FloatRnd::down&&x.sign_<0);
			BigFloat out=x;
			if(inc)
				out.step_mag(true);
			out.inex_=x.sign_>0?(inc?1:-1):(inc?-1:1);
			return out;
		}
		return from_parts_more(x.sign_,x.mag_,x.exp_,prec,rnd,true);
	}

	static BigFloat keep_more(int sign,BigInt mag,std::int64_t exp,
							  std::size_t prec,FloatRnd rnd){
		if(mag.sign()<0){
			mag=-mag;
			sign=-sign;
		}
		if(mag.bit_length()>prec)
			return from_parts_more(sign,std::move(mag),exp,prec,rnd,true);
		BigFloat tmp=from_limbs(sign,std::move(mag.limbs_),exp,prec,
								FloatRnd::zero);
		tmp.inex_=0;
		return keep_more(tmp,prec,rnd);
	}

	static BigFloat from_parts_more(int sign,BigInt mag,std::int64_t exp,
									std::size_t prec,FloatRnd rnd,bool more){
		if(mag.sign()<0){
			mag=-mag;
			sign=-sign;
		}
		return from_limbs(sign,std::move(mag.limbs_),exp,prec,rnd,more);
	}

	BigFloat int_part(FloatRnd rnd) const{
		if(sign_==0)
			return zero(prec_);
		if(exp_>=0)
			return from_int_exact(to_bigint(FloatRnd::zero),prec_);
		const std::size_t sh=neg_exp_size(exp_,
			"BigFloat::rint: exponent too small");
		BigInt q=mag_;
		const bool lost=low_nz(q,sh);
		detail::shr_ip(q.limbs_,sh);
		q.sign_=q.limbs_.empty()?0:1;
		bool inc=false;
		if(lost){
			if(rnd==FloatRnd::up&&sign_>0)
				inc=true;
			else if(rnd==FloatRnd::down&&sign_<0)
				inc=true;
			else if(rnd==FloatRnd::away)
				inc=true;
			else if(rnd==FloatRnd::nearest&&sh!=0){
				const bool guard=detail::test_bit(mag_.limbs_,sh-1u);
				const bool sticky=low_nz(mag_,sh-1u);
				inc=guard&&(sticky||q.is_odd());
			}
		}
		if(inc){
			detail::add_one_ip(q.limbs_);
			q.sign_=1;
		}
		if(q.is_zero())
			return zero(prec_);
		if(sign_<0)
			q=-q;
		return from_int_exact(q,prec_);
	}

	static BigFloat add_mag_exp(int as,BigInt am,std::int64_t ae,
								int bs,const BigInt&bm,std::int64_t be,
								std::size_t prec,FloatRnd rnd){
		if(am.sign()<0){
			am=-am;
			as=-as;
		}
		if(as==0||am.is_zero())
			return from_parts(bs,bm,be,prec,rnd);
		if(bs==0||bm.is_zero())
			return from_parts(as,std::move(am),ae,prec,rnd);
		if(as==bs){
			const std::int64_t ah=add_exp_sz(ae,am.bit_length(),
				"BigFloat::add: exponent overflow");
			const std::int64_t bh=add_exp_sz(be,bm.bit_length(),
				"BigFloat::add: exponent overflow");
			if(ah!=bh){
				const std::int64_t hh=std::max(ah,bh);
				const std::int64_t lh=std::min(ah,bh);
				const std::size_t gap=exp_gap(hh,lh,
					"BigFloat::add: exponent gap too large");
				if(gap>prec_add(prec,8,
					   "BigFloat::add: precision too large")){
					if(ah>bh)
						return keep_more(as,std::move(am),ae,prec,rnd);
					return keep_more(bs,bm,be,prec,rnd);
				}
			}
			const std::int64_t e=std::min(ae,be);
			detail::limbs_t out;
			if(ae==be){
				out=std::move(am.limbs_);
				detail::add_abs_ip(out,bm.limbs_);
			}else if(ae==e){
				out=std::move(am.limbs_);
				detail::add_shb_ip(out,bm.limbs_,exp_gap(be,e,
					"BigFloat::add: exponent gap too large"));
			}else{
				out=bm.limbs_;
				detail::add_shb_ip(out,am.limbs_,exp_gap(ae,e,
					"BigFloat::add: exponent gap too large"));
			}
			return from_limbs(as,std::move(out),e,prec,rnd);
		}
		const int cmp=cmp_mag_exp(am,ae,bm,be);
		if(cmp==0)
			return zero(prec);
		const std::int64_t e=std::min(ae,be);
		if(cmp>0){
			detail::limbs_t out=std::move(am.limbs_);
			if(ae!=e)
				detail::shl_ip(out,exp_gap(ae,e,
					"BigFloat::add: exponent gap too large"));
			detail::limbs_t tmp=bm.limbs_;
			if(be!=e)
				detail::shl_ip(tmp,exp_gap(be,e,
					"BigFloat::add: exponent gap too large"));
			detail::sub_abs_ip(out,tmp);
			return from_limbs(as,std::move(out),e,prec,rnd);
		}
		detail::limbs_t out=bm.limbs_;
		if(be!=e)
			detail::shl_ip(out,exp_gap(be,e,
				"BigFloat::add: exponent gap too large"));
		detail::limbs_t tmp=std::move(am.limbs_);
		if(ae!=e)
			detail::shl_ip(tmp,exp_gap(ae,e,
				"BigFloat::add: exponent gap too large"));
		detail::sub_abs_ip(out,tmp);
		return from_limbs(bs,std::move(out),e,prec,rnd);
	}

	static BigFloat add_same(const BigFloat&a,const BigFloat&b,
							 std::size_t prec,FloatRnd rnd){
		const std::int64_t ah=top_exp(a);
		const std::int64_t bh=top_exp(b);
		if(ah!=bh){
			const BigFloat&hi=(ah>bh)?a:b;
			const BigFloat&lo=(ah>bh)?b:a;
			const std::size_t gap=exp_gap(top_exp(hi),top_exp(lo),
				"BigFloat::add: exponent gap too large");
			if(gap>prec_add(prec,8,"BigFloat::add: precision too large")){
				return keep_more(hi,prec,rnd);
			}
			if(gap>64u)
				return add_clip(hi,lo,prec,rnd);
		}
		const std::int64_t e=std::min(a.exp_,b.exp_);
		detail::limbs_t out;
		if(a.exp_==b.exp_){
			out=a.mag_.limbs_;
			detail::add_abs_ip(out,b.mag_.limbs_);
		}else if(a.exp_==e){
			out=a.mag_.limbs_;
			detail::add_shb_ip(out,b.mag_.limbs_,exp_gap(b.exp_,e,
				"BigFloat::add: exponent gap too large"));
		}else{
			out=b.mag_.limbs_;
			detail::add_shb_ip(out,a.mag_.limbs_,exp_gap(a.exp_,e,
				"BigFloat::add: exponent gap too large"));
		}
		return from_limbs(a.sign_,std::move(out),e,prec,rnd);
	}

	static bool low_bits_nz(const detail::limbs_t&x,
							std::size_t bits) noexcept{
		if(bits==0||x.empty())
			return false;
		const std::size_t full=bits/64u;
		const unsigned rem=static_cast<unsigned>(bits%64u);
		const std::size_t n=std::min(full,x.size());
		for(std::size_t i=0;i<n;++i){
			if(x[i]!=0)
				return true;
		}
		if(rem!=0&&full<x.size()){
			const std::uint64_t mask=(std::uint64_t(1)<<rem)-1u;
			return (x[full]&mask)!=0;
		}
		return false;
	}

	static bool add_shr_ip(detail::limbs_t&out,
						   const detail::limbs_t&x,std::size_t sh){
		const bool more=low_bits_nz(x,sh);
		const std::size_t limb=sh/64u;
		const unsigned bits=static_cast<unsigned>(sh%64u);
		if(limb>=x.size())
			return more;
		const std::size_t n=x.size()-limb;
		if(out.size()<n)
			out.resize(n,0);
		std::uint64_t carry=0;
		for(std::size_t i=0;i<n;++i){
			const std::size_t src=limb+i;
			std::uint64_t part=x[src];
			if(bits!=0){
				part>>=bits;
				if(src+1u<x.size())
					part|=x[src+1u]<<(64u-bits);
			}
			std::uint64_t t=0;
			const std::uint64_t c1=detail::addc_u64(0,out[i],part,&t);
			std::uint64_t v=0;
			const std::uint64_t c2=detail::addc_u64(0,t,carry,&v);
			out[i]=v;
			MINI_MP_ASSERT((c1&c2)==0);
			carry=(c1|c2);
		}
		std::size_t i=n;
		while(carry!=0){
			if(i>=out.size())
				out.push_back(0);
			std::uint64_t v=0;
			carry=detail::addc_u64(0,out[i],carry,&v);
			out[i]=v;
			++i;
		}
		detail::trim_lz(out);
		return more;
	}

	static BigFloat add_clip(const BigFloat&hi,const BigFloat&lo,
							 std::size_t prec,FloatRnd rnd){
		const std::size_t keep=prec_add(prec,3,
			"BigFloat::add: precision too large");
		const std::int64_t e=sub_exp_sz(top_exp(hi),keep,
			"BigFloat::add: exponent overflow");
		detail::limbs_t out=hi.mag_.limbs_;
		if(hi.exp_!=e)
			detail::shl_ip(out,exp_gap(hi.exp_,e,
				"BigFloat::add: exponent gap too large"));
		bool more=false;
		if(lo.exp_>=e){
			detail::add_shb_ip(out,lo.mag_.limbs_,exp_gap(lo.exp_,e,
				"BigFloat::add: exponent gap too large"));
		}else{
			more=add_shr_ip(out,lo.mag_.limbs_,exp_gap(e,lo.exp_,
				"BigFloat::add: exponent gap too large"));
		}
		return from_limbs(hi.sign_,std::move(out),e,prec,rnd,more);
	}

	void set_int(const BigInt&v,FloatRnd rnd){
		kind_=FloatKind::finite;
		sign_=v.sign();
		exp_=0;
		mag_=v.abs();
		round_self(rnd);
	}

	static BigFloat from_int_exact(const BigInt&v,std::size_t min_prec){
		const std::size_t bits=std::max<std::size_t>(v.bit_length(),1u);
		return BigFloat(v,std::max(chk_prec(min_prec),bits),FloatRnd::zero);
	}

	static bool to_i64_checked(std::int64_t&out,const BigInt&v){
		if(v.sign()>=0){
			if(!v.fits_u64())
				return false;
			const std::uint64_t u=v.to_u64();
			if(u>static_cast<std::uint64_t>(
				   std::numeric_limits<std::int64_t>::max()))
				return false;
			out=static_cast<std::int64_t>(u);
			return true;
		}
		BigInt a=-v;
		if(!a.fits_u64())
			return false;
		const std::uint64_t u=a.to_u64();
		const std::uint64_t min_abs=std::uint64_t(1)<<63u;
		if(u==min_abs){
			out=std::numeric_limits<std::int64_t>::min();
			return true;
		}
		if(u>static_cast<std::uint64_t>(
			   std::numeric_limits<std::int64_t>::max()))
			return false;
		out=-static_cast<std::int64_t>(u);
		return true;
	}

	static bool integer_is_odd(const BigFloat&x){
		MINI_MP_ASSERT(x.kind_==FloatKind::finite&&x.is_integer());
		if(x.sign_==0)
			return false;
		if(x.exp_>0)
			return false;
		if(x.exp_==0)
			return x.mag_.is_odd();
		const std::size_t sh=neg_exp_size(x.exp_,
			"BigFloat: exponent too small");
		return detail::test_bit(x.mag_.limbs_,sh);
	}

	static bool exact_pow2_exp(std::int64_t&out,const BigFloat&x){
		if(x.kind_!=FloatKind::finite||x.sign_<=0)
			return false;
		if(x.mag_.popcount()!=1u)
			return false;
		out=add_exp_sz(x.exp_,x.mag_.ctz(),
					   "BigFloat: exponent overflow");
		return true;
	}

	static bool series_done(const BigFloat&term,const BigFloat&sum,
							std::size_t work){
		if(term.sign_==0)
			return true;
		if(sum.sign_==0)
			return false;
		const std::int64_t te=top_exp(term);
		const std::int64_t se=top_exp(sum);
		if(se<=te)
			return false;
		return exp_gap(se,te,"BigFloat: series exponent gap too large")>
			prec_add(work,8,"BigFloat: series precision too large");
	}

	static std::size_t series_cap(std::size_t work,const char*msg){
		return prec_add(mul_sz_u32(work,4u,msg),96u,msg);
	}

	static std::uint64_t u64_from_size(std::size_t v,const char*msg){
		if constexpr(sizeof(std::size_t)>sizeof(std::uint64_t)){
			if(v>std::numeric_limits<std::uint64_t>::max())
				detail::throw_ovf(msg);
		}
		return static_cast<std::uint64_t>(v);
	}

	static std::uint64_t mul_u64_checked(std::uint64_t a,std::uint64_t b,
										 const char*msg){
		if(a!=0&&b>std::numeric_limits<std::uint64_t>::max()/a)
			detail::throw_ovf(msg);
		return a*b;
	}

	static BigFloat div_ui(const BigFloat&x,std::uint64_t d,
						   std::size_t prec,FloatRnd rnd){
		MINI_MP_ASSERT(d!=0);
		if(d==1u)
			return x.rounded(prec,rnd);
		if(x.kind_!=FloatKind::finite)
			return div(x,BigFloat(BigInt::from_u64(d),prec),
					   prec,rnd);
		if(x.sign_==0)
			return zero(prec);
		if(std::has_single_bit(d)){
			return x.ldexp(-static_cast<std::int64_t>(
				std::countr_zero(d))).rounded(prec,rnd);
		}
		BigFloat q=from_ratio(x.sign_,x.mag_,BigInt::from_u64(d),
							  prec,rnd);
		const int qix=q.inex_;
		q.exp_=add_exp(q.exp_,x.exp_,
			"BigFloat::div_ui: exponent overflow");
		q.inex_=qix;
		return q;
	}

	static BigFloat mul_ui(const BigFloat&x,std::uint64_t m,
						   std::size_t prec,FloatRnd rnd){
		if(m==0u)
			return zero(prec);
		if(m==1u)
			return x.rounded(prec,rnd);
		if(x.kind_!=FloatKind::finite)
			return mul(x,BigFloat(BigInt::from_u64(m),prec),
					   prec,rnd);
		if(x.sign_==0)
			return zero(prec);
		if(std::has_single_bit(m)){
			return x.ldexp(static_cast<std::int64_t>(
				std::countr_zero(m))).rounded(prec,rnd);
		}
		detail::limbs_t mag;
		detail::mulbl_in(x.mag_.limbs_,m,mag);
		return from_limbs(x.sign_,std::move(mag),x.exp_,prec,rnd);
	}

	static BigFloat atanh_series(const BigFloat&z,std::size_t work){
		if(z.sign_==0)
			return zero(work);
		const BigFloat z2=sqr(z,work,FloatRnd::nearest);
		BigFloat term=z.rounded(work,FloatRnd::nearest);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::atanh: precision too large");
		for(std::size_t n=1;n<cap;++n){
			term=mul(term,z2,work,FloatRnd::nearest);
			const BigFloat addend=div_ui(term,
				u64_from_size(2u*n+1u,
					"BigFloat::atanh: series too long"),
				work,FloatRnd::nearest);
			sum=add(sum,addend,work,FloatRnd::nearest);
			if(series_done(addend,sum,work))
				break;
		}
		return sum;
	}

	static std::size_t log_sqrt_steps(std::size_t work) noexcept{
		const unsigned w=std::bit_width(work);
		return static_cast<std::size_t>(w>4u?w-3u:1u);
	}

	static BigFloat log_unit(BigFloat x,std::size_t work){
		if(x.sign_==0)
			return inf(-1,work);
		if(compare(x,one(work))==0)
			return zero(work);
		const std::size_t r=log_sqrt_steps(work);
		for(std::size_t i=0;i<r;++i)
			x=sqrt(x,work,FloatRnd::nearest);
		const BigFloat z=div(sub(x,one(work),work,FloatRnd::nearest),
							 add(x,one(work),work,FloatRnd::nearest),
							 work,FloatRnd::nearest);
		return atanh_series(z,work).ldexp(
			static_cast<std::int64_t>(r+1u));
	}

	static BigFloat log2_calc(std::size_t work){
		return log2_bsplit_calc(work).rounded(work,FloatRnd::nearest);
	}

	struct Log2Split{
		BigInt p;
		BigInt q;
		BigInt t;
	};

	static void strip_common_twos(Log2Split&s){
		if(s.t.is_zero()||s.q.is_zero())
			return;
		std::size_t v=s.t.abs().ctz();
		v=std::min(v,s.q.ctz());
		v=std::min(v,s.p.abs().ctz());
		if(v!=0u){
			s.t>>=v;
			s.q>>=v;
			s.p>>=v;
		}
	}

	static Log2Split log2_split(std::size_t a,std::size_t b){
		if(b==a+1u){
			const std::uint64_t k=u64_from_size(a,
				"BigFloat::const_log2: too many terms");
			BigInt p;
			if(k==0u){
				p=BigInt(3);
			}else{
				p=BigInt::from_u64(k);
				p=-p;
			}
			if(k>(std::numeric_limits<std::uint64_t>::max()-1u)/2u)
				detail::throw_ovf("BigFloat::const_log2: too many terms");
			BigInt q=BigInt::from_u64(2u*k+1u);
			q<<=2u;
			return {p,q,p};
		}
		const std::size_t m=a+(b-a)/2u;
		Log2Split lo=log2_split(a,m);
		Log2Split hi=log2_split(m,b);
		Log2Split out;
		out.p=lo.p*hi.p;
		out.q=lo.q*hi.q;
		out.t=lo.t*hi.q+hi.t*lo.p;
		strip_common_twos(out);
		return out;
	}

	static BigFloat log2_bsplit_calc(std::size_t work){
		const std::size_t terms=std::max<std::size_t>(2u,work/3u+3u);
		Log2Split s=log2_split(0,terms);
		return div(BigFloat(s.t,work,FloatRnd::nearest),
				   BigFloat(s.q,work,FloatRnd::nearest),
				   work,FloatRnd::nearest);
	}

	struct PiSplit{
		BigInt p;
		BigInt q;
		BigInt t;
	};

	static PiSplit pi_split(std::size_t a,std::size_t b){
		static constexpr std::uint64_t c3_24=10939058860032000ULL;
		if(b==a+1u){
			if(a==0)
				return {BigInt(1),BigInt(1),BigInt(13591409)};
			const std::uint64_t k=u64_from_size(a,
				"BigFloat::const_pi: too many terms");
			if(k>std::numeric_limits<std::uint64_t>::max()/6u)
				detail::throw_ovf("BigFloat::const_pi: too many terms");
			BigInt p=BigInt::from_u64(6u*k-5u)*
					 BigInt::from_u64(2u*k-1u)*
					 BigInt::from_u64(6u*k-1u);
			BigInt q=BigInt::from_u64(k);
			q=q*q*q*BigInt::from_u64(c3_24);
			BigInt lin=BigInt::from_u64(13591409u)+
				BigInt::from_u64(545140134u)*BigInt::from_u64(k);
			BigInt t=p*lin;
			if((k&1u)!=0u)
				t=-t;
			return {std::move(p),std::move(q),std::move(t)};
		}
		const std::size_t m=a+(b-a)/2u;
		PiSplit lo=pi_split(a,m);
		PiSplit hi=pi_split(m,b);
		PiSplit out;
		out.p=lo.p*hi.p;
		out.q=lo.q*hi.q;
		out.t=lo.t*hi.q+lo.p*hi.t;
		return out;
	}

	static BigFloat pi_calc(std::size_t work){
		const std::size_t terms=std::max<std::size_t>(1u,work/47u+3u);
		PiSplit s=pi_split(0,terms);
		BigFloat q(s.q,work,FloatRnd::nearest);
		BigFloat t(s.t,work,FloatRnd::nearest);
		BigFloat c=mul(BigFloat(426880,work),
					   sqrt(BigFloat(10005,work),work,FloatRnd::nearest),
					   work,FloatRnd::nearest);
		return div(mul(q,c,work,FloatRnd::nearest),t,work,
				   FloatRnd::nearest).rounded(work,FloatRnd::nearest);
	}

	static BigFloat cache_pi(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,32,
			"BigFloat::const_pi: precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=pi_calc(work);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_log2(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,32,
			"BigFloat::const_log2: precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=log2_calc(work);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_inv_log2(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: inverse log2 precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=recip(const_log2(work,FloatRnd::nearest),
					  work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_log10(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,32,
			"BigFloat: log10 constant precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=log(BigFloat(10,work),work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_inv_log10(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: inverse log10 precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=recip(cache_log10(work,FloatRnd::nearest),
					  work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_sqrt_pi(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: sqrt(pi) precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=sqrt(const_pi(work,FloatRnd::nearest),work,
					 FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_inv_sqrt_pi(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: inverse sqrt(pi) precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=recip(cache_sqrt_pi(work,FloatRnd::nearest),
					  work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_erf_scale(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: erf scale precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=cache_inv_sqrt_pi(work,FloatRnd::nearest).ldexp(1);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_sqrt_2pi(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: sqrt(2*pi) precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=sqrt(const_pi(work,FloatRnd::nearest).ldexp(1),
					 work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_e(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,32,
			"BigFloat: e constant precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=exp_kernel(one(work),work);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static BigFloat cache_inv_e(std::size_t prec,FloatRnd rnd){
		static std::mutex mtx;
		static BigFloat val;
		static std::size_t have=0;
		const std::size_t work=prec_add(prec,16,
			"BigFloat: inverse e precision too large");
		std::lock_guard<std::mutex> lock(mtx);
		if(have<work){
			val=recip(cache_e(work,FloatRnd::nearest),
					  work,FloatRnd::nearest);
			have=work;
		}
		return val.rounded(prec,rnd);
	}

	static std::size_t exp_steps(std::size_t work) noexcept{
		const unsigned w=std::bit_width(work);
		return static_cast<std::size_t>(w>5u?w-4u:1u);
	}

	static BigFloat exp_kernel(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return one(work);
		const bool neg=x.sign_<0;
		BigFloat z=neg?x.neg():x;
		const std::size_t s=exp_steps(work);
		z=z.ldexp(-static_cast<std::int64_t>(s));
		BigFloat term=one(work);
		BigFloat sum=one(work);
		const std::size_t cap=series_cap(work,
			"BigFloat::exp: precision too large");
		for(std::size_t n=1;n<cap;++n){
			term=div_ui(mul(term,z,work,FloatRnd::nearest),
						u64_from_size(n,
							"BigFloat::exp: series too long"),
						work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		for(std::size_t i=0;i<s;++i)
			sum=sqr(sum,work,FloatRnd::nearest);
		if(neg)
			sum=recip(sum,work,FloatRnd::nearest);
		return sum;
	}

	static BigFloat expm1_kernel(const BigFloat&x,std::size_t prec,
								 FloatRnd rnd){
		const std::size_t work=x.precision();
		BigFloat term=x;
		BigFloat sum=x;
		const std::size_t cap=series_cap(work,
			"BigFloat::expm1: precision too large");
		for(std::size_t n=2;n<cap;++n){
			term=div_ui(mul(term,x,work,FloatRnd::nearest),
						u64_from_size(n,
							"BigFloat::expm1: series too long"),
						work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return sum.rounded(prec,rnd);
	}

	static std::size_t trig_work_prec(const BigFloat&x,std::size_t prec){
		std::size_t extra=80;
		if(x.kind_==FloatKind::finite&&x.sign_!=0){
			const std::int64_t e=x.ilogb();
			if(e>0){
				extra=prec_add(extra,
					to_size_u64(static_cast<std::uint64_t>(e),
						"BigFloat trigonometry: argument too large"),
					"BigFloat trigonometry: precision too large");
			}
		}
		return prec_add(prec,extra,
			"BigFloat trigonometry: precision too large");
	}

	static BigFloat sin_kernel(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return zero(work);
		const BigFloat nz2=sqr(x,work,FloatRnd::nearest).neg();
		BigFloat term=x.rounded(work,FloatRnd::nearest);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::sin: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const std::uint64_t a=u64_from_size(2u*n,
				"BigFloat::sin: series too long");
			const std::uint64_t b=u64_from_size(2u*n+1u,
				"BigFloat::sin: series too long");
			term=div_ui(mul(term,nz2,work,FloatRnd::nearest),
						mul_u64_checked(a,b,
							"BigFloat::sin: series too long"),
						work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return sum;
	}

	static BigFloat cos_kernel(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return one(work);
		const BigFloat nz2=sqr(x,work,FloatRnd::nearest).neg();
		BigFloat term=one(work);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::cos: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const std::uint64_t a=u64_from_size(2u*n-1u,
				"BigFloat::cos: series too long");
			const std::uint64_t b=u64_from_size(2u*n,
				"BigFloat::cos: series too long");
			term=div_ui(mul(term,nz2,work,FloatRnd::nearest),
						mul_u64_checked(a,b,
							"BigFloat::cos: series too long"),
						work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return sum;
	}

	static std::pair<BigFloat,BigFloat> sin_cos_kernel(
		const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return {zero(work),one(work)};
		const BigFloat z2=sqr(x,work,FloatRnd::nearest);
		const BigFloat nz2=z2.neg();
		BigFloat st=x.rounded(work,FloatRnd::nearest);
		BigFloat ss=st;
		BigFloat ct=one(work);
		BigFloat cs=ct;
		const std::size_t cap=series_cap(work,
			"BigFloat::sin_cos: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const std::uint64_t a=u64_from_size(2u*n,
				"BigFloat::sin_cos: series too long");
			const std::uint64_t b=u64_from_size(2u*n+1u,
				"BigFloat::sin_cos: series too long");
			const std::uint64_t sd=mul_u64_checked(a,b,
				"BigFloat::sin_cos: series too long");
			st=div_ui(mul(st,nz2,work,FloatRnd::nearest),
					  sd,work,FloatRnd::nearest);
			ss=add(ss,st,work,FloatRnd::nearest);

			const std::uint64_t cd=mul_u64_checked(a-1u,a,
				"BigFloat::sin_cos: series too long");
			ct=div_ui(mul(ct,nz2,work,FloatRnd::nearest),
					  cd,work,FloatRnd::nearest);
			cs=add(cs,ct,work,FloatRnd::nearest);
			if(series_done(st,ss,work)&&series_done(ct,cs,work))
				break;
		}
		return {std::move(ss),std::move(cs)};
	}

	static BigFloat atan_series(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return zero(work);
		const BigFloat z2=sqr(x,work,FloatRnd::nearest).neg();
		BigFloat term=x.rounded(work,FloatRnd::nearest);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::atan: precision too large");
		for(std::size_t n=1;n<cap;++n){
			term=mul(term,z2,work,FloatRnd::nearest);
			BigFloat addend=div_ui(term,
				u64_from_size(2u*n+1u,
					"BigFloat::atan: series too long"),
				work,FloatRnd::nearest);
			sum=add(sum,addend,work,FloatRnd::nearest);
			if(series_done(addend,sum,work))
				break;
		}
		return sum;
	}

	static BigFloat asin_series(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return zero(work);
		const BigFloat x2=sqr(x,work,FloatRnd::nearest);
		BigFloat term=x.rounded(work,FloatRnd::nearest);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::asin: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const std::uint64_t a=u64_from_size(2u*n-1u,
				"BigFloat::asin: series too long");
			const std::uint64_t num=mul_u64_checked(a,a,
				"BigFloat::asin: series too long");
			const std::uint64_t den=mul_u64_checked(
				u64_from_size(2u*n,"BigFloat::asin: series too long"),
				u64_from_size(2u*n+1u,"BigFloat::asin: series too long"),
				"BigFloat::asin: series too long");
			term=div_ui(mul_ui(mul(term,x2,work,FloatRnd::nearest),
							   num,work,FloatRnd::nearest),
						den,work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return sum;
	}

	static BigFloat asinh_series(const BigFloat&x,std::size_t work){
		if(x.sign_==0)
			return zero(work);
		const BigFloat nx2=sqr(x,work,FloatRnd::nearest).neg();
		BigFloat term=x.rounded(work,FloatRnd::nearest);
		BigFloat sum=term;
		const std::size_t cap=series_cap(work,
			"BigFloat::asinh: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const std::uint64_t a=u64_from_size(2u*n-1u,
				"BigFloat::asinh: series too long");
			const std::uint64_t num=mul_u64_checked(a,a,
				"BigFloat::asinh: series too long");
			const std::uint64_t den=mul_u64_checked(
				u64_from_size(2u*n,"BigFloat::asinh: series too long"),
				u64_from_size(2u*n+1u,"BigFloat::asinh: series too long"),
				"BigFloat::asinh: series too long");
			term=div_ui(mul_ui(mul(term,nx2,work,FloatRnd::nearest),
							   num,work,FloatRnd::nearest),
						den,work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return sum;
	}

	static BigFloat atan_pos(const BigFloat&x,std::size_t work){
		const BigFloat onev=one(work);
		if(compare(x,onev)==0)
			return const_pi(work,FloatRnd::nearest).ldexp(-2);
		if(compare(x,onev)>0){
			BigFloat hpi=const_pi(work,FloatRnd::nearest).ldexp(-1);
			return sub(hpi,atan_pos(recip(x,work,FloatRnd::nearest),
									work),work,FloatRnd::nearest);
		}
		BigFloat z=x.rounded(work,FloatRnd::nearest);
		std::size_t scale=0;
		const BigFloat half=onev.ldexp(-1);
		while(cmpabs(z,half)>0&&scale<16u){
			BigFloat den=add(onev,sqrt(add(onev,
				sqr(z,work,FloatRnd::nearest),work,FloatRnd::nearest),
				work,FloatRnd::nearest),work,FloatRnd::nearest);
			z=div(z,den,work,FloatRnd::nearest);
			++scale;
		}
		return atan_series(z,work).ldexp(static_cast<std::int64_t>(scale));
	}

	static bool positive_half_index(std::uint64_t&idx,const BigFloat&x){
		if(x.kind_!=FloatKind::finite||x.sign_<=0)
			return false;
		BigFloat twice=x.ldexp(1);
		if(!twice.is_integer()||!twice.fits_u64(FloatRnd::zero))
			return false;
		const std::uint64_t m=twice.to_u64(FloatRnd::zero);
		if((m&1u)==0u)
			return false;
		idx=(m-1u)/2u;
		return true;
	}

	static BigFloat gamma_half_integer(std::uint64_t idx,std::size_t work){
		BigFloat out=cache_sqrt_pi(work,FloatRnd::nearest);
		for(std::uint64_t k=0;k<idx;++k){
			BigFloat factor(2u*k+1u,work,FloatRnd::nearest);
			factor=factor.ldexp(-1);
			out=mul(out,factor,work,FloatRnd::nearest);
		}
		return out;
	}

	static BigFloat gamma_lanczos(const BigFloat&x,std::size_t prec,
								  FloatRnd rnd){
		static constexpr const char*coef[]={
			"0.99999999999980993227684700473478",
			"676.52036812188509856700919044402",
			"-1259.1392167224028704715607875528",
			"771.3234287776530788486528258894",
			"-176.61502916214059906584551354",
			"12.507343278686904814458936853",
			"-0.13857109526572011689554707",
			"0.000009984369578019570859563",
			"0.000000150563273514931155834"
		};
		const std::size_t work=x.precision();
		BigFloat z=sub(x,one(work),work,FloatRnd::nearest);
		BigFloat sum(coef[0],work);
		for(std::size_t i=1;i<sizeof(coef)/sizeof(coef[0]);++i){
			BigFloat den=add(z,BigFloat(BigInt::from_u64(
				static_cast<std::uint64_t>(i)),work),
				work,FloatRnd::nearest);
			sum=add(sum,div(BigFloat(coef[i],work),den,work,
							FloatRnd::nearest),work,FloatRnd::nearest);
		}
		BigFloat t=add(z,BigFloat("7.5",work),work,FloatRnd::nearest);
		BigFloat p=add(z,one(work).ldexp(-1),work,FloatRnd::nearest);
		BigFloat out=mul(cache_sqrt_2pi(work,FloatRnd::nearest),
						 mul(pow(t,p,work,FloatRnd::nearest),
							 exp(t.neg(),work,FloatRnd::nearest),
							 work,FloatRnd::nearest),
						 work,FloatRnd::nearest);
		out=mul(out,sum,work,FloatRnd::nearest);
		return out.rounded(prec,rnd);
	}

	static BigFloat gamma_spouge(BigFloat x,std::size_t prec,FloatRnd rnd){
		const std::size_t work=x.precision();
		BigFloat prod=one(work);
		while(compare(x,one(work))<=0){
			prod=mul(prod,x,work,FloatRnd::nearest);
			x=add(x,one(work),work,FloatRnd::nearest);
		}
		BigFloat z=sub(x,one(work),work,FloatRnd::nearest);
		const std::size_t a=std::max<std::size_t>(8u,work/3u+8u);
		BigFloat sum=cache_sqrt_2pi(work,FloatRnd::nearest);
		BigInt fact(1);
		for(std::size_t k=1;k<a;++k){
			const std::size_t ak=a-k;
			BigFloat b(BigInt::from_u64(u64_from_size(ak,
				"BigFloat::gamma: too many terms")),work);
			BigFloat coeff=mul(exp(b,work,FloatRnd::nearest),
							   pow_ui(b,u64_from_size(k,
								   "BigFloat::gamma: too many terms"),
								   work,FloatRnd::nearest),
							   work,FloatRnd::nearest);
			coeff=div(coeff,sqrt(b,work,FloatRnd::nearest),
					  work,FloatRnd::nearest);
			coeff=div(coeff,BigFloat(fact,work),work,FloatRnd::nearest);
			if((k&1u)==0u)
				coeff=coeff.neg();
			sum=add(sum,div(coeff,add(z,BigFloat(
				BigInt::from_u64(u64_from_size(k,
					"BigFloat::gamma: too many terms")),work),
				work,FloatRnd::nearest),work,FloatRnd::nearest),
				work,FloatRnd::nearest);
			fact*=BigInt::from_u64(u64_from_size(k,
				"BigFloat::gamma: too many terms"));
		}
		BigFloat za=add(z,BigFloat(BigInt::from_u64(
			u64_from_size(a,"BigFloat::gamma: too many terms")),work),
			work,FloatRnd::nearest);
		BigFloat ze=add(z,one(work).ldexp(-1),work,FloatRnd::nearest);
		BigFloat pref=mul(pow(za,ze,work,FloatRnd::nearest),
						  exp(za.neg(),work,FloatRnd::nearest),
						  work,FloatRnd::nearest);
		BigFloat out=mul(pref,sum,work,FloatRnd::nearest);
		if(compare(prod,one(work))!=0)
			out=div(out,prod,work,FloatRnd::nearest);
		return out.rounded(prec,rnd);
	}

	static std::size_t erf_work_prec(std::size_t prec,double x){
		std::size_t extra=48u;
		if(std::isfinite(x)){
			const double ax=std::fabs(x);
			const double need=std::ceil(ax*ax*1.4426950408889634074);
			const std::size_t term_bits=
				need>static_cast<double>(
					std::numeric_limits<std::size_t>::max()/2u)
					?std::numeric_limits<std::size_t>::max()/2u
					:static_cast<std::size_t>(need);
			extra=std::max<std::size_t>(
				extra,term_bits+std::bit_width(prec)+32u);
		}
		return prec_add(prec,extra,
			"BigFloat::erf: precision too large");
	}

	static BigFloat erf_series(const BigFloat&x,std::size_t prec,
							   FloatRnd rnd){
		const std::size_t work=x.precision();
		const BigFloat xsq=sqr(x,work,FloatRnd::nearest);
		const BigFloat nx2=xsq.neg();
		BigFloat term=x;
		BigFloat sum=x;
		const std::size_t cap=series_cap(work,
			"BigFloat::erf: precision too large");
		for(std::size_t n=1;n<cap;++n){
			const BigFloat num=mul(term,nx2,work,FloatRnd::nearest);
			const std::uint64_t den=mul_u64_checked(
				u64_from_size(n,"BigFloat::erf: series too long"),
				u64_from_size(2u*n+1u,"BigFloat::erf: series too long"),
				"BigFloat::erf: series too long");
			term=mul_ui(div_ui(num,den,work,FloatRnd::nearest),
						u64_from_size(2u*n-1u,
							"BigFloat::erf: series too long"),
						work,FloatRnd::nearest);
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
		}
		return mul(cache_erf_scale(work,FloatRnd::nearest),sum,prec,rnd);
	}

	static BigFloat erfc_cf(const BigFloat&x,std::size_t work){
		const BigFloat half=one(work).ldexp(-1);
		const BigFloat tiny=epsilon(work).ldexp(8);
		const BigFloat z=sqr(x,work,FloatRnd::nearest);
		BigFloat b=add(z,half,work,FloatRnd::nearest);
		BigFloat c=recip(tiny,work,FloatRnd::nearest);
		BigFloat d=recip(b,work,FloatRnd::nearest);
		BigFloat h=d;
		const BigFloat eps=epsilon(work).ldexp(6);
		const std::size_t cap=series_cap(work,
			"BigFloat::erfc: precision too large");
		for(std::size_t i=1;i<cap;++i){
			BigFloat fi(u64_from_size(i,
				"BigFloat::erfc: continued fraction too long"),work);
			BigFloat an=mul(fi.neg(),sub(fi,half,work,
										 FloatRnd::nearest),
							work,FloatRnd::nearest);
			b=add(b,BigFloat(2,work),work,FloatRnd::nearest);
			d=add(mul(an,d,work,FloatRnd::nearest),b,work,
				  FloatRnd::nearest);
			if(d.sign_==0)
				d=tiny;
			c=add(b,div(an,c,work,FloatRnd::nearest),work,
				  FloatRnd::nearest);
			if(c.sign_==0)
				c=tiny;
			d=recip(d,work,FloatRnd::nearest);
			BigFloat del=mul(d,c,work,FloatRnd::nearest);
			h=mul(h,del,work,FloatRnd::nearest);
			if(cmpabs(sub(del,one(work),work,FloatRnd::nearest),eps)<=0)
				break;
		}
		BigFloat pref=div(mul(exp(z.neg(),work,FloatRnd::nearest),x,
							  work,FloatRnd::nearest),
						  cache_sqrt_pi(work,FloatRnd::nearest),
						  work,FloatRnd::nearest);
		return mul(pref,h,work,FloatRnd::nearest);
	}

	static BigFloat erfc_asymp(const BigFloat&x,std::size_t work){
		const BigFloat xsq=sqr(x,work,FloatRnd::nearest);
		const BigFloat den=xsq.ldexp(1);
		BigFloat term=one(work);
		BigFloat sum=term;
		BigFloat prev=term.abs();
		const std::size_t cap=series_cap(work,
			"BigFloat::erfc: precision too large");
		for(std::size_t n=1;n<cap;++n){
			term=div(mul_ui(term,u64_from_size(2u*n-1u,
				"BigFloat::erfc: series too long"),
				work,FloatRnd::nearest).neg(),den,work,FloatRnd::nearest);
			BigFloat at=term.abs();
			if(n>1u&&cmpabs(at,prev)>0)
				break;
			sum=add(sum,term,work,FloatRnd::nearest);
			if(series_done(term,sum,work))
				break;
			prev=std::move(at);
		}
		BigFloat pref=mul(div(exp(xsq.neg(),work,FloatRnd::nearest),
							  x,work,FloatRnd::nearest),
						  cache_inv_sqrt_pi(work,FloatRnd::nearest),
						  work,FloatRnd::nearest);
		return mul(pref,sum,work,FloatRnd::nearest);
	}

	static bool eq_ci(std::string_view a,std::string_view b) noexcept{
		if(a.size()!=b.size())
			return false;
		for(std::size_t i=0;i<a.size();++i){
			const unsigned char ac=static_cast<unsigned char>(a[i]);
			const unsigned char bc=static_cast<unsigned char>(b[i]);
			if(static_cast<char>(std::tolower(ac))!=
			   static_cast<char>(std::tolower(bc)))
				return false;
		}
		return true;
	}

	static std::int64_t parse_i64(std::string_view s,const char*msg){
		if(s.empty())
			detail::throw_inv(msg);
		bool neg=false;
		if(s.front()=='+'||s.front()=='-'){
			neg=s.front()=='-';
			s.remove_prefix(1);
			if(s.empty())
				detail::throw_inv(msg);
		}
		const std::uint64_t lim=neg?
			(static_cast<std::uint64_t>(
				std::numeric_limits<std::int64_t>::max())+1u):
			static_cast<std::uint64_t>(
				std::numeric_limits<std::int64_t>::max());
		std::uint64_t v=0;
		for(char ch : s){
			if(ch<'0'||ch>'9')
				detail::throw_inv(msg);
			const std::uint64_t d=static_cast<std::uint64_t>(ch-'0');
			if(v>(lim-d)/10u)
				detail::throw_ovf(msg);
			v=v*10u+d;
		}
		if(!neg)
			return static_cast<std::int64_t>(v);
		if(v==lim)
			return std::numeric_limits<std::int64_t>::min();
		return -static_cast<std::int64_t>(v);
	}

	static BigInt pow10_u64(std::uint64_t n){
		return mini_mp::pow(BigInt(10),n);
	}

	static BigFloat parse_bin(int sign,std::string_view text,
							  std::size_t prec,FloatRnd rnd){
		const std::size_t ppos=text.find_first_of("pP");
		std::string_view body=text.substr(0,ppos);
		std::int64_t be=0;
		if(ppos!=std::string_view::npos)
			be=parse_i64(text.substr(ppos+1),
						 "BigFloat::parse: malformed binary exponent");
		if(body.empty())
			detail::throw_inv("BigFloat::parse: malformed binary");
		const std::size_t dot=body.find('.');
		std::string digits;
		std::size_t frac=0;
		if(dot==std::string_view::npos){
			digits.assign(body.begin(),body.end());
		}else{
			digits.assign(body.begin(),body.begin()+
				static_cast<std::ptrdiff_t>(dot));
			digits.append(body.begin()+static_cast<std::ptrdiff_t>(dot+1u),
						  body.end());
			frac=body.size()-dot-1u;
		}
		if(digits.empty())
			detail::throw_inv("BigFloat::parse: malformed binary");
		for(char ch : digits){
			if(ch!='0'&&ch!='1')
				detail::throw_inv("BigFloat::parse: malformed binary");
		}
		BigInt mag=BigInt::parse(digits,2);
		if(mag.is_zero())
			return zero(prec);
		be=sub_exp_sz(be,frac,"BigFloat::parse: exponent overflow");
		return from_parts(sign,std::move(mag),be,prec,rnd);
	}

	static BigFloat parse_dec(int sign,std::string_view text,
							  std::size_t prec,FloatRnd rnd){
		const std::size_t epos=text.find_first_of("eE");
		std::string_view body=text.substr(0,epos);
		std::int64_t de=0;
		if(epos!=std::string_view::npos)
			de=parse_i64(text.substr(epos+1),
						 "BigFloat::parse: malformed decimal exponent");
		if(body.empty())
			detail::throw_inv("BigFloat::parse: malformed decimal");
		const std::size_t dot=body.find('.');
		std::string digits;
		std::size_t frac=0;
		if(dot==std::string_view::npos){
			digits.assign(body.begin(),body.end());
		}else{
			digits.assign(body.begin(),body.begin()+
				static_cast<std::ptrdiff_t>(dot));
			digits.append(body.begin()+static_cast<std::ptrdiff_t>(dot+1u),
						  body.end());
			frac=body.size()-dot-1u;
		}
		if(digits.empty())
			detail::throw_inv("BigFloat::parse: malformed decimal");
		for(char ch : digits){
			if(ch<'0'||ch>'9')
				detail::throw_inv("BigFloat::parse: malformed decimal");
		}
		BigInt mag=BigInt::parse(digits,10);
		if(mag.is_zero())
			return zero(prec);
		de=sub_exp_sz(de,frac,"BigFloat::parse: exponent overflow");
		if(de>=0){
			mag*=pow10_u64(static_cast<std::uint64_t>(de));
			return from_parts(sign,std::move(mag),0,prec,rnd);
		}
		const std::uint64_t den_exp=abs_i64(de);
		const BigInt den=pow10_u64(den_exp);
		return from_ratio(sign,std::move(mag),den,prec,rnd);
	}

	static BigFloat from_ratio(int sign,BigInt num,const BigInt&den,
							   std::size_t prec,FloatRnd rnd){
		if(den.sign()<=0)
			detail::throw_dom("BigFloat: non-positive denominator");
		if(num.sign()<0){
			num=-num;
			sign=-sign;
		}
		if(num.is_zero())
			return zero(prec);
		const std::size_t want=prec;
		const std::size_t nb=num.bit_length();
		const std::size_t db=den.bit_length();
		std::size_t sh=0;
		if(nb<=db){
			const std::size_t d=db-nb;
			sh=prec_add(want,d,"BigFloat: precision too large");
		}else{
			const std::size_t have=nb-db;
			if(have<want)
				sh=want-have;
		}
		const bool d2sh=sh!=0&&den.limbs_.size()==2;
		if(sh!=0&&!d2sh)
			num<<=sh;
		std::pair<detail::limbs_t,detail::limbs_t> qr;
		if(den.limbs_.size()==1){
			qr=detail::dvmk_absl(num.limbs_,den.limbs_);
		}else if(den.limbs_.size()==2){
			qr=d2sh?detail::dvm2_shl_absl(num.limbs_,sh,den.limbs_)
				   :detail::dvm2_absl(num.limbs_,den.limbs_);
		}else{
			detail::ensure_at();
			qr=detail::use_bzdiv(num.limbs_.size(),den.limbs_.size())
				?detail::dvmbz_abs(num.limbs_,den.limbs_)
				:detail::dvmk_absl(num.limbs_,den.limbs_);
		}
		std::int64_t e=0;
		if(sh!=0)
			e=sub_exp_sz(e,sh,"BigFloat: exponent overflow");
		detail::limbs_t q=std::move(qr.first);
		if(q.empty())
			return zero(prec);
		const int os=(sign<0)?-1:1;
		bool rem=!qr.second.empty();
		if(detail::bit_length(q)>prec)
			return from_limbs(os,std::move(q),e,prec,rnd,rem);
		int ix=0;
		if(rem){
			bool inc=false;
			switch(rnd){
			case FloatRnd::nearest:{
				const int cmp=detail::cmp_abs_dbl(den.limbs_,qr.second);
				const bool odd=(q[0]&1u)!=0u;
				inc=cmp<0||(cmp==0&&odd);
				break;
			}
			case FloatRnd::zero:
				inc=false;
				break;
			case FloatRnd::up:
				inc=os>0;
				break;
			case FloatRnd::down:
				inc=os<0;
				break;
			case FloatRnd::away:
				inc=true;
				break;
			}
			if(inc){
				detail::add_one_ip(q);
				if(detail::bit_length(q)>prec){
					detail::shr_ip(q,std::size_t(1));
					e=add_exp(e,1,"BigFloat: exponent overflow");
				}
			}
			ix=os>0?(inc?1:-1):(inc?-1:1);
		}
		BigFloat out=from_exact_limbs(os,std::move(q),e,prec);
		out.inex_=ix;
		return out;
	}

	std::size_t dec_digits() const{
		if(prec_>std::numeric_limits<std::size_t>::max()/30103u)
			detail::throw_ovf("BigFloat::to_string: precision too large");
		return (prec_*30103u)/100000u+4u;
	}

	std::string to_bin_string() const{
		std::string out;
		if(sign_<0)
			out.push_back('-');
		out+="0b";
		out+=mag_.to_string(2);
		out.push_back('p');
		out+=std::to_string(exp_);
		return out;
	}

  private:
	int sign_=0;
	FloatKind kind_=FloatKind::finite;
	std::int64_t exp_=0;
	std::size_t prec_=default_prec;
	BigInt mag_;
	int inex_=0;
};

template<class Float>
inline Float BigRat::to_floating(FloatRnd rnd) const{
	static_assert(std::numeric_limits<Float>::radix==2,
				  "BigRat::to_floating requires binary floating point");
	static_assert(std::numeric_limits<Float>::digits<=62,
				  "BigRat::to_floating requires at most 62 precision bits");
	if(num_.is_zero())
		return Float(0);
	const int s=num_.sign();
	const std::size_t nb=detail::bit_length(num_.limbs_);
	const std::size_t db=detail::bit_length(den_.limbs_);
	if(nb>static_cast<std::size_t>(
		   std::numeric_limits<std::int64_t>::max())||
	   db>static_cast<std::size_t>(
		   std::numeric_limits<std::int64_t>::max())){
		return static_cast<Float>(
			BigFloat(*this,std::numeric_limits<Float>::digits,rnd).to_double());
	}

	std::int64_t e=static_cast<std::int64_t>(nb)-
				   static_cast<std::int64_t>(db);

	constexpr int prec=std::numeric_limits<Float>::digits;
	constexpr int work=prec;
	constexpr int max_e=std::numeric_limits<Float>::max_exponent-1;
	constexpr int min_e=std::numeric_limits<Float>::min_exponent-1;
	if(e>max_e){
		return s<0?-std::numeric_limits<Float>::infinity():
			std::numeric_limits<Float>::infinity();
	}
	if(e<min_e){
		return static_cast<Float>(
			BigFloat(*this,prec,rnd).to_double());
	}
	if(e>=0&&std::max(num_.limbs_.size(),den_.limbs_.size())>32u){
		return static_cast<Float>(
			BigFloat(*this,prec,rnd).to_double());
	}

	auto div_abs=[](const detail::limbs_t&u,const detail::limbs_t&v)
		->std::pair<detail::limbs_t,detail::limbs_t>{
		if(v.size()==1)
			return detail::dvmk_absl(u,v);
		if(v.size()==2)
			return detail::dvm2_absl(u,v);
		detail::ensure_at();
		return detail::use_bzdiv(u.size(),v.size())
			?detail::dvmbz_abs(u,v)
			:detail::dvmk_absl(u,v);
	};

	const std::int64_t shift=static_cast<std::int64_t>(work-1)-e;
	std::pair<detail::limbs_t,detail::limbs_t> qr;
	detail::limbs_t scaled;
	detail::limbs_t scaled_den;
	const detail::limbs_t*divisor=&den_.limbs_;
	if(shift>=0){
		const std::size_t sh=checked_size(
			static_cast<std::uint64_t>(shift),
			"BigRat::to_floating: exponent too large");
		if(den_.limbs_.size()==2&&sh!=0){
			qr=detail::dvm2_shl_absl(num_.limbs_,sh,den_.limbs_);
		}else{
			if(sh==0)
				scaled=num_.limbs_;
			else
				detail::shl_into(scaled,num_.limbs_,sh);
			qr=div_abs(scaled,den_.limbs_);
		}
	}else{
		detail::shl_into(scaled_den,den_.limbs_,checked_size(abs_i64(shift),
			"BigRat::to_floating: exponent too small"));
		divisor=&scaled_den;
		qr=div_abs(num_.limbs_,scaled_den);
	}

	std::uint64_t q=qr.first.empty()?0:qr.first[0];
	const std::uint64_t lower=std::uint64_t(1)<<(work-1);
	if(q<lower){
		--e;
		if(e<min_e){
			return static_cast<Float>(
				BigFloat(*this,prec,rnd).to_double());
		}
		q<<=1u;
		detail::shl_ip(qr.second,1u);
		if(detail::cmp_abs(qr.second,*divisor)>=0){
			detail::sub_abs_ip(qr.second,*divisor);
			++q;
		}
	}
	const bool rem=!qr.second.empty();
	bool inc=false;
	switch(rnd){
	case FloatRnd::nearest:{
		if(rem){
			const int cmp=detail::cmp_abs_dbl(*divisor,qr.second);
			inc=cmp<0||(cmp==0&&((q&1u)!=0));
		}
		break;
	}
	case FloatRnd::zero:
		inc=false;
		break;
	case FloatRnd::up:
		inc=rem&&s>0;
		break;
	case FloatRnd::down:
		inc=rem&&s<0;
		break;
	case FloatRnd::away:
		inc=rem;
		break;
	}

	std::uint64_t mant=q;
	if(inc){
		++mant;
		if(mant==(std::uint64_t(1)<<prec)){
			mant>>=1u;
			++e;
			if(e>max_e){
				return s<0?-std::numeric_limits<Float>::infinity():
					std::numeric_limits<Float>::infinity();
			}
		}
	}

	Float out=std::ldexp(static_cast<Float>(mant),
						 static_cast<int>(e-(prec-1)));
	return s<0?-out:out;
}

inline double BigRat::to_double(FloatRnd rnd) const{
	return to_floating<double>(rnd);
}

inline float BigRat::to_float(FloatRnd rnd) const{
	return to_floating<float>(rnd);
}

inline BigFloat abs(const BigFloat&x){ return x.abs(); }
inline BigFloat copy_sign(const BigFloat&x,const BigFloat&sign,
						  FloatRnd rnd=FloatRnd::nearest){
	return x.copy_sign(sign,rnd);
}
inline BigFloat set_sign(const BigFloat&x,int sign,
						 FloatRnd rnd=FloatRnd::nearest){
	return x.with_sign(sign,rnd);
}
inline BigFloat floor(const BigFloat&x){ return x.floor(); }
inline BigFloat ceil(const BigFloat&x){ return x.ceil(); }
inline BigFloat trunc(const BigFloat&x){ return x.trunc(); }
inline BigFloat rint(const BigFloat&x,FloatRnd rnd=FloatRnd::nearest){
	return x.rint(rnd);
}
inline BigFloat frac(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return x.frac(prec,rnd);
}
inline BigFloat ldexp(const BigFloat&x,std::int64_t bits){
	return x.ldexp(bits);
}
inline BigFloat scalbn(const BigFloat&x,std::int64_t bits){
	return x.scalbn(bits);
}
inline std::pair<BigFloat,std::int64_t> frexp(
	const BigFloat&x,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return x.frexp(prec,rnd);
}
inline std::int64_t ilogb(const BigFloat&x){
	return x.ilogb();
}
inline BigRat to_bigrat(const BigFloat&x){
	return x.to_bigrat();
}
inline BigFloat epsilon(std::size_t prec=BigFloat::default_prec){
	return BigFloat::epsilon(prec);
}
inline BigFloat ulp(const BigFloat&x){
	return x.ulp();
}
inline BigFloat sqr(const BigFloat&x,std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sqr(x,prec,rnd);
}
inline BigFloat fma(const BigFloat&a,const BigFloat&b,const BigFloat&c,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::fma(a,b,c,prec,rnd);
}
inline BigFloat sqrt(const BigFloat&x,std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sqrt(x,prec,rnd);
}
inline BigFloat rootn(const BigFloat&x,std::uint32_t k,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::rootn(x,k,prec,rnd);
}
inline BigFloat cbrt(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::cbrt(x,prec,rnd);
}
inline BigFloat recip(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::recip(x,prec,rnd);
}
inline BigFloat pow_ui(const BigFloat&x,std::uint64_t n,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::pow_ui(x,n,prec,rnd);
}
inline BigFloat pow_si(const BigFloat&x,std::int64_t n,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::pow_si(x,n,prec,rnd);
}
inline BigFloat const_pi(std::size_t prec=BigFloat::default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::const_pi(prec,rnd);
}
inline BigFloat const_log2(std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::const_log2(prec,rnd);
}
inline BigFloat exp(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::exp(x,prec,rnd);
}
inline BigFloat exp2(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::exp2(x,prec,rnd);
}
inline BigFloat exp10(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::exp10(x,prec,rnd);
}
inline BigFloat expm1(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::expm1(x,prec,rnd);
}
inline BigFloat log(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::log(x,prec,rnd);
}
inline BigFloat log2(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::log2(x,prec,rnd);
}
inline BigFloat log10(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::log10(x,prec,rnd);
}
inline BigFloat log1p(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::log1p(x,prec,rnd);
}
inline BigFloat pow(const BigFloat&x,const BigFloat&y,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::pow(x,y,prec,rnd);
}
inline std::pair<BigFloat,BigFloat> sin_cos(
	const BigFloat&x,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sin_cos(x,prec,rnd);
}
inline BigFloat sin(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sin(x,prec,rnd);
}
inline BigFloat cos(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::cos(x,prec,rnd);
}
inline BigFloat tan(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::tan(x,prec,rnd);
}
inline BigFloat asin(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::asin(x,prec,rnd);
}
inline BigFloat acos(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::acos(x,prec,rnd);
}
inline BigFloat atan(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::atan(x,prec,rnd);
}
inline BigFloat atan2(const BigFloat&y,const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::atan2(y,x,prec,rnd);
}
inline BigFloat sinh(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sinh(x,prec,rnd);
}
inline BigFloat cosh(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::cosh(x,prec,rnd);
}
inline std::pair<BigFloat,BigFloat> sinh_cosh(
	const BigFloat&x,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::sinh_cosh(x,prec,rnd);
}
inline BigFloat tanh(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::tanh(x,prec,rnd);
}
inline BigFloat asinh(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::asinh(x,prec,rnd);
}
inline BigFloat acosh(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::acosh(x,prec,rnd);
}
inline BigFloat atanh(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::atanh(x,prec,rnd);
}
inline BigFloat gamma(const BigFloat&x,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::gamma(x,prec,rnd);
}
inline BigFloat erf(const BigFloat&x,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::erf(x,prec,rnd);
}
inline BigFloat erfc(const BigFloat&x,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::erfc(x,prec,rnd);
}
inline BigFloat dim(const BigFloat&a,const BigFloat&b,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::dim(a,b,prec,rnd);
}
inline BigFloat min(const BigFloat&a,const BigFloat&b,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::min(a,b,prec,rnd);
}
inline BigFloat max(const BigFloat&a,const BigFloat&b,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::max(a,b,prec,rnd);
}
inline BigFloat fmod(const BigFloat&a,const BigFloat&b,
					 std::size_t prec=BigFloat::default_prec,
					 FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::fmod(a,b,prec,rnd);
}
inline BigFloat remainder(const BigFloat&a,const BigFloat&b,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::remainder(a,b,prec,rnd);
}
inline std::pair<BigFloat,BigFloat> modf(
	const BigFloat&x,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::modf(x,prec,rnd);
}
inline BigFloat hypot(const BigFloat&a,const BigFloat&b,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigFloat::hypot(a,b,prec,rnd);
}
inline int cmpabs(const BigFloat&a,const BigFloat&b){
	return BigFloat::cmpabs(a,b);
}
inline BigFloat next_up(const BigFloat&x){ return x.next_up(); }
inline BigFloat next_down(const BigFloat&x){ return x.next_down(); }
inline BigFloat next_toward(const BigFloat&x,const BigFloat&to){
	return x.next_toward(to);
}





class BigComplex{
  public:
	BigComplex() :
		real_(BigFloat::zero()),imag_(BigFloat::zero()){}

	BigComplex(BigFloat real,BigFloat imag=BigFloat::zero()) :
		real_(std::move(real)),imag_(std::move(imag)){
		const std::size_t p=precision();
		real_=real_.rounded(p);
		imag_=imag_.rounded(p);
	}

	explicit BigComplex(const BigInt&real,
						std::size_t prec=BigFloat::default_prec) :
		real_(real,prec),imag_(BigFloat::zero(prec)){}

	explicit BigComplex(const BigRat&real,
						std::size_t prec=BigFloat::default_prec) :
		real_(real,prec),imag_(BigFloat::zero(prec)){}

	explicit BigComplex(double real,
						std::size_t prec=BigFloat::default_prec) :
		real_(real,prec),imag_(BigFloat::zero(prec)){}

	const BigFloat&real() const noexcept{ return real_; }
	const BigFloat&imag() const noexcept{ return imag_; }
	BigFloat&real() noexcept{ return real_; }
	BigFloat&imag() noexcept{ return imag_; }

	std::size_t precision() const noexcept{
		return std::max(real_.precision(),imag_.precision());
	}

	bool is_zero() const noexcept{
		return real_.is_zero()&&imag_.is_zero();
	}

	BigComplex rounded(std::size_t prec,
					   FloatRnd rnd=FloatRnd::nearest) const{
		return BigComplex(real_.rounded(prec,rnd),imag_.rounded(prec,rnd));
	}

	BigComplex conj() const{ return BigComplex(real_,-imag_); }
	BigComplex neg() const{ return BigComplex(-real_,-imag_); }

	BigFloat abs(std::size_t prec=BigFloat::default_prec,
				 FloatRnd rnd=FloatRnd::nearest) const{
		return mini_mp::hypot(real_,imag_,prec,rnd);
	}

	static BigComplex add(const BigComplex&a,const BigComplex&b,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max({prec,a.precision(),b.precision()});
		return BigComplex(
			BigFloat::add(a.real_,b.real_,work,FloatRnd::nearest),
			BigFloat::add(a.imag_,b.imag_,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex sub(const BigComplex&a,const BigComplex&b,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max({prec,a.precision(),b.precision()});
		return BigComplex(
			BigFloat::sub(a.real_,b.real_,work,FloatRnd::nearest),
			BigFloat::sub(a.imag_,b.imag_,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex mul(const BigComplex&a,const BigComplex&b,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(a.imag_.is_zero())
			return mul_real(b,a.real_,prec,rnd);
		if(b.imag_.is_zero())
			return mul_real(a,b.real_,prec,rnd);
		if(a.real_.is_zero())
			return BigComplex(-BigFloat::mul(a.imag_,b.imag_,prec,rnd),
							  BigFloat::mul(a.imag_,b.real_,prec,rnd));
		if(b.real_.is_zero())
			return BigComplex(-BigFloat::mul(a.imag_,b.imag_,prec,rnd),
							  BigFloat::mul(a.real_,b.imag_,prec,rnd));
		const std::size_t work=std::max({prec,a.precision(),b.precision()})+16u;
		const BigFloat ar=a.real_.rounded(work);
		const BigFloat ai=a.imag_.rounded(work);
		const BigFloat br=b.real_.rounded(work);
		const BigFloat bi=b.imag_.rounded(work);
		if(work<=256u){
			const BigFloat rr=BigFloat::sub(BigFloat::mul(ar,br,work),
											BigFloat::mul(ai,bi,work),work);
			const BigFloat ii=BigFloat::add(BigFloat::mul(ar,bi,work),
											BigFloat::mul(ai,br,work),work);
			return BigComplex(rr,ii).rounded(prec,rnd);
		}
		const BigFloat ac=BigFloat::mul(ar,br,work);
		const BigFloat bd=BigFloat::mul(ai,bi,work);
		const BigFloat ab=BigFloat::add(ar,ai,work);
		const BigFloat cd=BigFloat::add(br,bi,work);
		const BigFloat rr=BigFloat::sub(ac,bd,work);
		const BigFloat ii=BigFloat::sub(
			BigFloat::sub(BigFloat::mul(ab,cd,work),ac,work),bd,work);
		return BigComplex(rr,ii).rounded(prec,rnd);
	}

	static BigComplex mul_real(const BigComplex&a,const BigFloat&b,
							   std::size_t prec=BigFloat::default_prec,
							   FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max({prec,a.precision(),b.precision()})+8u;
		return BigComplex(BigFloat::mul(a.real_,b,work,FloatRnd::nearest),
						  BigFloat::mul(a.imag_,b,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex div(const BigComplex&a,const BigComplex&b,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(b.is_zero())
			detail::throw_dom("BigComplex::div: division by zero");
		if(b.imag_.is_zero()){
			const std::size_t work=std::max({prec,a.precision(),
											 b.real_.precision()})+16u;
			return BigComplex(BigFloat::div(a.real_,b.real_,work),
							  BigFloat::div(a.imag_,b.real_,work)).
				rounded(prec,rnd);
		}
		if(a.is_zero())
			return BigComplex(BigFloat::zero(prec),BigFloat::zero(prec));
		const std::size_t work=std::max({prec,a.precision(),b.precision()})+32u;
		const BigFloat ar=a.real_.rounded(work);
		const BigFloat ai=a.imag_.rounded(work);
		const BigFloat br=b.real_.rounded(work);
		const BigFloat bi=b.imag_.rounded(work);
		const BigFloat den=BigFloat::fma(br,br,BigFloat::sqr(bi,work),work);
		const BigFloat invden=BigFloat::recip(den,work,FloatRnd::nearest);
		const BigFloat rr=BigFloat::mul(
			BigFloat::add(BigFloat::mul(ar,br,work),
						  BigFloat::mul(ai,bi,work),work),invden,work);
		const BigFloat ii=BigFloat::mul(
			BigFloat::sub(BigFloat::mul(ai,br,work),
						  BigFloat::mul(ar,bi,work),work),invden,work);
		return BigComplex(rr,ii).rounded(prec,rnd);
	}

	static BigComplex exp(const BigComplex&z,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero())
			return BigComplex(mini_mp::exp(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		if(z.real_.is_zero()){
			const std::size_t work=std::max(prec,z.precision())+16u;
			const auto sc=mini_mp::sin_cos(z.imag_,work,FloatRnd::nearest);
			return BigComplex(sc.second,sc.first).rounded(prec,rnd);
		}
		const std::size_t work=std::max(prec,z.precision())+32u;
		const BigFloat er=mini_mp::exp(z.real_,work,FloatRnd::nearest);
		const auto sc=mini_mp::sin_cos(z.imag_,work,FloatRnd::nearest);
		return BigComplex(BigFloat::mul(er,sc.second,work),
						  BigFloat::mul(er,sc.first,work)).rounded(prec,rnd);
	}

	static BigFloat arg(const BigComplex&z,
						std::size_t prec=BigFloat::default_prec,
						FloatRnd rnd=FloatRnd::nearest){
		if(z.is_zero())
			return BigFloat::zero(prec);
		return mini_mp::atan2(z.imag_,z.real_,prec,rnd);
	}

	static BigComplex polar(const BigFloat&r,const BigFloat&theta,
							std::size_t prec=BigFloat::default_prec,
							FloatRnd rnd=FloatRnd::nearest){
		if(r.is_zero())
			return zero(prec);
		const std::size_t work=std::max({prec,r.precision(),
										 theta.precision()})+16u;
		const auto sc=mini_mp::sin_cos(theta,work,FloatRnd::nearest);
		return BigComplex(BigFloat::mul(r,sc.second,work,FloatRnd::nearest),
						  BigFloat::mul(r,sc.first,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex log(const BigComplex&z,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(z.is_zero())
			return BigComplex(BigFloat::inf(-1,prec),BigFloat::zero(prec));
		if(z.imag_.is_zero()){
			if(z.real_.sign()>0)
				return BigComplex(mini_mp::log(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			if(z.real_.sign()<0){
				const std::size_t work=std::max(prec,z.precision())+32u;
				return BigComplex(mini_mp::log(z.real_.abs(),work,
											   FloatRnd::nearest),
								  mini_mp::const_pi(work,FloatRnd::nearest)).
					rounded(prec,rnd);
			}
			return BigComplex(mini_mp::log(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		}
		if(z.real_.is_zero()){
			const std::size_t work=std::max(prec,z.precision())+32u;
			BigFloat a=mini_mp::log(z.imag_.abs(),work,FloatRnd::nearest);
			BigFloat p=mini_mp::const_pi(work,FloatRnd::nearest).ldexp(-1);
			if(z.imag_.sign()<0)
				p=p.neg();
			return BigComplex(a,p).rounded(prec,rnd);
		}
		const std::size_t work=std::max(prec,z.precision())+64u;
		return BigComplex(mini_mp::log(z.abs(work,FloatRnd::nearest),
									   work,FloatRnd::nearest),
						  arg(z,work,FloatRnd::nearest)).rounded(prec,rnd);
	}

	static BigComplex sqrt(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero()){
			if(z.real_.sign()>=0)
				return BigComplex(mini_mp::sqrt(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::sqrt(z.real_.abs(),prec,rnd));
		}
		if(z.real_.is_zero()){
			const std::size_t work=std::max(prec,z.precision())+16u;
			BigFloat t=mini_mp::sqrt(z.imag_.abs().ldexp(-1),work,
									 FloatRnd::nearest);
			BigFloat u=t;
			if(z.imag_.sign()<0)
				u=u.neg();
			return BigComplex(t,u).rounded(prec,rnd);
		}
		const std::size_t work=std::max(prec,z.precision())+48u;
		const BigFloat ar=z.real_.rounded(work,FloatRnd::nearest);
		const BigFloat ai=z.imag_.rounded(work,FloatRnd::nearest);
		const BigFloat r=mini_mp::hypot(ar,ai,work,FloatRnd::nearest);
		BigFloat rr;
		BigFloat ii;
		if(ar.sign()>=0){
			const BigFloat t=BigFloat::add(r,ar,work,FloatRnd::nearest);
			const BigFloat den=mini_mp::sqrt(t.ldexp(1),work,
											 FloatRnd::nearest);
			rr=den.ldexp(-1);
			ii=BigFloat::div(ai,den,work,FloatRnd::nearest);
		}else{
			const BigFloat t=BigFloat::sub(r,ar,work,FloatRnd::nearest);
			const BigFloat den=mini_mp::sqrt(t.ldexp(1),work,
											 FloatRnd::nearest);
			rr=BigFloat::div(ai.abs(),den,work,FloatRnd::nearest);
			ii=den.ldexp(-1);
			if(ai.sign()<0)
				ii=ii.neg();
		}
		return BigComplex(rr,ii).rounded(prec,rnd);
	}

	static BigComplex pow_ui(const BigComplex&z,std::uint64_t n,
							 std::size_t prec=BigFloat::default_prec,
							 FloatRnd rnd=FloatRnd::nearest){
		if(n==0)
			return one(prec);
		if(n==1)
			return z.rounded(prec,rnd);
		if(n==2)
			return mul(z,z,prec,rnd);
		if(n==3)
			return mul(mul(z,z,prec,FloatRnd::nearest),z,prec,rnd);
		const std::size_t work=std::max(prec,z.precision())+
			static_cast<std::size_t>(std::bit_width(n))+16u;
		BigComplex out=one(work);
		BigComplex base=z.rounded(work,FloatRnd::nearest);
		while(n!=0){
			if((n&1u)!=0u)
				out=mul(out,base,work,FloatRnd::nearest);
			n>>=1u;
			if(n!=0)
				base=mul(base,base,work,FloatRnd::nearest);
		}
		return out.rounded(prec,rnd);
	}

	static BigComplex pow_si(const BigComplex&z,std::int64_t n,
							 std::size_t prec=BigFloat::default_prec,
							 FloatRnd rnd=FloatRnd::nearest){
		if(n>=0)
			return pow_ui(z,static_cast<std::uint64_t>(n),prec,rnd);
		const std::size_t work=std::max(prec,z.precision())+
			static_cast<std::size_t>(std::bit_width(abs_i64(n)))+24u;
		return div(one(work),pow_ui(z,abs_i64(n),work,FloatRnd::nearest),
				   prec,rnd);
	}

	static BigComplex pow(const BigComplex&x,const BigComplex&y,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(y.is_zero())
			return one(prec);
		if(x.is_zero()){
			if(y.imag_.is_zero()&&y.real_.sign()>0)
				return zero(prec);
			return nan(prec);
		}
		if(y.imag_.is_zero()){
			if(x.imag_.is_zero()&&x.real_.sign()>0)
				return BigComplex(mini_mp::pow(x.real_,y.real_,prec,rnd),
								  BigFloat::zero(prec));
			if(y.real_.is_finite()&&y.real_.is_integer()&&
			   y.real_.fits_i64(FloatRnd::zero))
				return pow_si(x,y.real_.to_i64(FloatRnd::zero),prec,rnd);
			if(y.real_.is_finite()){
				const std::size_t cp=std::max({prec,x.precision(),
											   y.precision()});
				const BigFloat half=BigFloat::one(cp).ldexp(-1);
				if(compare(y.real_,half)==0)
					return sqrt(x,prec,rnd);
				if(compare(y.real_,half.neg())==0){
					const std::size_t work=cp+32u;
					return div(one(work),sqrt(x,work,FloatRnd::nearest),
							   prec,rnd);
				}
			}
		}
		const std::size_t work=std::max({prec,x.precision(),y.precision()})+
			96u;
		BigComplex t=mul(y.rounded(work,FloatRnd::nearest),
						 log(x,work,FloatRnd::nearest),work,
						 FloatRnd::nearest);
		return exp(t,prec,rnd);
	}

	static std::pair<BigComplex,BigComplex> sin_cos(
		const BigComplex&z,std::size_t prec=BigFloat::default_prec,
		FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero()){
			const auto sc=mini_mp::sin_cos(z.real_,prec,rnd);
			return {BigComplex(sc.first,BigFloat::zero(prec)),
					BigComplex(sc.second,BigFloat::zero(prec))};
		}
		if(z.real_.is_zero()){
			const auto hc=mini_mp::sinh_cosh(z.imag_,prec,rnd);
			return {BigComplex(BigFloat::zero(prec),hc.first),
					BigComplex(hc.second,BigFloat::zero(prec))};
		}
		const std::size_t work=std::max(prec,z.precision())+48u;
		const auto sc=mini_mp::sin_cos(z.real_,work,FloatRnd::nearest);
		const auto hc=mini_mp::sinh_cosh(z.imag_,work,FloatRnd::nearest);
		BigComplex s(BigFloat::mul(sc.first,hc.second,work,
								   FloatRnd::nearest),
					 BigFloat::mul(sc.second,hc.first,work,
								   FloatRnd::nearest));
		BigComplex c(BigFloat::mul(sc.second,hc.second,work,
								   FloatRnd::nearest),
					 BigFloat::mul(sc.first,hc.first,work,
								   FloatRnd::nearest).neg());
		return {s.rounded(prec,rnd),c.rounded(prec,rnd)};
	}

	static BigComplex sin(const BigComplex&z,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		return sin_cos(z,prec,rnd).first;
	}

	static BigComplex cos(const BigComplex&z,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		return sin_cos(z,prec,rnd).second;
	}

	static BigComplex tan(const BigComplex&z,
						  std::size_t prec=BigFloat::default_prec,
						  FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero())
			return BigComplex(mini_mp::tan(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		if(z.real_.is_zero())
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::tanh(z.imag_,prec,rnd));
		const std::size_t work=std::max(prec,z.precision())+64u;
		const auto sc=mini_mp::sin_cos(z.real_.rounded(work,
			FloatRnd::nearest).ldexp(1),work,FloatRnd::nearest);
		const auto hc=mini_mp::sinh_cosh(z.imag_.rounded(work,
			FloatRnd::nearest).ldexp(1),work,FloatRnd::nearest);
		const BigFloat den=BigFloat::add(sc.second,hc.second,work,
										 FloatRnd::nearest);
		return BigComplex(BigFloat::div(sc.first,den,work,FloatRnd::nearest),
						  BigFloat::div(hc.first,den,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex asin(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max(prec,z.precision())+80u;
		if(z.imag_.is_zero()&&z.real_.is_finite()){
			const int c=mini_mp::cmpabs(z.real_,BigFloat::one(work));
			if(c<=0)
				return BigComplex(mini_mp::asin(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			BigFloat hpi=mini_mp::const_pi(work,FloatRnd::nearest).
				ldexp(-1);
			if(z.real_.sign()<0)
				hpi=hpi.neg();
			return BigComplex(hpi,mini_mp::acosh(z.real_.abs(),work,
												FloatRnd::nearest)).
				rounded(prec,rnd);
		}
		if(z.real_.is_zero())
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::asinh(z.imag_,prec,rnd));
		BigComplex t=sub(one(work),mul(z,z,work,FloatRnd::nearest),
						 work,FloatRnd::nearest);
		t=sqrt(t,work,FloatRnd::nearest);
		BigComplex u=add(mul_i(z),t,work,FloatRnd::nearest);
		return div_i(log(u,work,FloatRnd::nearest)).rounded(prec,rnd);
	}

	static BigComplex acos(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max(prec,z.precision())+80u;
		if(z.imag_.is_zero()&&z.real_.is_finite()){
			const int c=mini_mp::cmpabs(z.real_,BigFloat::one(work));
			if(c<=0)
				return BigComplex(mini_mp::acos(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			BigFloat re=z.real_.sign()>0?BigFloat::zero(work):
				mini_mp::const_pi(work,FloatRnd::nearest);
			BigFloat im=mini_mp::acosh(z.real_.abs(),work,
									   FloatRnd::nearest).neg();
			return BigComplex(re,im).rounded(prec,rnd);
		}
		BigComplex a=asin(z,work,FloatRnd::nearest);
		BigFloat hpi=mini_mp::const_pi(work,FloatRnd::nearest).ldexp(-1);
		return BigComplex(BigFloat::sub(hpi,a.real_,work,
										FloatRnd::nearest),
						  a.imag_.neg()).rounded(prec,rnd);
	}

	static BigComplex atan(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero())
			return BigComplex(mini_mp::atan(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		const std::size_t work=std::max(prec,z.precision())+80u;
		if(z.real_.is_zero()&&z.imag_.is_finite()){
			const int c=mini_mp::cmpabs(z.imag_,BigFloat::one(work));
			if(c<0)
				return BigComplex(BigFloat::zero(prec),
								  mini_mp::atanh(z.imag_,prec,rnd));
			if(c==0)
				return BigComplex(BigFloat::zero(prec),
								  BigFloat::inf(z.imag_.sign(),prec));
			BigFloat re=mini_mp::const_pi(work,FloatRnd::nearest).
				ldexp(-1);
			if(z.imag_.sign()<0)
				re=re.neg();
			BigFloat im=mini_mp::atanh(BigFloat::recip(z.imag_,work,
				FloatRnd::nearest),work,FloatRnd::nearest);
			return BigComplex(re,im).rounded(prec,rnd);
		}
		return div_i(atanh(mul_i(z),work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static std::pair<BigComplex,BigComplex> sinh_cosh(
		const BigComplex&z,std::size_t prec=BigFloat::default_prec,
		FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero()){
			const auto hc=mini_mp::sinh_cosh(z.real_,prec,rnd);
			return {BigComplex(hc.first,BigFloat::zero(prec)),
					BigComplex(hc.second,BigFloat::zero(prec))};
		}
		if(z.real_.is_zero()){
			const auto sc=mini_mp::sin_cos(z.imag_,prec,rnd);
			return {BigComplex(BigFloat::zero(prec),sc.first),
					BigComplex(sc.second,BigFloat::zero(prec))};
		}
		const std::size_t work=std::max(prec,z.precision())+48u;
		const auto hc=mini_mp::sinh_cosh(z.real_,work,FloatRnd::nearest);
		const auto sc=mini_mp::sin_cos(z.imag_,work,FloatRnd::nearest);
		BigComplex sh(BigFloat::mul(hc.first,sc.second,work,
									FloatRnd::nearest),
					  BigFloat::mul(hc.second,sc.first,work,
									FloatRnd::nearest));
		BigComplex ch(BigFloat::mul(hc.second,sc.second,work,
									FloatRnd::nearest),
					  BigFloat::mul(hc.first,sc.first,work,
									FloatRnd::nearest));
		return {sh.rounded(prec,rnd),ch.rounded(prec,rnd)};
	}

	static BigComplex sinh(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		return sinh_cosh(z,prec,rnd).first;
	}

	static BigComplex cosh(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		return sinh_cosh(z,prec,rnd).second;
	}

	static BigComplex tanh(const BigComplex&z,
						   std::size_t prec=BigFloat::default_prec,
						   FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero())
			return BigComplex(mini_mp::tanh(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		if(z.real_.is_zero())
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::tan(z.imag_,prec,rnd));
		const std::size_t work=std::max(prec,z.precision())+64u;
		const auto hc=mini_mp::sinh_cosh(z.real_.rounded(work,
			FloatRnd::nearest).ldexp(1),work,FloatRnd::nearest);
		const auto sc=mini_mp::sin_cos(z.imag_.rounded(work,
			FloatRnd::nearest).ldexp(1),work,FloatRnd::nearest);
		const BigFloat den=BigFloat::add(hc.second,sc.second,work,
										 FloatRnd::nearest);
		return BigComplex(BigFloat::div(hc.first,den,work,FloatRnd::nearest),
						  BigFloat::div(sc.first,den,work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex asinh(const BigComplex&z,
							std::size_t prec=BigFloat::default_prec,
							FloatRnd rnd=FloatRnd::nearest){
		if(z.imag_.is_zero())
			return BigComplex(mini_mp::asinh(z.real_,prec,rnd),
							  BigFloat::zero(prec));
		if(z.real_.is_zero()&&z.imag_.is_finite()&&
		   mini_mp::cmpabs(z.imag_,BigFloat::one(z.precision()))<=0)
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::asin(z.imag_,prec,rnd));
		const std::size_t work=std::max(prec,z.precision())+80u;
		return div_i(asin(mul_i(z),work,FloatRnd::nearest)).
			rounded(prec,rnd);
	}

	static BigComplex acosh(const BigComplex&z,
							std::size_t prec=BigFloat::default_prec,
							FloatRnd rnd=FloatRnd::nearest){
		const std::size_t work=std::max(prec,z.precision())+80u;
		if(z.imag_.is_zero()&&z.real_.is_finite()){
			const BigFloat onev=BigFloat::one(work);
			if(compare(z.real_,onev)>=0)
				return BigComplex(mini_mp::acosh(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			if(compare(z.real_,onev.neg())<=0)
				return BigComplex(mini_mp::acosh(z.real_.abs(),work,
												FloatRnd::nearest),
								  mini_mp::const_pi(work,
													FloatRnd::nearest)).
					rounded(prec,rnd);
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::acos(z.real_,prec,rnd));
		}
		BigComplex zp=add(z.rounded(work,FloatRnd::nearest),one(work),
						  work,FloatRnd::nearest);
		BigComplex zm=sub(z.rounded(work,FloatRnd::nearest),one(work),
						  work,FloatRnd::nearest);
		BigComplex t=mul(sqrt(zp,work,FloatRnd::nearest),
						 sqrt(zm,work,FloatRnd::nearest),work,
						 FloatRnd::nearest);
		return log(add(z.rounded(work,FloatRnd::nearest),t,work,
					   FloatRnd::nearest),prec,rnd);
	}

	static BigComplex atanh(const BigComplex&z,
							std::size_t prec=BigFloat::default_prec,
							FloatRnd rnd=FloatRnd::nearest){
		if(z.real_.is_zero())
			return BigComplex(BigFloat::zero(prec),
							  mini_mp::atan(z.imag_,prec,rnd));
		const std::size_t work=std::max(prec,z.precision())+80u;
		if(z.imag_.is_zero()&&z.real_.is_finite()){
			const int c=mini_mp::cmpabs(z.real_,BigFloat::one(work));
			if(c<0)
				return BigComplex(mini_mp::atanh(z.real_,prec,rnd),
								  BigFloat::zero(prec));
			if(c==0)
				return BigComplex(BigFloat::inf(z.real_.sign(),prec),
								  BigFloat::zero(prec));
			BigFloat re=mini_mp::atanh(BigFloat::recip(z.real_,work,
				FloatRnd::nearest),work,FloatRnd::nearest);
			BigFloat im=mini_mp::const_pi(work,FloatRnd::nearest).ldexp(-1);
			return BigComplex(re,im).rounded(prec,rnd);
		}
		BigComplex zp=add(one(work),z.rounded(work,FloatRnd::nearest),
						  work,FloatRnd::nearest);
		BigComplex zm=sub(one(work),z.rounded(work,FloatRnd::nearest),
						  work,FloatRnd::nearest);
		BigComplex l=log(div(zp,zm,work,FloatRnd::nearest),work,
						 FloatRnd::nearest);
		return mul_real(l,BigFloat::one(work).ldexp(-1),prec,rnd);
	}

	BigComplex&operator+=(const BigComplex&rhs){
		*this=add(*this,rhs,precision());
		return *this;
	}
	BigComplex&operator-=(const BigComplex&rhs){
		*this=sub(*this,rhs,precision());
		return *this;
	}
	BigComplex&operator*=(const BigComplex&rhs){
		*this=mul(*this,rhs,precision());
		return *this;
	}
	BigComplex&operator/=(const BigComplex&rhs){
		*this=div(*this,rhs,precision());
		return *this;
	}

	friend BigComplex operator+(const BigComplex&a,const BigComplex&b){
		return add(a,b,std::max(a.precision(),b.precision()));
	}
	friend BigComplex operator-(const BigComplex&a,const BigComplex&b){
		return sub(a,b,std::max(a.precision(),b.precision()));
	}
	friend BigComplex operator*(const BigComplex&a,const BigComplex&b){
		return mul(a,b,std::max(a.precision(),b.precision()));
	}
	friend BigComplex operator/(const BigComplex&a,const BigComplex&b){
		return div(a,b,std::max(a.precision(),b.precision()));
	}
	friend BigComplex operator-(const BigComplex&z){ return z.neg(); }

	friend bool operator==(const BigComplex&a,const BigComplex&b){
		return a.real_==b.real_&&a.imag_==b.imag_;
	}
	friend bool operator!=(const BigComplex&a,const BigComplex&b){
		return !(a==b);
	}

  private:
	static std::uint64_t abs_i64(std::int64_t v) noexcept{
		return v<0?
			static_cast<std::uint64_t>(-(v+1))+std::uint64_t(1):
			static_cast<std::uint64_t>(v);
	}

	static BigComplex zero(std::size_t prec){
		return BigComplex(BigFloat::zero(prec),BigFloat::zero(prec));
	}

	static BigComplex one(std::size_t prec){
		return BigComplex(BigFloat::one(prec),BigFloat::zero(prec));
	}

	static BigComplex nan(std::size_t prec){
		return BigComplex(BigFloat::nan(prec),BigFloat::nan(prec));
	}

	static BigComplex mul_i(const BigComplex&z){
		return BigComplex(z.imag_.neg(),z.real_);
	}

	static BigComplex div_i(const BigComplex&z){
		return BigComplex(z.imag_,z.real_.neg());
	}

	BigFloat real_;
	BigFloat imag_;
};

inline BigFloat abs(const BigComplex&z,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return z.abs(prec,rnd);
}
inline BigComplex conj(const BigComplex&z){ return z.conj(); }
inline BigComplex exp(const BigComplex&z,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::exp(z,prec,rnd);
}
inline BigFloat arg(const BigComplex&z,
					std::size_t prec=BigFloat::default_prec,
					FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::arg(z,prec,rnd);
}
inline BigComplex polar(const BigFloat&r,const BigFloat&theta,
						std::size_t prec=BigFloat::default_prec,
						FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::polar(r,theta,prec,rnd);
}
inline BigComplex log(const BigComplex&z,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::log(z,prec,rnd);
}
inline BigComplex sqrt(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::sqrt(z,prec,rnd);
}
inline BigComplex pow_ui(const BigComplex&z,std::uint64_t n,
						 std::size_t prec=BigFloat::default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::pow_ui(z,n,prec,rnd);
}
inline BigComplex pow_si(const BigComplex&z,std::int64_t n,
						 std::size_t prec=BigFloat::default_prec,
						 FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::pow_si(z,n,prec,rnd);
}
inline BigComplex pow(const BigComplex&x,const BigComplex&y,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::pow(x,y,prec,rnd);
}
inline std::pair<BigComplex,BigComplex> sin_cos(
	const BigComplex&z,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::sin_cos(z,prec,rnd);
}
inline BigComplex sin(const BigComplex&z,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::sin(z,prec,rnd);
}
inline BigComplex cos(const BigComplex&z,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::cos(z,prec,rnd);
}
inline BigComplex tan(const BigComplex&z,
					  std::size_t prec=BigFloat::default_prec,
					  FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::tan(z,prec,rnd);
}
inline BigComplex asin(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::asin(z,prec,rnd);
}
inline BigComplex acos(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::acos(z,prec,rnd);
}
inline BigComplex atan(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::atan(z,prec,rnd);
}
inline std::pair<BigComplex,BigComplex> sinh_cosh(
	const BigComplex&z,std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::sinh_cosh(z,prec,rnd);
}
inline BigComplex sinh(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::sinh(z,prec,rnd);
}
inline BigComplex cosh(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::cosh(z,prec,rnd);
}
inline BigComplex tanh(const BigComplex&z,
					   std::size_t prec=BigFloat::default_prec,
					   FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::tanh(z,prec,rnd);
}
inline BigComplex asinh(const BigComplex&z,
						std::size_t prec=BigFloat::default_prec,
						FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::asinh(z,prec,rnd);
}
inline BigComplex acosh(const BigComplex&z,
						std::size_t prec=BigFloat::default_prec,
						FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::acosh(z,prec,rnd);
}
inline BigComplex atanh(const BigComplex&z,
						std::size_t prec=BigFloat::default_prec,
						FloatRnd rnd=FloatRnd::nearest){
	return BigComplex::atanh(z,prec,rnd);
}

namespace BigNT{

enum class PolyMulAlg{
	auto_select,
	schoolbook,
	karatsuba,
	ntt,
};

struct Factor{
	BigInt prime;
	std::uint32_t exp=0;
};

inline BigInt mod_pos(const BigInt&a,const BigInt&m){
	if(m.sign()<=0)
		detail::throw_dom("BigNT::mod_pos: non-positive modulus");
	BigInt r=a%m;
	if(r.sign()<0)
		r+=m;
	return r;
}

inline BigInt abs_sub(const BigInt&a,const BigInt&b){
	return a>=b?a-b:b-a;
}

namespace detail_nt{

inline std::uint64_t mod_u64(detail::u128 x,std::uint64_t mod) noexcept{
	std::uint64_t r=0;
	(void)detail::udiv128(x.hi,x.lo,mod,&r);
	return r;
}

inline std::uint64_t add_mod(std::uint64_t a,std::uint64_t b,
							 std::uint64_t mod) noexcept{
	std::uint64_t lo=0;
	const std::uint64_t hi=detail::addc_u64(0,a,b,&lo);
	if(hi==0){
		if(lo>=mod)
			lo-=mod;
		return lo;
	}
	return lo-mod;
}

inline std::uint64_t sub_mod(std::uint64_t a,std::uint64_t b,
							 std::uint64_t mod) noexcept{
	if(a>=b)
		return a-b;
	return mod-(b-a);
}

inline std::uint64_t neg_mod(std::uint64_t a,std::uint64_t mod) noexcept{
	return a==0?0:mod-a;
}

inline std::uint64_t mul_mod(std::uint64_t a,std::uint64_t b,
							 std::uint64_t mod) noexcept{
	return detail::mulmod64(a,b,mod);
}

inline std::uint64_t primitive_root(std::uint64_t p);

struct NttRootCacheEntry{
	std::uint64_t mod=0;
	std::uint64_t root=0;
};

inline std::uint64_t primitive_root_cached(std::uint64_t mod){
	static std::mutex mtx;
	static std::vector<NttRootCacheEntry> cache;
	{
		std::lock_guard<std::mutex> lock(mtx);
		for(const auto&e : cache){
			if(e.mod==mod)
				return e.root;
		}
	}
	const std::uint64_t root=primitive_root(mod);
	{
		std::lock_guard<std::mutex> lock(mtx);
		cache.push_back({mod,root});
	}
	return root;
}

inline void trim(std::vector<std::uint64_t>&a){
	while(!a.empty()&&a.back()==0)
		a.pop_back();
}

inline std::vector<std::uint64_t> add_vec(const std::vector<std::uint64_t>&a,
										  const std::vector<std::uint64_t>&b,
										  std::uint64_t mod){
	std::vector<std::uint64_t> out(std::max(a.size(),b.size()),0);
	for(std::size_t i=0;i<out.size();++i){
		const std::uint64_t av=i<a.size()?a[i]:0;
		const std::uint64_t bv=i<b.size()?b[i]:0;
		out[i]=add_mod(av,bv,mod);
	}
	trim(out);
	return out;
}

inline std::vector<std::uint64_t> sub_vec(const std::vector<std::uint64_t>&a,
										  const std::vector<std::uint64_t>&b,
										  std::uint64_t mod){
	std::vector<std::uint64_t> out(std::max(a.size(),b.size()),0);
	for(std::size_t i=0;i<out.size();++i){
		const std::uint64_t av=i<a.size()?a[i]:0;
		const std::uint64_t bv=i<b.size()?b[i]:0;
		out[i]=sub_mod(av,bv,mod);
	}
	trim(out);
	return out;
}

inline void add_shift(std::vector<std::uint64_t>&out,
					  const std::vector<std::uint64_t>&src,
					  std::size_t shift,std::uint64_t mod){
	if(src.empty())
		return;
	if(out.size()<src.size()+shift)
		out.resize(src.size()+shift,0);
	for(std::size_t i=0;i<src.size();++i)
		out[i+shift]=add_mod(out[i+shift],src[i],mod);
}

inline std::vector<std::uint64_t> mul_school(
	const std::vector<std::uint64_t>&a,
	const std::vector<std::uint64_t>&b,std::uint64_t mod){
	if(a.empty()||b.empty())
		return {};
#if MINI_MP_DETAIL_HAS_UINT128
	if(mod<=std::numeric_limits<std::uint32_t>::max()){
		const std::vector<std::uint64_t>*bp=&b;
		std::vector<std::uint64_t> bnorm;
		for(std::uint64_t x : b){
			if(x>=mod){
				bnorm=b;
				for(std::uint64_t&y : bnorm)
					y%=mod;
				bp=&bnorm;
				break;
			}
		}
		std::vector<unsigned __int128> acc(a.size()+b.size()-1u,0);
		for(std::size_t i=0;i<a.size();++i){
			std::uint64_t ai64=a[i];
			if(ai64>=mod)
				ai64%=mod;
			if(ai64==0)
				continue;
			const unsigned __int128 ai=ai64;
			for(std::size_t j=0;j<bp->size();++j){
				const std::uint64_t bj=(*bp)[j];
				if(bj!=0)
					acc[i+j]+=ai*static_cast<unsigned __int128>(bj);
			}
		}
		std::vector<std::uint64_t> out(acc.size(),0);
		for(std::size_t i=0;i<acc.size();++i)
			out[i]=static_cast<std::uint64_t>(acc[i]%mod);
		trim(out);
		return out;
	}
#endif
	std::vector<std::uint64_t> out(a.size()+b.size()-1u,0);
	for(std::size_t i=0;i<a.size();++i){
		if(a[i]==0)
			continue;
		for(std::size_t j=0;j<b.size();++j){
			if(b[j]==0)
				continue;
			const std::uint64_t p=mul_mod(a[i],b[j],mod);
			out[i+j]=add_mod(out[i+j],p,mod);
		}
	}
	trim(out);
	return out;
}

inline std::vector<std::uint64_t> slice(const std::vector<std::uint64_t>&a,
										std::size_t lo,std::size_t hi){
	if(lo>=a.size()||lo>=hi)
		return {};
	hi=std::min(hi,a.size());
	return std::vector<std::uint64_t>(
		a.begin()+static_cast<std::ptrdiff_t>(lo),
		a.begin()+static_cast<std::ptrdiff_t>(hi));
}

inline std::vector<std::uint64_t> mul_kar(
	const std::vector<std::uint64_t>&a,
	const std::vector<std::uint64_t>&b,std::uint64_t mod){
	if(a.empty()||b.empty())
		return {};
	if(std::min(a.size(),b.size())<32u)
		return mul_school(a,b,mod);
	const std::size_t n=std::max(a.size(),b.size());
	const std::size_t m=n/2u;
	const auto a0=slice(a,0,m);
	const auto a1=slice(a,m,a.size());
	const auto b0=slice(b,0,m);
	const auto b1=slice(b,m,b.size());
	const auto z0=mul_kar(a0,b0,mod);
	const auto z2=mul_kar(a1,b1,mod);
	const auto s0=add_vec(a0,a1,mod);
	const auto s1=add_vec(b0,b1,mod);
	auto z1=mul_kar(s0,s1,mod);
	z1=sub_vec(sub_vec(z1,z0,mod),z2,mod);
	std::vector<std::uint64_t> out(a.size()+b.size()-1u,0);
	add_shift(out,z0,0,mod);
	add_shift(out,z1,m,mod);
	add_shift(out,z2,2u*m,mod);
	trim(out);
	return out;
}

inline bool is_prime_u64(std::uint64_t n){
	if(n<2)
		return false;
	constexpr std::array<std::uint64_t,12> small={
		2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u};
	for(std::uint64_t p : small){
		if(n==p)
			return true;
		if(n%p==0)
			return false;
	}
	std::uint64_t d=n-1u;
	unsigned s=0;
	while((d&1u)==0u){
		d>>=1;
		++s;
	}
	constexpr std::array<std::uint64_t,7> wit={
		2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL};
	for(std::uint64_t a : wit){
		if(a%n==0)
			continue;
		std::uint64_t x=detail::powmod64(a%n,d,n);
		if(x==1u||x==n-1u)
			continue;
		bool comp=true;
		for(unsigned r=1;r<s;++r){
			x=mul_mod(x,x,n);
			if(x==n-1u){
				comp=false;
				break;
			}
		}
		if(comp)
			return false;
	}
	return true;
}

inline std::vector<std::uint64_t> factor_u64(std::uint64_t n){
	std::vector<std::uint64_t> f;
	for(std::uint64_t p=2;p<=n/p;p=(p==2)?3u:p+2u){
		if(n%p!=0)
			continue;
		f.push_back(p);
		while(n%p==0)
			n/=p;
	}
	if(n>1)
		f.push_back(n);
	return f;
}

inline std::uint64_t primitive_root(std::uint64_t p){
	if(p==2)
		return 1;
	const std::uint64_t phi=p-1u;
	const auto fac=factor_u64(phi);
	for(std::uint64_t g=2;g<p;++g){
		bool ok=true;
		for(std::uint64_t q : fac){
			if(detail::powmod64(g,phi/q,p)==1u){
				ok=false;
				break;
			}
		}
		if(ok)
			return g;
	}
	return 0;
}

inline void ntt(std::vector<std::uint64_t>&a,bool invert,
				std::uint64_t mod,std::uint64_t root){
	const std::size_t n=a.size();
	for(std::size_t i=1,j=0;i<n;++i){
		std::size_t bit=n>>1u;
		for(;j&bit;bit>>=1u)
			j^=bit;
		j^=bit;
		if(i<j)
			std::swap(a[i],a[j]);
	}
	for(std::size_t len=2;len<=n;len<<=1u){
		std::uint64_t wlen=detail::powmod64(
			root,static_cast<std::uint64_t>(n/len),mod);
		if(invert)
			wlen=detail::powmod64(wlen,mod-2u,mod);
		for(std::size_t i=0;i<n;i+=len){
			std::uint64_t w=1;
			for(std::size_t j=0;j<len/2u;++j){
				const std::uint64_t u=a[i+j];
				const std::uint64_t v=mul_mod(a[i+j+len/2u],w,mod);
				a[i+j]=add_mod(u,v,mod);
				a[i+j+len/2u]=sub_mod(u,v,mod);
				w=mul_mod(w,wlen,mod);
			}
		}
	}
	if(invert){
		const std::uint64_t inv_n=detail::powmod64(
			static_cast<std::uint64_t>(n)%mod,mod-2u,mod);
		for(std::uint64_t&x : a)
			x=mul_mod(x,inv_n,mod);
	}
}

inline void ntt_mont(std::vector<std::uint64_t>&a,bool invert,
					 std::uint64_t mod,std::uint64_t root,
					 const detail::Mont64Ctx&mc){
	const std::size_t n=a.size();
	for(std::size_t i=1,j=0;i<n;++i){
		std::size_t bit=n>>1u;
		for(;j&bit;bit>>=1u)
			j^=bit;
		j^=bit;
		if(i<j)
			std::swap(a[i],a[j]);
	}
	const std::uint64_t root_inv=detail::powmod64(root,mod-2u,mod);
	for(std::size_t len=2;len<=n;len<<=1u){
		const std::uint64_t wn=detail::powmod64(
			invert?root_inv:root,static_cast<std::uint64_t>(n/len),mod);
		const std::uint64_t wlen=detail::mto_u64(wn,mc);
		for(std::size_t i=0;i<n;i+=len){
			std::uint64_t w=detail::mto_u64(1,mc);
			for(std::size_t j=0;j<len/2u;++j){
				const std::uint64_t u=a[i+j];
				const std::uint64_t v=detail::mmul_u64(a[i+j+len/2u],w,mc);
				a[i+j]=add_mod(u,v,mod);
				a[i+j+len/2u]=sub_mod(u,v,mod);
				w=detail::mmul_u64(w,wlen,mc);
			}
		}
	}
}

inline bool mul_ntt(std::vector<std::uint64_t>&out,
					const std::vector<std::uint64_t>&a,
					const std::vector<std::uint64_t>&b,std::uint64_t mod){
	if(a.empty()||b.empty()){
		out.clear();
		return true;
	}
	if(mod<3||!is_prime_u64(mod))
		return false;
	std::size_t n=1;
	const std::size_t need=a.size()+b.size()-1u;
	while(n<need)
		n<<=1u;
	if(((mod-1u)%static_cast<std::uint64_t>(n))!=0u)
		return false;
	const std::uint64_t g=primitive_root_cached(mod);
	if(g==0)
		return false;
	const std::uint64_t root=detail::powmod64(
		g,(mod-1u)/static_cast<std::uint64_t>(n),mod);
	const detail::Mont64Ctx mc=detail::mk_m64ctx(mod);
	std::vector<std::uint64_t> fa(a.begin(),a.end());
	std::vector<std::uint64_t> fb(b.begin(),b.end());
	fa.resize(n,0);
	fb.resize(n,0);
	for(std::uint64_t&x : fa)
		x=detail::mto_u64(x,mc);
	for(std::uint64_t&x : fb)
		x=detail::mto_u64(x,mc);
	ntt_mont(fa,false,mod,root,mc);
	ntt_mont(fb,false,mod,root,mc);
	for(std::size_t i=0;i<n;++i)
		fa[i]=detail::mmul_u64(fa[i],fb[i],mc);
	ntt_mont(fa,true,mod,root,mc);
	const std::uint64_t inv_n=detail::mto_u64(detail::powmod64(
		static_cast<std::uint64_t>(n)%mod,mod-2u,mod),mc);
	for(std::uint64_t&x : fa)
		x=detail::mfr_u64(detail::mmul_u64(x,inv_n,mc),mc);
	fa.resize(need);
	trim(fa);
	out=std::move(fa);
	return true;
}

inline std::uint64_t inv_mod_prime(std::uint64_t a,std::uint64_t mod){
	a%=mod;
	if(a==0)
		detail::throw_dom("BigNT::ModPoly: non-invertible leading term");
	std::uint64_t out=0;
	if(!detail::invmod64(&out,a,mod))
		detail::throw_dom("BigNT::ModPoly: non-invertible leading term");
	return out;
}

inline void make_monic_vec(std::vector<std::uint64_t>&f,std::uint64_t mod){
	trim(f);
	if(f.empty())
		return;
	const std::uint64_t lc=f.back();
	if(lc==1u)
		return;
	const std::uint64_t inv=inv_mod_prime(lc,mod);
	for(std::uint64_t&x : f)
		x=mul_mod(x,inv,mod);
	trim(f);
}

inline std::vector<std::uint64_t> rem_monic_deg3_vec(
	std::vector<std::uint64_t> r,const std::vector<std::uint64_t>&f,
	std::uint64_t mod){
	trim(r);
	if(r.size()<4u)
		return r;
	for(std::size_t k=r.size();k>3u;--k){
		const std::uint64_t lead=r[k-1u];
		if(lead==0)
			continue;
		const std::size_t sh=k-4u;
		r[sh]=sub_mod(r[sh],mul_mod(lead,f[0],mod),mod);
		r[sh+1u]=sub_mod(r[sh+1u],mul_mod(lead,f[1],mod),mod);
		r[sh+2u]=sub_mod(r[sh+2u],mul_mod(lead,f[2],mod),mod);
		r[k-1u]=0;
	}
	r.resize(3u);
	trim(r);
	return r;
}

inline std::vector<std::uint64_t> mul_rem_monic_deg3_vec(
	const std::vector<std::uint64_t>&a,
	const std::vector<std::uint64_t>&b,
	const std::vector<std::uint64_t>&f,std::uint64_t mod){
	if(a.empty()||b.empty())
		return {};
	std::array<std::uint64_t,5> r={0,0,0,0,0};
	for(std::size_t i=0;i<a.size();++i){
		const std::uint64_t ai=a[i];
		if(ai==0)
			continue;
		for(std::size_t j=0;j<b.size();++j){
			if(b[j]!=0)
				r[i+j]=add_mod(r[i+j],mul_mod(ai,b[j],mod),mod);
		}
	}
	for(std::size_t k=5u;k>3u;--k){
		const std::uint64_t lead=r[k-1u];
		if(lead==0)
			continue;
		const std::size_t sh=k-4u;
		r[sh]=sub_mod(r[sh],mul_mod(lead,f[0],mod),mod);
		r[sh+1u]=sub_mod(r[sh+1u],mul_mod(lead,f[1],mod),mod);
		r[sh+2u]=sub_mod(r[sh+2u],mul_mod(lead,f[2],mod),mod);
	}
	std::vector<std::uint64_t> out;
	if(r[2]!=0)
		out={r[0],r[1],r[2]};
	else if(r[1]!=0)
		out={r[0],r[1]};
	else if(r[0]!=0)
		out={r[0]};
	return out;
}

inline std::array<std::uint64_t,3> to_deg3_array(
	const std::vector<std::uint64_t>&v) noexcept{
	return {v.size()>0u?v[0]:0u,v.size()>1u?v[1]:0u,
			v.size()>2u?v[2]:0u};
}

inline std::vector<std::uint64_t> from_deg3_array(
	const std::array<std::uint64_t,3>&v){
	if(v[2]!=0)
		return {v[0],v[1],v[2]};
	if(v[1]!=0)
		return {v[0],v[1]};
	if(v[0]!=0)
		return {v[0]};
	return {};
}

inline std::array<std::uint64_t,3> mul_rem_monic_deg3_arr(
	const std::array<std::uint64_t,3>&a,
	const std::array<std::uint64_t,3>&b,
	const std::vector<std::uint64_t>&f,std::uint64_t mod){
	std::array<std::uint64_t,5> r={0,0,0,0,0};
	for(std::size_t i=0;i<3u;++i){
		if(a[i]==0)
			continue;
		for(std::size_t j=0;j<3u;++j){
			if(b[j]!=0)
				r[i+j]=add_mod(r[i+j],mul_mod(a[i],b[j],mod),mod);
		}
	}
	for(std::size_t k=5u;k>3u;--k){
		const std::uint64_t lead=r[k-1u];
		if(lead==0)
			continue;
		const std::size_t sh=k-4u;
		r[sh]=sub_mod(r[sh],mul_mod(lead,f[0],mod),mod);
		r[sh+1u]=sub_mod(r[sh+1u],mul_mod(lead,f[1],mod),mod);
		r[sh+2u]=sub_mod(r[sh+2u],mul_mod(lead,f[2],mod),mod);
	}
	return {r[0],r[1],r[2]};
}

inline std::vector<std::uint64_t> rem_vec(std::vector<std::uint64_t> r,
										  const std::vector<std::uint64_t>&f,
										  std::uint64_t mod){
	trim(r);
	if(f.empty())
		detail::throw_dom("BigNT::ModPoly::rem: division by zero");
	if(r.size()<f.size())
		return r;
	if(f.size()==1u)
		return {};
	if(f.size()==4u&&f.back()==1u)
		return rem_monic_deg3_vec(std::move(r),f,mod);
	const bool monic=f.back()==1u;
	const std::uint64_t inv_lc=monic?1u:inv_mod_prime(f.back(),mod);
	const std::size_t deg=f.size()-1u;
	for(std::size_t k=r.size();k>=f.size();--k){
		const std::size_t sh=k-f.size();
		const std::uint64_t lead=r[k-1u];
		if(lead!=0){
			const std::uint64_t c=monic?lead:mul_mod(lead,inv_lc,mod);
			for(std::size_t i=0;i<deg;++i){
				const std::uint64_t t=mul_mod(c,f[i],mod);
				r[sh+i]=sub_mod(r[sh+i],t,mod);
			}
			r[k-1u]=0;
		}
	}
	r.resize(deg);
	trim(r);
	return r;
}

inline std::vector<std::uint64_t> gcd_vec(std::vector<std::uint64_t> a,
										  std::vector<std::uint64_t> b,
										  std::uint64_t mod){
	trim(a);
	trim(b);
	if(a.empty()){
		make_monic_vec(b,mod);
		return b;
	}
	if(b.empty()){
		make_monic_vec(a,mod);
		return a;
	}
	if(a.size()<b.size())
		a.swap(b);
	for(;;){
		if(b.size()==1u)
			return {1u};
		std::vector<std::uint64_t> r=rem_vec(std::move(a),b,mod);
		if(r.empty()){
			make_monic_vec(b,mod);
			return b;
		}
		if(r.size()==1u)
			return {1u};
		a=std::move(b);
		b=std::move(r);
	}
}

inline std::vector<std::uint64_t> mul_rem_vec(
	const std::vector<std::uint64_t>&a,
	const std::vector<std::uint64_t>&b,
	const std::vector<std::uint64_t>&f,
	std::uint64_t mod,PolyMulAlg alg){
	if(f.size()==4u&&f.back()==1u&&a.size()<=3u&&b.size()<=3u)
		return mul_rem_monic_deg3_vec(a,b,f,mod);
	std::vector<std::uint64_t> prod;
	bool have_prod=false;
	if(alg==PolyMulAlg::ntt||
	   (alg==PolyMulAlg::auto_select&&std::min(a.size(),b.size())>=256u)){
		have_prod=mul_ntt(prod,a,b,mod);
		if(!have_prod&&alg==PolyMulAlg::ntt)
				detail::throw_dom("BigNT::ModPoly::mul_ntt: unsupported modulus");
	}
	if(!have_prod&&!(a.empty()||b.empty())){
		if(alg==PolyMulAlg::schoolbook||std::min(a.size(),b.size())<96u)
			prod=mul_school(a,b,mod);
		else
			prod=mul_kar(a,b,mod);
	}
	return rem_vec(std::move(prod),f,mod);
}

inline std::vector<std::uint32_t> small_primes(std::uint32_t limit){
	std::vector<std::uint32_t> primes;
	if(limit<2)
		return primes;
	std::vector<unsigned char> comp(static_cast<std::size_t>(limit)+1u,0);
	for(std::uint32_t i=2;i<=limit;++i){
		if(comp[i])
			continue;
		primes.push_back(i);
		if(i>limit/i)
			continue;
		for(std::uint32_t j=i*i;j<=limit;j+=i)
			comp[j]=1;
	}
	return primes;
}

inline const std::vector<std::uint32_t>&trial_primes(){
	static const std::vector<std::uint32_t> primes=small_primes(1000u);
	return primes;
}

inline std::uint64_t pollard_rho_u64(std::uint64_t n,std::uint64_t seed,
									 std::uint64_t c_in,std::uint64_t batch,
									 std::uint64_t max_iters){
	if((n&1u)==0u)
		return 2;
	if(is_prime_u64(n))
		return n;
	if(batch==0)
		batch=1;
	std::uint64_t y=seed%n;
	if(y==0)
		y=2;
	std::uint64_t c=c_in%n;
	if(c==0)
		c=1;
	auto f=[&](std::uint64_t x){
		return add_mod(mul_mod(x,x,n),c,n);
	};
	std::uint64_t g=1,q=1,x=0,ys=y,r=1,iter=0;
	while(g==1&&iter<max_iters){
		x=y;
		for(std::uint64_t i=0;i<r;++i)
			y=f(y);
		std::uint64_t k=0;
		while(k<r&&g==1){
			ys=y;
			const std::uint64_t lim=std::min(batch,r-k);
			for(std::uint64_t i=0;i<lim;++i){
				y=f(y);
				const std::uint64_t d=x>y?x-y:y-x;
				q=mul_mod(q,d,n);
			}
			g=std::gcd(q,n);
			k+=lim;
			iter+=lim;
		}
		r<<=1u;
	}
	if(g==n){
		do{
			ys=f(ys);
			const std::uint64_t d=x>ys?x-ys:ys-x;
			g=std::gcd(d,n);
		}while(g==1);
	}
	return (g==1||g==n)?0:g;
}

inline std::uint64_t pollard_pm1_u64(std::uint64_t n,std::uint32_t bound,
									 std::uint64_t a){
	if((n&1u)==0u)
		return 2;
	if(is_prime_u64(n))
		return n;
	a%=n;
	if(a<2)
		a=2;
	std::uint32_t steps=0;
	for(std::uint32_t p : small_primes(bound)){
		std::uint64_t pp=p;
		while(pp<=static_cast<std::uint64_t>(bound)/p)
			pp*=p;
		a=detail::powmod64(a,pp,n);
		if((++steps&63u)==0u){
			const std::uint64_t g=std::gcd(a==0?n-1u:a-1u,n);
			if(g>1&&g<n)
				return g;
			if(g==n)
				return 0;
		}
	}
	const std::uint64_t g=std::gcd(a==0?n-1u:a-1u,n);
	return (g>1&&g<n)?g:0;
}

inline void factor_u64_rec(std::uint64_t n,std::vector<std::uint64_t>&out){
	if(n==1)
		return;
	if(is_prime_u64(n)){
		out.push_back(n);
		return;
	}
	for(std::uint32_t p : trial_primes()){
		if(n%p!=0)
			continue;
		do{
			out.push_back(p);
			n/=p;
		}while(n%p==0);
		factor_u64_rec(n,out);
		return;
	}
	std::uint64_t d=pollard_pm1_u64(n,10000u,2u);
	for(std::uint64_t s=2;(d==0||d==n)&&s<128u;++s)
		d=pollard_rho_u64(n,s,2u*s+1u,128u,1ULL<<20);
	if(d==0||d==n){
		out.push_back(n);
		return;
	}
	factor_u64_rec(d,out);
	factor_u64_rec(n/d,out);
}

inline BigInt u64(std::uint64_t x){ return BigInt::from_u64(x); }

inline BigInt sigma_power(std::uint64_t n,std::uint32_t k){
	BigInt s;
	for(std::uint64_t d=1;d<=n/d;++d){
		if(n%d!=0)
			continue;
		s+=pow(u64(d),k);
		const std::uint64_t e=n/d;
		if(e!=d)
			s+=pow(u64(e),k);
	}
	return s;
}

inline std::vector<BigInt> sigma_powers(std::size_t limit,std::uint32_t k){
	std::vector<BigInt> out(limit+1u);
	for(std::size_t d=1;d<=limit;++d){
		const BigInt term=pow(u64(static_cast<std::uint64_t>(d)),k);
		for(std::size_t m=d;m<=limit;m+=d)
			out[m]+=term;
	}
	return out;
}

inline std::vector<BigInt> mul_trunc(const std::vector<BigInt>&a,
									 const std::vector<BigInt>&b,
									 std::size_t n){
	if(a.empty()||b.empty()||n==0)
		return {};
	std::vector<BigInt> out(std::min(n,a.size()+b.size()-1u));
	for(std::size_t i=0;i<a.size();++i){
		if(a[i].is_zero())
			continue;
		for(std::size_t j=0;j<b.size()&&i+j<out.size();++j){
			if(!b[j].is_zero())
				out[i+j]+=a[i]*b[j];
		}
	}
	return out;
}

} 

class ModPoly{
  public:
	ModPoly()=default;
	explicit ModPoly(std::uint64_t mod) : mod_(mod){ check_mod(); }
	ModPoly(std::vector<std::uint64_t> coeff,std::uint64_t mod) :
		coeff_(std::move(coeff)),mod_(mod){
		check_mod();
		for(std::uint64_t&x : coeff_)
			x%=mod_;
		normalize();
	}
	ModPoly(std::initializer_list<std::uint64_t> coeff,std::uint64_t mod) :
		ModPoly(std::vector<std::uint64_t>(coeff),mod){}

	const std::vector<std::uint64_t>&coeff() const noexcept{ return coeff_; }
	std::vector<std::uint64_t>&coeff() noexcept{ return coeff_; }
	std::uint64_t modulus() const noexcept{ return mod_; }
	bool is_zero() const noexcept{ return coeff_.empty(); }
	std::size_t size() const noexcept{ return coeff_.size(); }
	std::size_t degree() const noexcept{
		return coeff_.empty()?BigInt::npos:coeff_.size()-1u;
	}

	std::uint64_t operator[](std::size_t i) const noexcept{
		return i<coeff_.size()?coeff_[i]:0;
	}

	void normalize(){ detail_nt::trim(coeff_); }

	static ModPoly x(std::uint64_t mod){ return ModPoly({0,1},mod); }
	static ModPoly one(std::uint64_t mod){ return ModPoly({1},mod); }

	static ModPoly add(const ModPoly&a,const ModPoly&b){
		check_same(a,b);
		return ModPoly(detail_nt::add_vec(a.coeff_,b.coeff_,a.mod_),a.mod_);
	}

	static ModPoly sub(const ModPoly&a,const ModPoly&b){
		check_same(a,b);
		return ModPoly(detail_nt::sub_vec(a.coeff_,b.coeff_,a.mod_),a.mod_);
	}

	static ModPoly mul(const ModPoly&a,const ModPoly&b,
					   PolyMulAlg alg=PolyMulAlg::auto_select){
		check_same(a,b);
		std::vector<std::uint64_t> out;
		if(alg==PolyMulAlg::ntt||
		   (alg==PolyMulAlg::auto_select&&
			std::min(a.size(),b.size())>=256u)){
			if(detail_nt::mul_ntt(out,a.coeff_,b.coeff_,a.mod_))
				return ModPoly(std::move(out),a.mod_);
			if(alg==PolyMulAlg::ntt)
				detail::throw_dom("BigNT::ModPoly::mul_ntt: unsupported modulus");
		}
		if(alg==PolyMulAlg::schoolbook||
		   std::min(a.size(),b.size())<96u)
			out=detail_nt::mul_school(a.coeff_,b.coeff_,a.mod_);
		else
			out=detail_nt::mul_kar(a.coeff_,b.coeff_,a.mod_);
		return ModPoly(std::move(out),a.mod_);
	}

	static std::pair<ModPoly,ModPoly> divrem(const ModPoly&a,const ModPoly&b){
		check_same(a,b);
		if(b.is_zero())
			detail::throw_dom("BigNT::ModPoly::divrem: division by zero");
		ModPoly r=a;
		ModPoly q(a.mod_);
		if(r.size()<b.size())
			return {q,r};
		q.coeff_.assign(r.size()-b.size()+1u,0);
		const std::uint64_t inv_lc=
			detail_nt::inv_mod_prime(b.coeff_.back(),a.mod_);
		for(std::size_t k=r.size();k>=b.size();--k){
			const std::size_t sh=k-b.size();
			const std::uint64_t lead=r.coeff_[k-1u];
			if(lead==0)
				continue;
			const std::uint64_t c=detail_nt::mul_mod(lead,inv_lc,a.mod_);
			q.coeff_[sh]=c;
			for(std::size_t i=0;i<b.size();++i){
				const std::uint64_t t=detail_nt::mul_mod(c,b.coeff_[i],a.mod_);
				r.coeff_[i+sh]=detail_nt::sub_mod(r.coeff_[i+sh],t,a.mod_);
			}
		}
		r.coeff_.resize(b.size()-1u);
		r.normalize();
		q.normalize();
		return {q,r};
	}

	ModPoly rem(const ModPoly&mod_poly) const{
		check_same(*this,mod_poly);
		return ModPoly(detail_nt::rem_vec(coeff_,mod_poly.coeff_,mod_),
					   mod_);
	}

	static ModPoly gcd(const ModPoly&a,const ModPoly&b){
		check_same(a,b);
		return ModPoly(detail_nt::gcd_vec(a.coeff_,b.coeff_,a.mod_),a.mod_);
	}

	ModPoly pow_mod(BigInt exp,const ModPoly&mod_poly,
					PolyMulAlg alg=PolyMulAlg::auto_select) const{
		check_same(*this,mod_poly);
		if(exp.sign()<0)
			detail::throw_dom("BigNT::ModPoly::pow_mod: negative exponent");
		ModPoly base=rem(mod_poly);
		if(exp.is_zero())
			return one(mod_).rem(mod_poly);
		if(exp.is_one())
			return base;
		if(mod_poly.size()==2u){
			const std::uint64_t inv_lc=
				detail_nt::inv_mod_prime(mod_poly.coeff_[1],mod_);
			const std::uint64_t xval=detail_nt::neg_mod(
				detail_nt::mul_mod(mod_poly.coeff_[0],inv_lc,mod_),mod_);
			const std::uint64_t val=exp.fits_u64()?
				detail::powmod64(xval,exp.to_u64(),mod_):
				modpow(BigInt::from_u64(xval),exp,
					   BigInt::from_u64(mod_)).to_u64();
			return ModPoly({val},mod_);
		}
		if(mod_poly.size()==4u&&mod_poly.coeff_[3]==1u){
			std::array<std::uint64_t,3> out{1u,0,0};
			const std::array<std::uint64_t,3> base3=
				detail_nt::to_deg3_array(base.coeff_);
			if(exp.fits_u64()){
				const std::uint64_t e=exp.to_u64();
				const unsigned bits=64u-std::countl_zero(e);
				for(unsigned ii=bits;ii>0;--ii){
					const unsigned i=ii-1u;
					out=detail_nt::mul_rem_monic_deg3_arr(
						out,out,mod_poly.coeff_,mod_);
					if(((e>>i)&1u)!=0u)
						out=detail_nt::mul_rem_monic_deg3_arr(
							out,base3,mod_poly.coeff_,mod_);
				}
			}else{
				const std::size_t bits=exp.bit_length();
				for(std::size_t ii=bits;ii>0;--ii){
					const std::size_t i=ii-1u;
					out=detail_nt::mul_rem_monic_deg3_arr(
						out,out,mod_poly.coeff_,mod_);
					if(exp.test_bit(i))
						out=detail_nt::mul_rem_monic_deg3_arr(
							out,base3,mod_poly.coeff_,mod_);
				}
			}
			return ModPoly(detail_nt::from_deg3_array(out),mod_);
		}
		std::vector<std::uint64_t> out{1u};
		if(exp.fits_u64()){
			const std::uint64_t e=exp.to_u64();
			const unsigned bits=64u-std::countl_zero(e);
			for(unsigned ii=bits;ii>0;--ii){
				const unsigned i=ii-1u;
				out=detail_nt::mul_rem_vec(out,out,mod_poly.coeff_,mod_,alg);
				if(((e>>i)&1u)!=0u)
					out=detail_nt::mul_rem_vec(out,base.coeff_,
											   mod_poly.coeff_,mod_,alg);
			}
		}else{
			const std::size_t bits=exp.bit_length();
			for(std::size_t ii=bits;ii>0;--ii){
				const std::size_t i=ii-1u;
				out=detail_nt::mul_rem_vec(out,out,mod_poly.coeff_,mod_,alg);
				if(exp.test_bit(i))
					out=detail_nt::mul_rem_vec(out,base.coeff_,
											   mod_poly.coeff_,mod_,alg);
			}
		}
		return ModPoly(std::move(out),mod_);
	}

	static ModPoly x_pow_mod(const BigInt&exp,const ModPoly&mod_poly,
							 PolyMulAlg alg=PolyMulAlg::auto_select){
		return x(mod_poly.mod_).pow_mod(exp,mod_poly,alg);
	}

	ModPoly&operator+=(const ModPoly&rhs){
		*this=add(*this,rhs);
		return *this;
	}
	ModPoly&operator-=(const ModPoly&rhs){
		*this=sub(*this,rhs);
		return *this;
	}
	ModPoly&operator*=(const ModPoly&rhs){
		*this=mul(*this,rhs);
		return *this;
	}

	friend ModPoly operator+(const ModPoly&a,const ModPoly&b){
		return add(a,b);
	}
	friend ModPoly operator-(const ModPoly&a,const ModPoly&b){
		return sub(a,b);
	}
	friend ModPoly operator*(const ModPoly&a,const ModPoly&b){
		return mul(a,b);
	}
	friend bool operator==(const ModPoly&a,const ModPoly&b){
		return a.mod_==b.mod_&&a.coeff_==b.coeff_;
	}
	friend bool operator!=(const ModPoly&a,const ModPoly&b){
		return !(a==b);
	}

  private:
	void check_mod() const{
		if(mod_<2)
			detail::throw_dom("BigNT::ModPoly: modulus must be >= 2");
	}

	static void check_same(const ModPoly&a,const ModPoly&b){
		if(a.mod_!=b.mod_)
			detail::throw_dom("BigNT::ModPoly: modulus mismatch");
	}

	std::vector<std::uint64_t> coeff_;
	std::uint64_t mod_=2;
};

inline ModPoly poly_add(const ModPoly&a,const ModPoly&b){
	return ModPoly::add(a,b);
}
inline ModPoly poly_sub(const ModPoly&a,const ModPoly&b){
	return ModPoly::sub(a,b);
}
inline ModPoly poly_mul(const ModPoly&a,const ModPoly&b,
						PolyMulAlg alg=PolyMulAlg::auto_select){
	return ModPoly::mul(a,b,alg);
}
inline std::pair<ModPoly,ModPoly> poly_divrem(const ModPoly&a,
											  const ModPoly&b){
	return ModPoly::divrem(a,b);
}
inline ModPoly poly_rem(const ModPoly&a,const ModPoly&b){
	return ModPoly::divrem(a,b).second;
}
inline ModPoly poly_gcd(const ModPoly&a,const ModPoly&b){
	return ModPoly::gcd(a,b);
}
inline ModPoly poly_x_pow_mod(const BigInt&k,const ModPoly&f,
							  PolyMulAlg alg=PolyMulAlg::auto_select){
	return ModPoly::x_pow_mod(k,f,alg);
}

class BigIntPoly{
  public:
	BigIntPoly()=default;

	explicit BigIntPoly(BigInt mod) : mod_(std::move(mod)){
		check_mod();
	}

	BigIntPoly(std::vector<BigInt> coeff,BigInt mod) :
		coeff_(std::move(coeff)),mod_(std::move(mod)){
		check_mod();
		normalize();
	}

	BigIntPoly(std::initializer_list<BigInt> coeff,BigInt mod) :
		BigIntPoly(std::vector<BigInt>(coeff),std::move(mod)){}

	const std::vector<BigInt>&coeff() const noexcept{ return coeff_; }
	std::vector<BigInt>&coeff() noexcept{ return coeff_; }
	const BigInt&modulus() const noexcept{ return mod_; }
	bool is_zero() const noexcept{ return coeff_.empty(); }
	std::size_t size() const noexcept{ return coeff_.size(); }
	std::size_t degree() const noexcept{
		return coeff_.empty()?BigInt::npos:coeff_.size()-1u;
	}

	BigInt operator[](std::size_t i) const{
		return i<coeff_.size()?coeff_[i]:BigInt();
	}

	void normalize(){
		normalize_vec(coeff_,mod_);
	}

	static BigIntPoly x(const BigInt&mod){
		return BigIntPoly({BigInt(),BigInt(1)},mod);
	}

	static BigIntPoly one(const BigInt&mod){
		return BigIntPoly({BigInt(1)},mod);
	}

	static BigIntPoly add(const BigIntPoly&a,const BigIntPoly&b){
		check_same(a,b);
		if(a.use_modpoly_fast())
			return from_modpoly(ModPoly::add(a.to_modpoly(),b.to_modpoly()));
		std::vector<BigInt> out(std::max(a.size(),b.size()));
		for(std::size_t i=0;i<out.size();++i){
			const BigInt&av=i<a.size()?a.coeff_[i]:zero_ref();
			const BigInt&bv=i<b.size()?b.coeff_[i]:zero_ref();
			out[i]=add_mod_coeff(av,bv,a.mod_);
		}
		trim_vec(out);
		return BigIntPoly(std::move(out),a.mod_,normalized_tag{});
	}

	static BigIntPoly sub(const BigIntPoly&a,const BigIntPoly&b){
		check_same(a,b);
		if(a.use_modpoly_fast())
			return from_modpoly(ModPoly::sub(a.to_modpoly(),b.to_modpoly()));
		std::vector<BigInt> out(std::max(a.size(),b.size()));
		for(std::size_t i=0;i<out.size();++i){
			const BigInt&av=i<a.size()?a.coeff_[i]:zero_ref();
			const BigInt&bv=i<b.size()?b.coeff_[i]:zero_ref();
			out[i]=sub_mod_coeff(av,bv,a.mod_);
		}
		trim_vec(out);
		return BigIntPoly(std::move(out),a.mod_,normalized_tag{});
	}

	static BigIntPoly neg(const BigIntPoly&a){
		std::vector<BigInt> out(a.size());
		for(std::size_t i=0;i<a.size();++i)
			out[i]=neg_mod_coeff(a.coeff_[i],a.mod_);
		trim_vec(out);
		return BigIntPoly(std::move(out),a.mod_,normalized_tag{});
	}

	static BigIntPoly mul(const BigIntPoly&a,const BigIntPoly&b,
						  PolyMulAlg alg=PolyMulAlg::auto_select){
		check_same(a,b);
		if(a.is_zero()||b.is_zero())
			return BigIntPoly(a.mod_);
		if(a.use_modpoly_fast())
			return from_modpoly(ModPoly::mul(a.to_modpoly(),b.to_modpoly(),
											 alg));
		std::vector<BigInt> out=mul_raw_vec(a.coeff_,b.coeff_,alg);
		normalize_vec(out,a.mod_);
		return BigIntPoly(std::move(out),a.mod_,normalized_tag{});
	}

	static BigIntPoly scalar_mul(const BigIntPoly&a,const BigInt&c){
		if(a.is_zero())
			return BigIntPoly(a.mod_);
		const BigInt cm=mod_pos(c,a.mod_);
		if(cm.is_zero())
			return BigIntPoly(a.mod_);
		if(cm.is_one())
			return a;
		std::vector<BigInt> out(a.size());
		for(std::size_t i=0;i<a.size();++i)
			out[i]=mod_pos(a.coeff_[i]*cm,a.mod_);
		trim_vec(out);
		return BigIntPoly(std::move(out),a.mod_,normalized_tag{});
	}

	static std::pair<BigIntPoly,BigIntPoly> divrem(const BigIntPoly&a,
												   const BigIntPoly&b){
		check_same(a,b);
		if(b.is_zero())
			detail::throw_dom("BigNT::BigIntPoly::divrem: division by zero");
		if(a.use_modpoly_fast()){
			auto qr=ModPoly::divrem(a.to_modpoly(),b.to_modpoly());
			return {from_modpoly(qr.first),from_modpoly(qr.second)};
		}
		auto qr=divrem_vec(a.coeff_,b.coeff_,a.mod_);
		return {BigIntPoly(std::move(qr.first),a.mod_,normalized_tag{}),
				BigIntPoly(std::move(qr.second),a.mod_,normalized_tag{})};
	}

	BigIntPoly rem(const BigIntPoly&mod_poly) const{
		check_same(*this,mod_poly);
		if(mod_poly.is_zero())
			detail::throw_dom("BigNT::BigIntPoly::rem: division by zero");
		if(use_modpoly_fast())
			return from_modpoly(to_modpoly().rem(mod_poly.to_modpoly()));
		return BigIntPoly(rem_vec(coeff_,mod_poly.coeff_,mod_),mod_,
						  normalized_tag{});
	}

	static BigIntPoly gcd(const BigIntPoly&a,const BigIntPoly&b){
		check_same(a,b);
		if(a.use_modpoly_fast())
			return from_modpoly(ModPoly::gcd(a.to_modpoly(),b.to_modpoly()));
		if(a.is_zero())
			return make_monic(b);
		if(b.is_zero())
			return make_monic(a);
		std::vector<BigInt> x=a.coeff_;
		std::vector<BigInt> y=b.coeff_;
		if(x.size()<y.size())
			x.swap(y);
		while(!y.empty()){
			if(y.size()==1u){
				(void)invmod(y[0],a.mod_);
				return one(a.mod_);
			}
			std::vector<BigInt> r=rem_vec(std::move(x),y,a.mod_);
			if(r.empty())
				return make_monic(BigIntPoly(std::move(y),a.mod_,
											 normalized_tag{}));
			x=std::move(y);
			y=std::move(r);
		}
		return make_monic(BigIntPoly(std::move(x),a.mod_,normalized_tag{}));
	}

	static BigIntPoly make_monic(const BigIntPoly&a){
		if(a.is_zero())
			return a;
		const BigInt inv=invmod(a.coeff_.back(),a.mod_);
		return scalar_mul(a,inv);
	}

	BigIntPoly pow_mod(BigInt exp,const BigIntPoly&mod_poly,
					   PolyMulAlg alg=PolyMulAlg::auto_select) const{
		check_same(*this,mod_poly);
		if(exp.sign()<0)
			detail::throw_dom("BigNT::BigIntPoly::pow_mod: negative exponent");
		if(use_modpoly_fast())
			return from_modpoly(modpoly_pow_mod_generic(
				to_modpoly(),exp,mod_poly.to_modpoly(),alg));
		std::vector<BigInt> base=rem_vec(coeff_,mod_poly.coeff_,mod_);
		if(exp.is_zero())
			return one(mod_).rem(mod_poly);
		if(exp.is_one())
			return BigIntPoly(std::move(base),mod_,normalized_tag{});
		std::vector<BigInt> out{BigInt(1)};
		if(exp.fits_u64()){
			const std::uint64_t e=exp.to_u64();
			const unsigned bits=64u-std::countl_zero(e);
			for(unsigned ii=bits;ii>0;--ii){
				const unsigned i=ii-1u;
				out=mul_rem_vec(out,out,mod_poly.coeff_,mod_,alg);
				if(((e>>i)&1u)!=0u)
					out=mul_rem_vec(out,base,mod_poly.coeff_,mod_,alg);
			}
		}else{
			const std::size_t bits=exp.bit_length();
			for(std::size_t ii=bits;ii>0;--ii){
				const std::size_t i=ii-1u;
				out=mul_rem_vec(out,out,mod_poly.coeff_,mod_,alg);
				if(exp.test_bit(i))
					out=mul_rem_vec(out,base,mod_poly.coeff_,mod_,alg);
			}
		}
		return BigIntPoly(std::move(out),mod_,normalized_tag{});
	}

	static BigIntPoly x_pow_mod(const BigInt&exp,const BigIntPoly&mod_poly,
								PolyMulAlg alg=PolyMulAlg::auto_select){
		return x(mod_poly.mod_).pow_mod(exp,mod_poly,alg);
	}

	BigInt evaluate(const BigInt&x) const{
		BigInt xm=mod_pos(x,mod_);
		BigInt out;
		for(std::size_t i=coeff_.size();i>0;--i)
			out=mod_pos(out*xm+coeff_[i-1u],mod_);
		return out;
	}

	BigIntPoly&operator+=(const BigIntPoly&rhs){
		*this=add(*this,rhs);
		return *this;
	}
	BigIntPoly&operator-=(const BigIntPoly&rhs){
		*this=sub(*this,rhs);
		return *this;
	}
	BigIntPoly&operator*=(const BigIntPoly&rhs){
		*this=mul(*this,rhs);
		return *this;
	}

	friend BigIntPoly operator+(const BigIntPoly&a,const BigIntPoly&b){
		return add(a,b);
	}
	friend BigIntPoly operator-(const BigIntPoly&a,const BigIntPoly&b){
		return sub(a,b);
	}
	friend BigIntPoly operator-(const BigIntPoly&a){ return neg(a); }
	friend BigIntPoly operator*(const BigIntPoly&a,const BigIntPoly&b){
		return mul(a,b);
	}
	friend bool operator==(const BigIntPoly&a,const BigIntPoly&b){
		return a.mod_==b.mod_&&a.coeff_==b.coeff_;
	}
	friend bool operator!=(const BigIntPoly&a,const BigIntPoly&b){
		return !(a==b);
	}

  private:
	struct normalized_tag{};

	BigIntPoly(std::vector<BigInt> coeff,const BigInt&mod,normalized_tag) :
		coeff_(std::move(coeff)),mod_(mod){}

	static const BigInt&zero_ref(){
		static const BigInt z;
		return z;
	}

	void check_mod() const{
		if(mod_.sign()<=0||mod_.is_one())
			detail::throw_dom("BigNT::BigIntPoly: modulus must be >= 2");
	}

	static void check_same(const BigIntPoly&a,const BigIntPoly&b){
		if(a.mod_!=b.mod_)
			detail::throw_dom("BigNT::BigIntPoly: modulus mismatch");
	}

	bool use_modpoly_fast() const noexcept{
		return mod_.fits_u64();
	}

	ModPoly to_modpoly() const{
		const std::uint64_t m=mod_.to_u64();
		std::vector<std::uint64_t> c;
		c.reserve(coeff_.size());
		for(const BigInt&x : coeff_)
			c.push_back(x.to_u64());
		return ModPoly(std::move(c),m);
	}

	static BigIntPoly from_modpoly(const ModPoly&p){
		std::vector<BigInt> c;
		c.reserve(p.coeff().size());
		for(std::uint64_t x : p.coeff())
			c.push_back(BigInt::from_u64(x));
		return BigIntPoly(std::move(c),BigInt::from_u64(p.modulus()),
						  normalized_tag{});
	}

	static void trim_vec(std::vector<BigInt>&v){
		while(!v.empty()&&v.back().is_zero())
			v.pop_back();
	}

	static void normalize_vec(std::vector<BigInt>&v,const BigInt&mod){
		for(BigInt&x : v){
			if(!x.is_zero())
				x=mod_pos(std::move(x),mod);
		}
		trim_vec(v);
	}

	static BigInt add_mod_coeff(const BigInt&a,const BigInt&b,
								const BigInt&mod){
		if(a.is_zero())
			return b;
		if(b.is_zero())
			return a;
		BigInt s=a+b;
		if(s>=mod)
			s-=mod;
		return s;
	}

	static BigInt sub_mod_coeff(const BigInt&a,const BigInt&b,
								const BigInt&mod){
		if(b.is_zero())
			return a;
		if(a.is_zero())
			return b.is_zero()?BigInt():mod-b;
		if(a>=b)
			return a-b;
		return mod-(b-a);
	}

	static BigInt neg_mod_coeff(const BigInt&a,const BigInt&mod){
		return a.is_zero()?BigInt():mod-a;
	}

	static std::vector<BigInt> add_plain_vec(const std::vector<BigInt>&a,
											 const std::vector<BigInt>&b){
		std::vector<BigInt> out(std::max(a.size(),b.size()));
		for(std::size_t i=0;i<out.size();++i){
			if(i<a.size())
				out[i]+=a[i];
			if(i<b.size())
				out[i]+=b[i];
		}
		trim_vec(out);
		return out;
	}

	static std::vector<BigInt> sub_plain_vec(const std::vector<BigInt>&a,
											 const std::vector<BigInt>&b){
		std::vector<BigInt> out(std::max(a.size(),b.size()));
		for(std::size_t i=0;i<out.size();++i){
			if(i<a.size())
				out[i]+=a[i];
			if(i<b.size())
				out[i]-=b[i];
		}
		trim_vec(out);
		return out;
	}

	static void add_shift_plain_ip(std::vector<BigInt>&out,
								   const std::vector<BigInt>&src,
								   std::size_t shift){
		if(src.empty())
			return;
		if(out.size()<src.size()+shift)
			out.resize(src.size()+shift);
		for(std::size_t i=0;i<src.size();++i)
			out[i+shift]+=src[i];
	}

	static std::vector<BigInt> slice_vec(const std::vector<BigInt>&a,
										 std::size_t lo,std::size_t hi){
		if(lo>=a.size()||lo>=hi)
			return {};
		hi=std::min(hi,a.size());
		return std::vector<BigInt>(
			a.begin()+static_cast<std::ptrdiff_t>(lo),
			a.begin()+static_cast<std::ptrdiff_t>(hi));
	}

	static std::vector<BigInt> mul_school_raw(const std::vector<BigInt>&a,
											  const std::vector<BigInt>&b){
		if(a.empty()||b.empty())
			return {};
		std::vector<BigInt> out(a.size()+b.size()-1u);
		const std::vector<BigInt>*ap=&a;
		const std::vector<BigInt>*bp=&b;
		if(ap->size()<bp->size())
			std::swap(ap,bp);
		for(std::size_t i=0;i<ap->size();++i){
			if((*ap)[i].is_zero())
				continue;
			for(std::size_t j=0;j<bp->size();++j){
				if(!(*bp)[j].is_zero())
					out[i+j]+=(*ap)[i]*(*bp)[j];
			}
		}
		trim_vec(out);
		return out;
	}

	static std::vector<BigInt> mul_kar_raw(const std::vector<BigInt>&a,
										   const std::vector<BigInt>&b){
		if(a.empty()||b.empty())
			return {};
		if(std::min(a.size(),b.size())<24u)
			return mul_school_raw(a,b);
		const std::size_t n=std::max(a.size(),b.size());
		const std::size_t m=n/2u;
		const auto a0=slice_vec(a,0,m);
		const auto a1=slice_vec(a,m,a.size());
		const auto b0=slice_vec(b,0,m);
		const auto b1=slice_vec(b,m,b.size());
		const auto z0=mul_kar_raw(a0,b0);
		const auto z2=mul_kar_raw(a1,b1);
		const auto s0=add_plain_vec(a0,a1);
		const auto s1=add_plain_vec(b0,b1);
		auto z1=mul_kar_raw(s0,s1);
		z1=sub_plain_vec(sub_plain_vec(z1,z0),z2);
		std::vector<BigInt> out(a.size()+b.size()-1u);
		add_shift_plain_ip(out,z0,0);
		add_shift_plain_ip(out,z1,m);
		add_shift_plain_ip(out,z2,2u*m);
		trim_vec(out);
		return out;
	}

	static std::vector<BigInt> mul_raw_vec(const std::vector<BigInt>&a,
										   const std::vector<BigInt>&b,
										   PolyMulAlg alg){
		if(alg==PolyMulAlg::schoolbook||
		   std::min(a.size(),b.size())<64u)
			return mul_school_raw(a,b);
		return mul_kar_raw(a,b);
	}

	static std::pair<std::vector<BigInt>,std::vector<BigInt>> divrem_vec(
		std::vector<BigInt> r,const std::vector<BigInt>&f,
		const BigInt&mod){
		normalize_vec(r,mod);
		if(f.empty())
			detail::throw_dom("BigNT::BigIntPoly::divrem: division by zero");
		if(r.size()<f.size())
			return {{},std::move(r)};
		const std::size_t deg=f.size()-1u;
		const BigInt inv_lc=invmod(f.back(),mod);
		std::vector<BigInt> q(r.size()-deg);
		for(std::size_t k=r.size();k>=f.size();--k){
			const std::size_t sh=k-f.size();
			const BigInt lead=mod_pos(r[k-1u],mod);
			if(!lead.is_zero()){
				const BigInt c=mod_pos(lead*inv_lc,mod);
				q[sh]=c;
				for(std::size_t i=0;i<deg;++i){
					if(!f[i].is_zero())
						r[sh+i]-=c*f[i];
				}
			}
			r[k-1u]=BigInt();
			if(sh>0)
				r[k-2u]=mod_pos(std::move(r[k-2u]),mod);
		}
		r.resize(deg);
		normalize_vec(q,mod);
		normalize_vec(r,mod);
		return {std::move(q),std::move(r)};
	}

	static std::vector<BigInt> rem_vec(std::vector<BigInt> r,
									   const std::vector<BigInt>&f,
									   const BigInt&mod){
		normalize_vec(r,mod);
		if(f.empty())
			detail::throw_dom("BigNT::BigIntPoly::rem: division by zero");
		if(r.size()<f.size())
			return r;
		const std::size_t deg=f.size()-1u;
		const BigInt inv_lc=invmod(f.back(),mod);
		for(std::size_t k=r.size();k>=f.size();--k){
			const std::size_t sh=k-f.size();
			const BigInt lead=mod_pos(r[k-1u],mod);
			if(!lead.is_zero()){
				const BigInt c=mod_pos(lead*inv_lc,mod);
				for(std::size_t i=0;i<deg;++i){
					if(!f[i].is_zero())
						r[sh+i]-=c*f[i];
				}
			}
			r[k-1u]=BigInt();
			if(sh>0)
				r[k-2u]=mod_pos(std::move(r[k-2u]),mod);
		}
		r.resize(deg);
		normalize_vec(r,mod);
		return r;
	}

	static std::vector<BigInt> mul_rem_vec(const std::vector<BigInt>&a,
										   const std::vector<BigInt>&b,
										   const std::vector<BigInt>&f,
										   const BigInt&mod,PolyMulAlg alg){
		if(a.empty()||b.empty()||f.size()==1u)
			return {};
		std::vector<BigInt> prod=mul_raw_vec(a,b,alg);
		if(prod.size()<f.size()){
			normalize_vec(prod,mod);
			return prod;
		}
		return rem_vec(std::move(prod),f,mod);
	}

	static ModPoly modpoly_pow_mod_generic(ModPoly base,const BigInt&exp,
										   const ModPoly&f,PolyMulAlg alg){
		base=base.rem(f);
		if(exp.is_zero())
			return ModPoly::one(f.modulus()).rem(f);
		if(exp.is_one())
			return base;
		ModPoly out=ModPoly::one(f.modulus());
		if(exp.fits_u64()){
			const std::uint64_t e=exp.to_u64();
			const unsigned bits=64u-std::countl_zero(e);
			for(unsigned ii=bits;ii>0;--ii){
				const unsigned i=ii-1u;
				out=ModPoly::mul(out,out,alg).rem(f);
				if(((e>>i)&1u)!=0u)
					out=ModPoly::mul(out,base,alg).rem(f);
			}
		}else{
			const std::size_t bits=exp.bit_length();
			for(std::size_t ii=bits;ii>0;--ii){
				const std::size_t i=ii-1u;
				out=ModPoly::mul(out,out,alg).rem(f);
				if(exp.test_bit(i))
					out=ModPoly::mul(out,base,alg).rem(f);
			}
		}
		return out;
	}

	std::vector<BigInt> coeff_;
	BigInt mod_=BigInt(2);
};

inline BigIntPoly poly_add(const BigIntPoly&a,const BigIntPoly&b){
	return BigIntPoly::add(a,b);
}
inline BigIntPoly poly_sub(const BigIntPoly&a,const BigIntPoly&b){
	return BigIntPoly::sub(a,b);
}
inline BigIntPoly poly_mul(const BigIntPoly&a,const BigIntPoly&b,
						   PolyMulAlg alg=PolyMulAlg::auto_select){
	return BigIntPoly::mul(a,b,alg);
}
inline std::pair<BigIntPoly,BigIntPoly> poly_divrem(const BigIntPoly&a,
													const BigIntPoly&b){
	return BigIntPoly::divrem(a,b);
}
inline BigIntPoly poly_rem(const BigIntPoly&a,const BigIntPoly&b){
	return a.rem(b);
}
inline BigIntPoly poly_gcd(const BigIntPoly&a,const BigIntPoly&b){
	return BigIntPoly::gcd(a,b);
}
inline BigIntPoly poly_x_pow_mod(const BigInt&k,const BigIntPoly&f,
								 PolyMulAlg alg=PolyMulAlg::auto_select){
	return BigIntPoly::x_pow_mod(k,f,alg);
}

inline BigInt pollard_rho(const BigInt&n,std::uint64_t seed=2,
						  std::uint64_t c_in=1,
						  std::uint64_t batch=128,
						  std::uint64_t max_iters=1u<<20){
	if(n.sign()<=0)
		detail::throw_dom("BigNT::pollard_rho: non-positive input");
	if(batch==0)
		batch=1;
	if(n.is_even())
		return BigInt(2);
	if(n.fits_u64()){
		const std::uint64_t d=detail_nt::pollard_rho_u64(
			n.to_u64(),seed,c_in,batch,max_iters);
		return detail_nt::u64(d);
	}
	if(pr_prime(n)>0)
		return n;
	const BigInt one(1);
	BigInt y=mod_pos(detail_nt::u64(seed),n);
	BigInt c=mod_pos(detail_nt::u64(c_in),n);
	if(c.is_zero())
		c=one;
	BigInt g(1),q(1),x,ys;
	std::uint64_t r=1;
	std::uint64_t iter=0;
	auto f=[&](const BigInt&v){
		return mod_pos(v*v+c,n);
	};
	while(g.is_one()&&iter<max_iters){
		x=y;
		for(std::uint64_t i=0;i<r;++i)
			y=f(y);
		std::uint64_t k=0;
		while(k<r&&g.is_one()){
			ys=y;
			const std::uint64_t lim=std::min(batch,r-k);
			for(std::uint64_t i=0;i<lim;++i){
				y=f(y);
				q=mod_pos(q*abs_sub(x,y),n);
			}
			g=gcd(q,n);
			k+=lim;
			iter+=lim;
		}
		r<<=1u;
	}
	if(g==n){
		do{
			ys=f(ys);
			g=gcd(abs_sub(x,ys),n);
		}while(g.is_one());
	}
	if(g.is_one()||g==n)
		return BigInt();
	return g;
}

inline BigInt pollard_pm1(const BigInt&n,std::uint32_t bound=10000,
						  const BigInt&a0=BigInt(2)){
	if(n.sign()<=0)
		detail::throw_dom("BigNT::pollard_pm1: non-positive input");
	if(n.is_even())
		return BigInt(2);
	if(n.fits_u64()){
		std::uint64_t a=2;
		if(a0.sign()>0&&a0.fits_u64())
			a=a0.to_u64();
		const std::uint64_t d=detail_nt::pollard_pm1_u64(
			n.to_u64(),bound,a);
		return detail_nt::u64(d);
	}
	if(pr_prime(n)>0)
		return n;
	BigInt a=mod_pos(a0,n);
	std::uint32_t steps=0;
	for(std::uint32_t p : detail_nt::small_primes(bound)){
		std::uint64_t pp=p;
		while(pp<=static_cast<std::uint64_t>(bound)/p)
			pp*=p;
		a=modpow(a,detail_nt::u64(pp),n);
		if((++steps&255u)==0u){
			BigInt g=gcd(a-BigInt(1),n);
			if(g>BigInt(1)&&g<n)
				return g;
			if(g==n)
				return BigInt();
		}
	}
	BigInt g=gcd(a-BigInt(1),n);
	if(g>BigInt(1)&&g<n)
		return g;
	return BigInt();
}

inline void factor_rec(BigInt n,std::vector<BigInt>&out){
	if(n.is_one())
		return;
	if(n.fits_u64()){
		std::vector<std::uint64_t> ufac;
		detail_nt::factor_u64_rec(n.to_u64(),ufac);
		for(std::uint64_t p : ufac)
			out.push_back(detail_nt::u64(p));
		return;
	}
	if(pr_prime(n)>0){
		out.push_back(n);
		return;
	}
	for(std::uint32_t p : detail_nt::trial_primes()){
		if(n.mod_u32(p)!=0)
			continue;
		const BigInt bp=detail_nt::u64(p);
		do{
			out.push_back(bp);
			n/=bp;
		}while(!n.is_one()&&n.mod_u32(p)==0);
		if(n.is_one())
			return;
		if(pr_prime(n)>0){
			out.push_back(n);
			return;
		}
	}
	BigInt d=pollard_pm1(n,2000);
	for(std::uint64_t s=2;(d.is_zero()||d==n)&&s<64u;++s)
		d=pollard_rho(n,s,2u*s+1u);
	if(d.is_zero()||d==n){
		out.push_back(n);
		return;
	}
	factor_rec(d,out);
	factor_rec(n/d,out);
}

inline std::vector<Factor> factor(const BigInt&n){
	if(n.sign()<=0)
		detail::throw_dom("BigNT::factor: input must be positive");
	std::vector<BigInt> flat;
	factor_rec(n,flat);
	std::sort(flat.begin(),flat.end());
	std::vector<Factor> out;
	for(const BigInt&p : flat){
		if(out.empty()||out.back().prime!=p)
			out.push_back(Factor{p,1});
		else
			++out.back().exp;
	}
	return out;
}

struct CornacchiaRes{
	bool ok=false;
	BigInt x;
	BigInt y;
};

inline CornacchiaRes cornacchia_with_root(const BigInt&d,const BigInt&m,
										  const BigInt&root){
	if(d.sign()<=0||m.sign()<=0)
		detail::throw_dom("BigNT::cornacchia: non-positive input");
	BigInt r=dvm_knuth(root,m).second;
	if(r.sign()<0)
		r+=m;
	if(r>(m>>1u))
		r=m-r;
	BigInt a=m,b=r;
	const BigInt lim=isqrt(m);
	while(b>lim){
		const BigInt t=dvm_knuth(a,b).second;
		a=b;
		b=t;
	}
	const BigInt x=b;
	const BigInt rem=m-x*x;
	if(rem.sign()<0)
		return {};
	const auto qr=dvm_knuth(rem,d);
	if(!qr.second.is_zero())
		return {};
	const BigInt y=isqrt(qr.first);
	if(sqr_disp_noat(y)!=qr.first)
		return {};
	return {true,x,y};
}

inline CornacchiaRes cornacchia(const BigInt&d,const BigInt&m){
	BigInt r;
	if(!sqrtmod(&r,mod_pos(-d,m),m))
		return {};
	return cornacchia_with_root(d,m,r);
}

struct QForm{
	BigInt a;
	BigInt b;
	BigInt c;

	BigInt discriminant() const{ return b*b-BigInt(4)*a*c; }
};

inline bool is_reduced(const QForm&f){
	if(f.a.sign()<=0)
		return false;
	const BigInt ab=f.b.abs();
	if(ab>f.a||f.a>f.c)
		return false;
	if((ab==f.a||f.a==f.c)&&f.b.sign()<0)
		return false;
	return f.discriminant().sign()<0;
}

inline QForm reduce(QForm f){
	const BigInt D=f.discriminant();
	if(f.a.sign()<=0||D.sign()>=0)
		detail::throw_dom("BigNT::reduce: expected positive definite form");
	for(std::size_t it=0;it<100000u;++it){
		bool done=true;
		if(f.c<f.a){
			std::swap(f.a,f.c);
			f.b=-f.b;
			done=false;
		}
		if(f.b.abs()>f.a){
			const BigInt twice_a=f.a<<1u;
			f.b=fdiv_r(f.b,twice_a);
			if(f.b>f.a)
				f.b-=twice_a;
			f.c=divexact(f.b*f.b-D,twice_a<<1u);
			done=false;
		}
		if(done)
			break;
	}
	if((f.a==f.b.abs()||f.a==f.c)&&f.b.sign()<0)
		f.b=-f.b;
	if(!is_reduced(f))
		detail::throw_dom("BigNT::reduce: reduction did not converge");
	return f;
}

inline std::vector<QForm> reduced_forms(const BigInt&D,bool primitive=true){
	if(D.sign()>=0)
		detail::throw_dom("BigNT::reduced_forms: discriminant must be negative");
	const std::uint32_t d4=D.mod_u32(4);
	if(d4!=0u&&d4!=1u)
		detail::throw_dom("BigNT::reduced_forms: bad discriminant");
	const BigInt absD=D.abs();
	const BigInt lim_bi=isqrt(absD/BigInt(3));
	if(!lim_bi.fits_u64())
		detail::throw_ovf("BigNT::reduced_forms: discriminant too large");
	const std::uint64_t lim=lim_bi.to_u64()+1u;
	if(lim>static_cast<std::uint64_t>(
		   std::numeric_limits<std::int64_t>::max()))
		detail::throw_ovf("BigNT::reduced_forms: discriminant too large");
	std::vector<QForm> out;
	for(std::int64_t ai=1;ai<=static_cast<std::int64_t>(lim);++ai){
		const BigInt a(ai);
		for(std::int64_t bi=-ai;bi<=ai;++bi){
			if(((bi&1)-(static_cast<int>(d4)&1))&1)
				continue;
			const BigInt b(bi);
			const BigInt num=b*b-D;
			const BigInt den=BigInt(4)*a;
			auto qr=divmod(num,den);
			if(!qr.second.is_zero())
				continue;
			QForm f{a,b,qr.first};
			if(!is_reduced(f))
				continue;
			if(primitive){
				BigInt g=gcd(f.a,f.b.abs());
				g=gcd(g,f.c);
				if(!g.is_one())
					continue;
			}
			out.push_back(std::move(f));
		}
	}
	return out;
}

inline std::size_t class_number(const BigInt&D,bool primitive=true){
	return reduced_forms(D,primitive).size();
}

inline std::vector<BigInt> j_qexp(std::size_t terms){
	const std::size_t need=terms+3u;
	std::vector<BigInt> e4(need),e6(need);
	e4[0]=BigInt(1);
	e6[0]=BigInt(1);
	const auto sig3=detail_nt::sigma_powers(need-1u,3);
	const auto sig5=detail_nt::sigma_powers(need-1u,5);
	for(std::size_t n=1;n<need;++n){
		e4[n]=BigInt(240)*sig3[n];
		e6[n]=BigInt(-504)*sig5[n];
	}
	const auto e4_2=detail_nt::mul_trunc(e4,e4,need);
	const auto e4_3=detail_nt::mul_trunc(e4_2,e4,need);
	const auto e6_2=detail_nt::mul_trunc(e6,e6,need);
	std::vector<BigInt> delta(need);
	for(std::size_t i=0;i<need;++i)
		delta[i]=(e4_3[i]-e6_2[i])/BigInt(1728);
	std::vector<BigInt> d(terms+2u);
	for(std::size_t i=0;i<d.size();++i)
		d[i]=delta[i+1u];
	std::vector<BigInt> h(terms+2u);
	for(std::size_t n=0;n<h.size();++n){
		BigInt acc=e4_3[n];
		for(std::size_t i=1;i<=n&&i<d.size();++i)
			acc-=d[i]*h[n-i];
		h[n]=acc/d[0];
	}
	return h;
}

inline BigComplex j_invariant_q(const BigComplex&q,std::size_t terms,
								std::size_t prec=BigFloat::default_prec,
								FloatRnd rnd=FloatRnd::nearest){
	if(q.is_zero())
		detail::throw_dom("BigNT::j_invariant_q: q is zero");
	const std::size_t work=std::max(prec,q.precision())+32u;
	const auto coeff=j_qexp(terms);
	const BigComplex qw=q.rounded(work);
	BigComplex sum=BigComplex::div(
		BigComplex(BigFloat(coeff[0],work),BigFloat::zero(work)),qw,work);
	sum+=BigComplex(BigFloat(coeff[1],work),BigFloat::zero(work));
	if(coeff.size()>2u){
		BigComplex tail(BigFloat(coeff.back(),work),BigFloat::zero(work));
		for(std::size_t ii=coeff.size()-1u;ii>2u;--ii){
			tail=BigComplex::mul(qw,tail,work);
			tail.real()=BigFloat::add(tail.real(),BigFloat(coeff[ii-1u],work),
									  work,FloatRnd::nearest);
		}
		sum+=BigComplex::mul(qw,tail,work);
	}
	return sum.rounded(prec,rnd);
}

inline BigComplex modular_tau_reduce(const BigComplex&tau,
									 std::size_t prec=BigFloat::default_prec){
	const std::size_t work=std::max(prec,tau.precision())+32u;
	BigComplex z=tau.rounded(work);
	if(z.imag().sign()<=0)
		detail::throw_dom("BigNT::modular_tau_reduce: Im(tau) must be positive");
	for(std::size_t it=0;it<32u;++it){
		const BigInt nearest=z.real().to_bigint(FloatRnd::nearest);
		if(!nearest.is_zero()){
			z.real()=BigFloat::sub(z.real(),BigFloat(nearest,work),
								   work,FloatRnd::nearest);
		}
		const BigFloat n2=BigFloat::add(BigFloat::sqr(z.real(),work),
										BigFloat::sqr(z.imag(),work),
										work,FloatRnd::nearest);
		if(compare(n2,BigFloat::one(work))>=0)
			break;
		z=BigComplex::div(BigComplex(BigFloat(-1,work),BigFloat::zero(work)),
						  z,work);
	}
	return z.rounded(prec);
}

inline BigComplex j_invariant_tau(const BigComplex&tau,std::size_t terms,
								  std::size_t prec=BigFloat::default_prec,
								  FloatRnd rnd=FloatRnd::nearest){
	const std::size_t work=std::max(prec,tau.precision())+48u;
	const BigComplex tau0=modular_tau_reduce(tau,work);
	const BigFloat two_pi=BigFloat::mul(BigFloat(2,work),
		mini_mp::const_pi(work),work);
	const BigComplex z(-BigFloat::mul(two_pi,tau0.imag(),work),
					   BigFloat::mul(two_pi,tau0.real(),work));
	return j_invariant_q(BigComplex::exp(z,work),terms,prec,rnd);
}

inline std::vector<BigComplex> j_polynomial_from_roots(
	const std::vector<BigComplex>&roots,
	std::size_t prec=BigFloat::default_prec,
	FloatRnd rnd=FloatRnd::nearest){
	std::vector<BigComplex> poly;
	poly.emplace_back(BigFloat::one(prec),BigFloat::zero(prec));
	for(const BigComplex&r : roots){
		std::vector<BigComplex> next(poly.size()+1u,
			BigComplex(BigFloat::zero(prec),BigFloat::zero(prec)));
		for(std::size_t i=0;i<poly.size();++i){
			next[i]-=BigComplex::mul(poly[i],r,prec,rnd);
			next[i+1u]+=poly[i].rounded(prec,rnd);
		}
		poly=std::move(next);
	}
	return poly;
}

} 

namespace ec{

struct Curve{
	BigInt p;
	BigInt a;
	BigInt b;
};

struct AffinePoint{
	BigInt x;
	BigInt y;
	bool inf=true;
};

struct JacobianPoint{
	BigInt x;
	BigInt y;
	BigInt z;
	bool inf=true;
};

inline JacobianPoint to_jacobian(const AffinePoint&P);
inline AffinePoint to_affine(const JacobianPoint&P,const Curve&E);
inline JacobianPoint mul_wnaf_affine(const AffinePoint&P,BigInt k,
									 const Curve&E,unsigned width);

inline BigInt mod(const BigInt&x,const Curve&E){
	return BigNT::mod_pos(x,E.p);
}

inline BigInt addm(const BigInt&a,const BigInt&b,const Curve&E){
	return mod(a+b,E);
}

inline BigInt subm(const BigInt&a,const BigInt&b,const Curve&E){
	return mod(a-b,E);
}

inline BigInt mulm(const BigInt&a,const BigInt&b,const Curve&E){
	return mod(a*b,E);
}

inline BigInt sqrm(const BigInt&a,const Curve&E){ return mulm(a,a,E); }

inline BigInt invm(const BigInt&a,const Curve&E){
	return invmod(mod(a,E),E.p);
}

inline AffinePoint infinity(){ return {}; }

inline AffinePoint affine(BigInt x,BigInt y,const Curve&E){
	return {mod(x,E),mod(y,E),false};
}

inline bool is_on_curve(const AffinePoint&P,const Curve&E){
	if(P.inf)
		return true;
	const BigInt lhs=sqrm(P.y,E);
	const BigInt rhs=addm(addm(mulm(sqrm(P.x,E),P.x,E),
							   mulm(E.a,P.x,E),E),E.b,E);
	return lhs==rhs;
}

inline AffinePoint neg(const AffinePoint&P,const Curve&E){
	if(P.inf)
		return P;
	return {P.x,mod(-P.y,E),false};
}

inline AffinePoint dbl(const AffinePoint&P,const Curve&E){
	if(P.inf||mod(P.y,E).is_zero())
		return infinity();
	const BigInt l=mulm(BigInt(3)*sqrm(P.x,E)+E.a,
						invm(BigInt(2)*P.y,E),E);
	const BigInt x3=subm(sqrm(l,E),BigInt(2)*P.x,E);
	const BigInt y3=subm(mulm(l,P.x-x3,E),P.y,E);
	return {x3,y3,false};
}

inline AffinePoint add(const AffinePoint&P,const AffinePoint&Q,
					   const Curve&E){
	if(P.inf)
		return Q;
	if(Q.inf)
		return P;
	if(P.x==Q.x){
		if(mod(P.y+Q.y,E).is_zero())
			return infinity();
		return dbl(P,E);
	}
	const BigInt l=mulm(Q.y-P.y,invm(Q.x-P.x,E),E);
	const BigInt x3=subm(subm(sqrm(l,E),P.x,E),Q.x,E);
	const BigInt y3=subm(mulm(l,P.x-x3,E),P.y,E);
	return {x3,y3,false};
}

namespace detail_ec{

struct UPoint{
	std::uint64_t x=0;
	std::uint64_t y=0;
	bool inf=true;
};

inline std::uint64_t mod_bi_u64(const BigInt&x,const BigInt&m,
								std::uint64_t mod){
	if(x.sign()>=0&&x.fits_u64())
		return x.to_u64()%mod;
	return BigNT::mod_pos(x,m).to_u64();
}

inline std::uint64_t inv_u64(std::uint64_t a,std::uint64_t mod){
	a%=mod;
	if(a==0)
		detail::throw_dom("mini_mp::ec: non-invertible denominator");
	std::uint64_t out=0;
	if(!detail::invmod64(&out,a,mod))
		detail::throw_dom("mini_mp::ec: non-invertible denominator");
	return out;
}

inline UPoint dbl_u64(const UPoint&P,std::uint64_t a,std::uint64_t mod){
	if(P.inf||P.y==0)
		return {};
	using BigNT::detail_nt::add_mod;
	using BigNT::detail_nt::mul_mod;
	using BigNT::detail_nt::sub_mod;
	const std::uint64_t xx=mul_mod(P.x,P.x,mod);
	const std::uint64_t num=add_mod(mul_mod(3u%mod,xx,mod),a,mod);
	const std::uint64_t den=mul_mod(2u%mod,P.y,mod);
	const std::uint64_t l=mul_mod(num,inv_u64(den,mod),mod);
	const std::uint64_t x3=sub_mod(sub_mod(mul_mod(l,l,mod),P.x,mod),
								   P.x,mod);
	const std::uint64_t y3=sub_mod(mul_mod(l,sub_mod(P.x,x3,mod),mod),
								   P.y,mod);
	return {x3,y3,false};
}

inline UPoint add_u64(const UPoint&P,const UPoint&Q,std::uint64_t a,
					  std::uint64_t mod){
	if(P.inf)
		return Q;
	if(Q.inf)
		return P;
	using BigNT::detail_nt::add_mod;
	using BigNT::detail_nt::mul_mod;
	using BigNT::detail_nt::sub_mod;
	if(P.x==Q.x){
		if(add_mod(P.y,Q.y,mod)==0)
			return {};
		return dbl_u64(P,a,mod);
	}
	const std::uint64_t num=sub_mod(Q.y,P.y,mod);
	const std::uint64_t den=sub_mod(Q.x,P.x,mod);
	const std::uint64_t l=mul_mod(num,inv_u64(den,mod),mod);
	const std::uint64_t x3=sub_mod(sub_mod(mul_mod(l,l,mod),P.x,mod),
								   Q.x,mod);
	const std::uint64_t y3=sub_mod(mul_mod(l,sub_mod(P.x,x3,mod),mod),
								   P.y,mod);
	return {x3,y3,false};
}

inline bool mul_u64(AffinePoint*out,const AffinePoint&P,const BigInt&k,
					const Curve&E){
	MINI_MP_ASSERT(out!=nullptr);
	if(P.inf||k.is_zero()){
		*out={};
		return true;
	}
	if(!E.p.fits_u64())
		return false;
	const std::uint64_t mod=E.p.to_u64();
	if(mod<=3u||mod>static_cast<std::uint64_t>(
		   std::numeric_limits<std::int64_t>::max()))
		return false;
	const std::uint64_t a=mod_bi_u64(E.a,E.p,mod);
	const UPoint base{mod_bi_u64(P.x,E.p,mod),mod_bi_u64(P.y,E.p,mod),
					  false};
	UPoint acc;
	for(std::size_t i=k.bit_length();i>0;--i){
		acc=dbl_u64(acc,a,mod);
		if(k.test_bit(i-1u))
			acc=add_u64(acc,base,a,mod);
	}
	if(acc.inf)
		*out={};
	else
		*out={BigInt::from_u64(acc.x),BigInt::from_u64(acc.y),false};
	return true;
}

}

inline AffinePoint mul(const AffinePoint&P,BigInt k,const Curve&E){
	if(k.sign()<0)
		return mul(neg(P,E),-k,E);
	AffinePoint fast;
	if(detail_ec::mul_u64(&fast,P,k,E))
		return fast;
	return to_affine(mul_wnaf_affine(P,std::move(k),E,5u),E);
}

inline JacobianPoint to_jacobian(const AffinePoint&P){
	if(P.inf)
		return {};
	return {P.x,P.y,BigInt(1),false};
}

inline AffinePoint to_affine(const JacobianPoint&P,const Curve&E){
	if(P.inf||P.z.is_zero())
		return infinity();
	const BigInt zi=invm(P.z,E);
	const BigInt zi2=sqrm(zi,E);
	const BigInt zi3=mulm(zi2,zi,E);
	return {mulm(P.x,zi2,E),mulm(P.y,zi3,E),false};
}

inline JacobianPoint neg(const JacobianPoint&P,const Curve&E){
	if(P.inf)
		return P;
	return {P.x,mod(-P.y,E),P.z,false};
}

inline JacobianPoint dbl(const JacobianPoint&P,const Curve&E){
	if(P.inf||P.y.is_zero())
		return {};
	const BigInt XX=sqrm(P.x,E);
	const BigInt YY=sqrm(P.y,E);
	const BigInt YYYY=sqrm(YY,E);
	const BigInt ZZ=sqrm(P.z,E);
	const BigInt S=mulm(BigInt(2),
		subm(subm(sqrm(addm(P.x,YY,E),E),XX,E),YYYY,E),E);
	BigInt M;
	if(mod(E.a+BigInt(3),E).is_zero())
		M=mulm(BigInt(3),mulm(P.x-ZZ,P.x+ZZ,E),E);
	else
		M=addm(BigInt(3)*XX,mulm(E.a,sqrm(ZZ,E),E),E);
	const BigInt T=subm(sqrm(M,E),BigInt(2)*S,E);
	const BigInt X3=T;
	const BigInt Y3=subm(mulm(M,S-T,E),BigInt(8)*YYYY,E);
	const BigInt Z3=subm(subm(sqrm(addm(P.y,P.z,E),E),YY,E),ZZ,E);
	return {X3,Y3,Z3,false};
}

inline JacobianPoint add(const JacobianPoint&P,const JacobianPoint&Q,
						 const Curve&E){
	if(P.inf)
		return Q;
	if(Q.inf)
		return P;
	const BigInt Z1Z1=sqrm(P.z,E);
	const BigInt Z2Z2=sqrm(Q.z,E);
	const BigInt U1=mulm(P.x,Z2Z2,E);
	const BigInt U2=mulm(Q.x,Z1Z1,E);
	const BigInt S1=mulm(P.y,mulm(Q.z,Z2Z2,E),E);
	const BigInt S2=mulm(Q.y,mulm(P.z,Z1Z1,E),E);
	if(U1==U2){
		if(S1==S2)
			return dbl(P,E);
		return {};
	}
	const BigInt H=subm(U2,U1,E);
	const BigInt I=sqrm(addm(H,H,E),E);
	const BigInt J=mulm(H,I,E);
	const BigInt r=addm(subm(S2,S1,E),subm(S2,S1,E),E);
	const BigInt V=mulm(U1,I,E);
	const BigInt X3=subm(subm(sqrm(r,E),J,E),BigInt(2)*V,E);
	const BigInt Y3=subm(mulm(r,V-X3,E),BigInt(2)*mulm(S1,J,E),E);
	const BigInt Z3=mulm(
		subm(subm(sqrm(addm(P.z,Q.z,E),E),Z1Z1,E),Z2Z2,E),H,E);
	return {X3,Y3,Z3,false};
}

inline JacobianPoint add_mixed(const JacobianPoint&P,const AffinePoint&Q,
							   const Curve&E){
	if(P.inf)
		return to_jacobian(Q);
	if(Q.inf)
		return P;
	const BigInt Z1Z1=sqrm(P.z,E);
	const BigInt U2=mulm(Q.x,Z1Z1,E);
	const BigInt S2=mulm(Q.y,mulm(P.z,Z1Z1,E),E);
	if(P.x==U2){
		if(P.y==S2)
			return dbl(P,E);
		return {};
	}
	const BigInt H=subm(U2,P.x,E);
	const BigInt HH=sqrm(H,E);
	const BigInt I=mulm(BigInt(4),HH,E);
	const BigInt J=mulm(H,I,E);
	const BigInt r=addm(subm(S2,P.y,E),subm(S2,P.y,E),E);
	const BigInt V=mulm(P.x,I,E);
	const BigInt X3=subm(subm(sqrm(r,E),J,E),BigInt(2)*V,E);
	const BigInt Y3=subm(mulm(r,V-X3,E),BigInt(2)*mulm(P.y,J,E),E);
	const BigInt Z3=subm(subm(sqrm(addm(P.z,H,E),E),Z1Z1,E),HH,E);
	return {X3,Y3,Z3,false};
}

inline std::vector<int> wnaf_digits(BigInt k,unsigned width=5){
	MINI_MP_ASSERT(width>=2&&width<=16);
	std::vector<int> out;
	const std::uint32_t radix=std::uint32_t(1)<<width;
	const std::uint32_t half=radix>>1u;
	while(k.sign()>0){
		int digit=0;
		if(k.is_odd()){
			std::uint32_t u=k.mod_u32(radix);
			if(u>=half)
				digit=static_cast<int>(u)-static_cast<int>(radix);
			else
				digit=static_cast<int>(u);
			k-=BigInt(digit);
		}
		out.push_back(digit);
		k>>=1u;
	}
	return out;
}

inline JacobianPoint mul_wnaf_affine(const AffinePoint&P,BigInt k,
									 const Curve&E,unsigned width=5){
	if(P.inf||k.is_zero())
		return {};
	if(k.sign()<0)
		return mul_wnaf_affine(neg(P,E),-k,E,width);
	const std::size_t table_n=std::size_t(1)<<(width-2u);
	std::vector<AffinePoint> table;
	table.reserve(table_n);
	table.push_back(P);
	if(table_n>1u){
		const AffinePoint twoP=to_affine(dbl(to_jacobian(P),E),E);
		for(std::size_t i=1;i<table_n;++i)
			table.push_back(add(table.back(),twoP,E));
	}
	const auto naf=wnaf_digits(k,width);
	JacobianPoint out;
	for(std::size_t ii=naf.size();ii>0;--ii){
		out=dbl(out,E);
		const int d=naf[ii-1u];
		if(d>0)
			out=add_mixed(out,table[static_cast<std::size_t>(d/2)],E);
		else if(d<0)
			out=add_mixed(out,
				neg(table[static_cast<std::size_t>((-d)/2)],E),E);
	}
	return out;
}

inline JacobianPoint mul(const JacobianPoint&P,BigInt k,const Curve&E){
	if(P.inf||k.is_zero())
		return {};
	if(k.sign()<0)
		return mul(neg(P,E),-k,E);
	return mul_wnaf_affine(to_affine(P,E),std::move(k),E);
}

} 

} 





#if defined(MINI_MP_BENCH_MAIN)

#include<iomanip>
#include<iostream>

namespace mini_mp_bench_detail{

using mini_mp::BigInt;

inline std::uint64_t xorshift64(std::uint64_t&s) noexcept{
	s^=(s<<7);
	s^=(s>>9);
	s^=(s<<8);
	return s;
}

inline BigInt rnd_biex(std::uint64_t&seed,std::size_t limbs){
	if(limbs==0)
		return BigInt();
	std::uint64_t top=xorshift64(seed);
	if(top==0)
		top=1;
	BigInt x=BigInt::from_u64(top);
	for(std::size_t i=1;i<limbs;++i){
		x<<=64;
		x+=BigInt::from_u64(xorshift64(seed));
	}
	if((xorshift64(seed)&1u)!=0u)
		x=-x;
	return x;
}

inline std::uint64_t bench_sch(const BigInt&a,const BigInt&b,
										  int loops){
	volatile std::size_t sink=0;
	const auto t0=std::chrono::steady_clock::now();
	for(int i=0;i<loops;++i){
		BigInt c=mini_mp::mul_sbk(a,b);
		sink^=c.bit_length();
	}
	const auto t1=std::chrono::steady_clock::now();
	(void)sink;
	return static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
}

} 

int main(){
	using mini_mp::BigInt;
	using namespace mini_mp_bench_detail;

	mini_mp::detail::ensure_at();

	std::cout<<"mini_mp benchmark\n";
	const std::size_t tun_nth=mini_mp::detail::tun_ntt();
	std::cout<<"autotune.ntt_th=";
	if(tun_nth==mini_mp::detail::kNttOff){
		std::cout<<"disabled\n";
	}else{
		std::cout<<tun_nth<<"\n";
	}
#if MINI_MP_ENABLE_NTT
	const std::size_t tun_nsth=mini_mp::detail::tun_ntt_sq();
	std::cout<<"autotune.ntt_sq_th=";
	if(tun_nsth==mini_mp::detail::kNttOff){
		std::cout<<"disabled\n";
	}else{
		std::cout<<tun_nsth<<"\n";
	}
	std::cout<<"autotune.ntt_bits="
			 <<mini_mp::detail::tun_ntt_bits()<<"\n";
#endif
	std::cout<<"autotune.kar_th="
			 <<mini_mp::detail::tun_kar()<<"\n";
	std::cout<<"autotune.kar_rec="
			 <<mini_mp::detail::tun_krec()<<"\n";
	std::cout<<"autotune.kar_imb="
			 <<mini_mp::detail::tun_kar_imb()<<"\n";
#if MINI_MP_ENABLE_NTT
	std::cout<<"autotune.ntt_imb="
			 <<mini_mp::detail::tun_ntt_imb()<<"\n";
#endif
	std::cout<<"autotune.hl_min="
			 <<mini_mp::detail::tun_hmin()<<"\n";
	std::cout<<"autotune.hl_rnd="
			 <<mini_mp::detail::tun_hrnd()<<"\n";
	std::cout<<"autotune.gcd_small="
			 <<mini_mp::detail::tun_gcd_sm()<<"\n";
	std::cout<<"autotune.gcd_large="
			 <<mini_mp::detail::tun_gcd_lg()<<"\n";
	std::cout<<"autotune.gcd_qsmall="
			 <<mini_mp::detail::tun_gcd_qs()<<"\n";
	std::cout<<"autotune.bz_min="
			 <<mini_mp::detail::tun_bz_min()<<"\n";
	std::cout<<"autotune.bz_chunk="
			 <<mini_mp::detail::tun_bz_chunk()<<"\n";
	std::cout<<"autotune.prod_leaf="
			 <<mini_mp::detail::tun_prod_leaf()<<"\n";
	std::cout<<"autotune.factorial_tree="
			 <<mini_mp::detail::tun_fac_tree()<<"\n";
	std::cout<<"autotune.binomial_tree="
			 <<mini_mp::detail::tun_binom_tree()<<"\n";
	std::cout<<"autotune.pow_w5="
			 <<mini_mp::detail::tun_pow_w5()<<"\n";
	std::cout<<"autotune.pow_w6="
			 <<mini_mp::detail::tun_pow_w6()<<"\n";
	std::cout<<"autotune.d10_dc=";
	const std::size_t tun_d10=mini_mp::detail::tun_d10_dc();
	if(tun_d10==mini_mp::detail::kNttOff){
		std::cout<<"disabled\n";
	}else{
		std::cout<<tun_d10<<"\n";
	}
	std::cout<<"autotune.d10_prs=";
	const std::size_t tun_d10_prs=mini_mp::detail::tun_d10_prs();
	if(tun_d10_prs==mini_mp::detail::kNttOff){
		std::cout<<"disabled\n";
	}else{
		std::cout<<tun_d10_prs<<"\n";
	}
	std::cout<<"autotune.t3_th=";
	const std::size_t tun_t3=mini_mp::detail::tun_t3();
	if(tun_t3==mini_mp::detail::kNttOff){
		std::cout<<"disabled\n";
	}else{
		std::cout<<tun_t3<<"\n";
	}

	std::cout<<"\n[mul backend ns/op]\n";
	std::cout<<std::left<<std::setw(10)<<"limbs"<<std::setw(14)<<"schoolbook"
			 <<std::setw(14)<<"karatsuba"<<std::setw(14)<<"toom3";
#if MINI_MP_ENABLE_NTT
	std::cout<<std::setw(14)<<"ntt";
#endif
	std::cout<<"\n";

	std::uint64_t seed=0x123456789abcdef0ULL;
	constexpr std::array<std::size_t,8> kMulSizes={16u, 32u, 64u, 96u,
												   128u,192u,256u,384u};
	for(std::size_t limbs : kMulSizes){
		const BigInt a=rnd_biex(seed,limbs);
		const BigInt b=rnd_biex(seed,limbs);
		const int loops=(limbs<=32u)?12:(limbs<=96u)?8:(limbs<=192u)?5:3;

		const std::uint64_t t_school=bench_sch(a,b,loops);
		const std::uint64_t t_kar=
			mini_mp::detail::bench_mul(a,b,loops,false);
		const std::uint64_t t_t3=
			mini_mp::detail::bench_t3(a,b,loops);
#if MINI_MP_ENABLE_NTT
		const std::uint64_t t_ntt=
			mini_mp::detail::bench_mul(a,b,loops,true);
#endif

		std::cout<<std::left<<std::setw(10)<<limbs<<std::setw(14)
				 <<(t_school/static_cast<std::uint64_t>(loops))<<std::setw(14)
				 <<(t_kar/static_cast<std::uint64_t>(loops))<<std::setw(14)
				 <<(t_t3/static_cast<std::uint64_t>(loops));
#if MINI_MP_ENABLE_NTT
		std::cout<<std::setw(14)<<(t_ntt/static_cast<std::uint64_t>(loops));
#endif
		std::cout<<"\n";
	}

	std::cout<<"\n[gcd half-lehmer params total ns]\n";
	std::vector<std::pair<BigInt,BigInt>> samples;
	samples.reserve(24);
	constexpr std::array<std::size_t,4> kGcdSizes={8u,12u,24u,40u};
	for(std::size_t limbs : kGcdSizes){
		for(int i=0;i<6;++i){
			BigInt a=rnd_biex(seed,limbs).abs();
			BigInt b=rnd_biex(seed,limbs).abs();
			if(a==b)
				b+=BigInt(1);
			if(a<b)
				std::swap(a,b);
			samples.emplace_back(std::move(a),std::move(b));
		}
	}

	const std::size_t tuned_min=mini_mp::detail::tun_hmin();
	const std::size_t tun_rnd=mini_mp::detail::tun_hrnd();
	constexpr std::array<std::size_t,4> kMinCand={3u,4u,6u,8u};
	constexpr std::array<std::size_t,5> kRndCand={0u,4u,6u,8u,10u};
	std::cout<<std::left<<std::setw(10)<<"min"<<std::setw(10)<<"rounds"
			 <<std::setw(14)<<"time_ns"
			 <<"selected\n";
	for(std::size_t min_limbs : kMinCand){
		for(std::size_t rounds : kRndCand){
			const std::uint64_t t=
				mini_mp::detail::bench_gcd(samples,min_limbs,rounds);
			const bool is_sel=(min_limbs==tuned_min&&rounds==tun_rnd);
			std::cout<<std::left<<std::setw(10)<<min_limbs<<std::setw(10)
					 <<rounds<<std::setw(14)<<t<<(is_sel?"*":"")<<"\n";
		}
	}

	return 0;
}

#endif 


#endif 





