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
static_assert(sizeof(void*)==8,
			  "mini_mp currently supports only 64-bit targets.");

#if defined(_MSC_VER)&&defined(_M_X64)
#define MINI_MP_DETAIL_USE_MSVC_INTRIN 1
#include<intrin.h>
#else
#define MINI_MP_DETAIL_USE_MSVC_INTRIN 0
#endif

#if defined(__x86_64__)&&(defined(__GNUC__)||defined(__clang__))&&             \
	!defined(_MSC_VER)
#define MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY 1
#else
#define MINI_MP_DETAIL_USE_GNU_X86_ADDCARRY 0
#endif

#if !MINI_MP_DETAIL_USE_MSVC_INTRIN&&!defined(__SIZEOF_INT128__)
#error                                                                         \
	"mini_mp requires either MSVC x64 intrinsics or compiler support for unsigned __int128."
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
#else
	const unsigned __int128 p=
		static_cast<unsigned __int128>(a)*static_cast<unsigned __int128>(b);
	return {static_cast<std::uint64_t>(p),static_cast<std::uint64_t>(p>>64)};
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
#else
	const unsigned __int128 s=static_cast<unsigned __int128>(a)+
							  static_cast<unsigned __int128>(b)+
							  static_cast<unsigned __int128>(carry_in);
	*out=static_cast<std::uint64_t>(s);
	return static_cast<std::uint64_t>(s>>64);
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
#else
	const unsigned __int128 aa=static_cast<unsigned __int128>(a);
	const unsigned __int128 bb=static_cast<unsigned __int128>(b)+
							   static_cast<unsigned __int128>(borrow_in);
	const unsigned __int128 d=aa-bb;
	*out=static_cast<std::uint64_t>(d);
	return (aa<bb)?1u:0u;
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
#else
	const unsigned __int128 num=(static_cast<unsigned __int128>(hi)<<64)|
								static_cast<unsigned __int128>(lo);
	const std::uint64_t q=static_cast<std::uint64_t>(num/d);
	*rem_ptr=static_cast<std::uint64_t>(num%d);
	return q;
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
			for(size_type i=0;i<n;++i){
				ptr_[i]=first[static_cast<typename std::iterator_traits<It>::difference_type>(i)];
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
	for(std::size_t i=0;i<n;++i){
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
	for(std::size_t i=0;i<n;++i){
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
	for(std::size_t i=0;i<n;++i){
#if !MINI_MP_DETAIL_USE_MSVC_INTRIN
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(carry);
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
#else
		const u128 p=mul_u64(ap[i],v);
		std::uint64_t out=0;
		const std::uint64_t c=addc_u64(0,p.lo,carry,&out);
		rp[i]=out;
		std::uint64_t hi=0;
		const std::uint64_t c2=addc_u64(0,p.hi,c,&hi);
		MINI_MP_ASSERT(c2==0);
		carry=hi;
#endif
	}
	return carry;
}

inline std::uint64_t am_1n(std::uint64_t*rp,const std::uint64_t*ap,
								  std::size_t n,std::uint64_t v){
	std::uint64_t carry=0;
	for(std::size_t i=0;i<n;++i){
#if !MINI_MP_DETAIL_USE_MSVC_INTRIN
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(rp[i])+
									 static_cast<unsigned __int128>(carry);
		rp[i]=static_cast<std::uint64_t>(prod);
		carry=static_cast<std::uint64_t>(prod>>64);
#else
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
#endif
	}
	return carry;
}

inline std::uint64_t sm_1n(std::uint64_t*rp,const std::uint64_t*ap,
								  std::size_t n,std::uint64_t v){
	std::uint64_t borrow=0;
	for(std::size_t i=0;i<n;++i){
#if !MINI_MP_DETAIL_USE_MSVC_INTRIN
		const unsigned __int128 prod=static_cast<unsigned __int128>(ap[i])*v+
									 static_cast<unsigned __int128>(borrow);
		const std::uint64_t lo=static_cast<std::uint64_t>(prod);
		const std::uint64_t hi=static_cast<std::uint64_t>(prod>>64);
		const std::uint64_t cur=rp[i];
		const std::uint64_t out=cur-lo;
		const std::uint64_t b=(cur<lo)?1u:0u;
		rp[i]=out;
		borrow=hi+b;
#else
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
#endif
	}
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
		for(std::size_t i=0;i<out_size;++i){
			x[i]=x[i+limb_shift];
		}
		x.resize(out_size);
	}

	if(bit_shift!=0){
		std::uint64_t carry=0;
		for(std::size_t src=x.size();src>0;--src){
			const std::uint64_t cur=x[src-1];
			x[src-1]=(cur>>bit_shift)|carry;
			carry=cur<<(64u-bit_shift);
		}
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
#if !MINI_MP_DETAIL_USE_MSVC_INTRIN
		const unsigned __int128 z=
			static_cast<unsigned __int128>(x[i])*
			static_cast<std::uint64_t>(mul)+carry;
		x[i]=static_cast<std::uint64_t>(z);
		carry=static_cast<std::uint64_t>(z>>64);
#else
		const u128 p=mul_u64(x[i],static_cast<std::uint64_t>(mul));
		std::uint64_t lo=0;
		const std::uint64_t c1=addc_u64(0,p.lo,carry,&lo);
		x[i]=lo;

		std::uint64_t hi=0;
		const std::uint64_t c2=addc_u64(0,p.hi,c1,&hi);
		MINI_MP_ASSERT(c2==0);
		carry=hi;
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
#if !MINI_MP_DETAIL_USE_MSVC_INTRIN
		const unsigned __int128 z=
			static_cast<unsigned __int128>(x[i])*mul+carry;
		x[i]=static_cast<std::uint64_t>(z);
		carry=static_cast<std::uint64_t>(z>>64);
#else
		const u128 p=mul_u64(x[i],mul);
		std::uint64_t lo=0;
		const std::uint64_t c1=addc_u64(0,p.lo,carry,&lo);
		x[i]=lo;

		std::uint64_t hi=0;
		const std::uint64_t c2=addc_u64(0,p.hi,c1,&hi);
		MINI_MP_ASSERT(c2==0);
		carry=hi;
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

inline bool prs_d10_chunks(std::string_view digits,
						   std::vector<std::uint64_t>&chunks){
	chunks.clear();
	if(digits.empty())
		return false;

	auto prs_chunk=[&](std::size_t pos,std::size_t len,
					   std::uint64_t*v)->bool{
		std::uint64_t out_v=0;
		for(std::size_t i=0;i<len;++i){
			const char ch=digits[pos+i];
			if(ch<'0'||ch>'9')
				return false;
			out_v=out_v*10u+
				  static_cast<std::uint64_t>(ch-'0');
		}
		*v=out_v;
		return true;
	};

	chunks.reserve((digits.size()+kD10Len-1u)/kD10Len);
	std::size_t pos=0;
	std::size_t first_len=digits.size()%kD10Len;
	if(first_len==0)
		first_len=kD10Len;

	std::uint64_t v=0;
	if(!prs_chunk(pos,first_len,&v))
		return false;
	chunks.push_back(v);
	pos+=first_len;

	while(pos<digits.size()){
		if(!prs_chunk(pos,kD10Len,&v))
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
#if MINI_MP_DETAIL_USE_MSVC_INTRIN
	muladd_lb(x,kD10Base,hi);
	muladd_lb(x,kD10Base,lo);
#else
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
#endif
}

inline bool prs_d10_seq(std::string_view digits,limbs_t&out){
	out.clear();
	if(digits.empty())
		return false;

	auto prs_chunk=[&](std::size_t pos,std::size_t len,
					   std::uint64_t*v)->bool{
		std::uint64_t out_v=0;
		for(std::size_t i=0;i<len;++i){
			const char ch=digits[pos+i];
			if(ch<'0'||ch>'9')
				return false;
			out_v=out_v*10u+
				  static_cast<std::uint64_t>(ch-'0');
		}
		*v=out_v;
		return true;
	};

	out.reserve((digits.size()*3402u)/(64u*1024u)+2u);
	const std::size_t chunk_count=(digits.size()+kD10Len-1u)/kD10Len;
	std::size_t pos=0;
	std::size_t first_len=digits.size()%kD10Len;
	if(first_len==0)
		first_len=kD10Len;

	std::uint64_t v=0;
	if(!prs_chunk(pos,first_len,&v))
		return false;
	pos+=first_len;
	if((chunk_count&1u)!=0u){
		if(v!=0)
			out.push_back(v);
	}else{
		std::uint64_t lo=0;
		if(!prs_chunk(pos,kD10Len,&lo))
			return false;
		pos+=kD10Len;
		set_u128(out,d10_pair_val(v,lo));
	}

	while(pos<digits.size()){
		std::uint64_t hi=0;
		std::uint64_t lo=0;
		if(!prs_chunk(pos,kD10Len,&hi))
			return false;
		pos+=kD10Len;
		if(!prs_chunk(pos,kD10Len,&lo))
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
	const u128 p=mul_u64(a,b);
	std::uint64_t rem=0;
	(void)udiv128(p.hi,p.lo,mod,&rem);
	return rem;
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

	limbs_t out(un+vn,0);
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
	out.assign(need,0);
	out[un]=mul_1n(out.data(),up->data(),un,(*vp)[0]);
	for(std::size_t j=1;j<vn;++j){
		const std::uint64_t cy=am_1n(
			out.data()+static_cast<std::ptrdiff_t>(j),up->data(),un,(*vp)[j]);
		out[j+un]=cy;
	}
	out.resize(lb_nz(out.data(),need));
}

inline limbs_t sqrsbk_ab(const limbs_t&x){
	if(x.empty())
		return {};
	return mulsbk_ab(x,x);
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

inline constexpr std::size_t kKarRecB=24;

inline std::size_t tun_krec() noexcept;
inline std::size_t tun_kar() noexcept;
inline std::size_t tun_kar_imb() noexcept;
inline std::size_t tun_ntt_imb() noexcept;
inline std::size_t tun_bz_min() noexcept;
inline std::size_t tun_bz_chunk() noexcept;
inline std::size_t tun_prod_leaf() noexcept;
inline std::size_t tun_pow_w5() noexcept;
inline std::size_t tun_pow_w6() noexcept;
inline void ensure_at();

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

	const limbs_t z0=mulkar_rc(a0,b0,rec_b);
	const limbs_t z2=mulkar_rc(a1,b1,rec_b);
	const limbs_t a01=add_abs(a0,a1);
	const limbs_t b01=add_abs(b0,b1);

	limbs_t z1=mulkar_rc(a01,b01,rec_b);
	const limbs_t z0z2=add_abs(z0,z2);
	if(cmp_abs(z1,z0z2)<0){
		
		return mulsbk_ab(a,b);
	}
	z1=sub_abs(z1,z0z2);

	limbs_t out=z0;
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2*m);
	trim_lz(out);
	return out;
}

inline limbs_t mulkar_ab(const limbs_t&a,const limbs_t&b){
	if(a.empty()||b.empty())
		return {};
	return mulkar_rc(a,b,tun_krec());
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

	const limbs_t z0=sqrkar_rc(x0,rec_b);
	const limbs_t z2=sqrkar_rc(x1,rec_b);
	const limbs_t x01=add_abs(x0,x1);

	limbs_t z1=sqrkar_rc(x01,rec_b);
	const limbs_t z0z2=add_abs(z0,z2);
	if(cmp_abs(z1,z0z2)<0)
		return sqrsbk_ab(x);
	z1=sub_abs(z1,z0z2);

	limbs_t out=z0;
	add_sh_ip(out,z1,m);
	add_sh_ip(out,z2,2*m);
	trim_lz(out);
	return out;
}

inline limbs_t sqrkar_ab(const limbs_t&x){
	if(x.empty())
		return {};
	return sqrkar_rc(x,tun_krec());
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
	if(cmp_abs(u,v)<0)
		return;

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

inline limbs_t modk_absl(const limbs_t&u,const limbs_t&v){
	MINI_MP_ASSERT(!v.empty());
	if(u.empty())
		return {};
	if(cmp_abs(u,v)<0)
		return u;

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

inline void cpylo_in(const limbs_t&src,std::size_t count,
								limbs_t&out){
	const std::size_t n=std::min(count,src.size());
	out.resize(n);
	for(std::size_t i=0;i<n;++i){
		out[i]=src[i];
	}
	trim_lz(out);
}

inline void cpyhi_in(const limbs_t&src,std::size_t start,
								 limbs_t&out){
	if(start>=src.size()){
		out.clear();
		return;
	}
	const std::size_t n=src.size()-start;
	out.resize(n);
	for(std::size_t i=0;i<n;++i){
		out[i]=src[start+i];
	}
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
		std::uint64_t carry=0;
		for(std::size_t j=0;j<k;++j){
			const std::size_t idx=i+j;
			const u128 p=mul_u64(m,ctx.mod[j]);

			std::uint64_t t0=0;
			const std::uint64_t c1=addc_u64(0,t[idx],p.lo,&t0);
			std::uint64_t t1=0;
			const std::uint64_t c2=addc_u64(0,t0,carry,&t1);
			t[idx]=t1;

			std::uint64_t hi0=0;
			const std::uint64_t c3=addc_u64(0,p.hi,c1,&hi0);
			std::uint64_t hi1=0;
			const std::uint64_t c4=addc_u64(0,hi0,c2,&hi1);
			MINI_MP_ASSERT((c3&c4)==0);
			carry=hi1;
		}
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
inline constexpr std::size_t kHlmDef=4;
inline constexpr std::size_t kHlrDef=0;
inline constexpr std::size_t kBzDivMnDef=128;
inline constexpr std::size_t kBzChunkDef=3;
inline constexpr std::size_t kGcdSmMxDef=64;
inline constexpr std::size_t kGcdLgMxDef=8;
inline constexpr std::size_t kProdLeafDef=24;
inline constexpr std::size_t kFacTreeDef=192;
inline constexpr std::size_t kBinomTreeDef=160;
inline constexpr std::size_t kPowW5Def=384;
inline constexpr std::size_t kPowW6Def=1536;
inline constexpr std::size_t kD10DcDef=kNttOff;
inline constexpr std::size_t kD10PrsDef=kNttOff;
inline constexpr std::size_t kT3Def=kNttOff;

inline std::size_t tun_ntt() noexcept;
inline std::size_t tun_ntt_sq() noexcept;
inline std::size_t tun_ntt_bits() noexcept;
inline std::size_t tun_krec() noexcept;
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

namespace detail{
inline BigInt factorial_loop(std::uint64_t n);
inline BigInt factorial_tree(std::uint64_t n,std::size_t leaf);
inline BigInt binomial_loop(std::uint64_t n,std::uint64_t k);
inline BigInt binomial_tree(std::uint64_t n,std::uint64_t k,
							std::size_t leaf);
inline std::uint64_t bench_sqr_mul(const BigInt&a,int loops,bool use_ntt);
}

struct ExtGcdRes;

inline BigInt mul_sbk(const BigInt&a,const BigInt&b);
inline BigInt mul_kar(const BigInt&a,const BigInt&b);
inline BigInt mul_t3(const BigInt&a,const BigInt&b);
inline BigInt mul_ntt(const BigInt&a,const BigInt&b);
inline BigInt mul_disp(const BigInt&a,const BigInt&b);
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
inline std::pair<BigInt,BigInt> sqrtrem(const BigInt&n);
inline BigInt isqrt(const BigInt&n);
inline std::pair<BigInt,BigInt> rootrem(const BigInt&n,std::uint32_t k);
inline BigInt iroot(const BigInt&n,std::uint32_t k);
inline bool is_ppow(const BigInt&n);
inline bool is_square(const BigInt&n);
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
	bool test_bit(std::size_t bit_index) const noexcept{
		return detail::test_bit(limbs_,bit_index);
	}

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
	friend BigInt divexact(const BigInt&a,const BigInt&b);
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
		out.normalize();
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
	out.normalize();
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
	out.normalize();
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

	const BigInt w0=mul_kar(a0,b0);
	const BigInt w1=mul_kar(ax1,bx1);
	const BigInt wm1=mul_kar(axm1,bxm1);
	const BigInt wm2=mul_kar(axm2,bxm2);
	const BigInt wi=mul_kar(a2,b2);

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

	if(detail::cmp_abs(ua,vb)<0){
		return {BigInt(),a};
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

	if(detail::cmp_abs(ua,vb)<0){
		return BigInt();
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

	if(detail::cmp_abs(ua,vb)<0){
		return a;
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
	std::size_t kar_th=kKarTh;
	std::size_t kar_imb=kKarImb;
	std::size_t ntt_imb=kNttImb;
	std::size_t hl_min=kHlmDef;
	std::size_t hl_rnd=kHlrDef;
	std::size_t gcd_sm=kGcdSmMxDef;
	std::size_t gcd_lg=kGcdLgMxDef;
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

inline std::atomic<bool>&at_done() noexcept{
	static std::atomic<bool> done{
#if MINI_MP_ENABLE_AUTOTUNE
		false
#else
		true
#endif
	};
	return done;
}

inline bool&at_busy() noexcept{
	static thread_local bool busy=false;
	return busy;
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

inline void at_impl(){
	ATState tuned{};
	tuned.ntt_th=kNttOff;
	tuned.ntt_sq_th=kNttOff;
	tuned.ntt_bits=kNttBitsDef;
	tuned.kar_rec=kKarRecB;
	tuned.kar_th=kKarTh;
	tuned.kar_imb=kKarImb;
	tuned.ntt_imb=kNttImb;
	tuned.hl_min=kHlmDef;
	tuned.hl_rnd=kHlrDef;
	tuned.gcd_sm=kGcdSmMxDef;
	tuned.gcd_lg=kGcdLgMxDef;
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
	at_state()=tuned;

	{
		std::uint64_t seed=0x27bb2ee687b0b0fdULL;
		constexpr std::array<std::size_t,5> kRecCand={
			16u,24u,32u,40u,48u};
		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_rec=tuned.kar_rec;
		for(std::size_t rec : kRecCand){
			at_state().kar_rec=rec;
			std::uint64_t cost=0;
			std::uint64_t s=seed;
			for(std::size_t n : {64u,128u,256u}){
				const BigInt a=mk_bench(s,n);
				const BigInt b=mk_bench(s,n);
				const int loops=(n<=64u)?5:(n<=128u)?3:1;
				cost+=bench_mul(a,b,loops,false);
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
		constexpr std::array<std::size_t,11> kKarCand={
			64u,96u,128u,160u,192u,224u,256u,320u,384u,512u,640u};
		std::array<std::uint64_t,kKarCand.size()> school_ns{};
		std::array<std::uint64_t,kKarCand.size()> kar_ns{};

		for(std::size_t idx=0;idx<kKarCand.size();++idx){
			const std::size_t n=kKarCand[idx];
			const int loops=(n<=96u)?6:(n<=192u)?4:(n<=320u)?3:(n<=512u)?2:1;
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
			school_ns[idx]=(t_schsum+ops/2u)/ops;
			kar_ns[idx]=(t_kar_sum+ops/2u)/ops;
		}

		std::uint64_t best_cost=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_th=tuned.kar_th;
		for(std::size_t threshold : kKarCand){
			std::uint64_t cost=0;
			for(std::size_t i=0;i<kKarCand.size();++i){
				cost+=(kKarCand[i]>=threshold)?kar_ns[i]:school_ns[i];
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
		std::size_t best_th=kNttOff;
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
		constexpr std::array<std::size_t,5> kSizes={16u,24u,40u,64u,96u};
		samples.reserve(kSizes.size()*2);
		for(std::size_t limbs : kSizes){
			for(int i=0;i<2;++i){
				BigInt a=mk_bench(seed,limbs);
				BigInt b=mk_bench(seed,limbs);
				if(a==b)
					b+=BigInt(1);
				if(a<b)
					std::swap(a,b);
				samples.emplace_back(std::move(a),std::move(b));
			}
		}

		constexpr std::array<std::size_t,4> kMinCand={3u,4u,6u,8u};
		constexpr std::array<std::size_t,5> kRndCand={0u,4u,6u,8u,10u};

		std::uint64_t best_t=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_min=tuned.hl_min;
		std::size_t best_rnd=tuned.hl_rnd;

		for(std::size_t min_limbs : kMinCand){
			for(std::size_t rounds : kRndCand){
				const std::uint64_t t=
					bench_gcd(samples,min_limbs,rounds);
				if(t<best_t){
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

		constexpr std::array<std::size_t,4> kGcdSmCand={32u,48u,64u,96u};
		constexpr std::array<std::size_t,4> kGcdLgCand={4u,8u,12u,16u};
		best_t=std::numeric_limits<std::uint64_t>::max();
		std::size_t best_sm=tuned.gcd_sm;
		std::size_t best_lg=tuned.gcd_lg;
		for(std::size_t sm : kGcdSmCand){
			for(std::size_t lg : kGcdLgCand){
				at_state().gcd_sm=sm;
				at_state().gcd_lg=lg;
				const std::uint64_t t=
					bench_gcd(samples,tuned.hl_min,tuned.hl_rnd);
				if(t<best_t){
					best_t=t;
					best_sm=sm;
					best_lg=lg;
				}
			}
		}
		tuned.gcd_sm=best_sm;
		tuned.gcd_lg=best_lg;
		at_state().gcd_sm=best_sm;
		at_state().gcd_lg=best_lg;
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
		struct D10Smp{
			limbs_t x;
			std::string s;
			std::size_t n=0;
			std::uint64_t base_ns=0;
			std::array<std::uint64_t,kD10Cand.size()> dc_ns{};
			std::uint64_t prs_base_ns=0;
			std::array<std::uint64_t,kD10Cand.size()> prs_dc_ns{};
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

inline void ensure_at(){
#if MINI_MP_ENABLE_AUTOTUNE
	if(at_done().load(std::memory_order_acquire))
		return;
	if(at_busy())
		return;
	static std::once_flag once;
	std::call_once(once,[](){
		at_busy()=true;
		at_impl();
		at_busy()=false;
		at_done().store(true,std::memory_order_release);
	});
#endif
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

inline std::size_t tun_kar() noexcept{
	return at_state().kar_th;
}

inline std::size_t tun_kar_imb() noexcept{
	return at_state().kar_imb;
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
	detail::limbs_t qb;
	if(hl_rnd!=0){
		qb.reserve(b.limbs_.size()+1);
	}

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

		bool did_hleh=false;
		if(hl_rnd!=0&&a.limbs_.size()==b.limbs_.size()&&
		   b.limbs_.size()>=hl_min){
			for(std::size_t step=0;step<hl_rnd;++step){
				if(b.is_zero())
					break;
				if(a.limbs_.size()!=b.limbs_.size()||
				   b.limbs_.size()<hl_min){
					break;
				}

				std::uint64_t q=
					detail::leh_qest(a.limbs_,b.limbs_);
				if(q==0)
					q=1;

				if(q>1){
					detail::mulbl_in(b.limbs_,q,qb);
					std::size_t corr=0;
					while(q>1&&detail::cmp_abs(a.limbs_,qb)<0&&corr<4){
						--q;
						if(q>1){
							detail::sub_abs_ip(qb,b.limbs_);
						}
						++corr;
					}
					if(q>1&&detail::cmp_abs(a.limbs_,qb)<0){
						q=1;
					}
				}

				if(q==1){
					detail::sub_abs_ip(a.limbs_,b.limbs_);
				}else{
					detail::sub_abs_ip(a.limbs_,qb);
				}
				a.sign_=a.limbs_.empty()?0:1;
				std::swap(a,b);
				if(detail::cmp_abs(a.limbs_,b.limbs_)<0)
					std::swap(a,b);
				did_hleh=true;
			}
			if(did_hleh)
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
	base%=mod;
	if(base.sign()<0)
		base+=mod;
	if(mod.limbs_.size()==1&&mod.limbs_[0]==1u)
		return BigInt();

	BigInt one_mod(1);
	one_mod%=mod;
	if(one_mod.sign()<0)
		one_mod+=mod;
	if(exp.is_zero())
		return one_mod;

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
			detail::limbs_t one{1};
			detail::limbs_t base_bar;
			detail::limbs_t one_bar;
			detail::mont_to(base.limbs_,ctx,base_bar);
			detail::mont_to(one,ctx,one_bar);

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

inline std::pair<BigInt,BigInt> sqrtrem(const BigInt&n){
	if(n.sign()<0){
		detail::throw_dom("sqrtrem: negative input");
	}
	if(n.is_zero())
		return {BigInt(),BigInt()};
	if(n==BigInt(1))
		return {BigInt(1),BigInt()};

	const std::size_t init_bits=(n.bit_length()+1)/2;
	BigInt x=BigInt(1)<<init_bits;

	for(;;){
		BigInt y=(x+(n/x))>>1;
		if(y>=x)
			break;
		x=std::move(y);
	}

	BigInt x2=sqr_disp(x);
	while(x2>n){
		x-=BigInt(1);
		x2=sqr_disp(x);
	}
	for(;;){
		BigInt xp1=x+BigInt(1);
		BigInt xp1_sq=sqr_disp(xp1);
		if(xp1_sq>n)
			break;
		x=std::move(xp1);
		x2=std::move(xp1_sq);
	}

	return {x,n-x2};
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
	return sqrtrem(n).second.is_zero();
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

	BigInt d=n-BigInt(1);
	std::size_t s=0;
	while(d.is_even()){
		d>>=1;
		++s;
	}

	static constexpr std::uint64_t kBases[]={
		2ULL, 3ULL, 5ULL, 7ULL, 11ULL,13ULL,17ULL,19ULL,
		23ULL,29ULL,31ULL,37ULL,41ULL,43ULL,47ULL,53ULL};
	if(rounds<=0)
		rounds=1;
	const std::size_t max_rounds=sizeof(kBases)/sizeof(kBases[0]);
	const std::size_t use_rounds=static_cast<std::size_t>(
		std::min<int>(rounds,static_cast<int>(max_rounds)));

	const BigInt n_minus_1=n-BigInt(1);
	for(std::size_t i=0;i<use_rounds;++i){
		BigInt a=BigInt::from_u64(kBases[i]);
		if(a>=n)
			continue;

		BigInt x=modpow(a,d,n);
		if(x==BigInt(1)||x==n_minus_1)
			continue;

		bool witness=true;
		for(std::size_t r=1;r<s;++r){
			x=sqr_disp(x)%n;
			if(x==n_minus_1){
				witness=false;
				break;
			}
		}
		if(witness)
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





class BigRat{
  public:
	BigRat() : num_(0),den_(1){}
	BigRat(std::int64_t v) : num_(v),den_(1){}
	explicit BigRat(BigInt n) : num_(std::move(n)),den_(1){}

	BigRat(BigInt n,BigInt d) : num_(std::move(n)),den_(std::move(d)){
		canon();
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

	BigRat&operator+=(const BigRat&rhs){
		if(rhs.num_.is_zero())
			return *this;
		if(num_.is_zero()){
			*this=rhs;
			return *this;
		}

		
		
		
		
		
		const BigInt g=gcd(den_,rhs.den_);
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
		BigRat tmp=rhs;
		tmp.num_=-tmp.num_;
		return (*this+=tmp);
	}

	BigRat&operator*=(const BigRat&rhs){
		if(num_.is_zero()||rhs.num_.is_zero()){
			num_=BigInt();
			den_=BigInt(1);
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

	friend bool operator==(const BigRat&a,const BigRat&b){
		return a.num_==b.num_&&a.den_==b.den_;
	}
	friend bool operator!=(const BigRat&a,const BigRat&b){ return !(a==b); }
	friend bool operator<(const BigRat&a,const BigRat&b){
		return (a.num_*b.den_)<(b.num_*a.den_);
	}
	friend bool operator>(const BigRat&a,const BigRat&b){ return b<a; }
	friend bool operator<=(const BigRat&a,const BigRat&b){ return !(b<a); }
	friend bool operator>=(const BigRat&a,const BigRat&b){ return !(a<b); }

  private:
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





