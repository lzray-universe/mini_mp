// SPDX-License-Identifier: MIT
//
// Standalone coverage and timing test for mini_mp.hpp.
//
// Default build:
//   g++ -O2 -std=c++20 -DNDEBUG -DMINI_MP_ENABLE_SIMD=0 -DMINI_MP_ENABLE_NTT=0 test.cpp -o test
//
// Optional old/new cross-check in one translation unit:
//   g++ -O2 -std=c++20 -DNDEBUG -DMINI_MP_TEST_OLD_HEADER=\"mini_mp_old.hpp\" test.cpp -o test
//
// Runtime:
//   ./test
//   ./test --quick
//   ./test --cases 10000 --no-bench
//   ./test --bench-only

#ifndef MINI_MP_TEST_NEW_HEADER
#define MINI_MP_TEST_NEW_HEADER "mini_mp.hpp"
#endif

#define mini_mp mini_mp_new
#include MINI_MP_TEST_NEW_HEADER
#undef mini_mp

#ifdef MINI_MP_TEST_OLD_HEADER
#undef MINI_MP_HPP_INCLUDED
#define mini_mp mini_mp_old
#include MINI_MP_TEST_OLD_HEADER
#undef mini_mp
#define MINI_MP_TEST_HAVE_OLD 1
#else
#define MINI_MP_TEST_HAVE_OLD 0
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mp=mini_mp_new;

struct Options{
	std::size_t cases=2500;
	std::uint64_t seed=0x9e3779b97f4a7c15ull;
	bool bench=true;
	bool bench_only=false;
	bool quick=false;
	bool verbose=false;
};

struct TestState{
	std::uint64_t checks=0;
	std::uint64_t failed=0;
	std::vector<std::string> first_failures;

	void check(bool ok,const char*expr,const char*file,int line){
		++checks;
		if(ok)
			return;
		++failed;
		if(first_failures.size()<32u){
			std::ostringstream os;
			os<<file<<":"<<line<<" check failed: "<<expr;
			first_failures.push_back(os.str());
		}
	}

	template<class A,class B>
	void equal(const A&a,const B&b,const char*ea,const char*eb,
			   const char*file,int line){
		++checks;
		if(a==b)
			return;
		++failed;
		if(first_failures.size()<32u){
			std::ostringstream os;
			os<<file<<":"<<line<<" equality failed: "<<ea<<" == "<<eb;
			first_failures.push_back(os.str());
		}
	}
};

static TestState g_test;

#define CHECK(expr) g_test.check(static_cast<bool>(expr),#expr,__FILE__,__LINE__)
#define CHECK_EQ(a,b) g_test.equal((a),(b),#a,#b,__FILE__,__LINE__)

#ifndef MINI_MP_NO_EXCEPT
#define CHECK_THROWS(expr)                                                       \
	do{                                                                          \
		bool mini_mp_test_threw=false;                                           \
		try{                                                                     \
			(void)(expr);                                                        \
		}catch(const std::exception&){                                           \
			mini_mp_test_threw=true;                                             \
		}                                                                        \
		g_test.check(mini_mp_test_threw,#expr " throws",__FILE__,__LINE__);      \
	}while(false)
#else
#define CHECK_THROWS(expr) do{ (void)sizeof(expr); }while(false)
#endif

struct Rng{
	using result_type=std::uint64_t;
	std::uint64_t s;

	explicit Rng(std::uint64_t seed) : s(seed?seed:0x123456789abcdef0ull){}
	static constexpr result_type min(){ return 0; }
	static constexpr result_type max(){ return std::numeric_limits<result_type>::max(); }

	std::uint64_t operator()(){
		std::uint64_t x=s;
		x^=x>>12;
		x^=x<<25;
		x^=x>>27;
		s=x;
		return x*0x2545f4914f6cdd1dull;
	}

	std::uint64_t below(std::uint64_t n){
		return n==0?0:(operator()()%n);
	}

	bool bit(){ return (operator()()&1u)!=0; }
};

using BI=mp::BigInt;
using BR=mp::BigRat;
using BF=mp::BigFloat;
using BC=mp::BigComplex;
using FR=mp::FloatRnd;

static volatile std::uint64_t g_sink=0;

BI bi_from_i64(std::int64_t v){
	return BI(v);
}

BI random_bi(Rng&rng,std::size_t bits,bool signed_value=true){
	BI x=BI::rand_bits(bits,rng);
	if(bits!=0&&rng.bit())
		x|=(BI(1)<<(bits-1u));
	if(signed_value&&rng.bit())
		x=-x;
	return x;
}

BI positive_random_bi(Rng&rng,std::size_t bits){
	BI x=random_bi(rng,bits,false);
	if(x.is_zero())
		x=BI(1);
	return x;
}

BI positive_mod(BI x,const BI&m){
	x%=m;
	if(x.sign()<0)
		x+=m;
	return x;
}

bool is_pow2_positive(const BI&x){
	return x.sign()>0&&(x&(x-BI(1))).is_zero();
}

std::string s10(const BI&x){ return x.to_string(10); }
std::string s10(const BR&x){ return x.to_string(10); }
std::string s10(const BF&x){ return x.to_string(10,30); }

bool near_double(double actual,double expected,double rel=3e-11,double abs=3e-13){
	if(std::isnan(expected))
		return std::isnan(actual);
	if(std::isinf(expected))
		return std::isinf(actual)&&std::signbit(actual)==std::signbit(expected);
	const double diff=std::fabs(actual-expected);
	const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
	return diff<=abs||diff<=rel*scale;
}

void check_near(double actual,double expected,const char*label,
				double rel=3e-11,double abs=3e-13){
	++g_test.checks;
	if(near_double(actual,expected,rel,abs))
		return;
	++g_test.failed;
	if(g_test.first_failures.size()<32u){
		std::ostringstream os;
		os<<label<<" not near: "<<std::setprecision(17)<<actual
		  <<" vs "<<expected;
		g_test.first_failures.push_back(os.str());
	}
}

std::uint64_t powmod_u64(std::uint64_t a,std::uint64_t e,std::uint64_t m){
	std::uint64_t r=1%m;
	a%=m;
	while(e!=0){
		if(e&1u)
			r=mp::detail::mulmod64(r,a,m);
		e>>=1u;
		if(e)
			a=mp::detail::mulmod64(a,a,m);
	}
	return r;
}

std::uint64_t hash_bi(const BI&x){
	std::uint64_t h=static_cast<std::uint64_t>(x.bit_length()*1315423911u);
	h^=static_cast<std::uint64_t>(x.sign()+17)*0x9e3779b185ebca87ull;
	if(!x.is_zero())
		h^=x.abs().mod_u32(1000000007u);
	return h;
}

std::uint64_t hash_bf(const BF&x){
	std::uint64_t h=static_cast<std::uint64_t>(x.precision()*65537u);
	h^=static_cast<std::uint64_t>(x.sign()+19)*0x94d049bb133111ebull;
	if(x.is_finite()&&!x.is_zero()){
		h^=static_cast<std::uint64_t>(x.mantissa().bit_length());
		h^=static_cast<std::uint64_t>(x.exponent())*0xbf58476d1ce4e5b9ull;
	}
	return h;
}

std::uint64_t hash_br(const BR&x){
	return hash_bi(x.num())^(hash_bi(x.den())<<1);
}

std::uint64_t hash_bc(const BC&z){
	return hash_bf(z.real())^(hash_bf(z.imag())<<1);
}

std::uint64_t hash_poly(const mp::BigNT::ModPoly&p){
	std::uint64_t h=p.modulus()*0x9e3779b185ebca87ull;
	h^=static_cast<std::uint64_t>(p.size())*0xbf58476d1ce4e5b9ull;
	for(std::size_t i=0;i<p.coeff().size();i+=std::max<std::size_t>(1,p.coeff().size()/8u))
		h^=p.coeff()[i]+0x94d049bb133111ebull+(h<<6)+(h>>2);
	return h;
}

std::int64_t floor_div_i64(std::int64_t a,std::int64_t b){
	std::int64_t q=a/b;
	std::int64_t r=a%b;
	if(r!=0&&((r>0)!=(b>0)))
		--q;
	return q;
}

std::int64_t ceil_div_i64(std::int64_t a,std::int64_t b){
	std::int64_t q=a/b;
	std::int64_t r=a%b;
	if(r!=0&&((r>0)==(b>0)))
		++q;
	return q;
}

std::size_t bit_length_u64(std::uint64_t x){
	std::size_t n=0;
	while(x!=0){
		++n;
		x>>=1u;
	}
	return n;
}

std::size_t ctz_u64(std::uint64_t x){
	if(x==0)
		return BI::npos;
	std::size_t n=0;
	while((x&1u)==0){
		++n;
		x>>=1u;
	}
	return n;
}

std::size_t popcount_u64(std::uint64_t x){
	std::size_t n=0;
	while(x!=0){
		n+=x&1u;
		x>>=1u;
	}
	return n;
}

void test_bigint_parse_convert(){
	CHECK(BI().is_zero());
	CHECK(BI(1).is_one());
	CHECK(BI(-1).is_neg());
	CHECK_EQ(BI::from_u64(std::numeric_limits<std::uint64_t>::max()).to_string(16),
			 std::string("ffffffffffffffff"));
	CHECK_EQ(BI(std::numeric_limits<std::int64_t>::min()).to_string(10),
			 std::string("-9223372036854775808"));
	CHECK_EQ(BI::parse("  +0xFF  ",0).to_string(10),std::string("255"));
	CHECK_EQ(BI::parse("  -0Xff  ",16).to_string(10),std::string("-255"));
	CHECK_EQ(BI::parse("zzzz",36).to_string(10),std::string("1679615"));

	for(int base=2;base<=36;++base){
		for(std::int64_t v=-1024;v<=1024;++v){
			BI x(v);
			std::string text=x.to_string(base);
			BI y=BI::parse(text,base);
			CHECK_EQ(y.to_string(10),std::to_string(v));
			CHECK_EQ(y.to_string(base),text);
		}
	}

	for(std::uint64_t v : {0ull,1ull,2ull,3ull,15ull,16ull,17ull,
						   0xffffffffull,0x100000000ull,
						   std::numeric_limits<std::uint64_t>::max()}){
		BI x=BI::from_u64(v);
		CHECK(x.fits_u64());
		CHECK_EQ(x.to_u64(),v);
		CHECK_EQ(x.bit_length(),bit_length_u64(v));
		CHECK_EQ(x.ctz(),v==0?0u:ctz_u64(v));
		CHECK_EQ(x.popcount(),popcount_u64(v));
		for(std::size_t start=0;start<72;++start){
			std::size_t e1=BI::npos;
			for(std::size_t b=start;b<64;++b){
				if((v>>b)&1ull){
					e1=b;
					break;
				}
			}
			std::size_t e0=start;
			while(e0<64&&((v>>e0)&1ull))
				++e0;
			CHECK_EQ(x.scan1(start),e1);
			CHECK_EQ(x.scan0(start),e0);
		}
	}

	CHECK_THROWS(BI::parse("",10));
	CHECK_THROWS(BI::parse("-",10));
	CHECK_THROWS(BI::parse("102",2));
	CHECK_THROWS(BI::parse("12x",10));
	CHECK_THROWS(BI::parse("10",1));
	CHECK_THROWS(BI::parse("10",37));
	CHECK_THROWS(BI(-1).to_u64());
	CHECK_THROWS((BI(1)<<80).to_u64());
	CHECK_THROWS(BI(10).mod_u32(0));

	std::vector<std::uint8_t> empty;
	CHECK(BI::from_bytes(empty.data(),0).is_zero());
	std::vector<std::uint8_t> bytes={0,1,2,3,4,5,0xff,0};
	for(std::size_t n=0;n<=bytes.size();++n){
		BI msf=BI::from_bytes(bytes.data(),n,true);
		BI lsf=BI::from_bytes(bytes.data(),n,false);
		CHECK_EQ(BI::from_bytes(msf.to_bytes(true).data(),msf.to_bytes(true).size(),true),msf);
		CHECK_EQ(BI::from_bytes(lsf.to_bytes(false).data(),lsf.to_bytes(false).size(),false),lsf);
		BI neg=BI::from_bytes(bytes.data(),n,true,-1);
		CHECK_EQ(neg.abs(),msf);
	}

	const std::vector<std::uint8_t> word_data={
		0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0x10,0x20,0x30,0x40};
	for(std::size_t word_size : {1u,2u,3u,4u,6u}){
		const std::size_t count=word_data.size()/word_size;
		for(int word_order : {1,-1}){
			for(int byte_order : {1,-1}){
				for(std::size_t nails : {0u,1u,3u}){
					if(nails>=word_size*8u)
						continue;
					BI x=BI::import_words(word_data.data(),count,word_size,
										   word_order,byte_order,nails,1);
					std::vector<std::uint8_t> out=
						x.export_words(word_size,word_order,byte_order,nails);
					BI y=BI::import_words(out.data(),out.size()/word_size,
										   word_size,word_order,byte_order,
										   nails,1);
					CHECK_EQ(x,y);
					BI z=BI::import_words(word_data.data(),count,word_size,
										   word_order,byte_order,nails,-1);
					CHECK_EQ(z.abs(),x);
				}
			}
		}
	}

	CHECK_THROWS(BI::from_bytes(nullptr,1,true));
	CHECK_THROWS(BI::import_words(nullptr,1,1));
	CHECK_THROWS(BI::import_words(word_data.data(),1,0));
	CHECK_THROWS(BI::import_words(word_data.data(),1,1,0,1));
	CHECK_THROWS(BI::import_words(word_data.data(),1,1,1,0,8));
	CHECK_THROWS(BI(1).export_words(0));
	CHECK_THROWS(BI(1).export_words(1,0));
	CHECK_THROWS(BI(1).export_words(1,1,0,8));
}

void test_bigint_small_arithmetic(){
	for(std::int64_t a=-150;a<=150;++a){
		for(std::int64_t b=-150;b<=150;++b){
			BI A(a),B(b);
			CHECK_EQ((A+B).to_string(10),std::to_string(a+b));
			CHECK_EQ((A-B).to_string(10),std::to_string(a-b));
			CHECK_EQ((A*B).to_string(10),std::to_string(a*b));
			CHECK_EQ((-A).to_string(10),std::to_string(-a));
			CHECK_EQ((+A).to_string(10),std::to_string(a));
			CHECK_EQ(compare(A,B),(a<b?-1:(a>b?1:0)));
			CHECK_EQ(A==B,a==b);
			CHECK_EQ(A!=B,a!=b);
			CHECK_EQ(A<B,a<b);
			CHECK_EQ(A<=B,a<=b);
			CHECK_EQ(A>B,a>b);
			CHECK_EQ(A>=B,a>=b);
			if(b!=0){
				CHECK_EQ((A/B).to_string(10),std::to_string(a/b));
				CHECK_EQ((A%B).to_string(10),std::to_string(a%b));
				CHECK_EQ(mp::tdiv_q(A,B).to_string(10),std::to_string(a/b));
				CHECK_EQ(mp::tdiv_r(A,B).to_string(10),std::to_string(a%b));
				CHECK_EQ(mp::fdiv_q(A,B).to_string(10),
						 std::to_string(floor_div_i64(a,b)));
				CHECK_EQ(mp::cdiv_q(A,B).to_string(10),
						 std::to_string(ceil_div_i64(a,b)));
			}
		}
	}

	for(std::int64_t a=-256;a<=256;++a){
		BI A(a);
		CHECK_EQ((~A).to_string(10),std::to_string(~a));
		for(std::int64_t b=-256;b<=256;++b){
			BI B(b);
			CHECK_EQ((A&B).to_string(10),std::to_string(a&b));
			CHECK_EQ((A|B).to_string(10),std::to_string(a|b));
			CHECK_EQ((A^B).to_string(10),std::to_string(a^b));
		}
		for(std::size_t sh=0;sh<=12;++sh){
			const std::int64_t l=a<<sh;
			const std::int64_t r=a>>sh;
			CHECK_EQ((A<<sh).to_string(10),std::to_string(l));
			CHECK_EQ((A>>sh).to_string(10),std::to_string(r));
			BI t=A;
			t<<=sh;
			CHECK_EQ(t,A<<sh);
			t=A;
			t>>=sh;
			CHECK_EQ(t,A>>sh);
		}
	}
}

void check_division_invariants(const BI&a,const BI&b){
	auto qr=mp::divmod(a,b);
	CHECK_EQ(qr.first,mp::divk_q(a,b));
	CHECK_EQ(qr.second,mp::divk_r(a,b));
	CHECK_EQ(qr.first,a/b);
	CHECK_EQ(qr.second,a%b);
	CHECK_EQ(qr.first*b+qr.second,a);
	if(!qr.second.is_zero()){
		CHECK(qr.second.sign()==a.sign());
		CHECK(qr.second.abs()<b.abs());
	}

	auto s=mp::dvm_simp(a,b);
	auto k=mp::dvm_knuth(a,b);
	CHECK_EQ(s.first,qr.first);
	CHECK_EQ(s.second,qr.second);
	CHECK_EQ(k.first,qr.first);
	CHECK_EQ(k.second,qr.second);
	if((b.abs()&(b.abs()-BI(1))).is_zero()){
		auto p2=mp::dvm_p2(a,b);
		CHECK_EQ(p2.first,qr.first);
		CHECK_EQ(p2.second,qr.second);
	}

	auto tq=mp::tdiv_qr(a,b);
	auto fq=mp::fdiv_qr(a,b);
	auto cq=mp::cdiv_qr(a,b);
	CHECK_EQ(tq.first,qr.first);
	CHECK_EQ(tq.second,qr.second);
	CHECK_EQ(fq.first*b+fq.second,a);
	CHECK_EQ(cq.first*b+cq.second,a);
	CHECK_EQ(mp::fdiv_q(a,b),fq.first);
	CHECK_EQ(mp::fdiv_r(a,b),fq.second);
	CHECK_EQ(mp::cdiv_q(a,b),cq.first);
	CHECK_EQ(mp::cdiv_r(a,b),cq.second);
	if(!fq.second.is_zero()){
		CHECK(fq.second.sign()==b.sign());
		CHECK(fq.second.abs()<b.abs());
	}
	if(!cq.second.is_zero()){
		CHECK(cq.second.sign()!=b.sign());
		CHECK(cq.second.abs()<b.abs());
	}
}

void test_bigint_random_arithmetic(const Options&opt){
	Rng rng(opt.seed^0x72616e646f6d5f31ull);
	const std::size_t cases=opt.cases;
	for(std::size_t i=0;i<cases;++i){
		const std::size_t bits_a=static_cast<std::size_t>(rng.below(1536));
		const std::size_t bits_b=static_cast<std::size_t>(rng.below(1024));
		BI a=random_bi(rng,bits_a,true);
		BI b=random_bi(rng,bits_b,true);
		if(b.is_zero())
			b=BI(1);

		CHECK_EQ((a+b)-b,a);
		CHECK_EQ((a-b)+b,a);
		CHECK_EQ(a+b,b+a);
		CHECK_EQ(a*b,b*a);
		CHECK_EQ((a*b)/b,a);
		CHECK_EQ(mp::divexact(a*b,b),a);
		CHECK_EQ(mp::sqr_disp(a),a*a);

		if(std::min(a.abs().bit_length(),b.abs().bit_length())<=2048u){
			CHECK_EQ(mp::mul_sbk(a,b),a*b);
			CHECK_EQ(mp::mul_kar(a,b),a*b);
			CHECK_EQ(mp::mul_t3(a,b),a*b);
		}
#if MINI_MP_ENABLE_NTT
		if(std::min(a.abs().bit_length(),b.abs().bit_length())<=4096u)
			CHECK_EQ(mp::mul_ntt(a,b),a*b);
#endif

		check_division_invariants(a,b);
		const std::size_t sh=static_cast<std::size_t>(rng.below(320));
		CHECK_EQ(((a<<sh)>>sh),a);
		CHECK_EQ((a&b)|(a^b),a|b);
		CHECK_EQ((a|b)&(~a),b&(~a));
		CHECK_EQ((~a),-(a+BI(1)));
		CHECK_EQ(a.abs().sign(),a.is_zero()?0:1);
		CHECK_EQ(a.neg(),-a);
		CHECK_EQ(a.mod_u32(1000003u),
				 static_cast<std::uint32_t>(((a%BI(1000003)).sign()<0?
					 (a%BI(1000003))+BI(1000003):(a%BI(1000003))).to_u64()));
	}

	for(std::size_t bits : {0u,1u,63u,64u,65u,127u,128u,129u,255u,256u,
							511u,512u,1024u,2048u,4096u}){
		BI a=positive_random_bi(rng,bits);
		BI b=positive_random_bi(rng,bits/2u+1u);
		CHECK_EQ((a+b)-a,b);
		CHECK_EQ((a*b)/a,b);
		check_division_invariants(a,b);
	}
}

void test_number_theory(const Options&opt){
	const std::int64_t small_lim=opt.quick?80:220;
	for(std::int64_t a=-small_lim;a<=small_lim;++a){
		for(std::int64_t b=-small_lim;b<=small_lim;++b){
			BI A(a),B(b);
			const std::uint64_t aa=static_cast<std::uint64_t>(a<0?-a:a);
			const std::uint64_t bb=static_cast<std::uint64_t>(b<0?-b:b);
			const std::uint64_t g=std::gcd(aa,bb);
			CHECK_EQ(mp::gcd(A,B).to_string(10),std::to_string(g));
			if(a!=0||b!=0){
				auto eg=mp::extgcd(A,B);
				CHECK_EQ(eg.g,mp::gcd(A,B));
				CHECK_EQ(A*eg.x+B*eg.y,eg.g);
				BI s,t;
				BI gg=mp::gcdext(&s,&t,A,B);
				CHECK_EQ(gg,eg.g);
				CHECK_EQ(A*s+B*t,gg);
			}
			if(a!=0&&b!=0){
				const std::uint64_t l=aa/std::gcd(aa,bb)*bb;
				CHECK_EQ(mp::lcm(A,B).to_string(10),std::to_string(l));
			}
		}
	}

	for(std::uint64_t a=0;a<24;++a){
		for(std::uint64_t e=0;e<13;++e){
			BI p=mp::pow(BI::from_u64(a),e);
			std::uint64_t expected=1;
			for(std::uint64_t i=0;i<e;++i)
				expected*=a;
			CHECK_EQ(p.to_string(10),std::to_string(expected));
		}
	}

	const std::array<std::uint64_t,14> primes={
		2,3,5,7,11,13,17,19,97,101,257,65537,1000003,1000033};
	for(std::uint64_t p : primes){
		CHECK(mp::pr_prime(BI::from_u64(p),12)>0);
		CHECK_EQ(mp::next_prime(BI::from_u64(p-1),12),BI::from_u64(p));
		if(p>2){
			for(std::uint64_t a=1;a<std::min<std::uint64_t>(p,200);++a){
				BI inv;
				CHECK(mp::invert(&inv,BI::from_u64(a),BI::from_u64(p)));
				CHECK_EQ(positive_mod(BI::from_u64(a)*inv,BI::from_u64(p)),BI(1));
				CHECK_EQ(mp::invert(BI::from_u64(a),BI::from_u64(p)),inv);
				CHECK_EQ(mp::invmod(BI::from_u64(a),BI::from_u64(p)),inv);
				const BI exp((p-1)/2);
				const BI jm=mp::modpow(BI::from_u64(a),exp,BI::from_u64(p));
				const int jac=mp::jacobi(BI::from_u64(a),BI::from_u64(p));
				const int exp_j=jm.is_zero()?0:(jm==BI(1)?1:-1);
				CHECK_EQ(jac,exp_j);
				CHECK_EQ(mp::kronecker(BI::from_u64(a),BI::from_u64(p)),jac);
			}
		}
	}

	CHECK(!mp::invert(nullptr,BI(6),BI(9)));
	CHECK_THROWS(mp::invert(BI(6),BI(9)));
	CHECK_THROWS(mp::modpow(BI(2),BI(-1),BI(7)));
	CHECK_THROWS(mp::modpow(BI(2),BI(3),BI(0)));
	CHECK_THROWS(mp::jacobi(BI(1),BI(2)));

	for(std::uint64_t mod : {3u,5u,7u,11u,13u,17u,19u,29u,41u,97u}){
		for(std::uint64_t x=0;x<mod;++x){
			BI a=BI::from_u64((x*x)%mod);
			BI root;
			CHECK(mp::sqrtmod(&root,a,BI::from_u64(mod)));
			CHECK_EQ(positive_mod(root*root,BI::from_u64(mod)),a);
		}
	}
	CHECK(!mp::sqrtmod(nullptr,BI(3),BI(7)));
	CHECK_THROWS(mp::sqrtmod(BI(3),BI(7)));
	CHECK_THROWS(mp::sqrtmod(BI(1),BI(4)));

	if(opt.verbose)
		std::cerr<<"[test] number theory: small roots\n";
	const std::uint64_t root_lim=opt.quick?350u:2000u;
	for(std::uint64_t n=0;n<root_lim;++n){
		BI N=BI::from_u64(n);
		auto sr=mp::sqrtrem(N);
		CHECK_EQ(sr.first*sr.first+sr.second,N);
		CHECK(sr.second>=BI(0));
		CHECK(sr.second<(sr.first<<1)+BI(1));
		CHECK_EQ(mp::isqrt(N),sr.first);
		CHECK_EQ(mp::is_square(N),sr.second.is_zero());
		for(std::uint32_t k=1;k<=8;++k){
			auto rr=mp::rootrem(N,k);
			CHECK_EQ(mp::pow(rr.first,k)+rr.second,N);
			CHECK(rr.second>=BI(0));
			CHECK(mp::pow(rr.first+BI(1),k)>N);
			CHECK_EQ(mp::iroot(N,k),rr.first);
		}
	}
	if(opt.verbose)
		std::cerr<<"[test] number theory: large exact roots\n";
	Rng root_rng(opt.seed^0x726f6f745f736361ull);
	const std::vector<std::size_t> root_bits=opt.quick?
		std::vector<std::size_t>{1,7,15,31,32}:
		std::vector<std::size_t>{1,7,15,31,32};
	for(std::size_t bits : root_bits){
		BI base=positive_random_bi(root_rng,bits);
		const std::uint32_t max_root_k=4u;
		for(std::uint32_t k=2;k<=max_root_k;++k){
			const BI exact=mp::pow(base,k);
			auto rr=mp::rootrem(exact,k);
			CHECK_EQ(rr.first,base);
			CHECK(rr.second.is_zero());
			CHECK_EQ(mp::iroot(exact,k),base);
			auto nr=mp::rootrem(exact+BI(123),k);
			CHECK_EQ(mp::pow(nr.first,k)+nr.second,exact+BI(123));
			CHECK(nr.second>=BI(0));
			CHECK(mp::pow(nr.first+BI(1),k)>exact+BI(123));
		}
		for(std::uint32_t k : {3u,5u,7u}){
			CHECK(mp::is_ppow(-mp::pow(base,k)));
		}
	}
	if(opt.verbose)
		std::cerr<<"[test] number theory: large squares\n";
	for(std::size_t bits : (opt.quick?std::vector<std::size_t>{1024}:
						   std::vector<std::size_t>{1024,2048})){
		BI base=positive_random_bi(root_rng,bits);
		BI square=base*base;
		auto sr=mp::sqrtrem(square);
		CHECK_EQ(sr.first,base);
		CHECK(sr.second.is_zero());
		CHECK(mp::is_square(square));
	}
	CHECK_THROWS(mp::sqrtrem(BI(-1)));
	CHECK_THROWS(mp::rootrem(BI(4),0));
	CHECK_THROWS(mp::rootrem(BI(-4),2));
	CHECK_THROWS(mp::rootrem(BI(-8),3));

	for(std::uint64_t base=2;base<=30;++base){
		for(std::uint64_t e=2;e<=12;++e){
			BI v=mp::pow(BI::from_u64(base),e);
			CHECK(mp::is_ppow(v));
		}
	}
	CHECK(!mp::is_ppow(BI(6)));
	CHECK(mp::is_ppow(BI(0)));
	CHECK(mp::is_ppow(BI(1)));

	BI fact(1);
	for(std::uint64_t n=0;n<=80;++n){
		if(n>0)
			fact*=BI::from_u64(n);
		CHECK_EQ(mp::factorial(n),fact);
	}
	for(std::uint64_t n=0;n<=80;++n){
		for(std::uint64_t k=0;k<=n;++k){
			CHECK_EQ(mp::binomial(n,k),mp::binomial(n,n-k));
			if(k>0&&k<n){
				CHECK_EQ(mp::binomial(n,k),
						 mp::binomial(n-1,k-1)+mp::binomial(n-1,k));
			}
		}
		CHECK(mp::binomial(n,n+1).is_zero());
	}
	CHECK_EQ(mp::fibonacci(0),BI(0));
	CHECK_EQ(mp::fibonacci(1),BI(1));
	for(std::uint64_t n=2;n<=200;++n)
		CHECK_EQ(mp::fibonacci(n),mp::fibonacci(n-1)+mp::fibonacci(n-2));

	std::vector<std::pair<BI,BI>> eqs={{BI(2),BI(3)},{BI(3),BI(5)},{BI(2),BI(7)}};
	auto cr=mp::crt_solve(eqs);
	CHECK(cr.ok);
	CHECK_EQ(cr.value,BI(23));
	CHECK_EQ(cr.modulus,BI(105));
	BI cv,cm;
	CHECK(mp::crt_solve(&cv,&cm,eqs));
	CHECK_EQ(cv,cr.value);
	CHECK_EQ(cm,cr.modulus);
	CHECK_EQ(mp::crt(eqs).first,BI(23));
	std::vector<std::pair<BI,BI>> bad={{BI(0),BI(2)},{BI(1),BI(2)}};
	CHECK(!mp::crt_solve(bad).ok);
	CHECK_THROWS(mp::crt(bad));

	Rng rng(opt.seed^0x6e756d6265727468ull);
	for(std::size_t i=0;i<opt.cases/3u+1u;++i){
		BI a=random_bi(rng,static_cast<std::size_t>(rng.below(1024)),true);
		BI b=random_bi(rng,static_cast<std::size_t>(rng.below(1024)),true);
		if(a.is_zero()&&b.is_zero())
			b=BI(1);
		BI g=mp::gcd(a,b);
		CHECK((a%g).is_zero());
		CHECK((b%g).is_zero());
		auto eg=mp::extgcd(a,b);
		CHECK_EQ(eg.g,g);
		CHECK_EQ(a*eg.x+b*eg.y,g);
		BI l=mp::lcm(a,b);
		if(!a.is_zero()&&!b.is_zero())
			CHECK_EQ(mp::divexact(a.abs(),g)*b.abs(),l);
	}
}

struct Rat64{
	std::int64_t n=0;
	std::int64_t d=1;

	Rat64(std::int64_t nn=0,std::int64_t dd=1) : n(nn),d(dd){ norm(); }
	void norm(){
		if(d<0){
			d=-d;
			n=-n;
		}
		const std::int64_t g=std::gcd(n<0?-n:n,d);
		if(g!=0){
			n/=g;
			d/=g;
		}
		if(n==0)
			d=1;
	}
};

Rat64 operator+(Rat64 a,Rat64 b){ return Rat64(a.n*b.d+b.n*a.d,a.d*b.d); }
Rat64 operator-(Rat64 a,Rat64 b){ return Rat64(a.n*b.d-b.n*a.d,a.d*b.d); }
Rat64 operator*(Rat64 a,Rat64 b){ return Rat64(a.n*b.n,a.d*b.d); }
Rat64 operator/(Rat64 a,Rat64 b){ return Rat64(a.n*b.d,a.d*b.n); }

std::string rat64_string(Rat64 r){
	if(r.d==1)
		return std::to_string(r.n);
	return std::to_string(r.n)+"/"+std::to_string(r.d);
}

void test_bigrat(const Options&opt){
	CHECK_EQ(BR().to_string(),std::string("0"));
	CHECK_EQ(BR(BI(2),BI(4)).to_string(),std::string("1/2"));
	CHECK_EQ(BR(BI(2),BI(-4)).to_string(),std::string("-1/2"));
	CHECK_EQ(BR(BI(0),BI(-9)).to_string(),std::string("0"));
	CHECK_EQ(BR::parse("355/113").to_string(),std::string("355/113"));
	CHECK_EQ(BR::parse("1.25").to_string(),std::string("5/4"));
	CHECK_EQ(BR::parse(".125").to_string(),std::string("1/8"));
	CHECK_EQ(BR::parse("-.125").to_string(),std::string("-1/8"));
	CHECK_EQ(BR::parse("ff/10",16).to_string(),std::string("255/16"));
	CHECK_EQ(BR::from_double(0.5).to_string(),std::string("1/2"));
	CHECK_EQ(BR::from_float(-0.25f).to_string(),std::string("-1/4"));
	CHECK_THROWS(BR(BI(1),BI(0)));
	CHECK_THROWS(BR::parse(""));
	CHECK_THROWS(BR::parse("1//2"));
	CHECK_THROWS(BR::parse("1.2",16));
	CHECK_THROWS(BR::parse(".",10));
	CHECK_THROWS(BR::from_double(std::numeric_limits<double>::infinity()));
	CHECK_THROWS(BR(0).recip());

	const std::int64_t lim=opt.quick?8:14;
	for(std::int64_t an=-lim;an<=lim;++an){
		for(std::int64_t ad=1;ad<=lim;++ad){
			for(std::int64_t bn=-lim;bn<=lim;++bn){
				for(std::int64_t bd=1;bd<=lim;++bd){
					BR a{BI(an),BI(ad)};
					BR b{BI(bn),BI(bd)};
					Rat64 ra(an,ad),rb(bn,bd);
					CHECK_EQ((a+b).to_string(),rat64_string(ra+rb));
					CHECK_EQ((a-b).to_string(),rat64_string(ra-rb));
					CHECK_EQ((a*b).to_string(),rat64_string(ra*rb));
					if(bn!=0)
						CHECK_EQ((a/b).to_string(),rat64_string(ra/rb));
					CHECK_EQ((a<b),(ra.n*rb.d<rb.n*ra.d));
					CHECK_EQ((a==b),(ra.n==rb.n&&ra.d==rb.d));
				}
			}
		}
	}

	for(BR x : {BR::parse("-7/3"),BR::parse("-3/2"),BR::parse("-1/2"),
				BR::parse("0"),BR::parse("1/2"),BR::parse("3/2"),
				BR::parse("7/3")}){
		BI z=x.trunc();
		BI f=x.floor();
		BI c=x.ceil();
		if(!x.is_integer()){
			if(x.sign()>=0){
				CHECK(BR(z)<=x);
				CHECK(x<BR(z+BI(1)));
			}else{
				CHECK(BR(z)>=x);
				CHECK(x>BR(z-BI(1)));
			}
		}
		CHECK(BR(f)<=x);
		CHECK(x<BR(f+BI(1)));
		CHECK(BR(c- BI(1))<x);
		CHECK(x<=BR(c));
		CHECK_EQ(mp::floor(x),f);
		CHECK_EQ(mp::ceil(x),c);
		CHECK_EQ(mp::trunc(x),z);
		CHECK_EQ(x.to_bigint(FR::zero),z);
		CHECK_EQ(x.to_bigint(FR::down),f);
		CHECK_EQ(x.to_bigint(FR::up),c);
		CHECK_EQ(x.rint(FR::nearest),mp::rint(x,FR::nearest));
	}
	CHECK_EQ(BR::parse("5/2").to_bigint(FR::nearest),BI(2));
	CHECK_EQ(BR::parse("7/2").to_bigint(FR::nearest),BI(4));
	CHECK_EQ(BR::parse("-5/2").to_bigint(FR::nearest),BI(-2));
	CHECK_EQ(BR::parse("-7/2").to_bigint(FR::nearest),BI(-4));
	CHECK_EQ(BR::parse("5/2").to_bigint(FR::away),BI(3));
	CHECK_EQ(BR::parse("-5/2").to_bigint(FR::away),BI(-3));
	CHECK(BR(123).fits_u64());
	CHECK(BR(-123).fits_i64());
	CHECK_EQ(BR(123).to_u64(),123ull);
	CHECK_EQ(BR(-123).to_i64(),-123);
	check_near(BR::parse("1/8").to_double(),0.125,"BigRat::to_double");
	check_near(BR::parse("-1/4").to_float(),-0.25,"BigRat::to_float");
	CHECK_EQ(mp::abs(BR::parse("-2/3")).to_string(),std::string("2/3"));
	CHECK_EQ(mp::recip(BR::parse("-2/3")).to_string(),std::string("-3/2"));
	CHECK_EQ(mp::pow(BR::parse("-2/3"),3).to_string(),std::string("-8/27"));
	CHECK_EQ(mp::pow(BR::parse("-2/3"),-2).to_string(),std::string("9/4"));

	BR inc=BR::parse("1/3");
	CHECK_EQ((inc++).to_string(),std::string("1/3"));
	CHECK_EQ(inc.to_string(),std::string("4/3"));
	CHECK_EQ((++inc).to_string(),std::string("7/3"));
	CHECK_EQ((inc--).to_string(),std::string("7/3"));
	CHECK_EQ(inc.to_string(),std::string("4/3"));
	CHECK_EQ((--inc).to_string(),std::string("1/3"));
	CHECK_THROWS(BR(1)/BR(0));
	CHECK_THROWS(BR(0).pow_int(-1));
}

void test_bigfloat_core(){
	const std::size_t P=192;
	BF z;
	CHECK(z.is_zero());
	CHECK(BF::zero(P).is_zero());
	CHECK_EQ(BF::one(P).to_bigint(),BI(1));
	CHECK(BF::epsilon(P)>BF::zero(P));
	CHECK(BF::inf(1,P).is_inf());
	CHECK(BF::inf(-1,P).is_neg());
	CHECK(BF::nan(P).is_nan());
	CHECK_EQ(BF(123,P).to_bigint(),BI(123));
	CHECK_EQ(BF(BI::parse("12345678901234567890"),P).to_bigint(),
			 BI::parse("12345678901234567890"));
	CHECK_EQ(BF(BR::parse("3/2"),P).to_bigrat().to_string(),
			 std::string("3/2"));
	CHECK_EQ(BF::parse("0b1.001p5",P).to_bigint(),BI(36));
	CHECK_EQ(BF::from_parts(1,BI(31),-4,P).to_bigrat().to_string(),
			 std::string("31/16"));
	CHECK_EQ(BF::from_double(0.5,P).to_bigrat().to_string(),std::string("1/2"));
	CHECK_EQ(BF::parse("nan",P).to_string(),std::string("nan"));
	CHECK_EQ(BF::parse("-inf",P).to_string(),std::string("-inf"));
	CHECK_THROWS(BF::parse("",P));
	CHECK_THROWS(BF::zero(0));
	CHECK_THROWS(BF::parse("0b.",P));

	const std::vector<BR> dyadics={
		BR::parse("-4"),BR::parse("-2"),BR::parse("-1"),
		BR::parse("-1/2"),BR::parse("0"),BR::parse("1/2"),
		BR::parse("1"),BR::parse("2"),BR::parse("4")};
	for(const BR&ar : dyadics){
		for(const BR&br : dyadics){
			if(br.is_zero())
				continue;
			BF a(ar,P),b(br,P);
			CHECK_EQ((a+b).to_bigrat(),ar+br);
			CHECK_EQ((a-b).to_bigrat(),ar-br);
			CHECK_EQ((a*b).to_bigrat(),ar*br);
			const BR exact_q=ar/br;
			const BF bf_q=BF::div(a,b,P);
			if(is_pow2_positive(exact_q.den()))
				CHECK_EQ(bf_q.to_bigrat(),exact_q);
			else
				check_near(bf_q.to_double(),exact_q.to_double(),"BigFloat div",1e-14,1e-14);
			CHECK_EQ(BF::fma(a,b,BF(ar,P),P).to_bigrat(),ar*br+ar);
			BF c=a;
			c+=b;
			CHECK_EQ(c.to_bigrat(),ar+br);
			c=a;
			c-=b;
			CHECK_EQ(c.to_bigrat(),ar-br);
			c=a;
			c*=b;
			CHECK_EQ(c.to_bigrat(),ar*br);
			c=a;
			c/=b;
			if(is_pow2_positive(exact_q.den()))
				CHECK_EQ(c.to_bigrat(),exact_q);
			else
				check_near(c.to_double(),exact_q.to_double(),"BigFloat /=",1e-14,1e-14);
		}
	}

	CHECK_EQ(BF::parse("2.5",P).to_bigint(FR::nearest),BI(2));
	CHECK_EQ(BF::parse("3.5",P).to_bigint(FR::nearest),BI(4));
	CHECK_EQ(BF::parse("-2.5",P).to_bigint(FR::nearest),BI(-2));
	CHECK_EQ(BF::parse("-3.5",P).to_bigint(FR::nearest),BI(-4));
	CHECK_EQ(BF::parse("2.25",P).to_bigint(FR::zero),BI(2));
	CHECK_EQ(BF::parse("2.25",P).to_bigint(FR::up),BI(3));
	CHECK_EQ(BF::parse("-2.25",P).to_bigint(FR::down),BI(-3));
	CHECK_EQ(BF::parse("-2.25",P).to_bigint(FR::away),BI(-3));
	CHECK_EQ(BF::parse("123",P).to_i64(),123);
	CHECK_EQ(BF::parse("123",P).to_u64(),123ull);
	CHECK(BF::parse("123",P).fits_i64());
	CHECK(BF::parse("123",P).fits_u64());
	CHECK(!BF::inf().fits_i64());

	BF x=BF::parse("10.75",P);
	CHECK_EQ(x.floor().to_bigint(),BI(10));
	CHECK_EQ(x.ceil().to_bigint(),BI(11));
	CHECK_EQ(x.trunc().to_bigint(),BI(10));
	CHECK_EQ(x.frac(P).to_bigrat().to_string(),std::string("3/4"));
	auto mf=BF::modf(x,P);
	CHECK_EQ(mf.first.to_bigint(),BI(10));
	CHECK_EQ(mf.second.to_bigrat().to_string(),std::string("3/4"));
	CHECK_EQ(mp::floor(x).to_bigint(),BI(10));
	CHECK_EQ(mp::ceil(x).to_bigint(),BI(11));
	CHECK_EQ(mp::trunc(x).to_bigint(),BI(10));
	CHECK_EQ(mp::frac(x,P).to_bigrat().to_string(),std::string("3/4"));

	BF ld=mp::ldexp(BF::parse("1.5",P),10);
	CHECK_EQ(ld.to_bigrat().to_string(),std::string("1536"));
	CHECK_EQ(mp::scalbn(ld,-10).to_bigrat().to_string(),std::string("3/2"));
	auto fx=mp::frexp(BF::parse("12",P),P);
	CHECK_EQ(mp::ldexp(fx.first,fx.second).to_bigrat().to_string(),
			 std::string("12"));
	CHECK_EQ(mp::ilogb(BF::parse("12",P)),3);

	CHECK_EQ(mp::sqr(BF(12,P),P).to_bigint(),BI(144));
	CHECK_EQ(mp::sqrt(BF(144,P),P).to_bigint(),BI(12));
	CHECK_EQ(mp::cbrt(BF(125,P),P).to_bigint(),BI(5));
	CHECK_EQ(mp::cbrt(BF(-125,P),P).to_bigint(),BI(-5));
	CHECK_EQ(mp::rootn(BF(81,P),4,P).to_bigint(),BI(3));
	CHECK_EQ(mp::rootn(BF(-243,P),5,P).to_bigint(),BI(-3));
	CHECK(mp::rootn(BF(-16,P),4,P).is_nan());
	CHECK_EQ(mp::pow_ui(BF(3,P),5,P).to_bigint(),BI(243));
	CHECK_EQ(mp::pow_si(BF(2,P),-3,P).to_bigrat().to_string(),std::string("1/8"));
	CHECK_EQ(mp::recip(BF(4,P),P).to_bigrat().to_string(),std::string("1/4"));
	CHECK_EQ(mp::fmod(BF::parse("17.5",P),BF(5,P),P).to_bigrat().to_string(),
			 std::string("5/2"));
	CHECK_EQ(mp::remainder(BF::parse("17.5",P),BF(5,P),P).to_bigrat().to_string(),
			 std::string("-5/2"));
	CHECK_EQ(mp::dim(BF(5,P),BF(3,P),P).to_bigint(),BI(2));
	CHECK(mp::dim(BF(3,P),BF(5,P),P).is_zero());
	CHECK_EQ(mp::min(BF(3,P),BF(5,P),P).to_bigint(),BI(3));
	CHECK_EQ(mp::max(BF(3,P),BF(5,P),P).to_bigint(),BI(5));
	CHECK_EQ(mp::cmpabs(BF(-5,P),BF(3,P)),1);
	CHECK(mp::ulp(BF(1,P))>BF::zero(P));
	CHECK(mp::next_up(BF(1,P))>BF(1,P));
	CHECK(mp::next_down(BF(1,P))<BF(1,P));
	CHECK(mp::next_toward(BF(1,P),BF(2,P))>BF(1,P));
	CHECK_EQ(mp::copy_sign(BF(2,P),BF(-1,P)).sign(),-1);
	CHECK_EQ(mp::set_sign(BF(-2,P),1).sign(),1);
	CHECK_EQ(BF(123,P).rounded(64).precision(),64u);
	BF rp=BF::parse("1.1",P);
	rp.set_precision(80);
	CHECK_EQ(rp.precision(),80u);
	CHECK(BF::parse("1.1",64).inexact()!=0);

	CHECK(BF::sqrt(BF(-1,P),P).is_nan());
	CHECK((BF(1,P)/BF(0,P)).is_inf());
	CHECK((BF(0,P)/BF(0,P)).is_nan());
	CHECK_THROWS(BF::nan(P).to_bigint());
	CHECK_THROWS(BF::inf(1,P).to_bigrat());
	CHECK_THROWS(compare(BF::nan(P),BF(1,P)));
	CHECK_THROWS(mp::cmpabs(BF::nan(P),BF(1,P)));
	CHECK_THROWS(mp::ilogb(BF::zero(P)));
}

void test_bigfloat_transcendentals(const Options&opt){
	const std::size_t P=opt.quick?96u:160u;
	check_near(mp::const_pi(P).to_double(),3.14159265358979323846264338327950288,
			   "const_pi",1e-14,1e-14);
	check_near(mp::const_log2(P).to_double(),std::log(2.0),"const_log2",1e-14,1e-14);

	const std::vector<double> vals={-0.75,-0.5,-0.125,0.0,0.125,0.5,0.75,1.0,2.0};
	for(double v : vals){
		BF x(v,P);
		check_near(mp::exp(x,P).to_double(),std::exp(v),"exp");
		check_near(mp::exp2(x,P).to_double(),std::exp2(v),"exp2");
		check_near(mp::exp10(x,P).to_double(),std::pow(10.0,v),"exp10",2e-10,1e-12);
		check_near(mp::expm1(x,P).to_double(),std::expm1(v),"expm1");
		if(v>-1.0){
			check_near(mp::log1p(x,P).to_double(),std::log1p(v),"log1p",2e-11,2e-13);
		}
		if(v>0.0){
			check_near(mp::log(x,P).to_double(),std::log(v),"log");
			check_near(mp::log2(x,P).to_double(),std::log2(v),"log2");
			check_near(mp::log10(x,P).to_double(),std::log10(v),"log10");
			check_near(mp::sqrt(x,P).to_double(),std::sqrt(v),"sqrt");
		}
		check_near(mp::sin(x,P).to_double(),std::sin(v),"sin",2e-11,2e-13);
		check_near(mp::cos(x,P).to_double(),std::cos(v),"cos",2e-11,2e-13);
		check_near(mp::tan(x,P).to_double(),std::tan(v),"tan",5e-11,5e-13);
		check_near(mp::atan(x,P).to_double(),std::atan(v),"atan",2e-11,2e-13);
		if(std::fabs(v)<=1.0){
			check_near(mp::asin(x,P).to_double(),std::asin(v),"asin",5e-11,5e-13);
			check_near(mp::acos(x,P).to_double(),std::acos(v),"acos",5e-11,5e-13);
			check_near(mp::atanh(BF(v*0.5,P),P).to_double(),std::atanh(v*0.5),
					   "atanh",5e-11,5e-13);
		}
		check_near(mp::sinh(x,P).to_double(),std::sinh(v),"sinh",5e-11,5e-13);
		check_near(mp::cosh(x,P).to_double(),std::cosh(v),"cosh",5e-11,5e-13);
		check_near(mp::tanh(x,P).to_double(),std::tanh(v),"tanh",5e-11,5e-13);
		check_near(mp::asinh(x,P).to_double(),std::asinh(v),"asinh",5e-11,5e-13);
	}

	for(double v : {1.0,1.25,1.5,2.0,3.5,5.0}){
		check_near(mp::gamma(BF(v,P),P).to_double(),std::tgamma(v),
				   "gamma",3e-10,3e-12);
	}
	for(double v : {-1.0,-0.5,0.0,0.5,1.0,2.0}){
		check_near(mp::erf(BF(v,P),P).to_double(),std::erf(v),
				   "erf",3e-10,3e-12);
		check_near(mp::erfc(BF(v,P),P).to_double(),std::erfc(v),
				   "erfc",3e-10,3e-12);
	}
	check_near(mp::pow(BF(2.0,P),BF(0.5,P),P).to_double(),std::sqrt(2.0),
			   "pow");
	check_near(mp::atan2(BF(1,P),BF(-1,P),P).to_double(),std::atan2(1.0,-1.0),
			   "atan2");
	CHECK(mp::log(BF(-1,P),P).is_nan());
	CHECK(mp::log(BF::zero(P),P).is_inf());
	CHECK(mp::acosh(BF(0.5,P),P).is_nan());
	CHECK(mp::atanh(BF(1,P),P).is_inf());
	CHECK(mp::gamma(BF(-2,P),P).is_nan());
	CHECK(mp::erfc(BF::inf(1,P),P).is_zero());
}

void test_bigcomplex(){
	const std::size_t P=160;
	BC z(BF(3,P),BF(4,P));
	CHECK_EQ(z.real().to_bigint(),BI(3));
	CHECK_EQ(z.imag().to_bigint(),BI(4));
	CHECK_EQ(z.precision(),P);
	CHECK(!z.is_zero());
	CHECK_EQ(mp::conj(z).imag().to_bigint(),BI(-4));
	CHECK_EQ((-z).real().to_bigint(),BI(-3));
	CHECK_EQ((-z).imag().to_bigint(),BI(-4));
	CHECK_EQ(mp::abs(z,P).to_bigint(),BI(5));

	BC a(BF(1,P),BF(1,P));
	BC b(BF(1,P),BF(-1,P));
	BC prod=a*b;
	CHECK_EQ(prod.real().to_bigint(),BI(2));
	CHECK(prod.imag().is_zero());
	BC quot=a/a;
	CHECK_EQ(quot.real().to_bigint(),BI(1));
	CHECK(quot.imag().is_zero());
	BC sum=a+b;
	CHECK_EQ(sum.real().to_bigint(),BI(2));
	CHECK(sum.imag().is_zero());
	BC diff=a-b;
	CHECK(diff.real().is_zero());
	CHECK_EQ(diff.imag().to_bigint(),BI(2));

	BC c=a;
	c+=b;
	CHECK_EQ(c,sum);
	c-=b;
	CHECK_EQ(c,a);
	c*=b;
	CHECK_EQ(c,prod);
	c/=b;
	CHECK_EQ(c,a);

	BC ipi(BF::zero(P),mp::const_pi(P));
	BC e=mp::exp(ipi,P);
	check_near(e.real().to_double(),-1.0,"complex exp real",2e-10,2e-12);
	check_near(e.imag().to_double(),0.0,"complex exp imag",2e-10,2e-12);
	auto check_bc=[&](const BC&got,const std::complex<double>&want,
					  const char*label,double rel=3e-10,
					  double abs=3e-12){
		const std::string lr=std::string(label)+" real";
		const std::string li=std::string(label)+" imag";
		check_near(got.real().to_double(),want.real(),lr.c_str(),rel,abs);
		check_near(got.imag().to_double(),want.imag(),li.c_str(),rel,abs);
	};
	const BC w(BF(0.75,P),BF(-0.5,P));
	const std::complex<double> wd(0.75,-0.5);
	check_bc(mp::log(w,P),std::log(wd),"complex log");
	check_bc(mp::sqrt(w,P),std::sqrt(wd),"complex sqrt");
	check_bc(mp::pow(w,BC(BF(1.25,P),BF(-0.375,P)),P),
			 std::pow(wd,std::complex<double>(1.25,-0.375)),
			 "complex pow");
	check_bc(mp::sin(w,P),std::sin(wd),"complex sin");
	check_bc(mp::cos(w,P),std::cos(wd),"complex cos");
	check_bc(mp::tan(w,P),std::tan(wd),"complex tan");
	check_bc(mp::asin(w,P),std::asin(wd),"complex asin");
	check_bc(mp::acos(w,P),std::acos(wd),"complex acos");
	check_bc(mp::atan(w,P),std::atan(wd),"complex atan");
	check_bc(mp::sinh(w,P),std::sinh(wd),"complex sinh");
	check_bc(mp::cosh(w,P),std::cosh(wd),"complex cosh");
	check_bc(mp::tanh(w,P),std::tanh(wd),"complex tanh");
	check_bc(mp::asinh(w,P),std::asinh(wd),"complex asinh");
	check_bc(mp::acosh(w,P),std::acosh(wd),"complex acosh");
	check_bc(mp::atanh(w,P),std::atanh(wd),"complex atanh");
	check_bc(mp::sqrt(BC(BF(-4,P),BF::zero(P)),P),
			 std::complex<double>(0.0,2.0),"complex sqrt neg real");
	BC lneg=mp::log(BC(BF(-2,P),BF::zero(P)),P);
	check_near(lneg.real().to_double(),std::log(2.0),
			   "complex log neg real",2e-10,2e-12);
	check_near(lneg.imag().to_double(),std::acos(-1.0),
			   "complex log neg imag",2e-10,2e-12);
	check_bc(mp::asin(BC(BF(2,P),BF::zero(P)),P),
			 std::asin(std::complex<double>(2.0,0.0)),
			 "complex asin branch");
	check_bc(mp::acos(BC(BF(2,P),BF::zero(P)),P),
			 std::acos(std::complex<double>(2.0,0.0)),
			 "complex acos branch");
	check_bc(mp::atanh(BC(BF(2,P),BF::zero(P)),P),
			 std::atanh(std::complex<double>(2.0,0.0)),
			 "complex atanh branch");
	check_bc(mp::asinh(BC(BF::zero(P),BF(2,P)),P),
			 std::asinh(std::complex<double>(0.0,2.0)),
			 "complex asinh upper branch");
	check_bc(mp::asinh(BC(BF::zero(P),BF(-2,P)),P),
			 std::asinh(std::complex<double>(0.0,-2.0)),
			 "complex asinh lower branch");
	BC pw=mp::pow_si(a,5,P);
	check_bc(pw,std::pow(std::complex<double>(1.0,1.0),5),
			 "complex pow_si");
	BC pol=mp::polar(BF(2,P),BF(0.75,P),P);
	check_bc(pol,std::polar(2.0,0.75),"complex polar");
	check_near(mp::arg(pol,P).to_double(),0.75,"complex arg",2e-10,2e-12);
	BC rounded=z.rounded(80);
	CHECK_EQ(rounded.precision(),80u);
	rounded.real()=BF(5,80);
	CHECK_EQ(rounded.real().to_bigint(),BI(5));
	CHECK_THROWS(BC::div(z,BC(BF::zero(P),BF::zero(P)),P));
}

void test_bignt_and_ec(){
	using namespace mp::BigNT;
	CHECK_EQ(mod_pos(BI(-1),BI(7)),BI(6));
	CHECK_EQ(abs_sub(BI(10),BI(3)),BI(7));
	CHECK_EQ(abs_sub(BI(3),BI(10)),BI(7));

	ModPoly p({1,2,3},17);
	CHECK_EQ(p.modulus(),17ull);
	CHECK_EQ(p.size(),3u);
	CHECK_EQ(p.degree(),2u);
	CHECK_EQ(p[10],0ull);
	p.coeff().push_back(0);
	p.normalize();
	CHECK_EQ(p.size(),3u);
	CHECK(ModPoly(17).is_zero());

	ModPoly a({1,2},17);
	ModPoly b({3,4},17);
	ModPoly prod=poly_mul(a,b,PolyMulAlg::schoolbook);
	CHECK((prod.coeff()==std::vector<std::uint64_t>{3,10,8}));
	CHECK_EQ(ModPoly::mul(a,b,PolyMulAlg::karatsuba),prod);
	ModPoly sum=a+b;
	CHECK((sum.coeff()==std::vector<std::uint64_t>{4,6}));
	ModPoly dif=b-a;
	CHECK((dif.coeff()==std::vector<std::uint64_t>{2,2}));
	auto qr=poly_divrem(prod,a);
	CHECK_EQ(qr.first,b);
	CHECK(qr.second.is_zero());
	CHECK_EQ(poly_rem(prod,a),ModPoly(17));
	ModPoly common({3,1},17);
	ModPoly qa({1,0,1},17);
	ModPoly qb({2,1,1},17);
	ModPoly ga=common*qa;
	ModPoly gb=common*qb;
	CHECK_EQ(ModPoly::gcd(ga,gb),common);
	CHECK_EQ(poly_gcd(ga*ModPoly({5},17),gb*ModPoly({9},17)),common);
	CHECK_EQ(ModPoly::gcd(common*ModPoly({5},17),ModPoly(17)),common);
	CHECK_EQ(ModPoly::gcd(ModPoly(17),ModPoly(17)),ModPoly(17));
	CHECK_EQ(ModPoly::gcd(ModPoly({5},17),ModPoly({10},17)),
			 ModPoly({1},17));
	ModPoly f({1,0,1},17);
	CHECK_EQ(poly_x_pow_mod(BI(4),f).coeff(),std::vector<std::uint64_t>{1});
	CHECK_EQ(ModPoly::x(17).pow_mod(BI(4),f).coeff(),std::vector<std::uint64_t>{1});
	CHECK_EQ(ModPoly::one(17).coeff(),std::vector<std::uint64_t>{1});
	ModPoly lin({2,1},17);
	CHECK_EQ(ModPoly::x(17).pow_mod(BI(5),lin).coeff(),
			 std::vector<std::uint64_t>{2});

	std::vector<std::uint64_t> va(96),vb(80);
	for(std::size_t i=0;i<va.size();++i)
		va[i]=(i*i+3*i+1)%998244353u;
	for(std::size_t i=0;i<vb.size();++i)
		vb[i]=(7*i+11)%998244353u;
	ModPoly pa(va,998244353u),pb(vb,998244353u);
	CHECK_EQ(ModPoly::mul(pa,pb,PolyMulAlg::ntt),
			 ModPoly::mul(pa,pb,PolyMulAlg::schoolbook));
	CHECK_THROWS(ModPoly::mul(ModPoly({1,2},17),ModPoly({1},19)));
	CHECK_THROWS(ModPoly::gcd(ModPoly({1},17),ModPoly({1},19)));
	CHECK_THROWS(ModPoly::divrem(a,ModPoly(17)));
	CHECK_THROWS(ModPoly(1));
	CHECK_THROWS(ModPoly::x(15).pow_mod(BI(-1),f));

	const BI big_mod=(BI(1)<<100u)-BI(159);
	BigIntPoly ia({BI(-1),big_mod+BI(2),BI(0)},big_mod);
	CHECK_EQ(ia.size(),2u);
	CHECK_EQ(ia[0],big_mod-BI(1));
	CHECK_EQ(ia[1],BI(2));
	BigIntPoly ib({BI(3),BI(4)},big_mod);
	BigIntPoly ip=ia*ib;
	CHECK_EQ(ip.evaluate(BI(5)),
			 mp::BigNT::mod_pos(ia.evaluate(BI(5))*ib.evaluate(BI(5)),
								 big_mod));
	CHECK_EQ((ia+ib).evaluate(BI(7)),
			 mp::BigNT::mod_pos(ia.evaluate(BI(7))+ib.evaluate(BI(7)),
								 big_mod));
	CHECK_THROWS(BigIntPoly(BI(1)));

	BI root;
	CHECK(mp::sqrtmod(&root,BI(10),BI(13)));
	CHECK_EQ((root*root)%BI(13),BI(10));
	CHECK(!mp::sqrtmod(&root,BI(3),BI(7)));
	CHECK_THROWS(mp::sqrtmod(BI(3),BI(7)));

	auto fac=factor(BI(360));
	CHECK_EQ(fac.size(),3u);
	CHECK_EQ(fac[0].prime,BI(2));
	CHECK_EQ(fac[0].exp,3u);
	CHECK_EQ(fac[1].prime,BI(3));
	CHECK_EQ(fac[1].exp,2u);
	CHECK_EQ(fac[2].prime,BI(5));
	CHECK_EQ(fac[2].exp,1u);
	BI rho=pollard_rho(BI(8051));
	CHECK(!rho.is_zero());
	CHECK((BI(8051)%rho).is_zero());
	BI pm1=pollard_pm1(BI(91),100);
	if(!pm1.is_zero())
		CHECK((BI(91)%pm1).is_zero());
	CHECK_EQ(pollard_pm1(BI(13),100),BI(13));
	CHECK_THROWS(factor(BI(0)));

	auto cor=cornacchia(BI(1),BI(5));
	CHECK(cor.ok);
	CHECK_EQ(cor.x*cor.x+cor.y*cor.y,BI(5));
	mp::BigNT::QForm form{BI(2),BI(4),BI(3)};
	auto red=reduce(form);
	CHECK(is_reduced(red));
	CHECK_EQ(red.discriminant(),form.discriminant());
	CHECK_EQ(class_number(BI(-3)),1u);
	CHECK_EQ(class_number(BI(-4)),1u);
	CHECK_EQ(class_number(BI(-23)),3u);
	auto forms=reduced_forms(BI(-23));
	CHECK_EQ(forms.size(),3u);
	for(const auto&rf : forms)
		CHECK(is_reduced(rf));

	auto jq=j_qexp(5);
	CHECK(jq.size()>=5u);
	CHECK_EQ(jq[0],BI(1));
	CHECK_EQ(jq[1],BI(744));
	CHECK_EQ(jq[2],BI(196884));
	CHECK_EQ(jq[3],BI(21493760));
	std::vector<BC> roots={BC(BF(1,96)),BC(BF(2,96))};
	auto poly=j_polynomial_from_roots(roots,96);
	CHECK_EQ(poly.size(),3u);
	CHECK_EQ(poly[0].real().to_bigint(),BI(2));
	CHECK_EQ(poly[1].real().to_bigint(),BI(-3));
	CHECK_EQ(poly[2].real().to_bigint(),BI(1));
	BC tau(BF::zero(96),BF(1,96));
	BC tau_red=modular_tau_reduce(tau,96);
	CHECK(tau_red.imag().sign()>0);
	BC jtau=j_invariant_tau(tau,3,96);
	CHECK(jtau.real().is_finite());
	CHECK(jtau.imag().is_finite());

	using namespace mp::ec;
	Curve E{BI(17),BI(2),BI(2)};
	AffinePoint G=affine(BI(5),BI(1),E);
	CHECK(is_on_curve(G,E));
	CHECK(is_on_curve(infinity(),E));
	AffinePoint negG=neg(G,E);
	CHECK(add(G,negG,E).inf);
	AffinePoint two=dbl(G,E);
	CHECK(is_on_curve(two,E));
	CHECK_EQ(two.x,BI(6));
	CHECK_EQ(two.y,BI(3));
	CHECK_EQ(add(G,G,E).x,two.x);
	CHECK_EQ(add(G,G,E).y,two.y);
	AffinePoint three=add(two,G,E);
	CHECK_EQ(mul(G,BI(3),E).x,three.x);
	CHECK_EQ(mul(G,BI(3),E).y,three.y);
	AffinePoint neg_three=mul(G,BI(-3),E);
	CHECK_EQ(neg_three.x,neg(three,E).x);
	CHECK_EQ(neg_three.y,neg(three,E).y);
	CHECK(mul(G,BI(19),E).inf);
	JacobianPoint J=to_jacobian(G);
	CHECK(!J.inf);
	CHECK_EQ(to_affine(J,E).x,G.x);
	CHECK_EQ(to_affine(J,E).y,G.y);
	CHECK(to_affine(mul(J,BI(19),E),E).inf);
	CHECK_EQ(to_affine(add(J,J,E),E).x,two.x);
	CHECK_EQ(to_affine(add_mixed(J,G,E),E).x,two.x);
	auto naf=wnaf_digits(BI(29),5);
	BI recon;
	for(std::size_t i=naf.size();i>0;--i){
		recon<<=1u;
		recon+=BI(naf[i-1u]);
	}
	CHECK_EQ(recon,BI(29));

	const BI large_p=(BI(1)<<89u)-BI(1);
	Curve LE{large_p,BI(2),mp::BigNT::mod_pos(BI(49)-BI(125)-BI(10),
											   large_p)};
	AffinePoint LG=affine(BI(5),BI(7),LE);
	CHECK(is_on_curve(LG,LE));
	AffinePoint L2=dbl(LG,LE);
	AffinePoint L3=add(L2,LG,LE);
	AffinePoint Lm=mul(LG,BI(3),LE);
	CHECK_EQ(Lm.x,L3.x);
	CHECK_EQ(Lm.y,L3.y);
	CHECK_EQ(to_affine(mul(to_jacobian(LG),BI(3),LE),LE).x,L3.x);
}

template<class NewBI,class OldBI>
void cross_check_one_int(const std::string&as,const std::string&bs){
	NewBI an=NewBI::parse(as,10);
	NewBI bn=NewBI::parse(bs,10);
	OldBI ao=OldBI::parse(as,10);
	OldBI bo=OldBI::parse(bs,10);
	auto chk=[&](const auto&n,const auto&o,const char*label){
		++g_test.checks;
		if(n.to_string(10)==o.to_string(10))
			return;
		++g_test.failed;
		if(g_test.first_failures.size()<32u){
			std::ostringstream os;
			os<<"old/new mismatch "<<label<<": "<<n.to_string(10)
			  <<" vs "<<o.to_string(10);
			g_test.first_failures.push_back(os.str());
		}
	};
	chk(an+bn,ao+bo,"add");
	chk(an-bn,ao-bo,"sub");
	chk(an*bn,ao*bo,"mul");
	if(!bn.is_zero()){
		chk(an/bn,ao/bo,"div");
		chk(an%bn,ao%bo,"mod");
		chk(gcd(an,bn),gcd(ao,bo),"gcd");
		chk(lcm(an,bn),lcm(ao,bo),"lcm");
		chk(divexact(an*bn,bn),divexact(ao*bo,bo),"divexact");
	}
	const NewBI modn=NewBI::parse("1000003",10);
	const OldBI modo=OldBI::parse("1000003",10);
	chk(modpow(an,NewBI::parse("65537",10),modn),
		modpow(ao,OldBI::parse("65537",10),modo),"modpow");
	chk(pow(an,7),pow(ao,7),"pow");
}

void test_old_new_cross(const Options&opt){
#if MINI_MP_TEST_HAVE_OLD
	Rng rng(opt.seed^0x6f6c645f6e65775full);
	for(std::size_t i=0;i<opt.cases;++i){
		const BI a=random_bi(rng,static_cast<std::size_t>(rng.below(1024)),true);
		BI b=random_bi(rng,static_cast<std::size_t>(rng.below(768)),true);
		if(b.is_zero())
			b=BI(1);
		cross_check_one_int<mini_mp_new::BigInt,mini_mp_old::BigInt>(
			a.to_string(10),b.to_string(10));
	}
	for(std::string_view s : {"0","1","-1","355/113","-22/7","1.25","-.125"}){
		auto nr=mini_mp_new::BigRat::parse(s);
		auto orr=mini_mp_old::BigRat::parse(s);
		CHECK_EQ(nr.to_string(10),orr.to_string(10));
	}
	std::cerr<<"old/new cross-check enabled\n";
#else
	(void)opt;
#endif
}

void test_bigint_exceptions(){
	CHECK_THROWS(BI(1)/BI(0));
	CHECK_THROWS(BI(1)%BI(0));
	CHECK_THROWS(mp::divmod(BI(1),BI(0)));
	CHECK_THROWS(mp::dvm_simp(BI(1),BI(0)));
	CHECK_THROWS(mp::dvm_knuth(BI(1),BI(0)));
	CHECK_THROWS(mp::divexact(BI(10),BI(3)));
	CHECK_THROWS(mp::divexact(BI(10),BI(0)));
	CHECK_THROWS(mp::rand_range(BI(0)));
	CHECK_THROWS(mp::rand_range(BI(-1)));
}

template<class Fn>
void run_stage(const Options&opt,const char*name,Fn fn){
	if(opt.verbose)
		std::cerr<<"[test] "<<name<<"\n";
	try{
		fn();
	}catch(const std::exception&e){
		++g_test.failed;
		if(g_test.first_failures.size()<32u){
			std::ostringstream os;
			os<<"unexpected exception in "<<name<<": "<<e.what();
			g_test.first_failures.push_back(os.str());
		}
	}
}

void run_tests(const Options&opt){
	if(!opt.bench_only){
		run_stage(opt,"BigInt parse/convert",[](){ test_bigint_parse_convert(); });
		run_stage(opt,"BigInt small arithmetic",[](){ test_bigint_small_arithmetic(); });
		run_stage(opt,"BigInt random arithmetic",[&](){ test_bigint_random_arithmetic(opt); });
		run_stage(opt,"number theory",[&](){ test_number_theory(opt); });
		run_stage(opt,"BigRat",[&](){ test_bigrat(opt); });
		run_stage(opt,"BigFloat core",[](){ test_bigfloat_core(); });
		run_stage(opt,"BigFloat transcendentals",[&](){ test_bigfloat_transcendentals(opt); });
		run_stage(opt,"BigComplex",[](){ test_bigcomplex(); });
		run_stage(opt,"BigNT/ec",[](){ test_bignt_and_ec(); });
		run_stage(opt,"old/new cross",[&](){ test_old_new_cross(opt); });
		run_stage(opt,"BigInt exceptions",[](){ test_bigint_exceptions(); });
	}
}

template<class Fn>
void bench_one(const char*name,std::size_t bits,std::size_t loops,Fn fn){
	const auto t0=std::chrono::steady_clock::now();
	for(std::size_t i=0;i<loops;++i)
		g_sink^=fn(i);
	const auto t1=std::chrono::steady_clock::now();
	const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
	const double per=loops?static_cast<double>(ns)/static_cast<double>(loops):0.0;
	std::cout<<std::left<<std::setw(18)<<name
			 <<std::right<<std::setw(8)<<bits
			 <<std::setw(10)<<loops
			 <<std::setw(14)<<ns
			 <<std::setw(14)<<std::fixed<<std::setprecision(1)<<per
			 <<"\n";
}

void run_benchmarks(const Options&opt){
	if(!opt.bench)
		return;
	std::cout<<"\nTiming (nanoseconds):\n";
	std::cout<<std::left<<std::setw(18)<<"operation"
			 <<std::right<<std::setw(8)<<"bits"
			 <<std::setw(10)<<"loops"
			 <<std::setw(14)<<"total"
			 <<std::setw(14)<<"per-op"<<"\n";

	Rng rng(opt.seed^0x62656e63685f3131ull);
	const std::vector<std::size_t> scales=opt.quick?
		std::vector<std::size_t>{64,256,1024}:
		std::vector<std::size_t>{64,256,1024,4096,8192};
	for(std::size_t bits : scales){
		BI a=positive_random_bi(rng,bits);
		BI b=positive_random_bi(rng,bits);
		BI m=positive_random_bi(rng,std::max<std::size_t>(bits,64u))|BI(1);
		if(m<BI(3))
			m=BI(1000003);
		BI num=a*b+a;
		const std::string dec=a.to_string(10);
		const std::string hex=a.to_string(16);
		const std::size_t add_loops=std::max<std::size_t>(1,2000000/(bits+1));
		const std::size_t mul_loops=std::max<std::size_t>(1,200000/(bits+1));
		const std::size_t div_loops=std::max<std::size_t>(1,60000/(bits+1));
		const std::size_t gcd_loops=std::max<std::size_t>(1,100000/(bits+1));
		const std::size_t pow_loops=std::max<std::size_t>(1,20000/(bits+1));
		const std::size_t conv_loops=std::max<std::size_t>(1,30000/(bits+1));
		bench_one("BigInt parse10",bits,conv_loops,[&](std::size_t){
			return hash_bi(BI::parse(dec,10));
		});
		bench_one("BigInt format10",bits,conv_loops,[&](std::size_t){
			std::string s=a.to_string(10);
			return static_cast<std::uint64_t>(s.size());
		});
		bench_one("BigInt parse16",bits,conv_loops,[&](std::size_t){
			return hash_bi(BI::parse(hex,16));
		});
		bench_one("BigInt bytes",bits,conv_loops,[&](std::size_t){
			auto bytes=a.to_bytes(true);
			return hash_bi(BI::from_bytes(bytes.data(),bytes.size(),true));
		});
		bench_one("BigInt bitops",bits,add_loops,[&](std::size_t i){
			BI r=((a&b)|(a^b))^(BI::from_u64(static_cast<std::uint64_t>(i))&a);
			return hash_bi(r);
		});
		bench_one("BigInt shift",bits,add_loops,[&](std::size_t i){
			const std::size_t sh=(i&127u);
			BI r=(a<<sh)>>sh;
			return hash_bi(r);
		});
		bench_one("BigInt add",bits,add_loops,[&](std::size_t i){
			BI r=a+b+BI::from_u64(static_cast<std::uint64_t>(i));
			return hash_bi(r);
		});
		bench_one("BigInt sub",bits,add_loops,[&](std::size_t i){
			BI r=a-b-BI::from_u64(static_cast<std::uint64_t>(i));
			return hash_bi(r);
		});
		bench_one("BigInt mul",bits,mul_loops,[&](std::size_t i){
			BI r=(i&1u)?a*b:b*a;
			return hash_bi(r);
		});
		bench_one("BigInt mul_sbk",bits,std::max<std::size_t>(1,mul_loops/2u),
				  [&](std::size_t){
			return hash_bi(mp::mul_sbk(a,b));
		});
		bench_one("BigInt mul_kar",bits,std::max<std::size_t>(1,mul_loops/2u),
				  [&](std::size_t){
			return hash_bi(mp::mul_kar(a,b));
		});
		bench_one("BigInt mul_t3",bits,std::max<std::size_t>(1,mul_loops/4u),
				  [&](std::size_t){
			return hash_bi(mp::mul_t3(a,b));
		});
		bench_one("BigInt sqr",bits,mul_loops,[&](std::size_t){
			BI r=mp::sqr_disp(a);
			return hash_bi(r);
		});
		bench_one("BigInt divmod",bits,div_loops,[&](std::size_t){
			auto qr=mp::divmod(num,a);
			return hash_bi(qr.first)^hash_bi(qr.second);
		});
		bench_one("BigInt t/f/c div",bits,div_loops,[&](std::size_t){
			auto tq=mp::tdiv_qr(num,b);
			auto fq=mp::fdiv_qr(-num,b);
			auto cq=mp::cdiv_qr(num,-b);
			return hash_bi(tq.first)^hash_bi(fq.second)^hash_bi(cq.first);
		});
		bench_one("BigInt gcd",bits,gcd_loops,[&](std::size_t){
			BI r=mp::gcd(num,b);
			return hash_bi(r);
		});
		bench_one("BigInt lcm",bits,std::max<std::size_t>(1,gcd_loops/2u),
				  [&](std::size_t){
			return hash_bi(mp::lcm(num,b));
		});
		bench_one("BigInt extgcd",bits,std::max<std::size_t>(1,gcd_loops/4u),
				  [&](std::size_t){
			auto r=mp::extgcd(num,b);
			return hash_bi(r.g)^hash_bi(r.x);
		});
		bench_one("BigInt modpow",bits,pow_loops,[&](std::size_t){
			BI r=mp::modpow(a,BI(65537),m);
			return hash_bi(r);
		});
		bench_one("BigInt invert",bits,std::max<std::size_t>(1,pow_loops/2u),
				  [&](std::size_t){
			BI p=BI(1000003);
			BI inv;
			(void)mp::invert(&inv,a%p+BI(1),p);
			return hash_bi(inv);
		});
		bench_one("BigInt sqrtrem",bits,std::max<std::size_t>(1,mul_loops/8u),
				  [&](std::size_t){
			auto r=mp::sqrtrem(a*a);
			return hash_bi(r.first);
		});
		if(bits<=1024u){
			bench_one("BigInt rootrem3",bits,std::max<std::size_t>(1,mul_loops/16u),
					  [&](std::size_t){
				auto r=mp::rootrem(a*a*a,3);
				return hash_bi(r.first);
			});
		}
	}

	std::cout<<"\nCombinatorics / primality timing:\n";
	std::cout<<std::left<<std::setw(18)<<"operation"
			 <<std::right<<std::setw(8)<<"n"
			 <<std::setw(10)<<"loops"
			 <<std::setw(14)<<"total"
			 <<std::setw(14)<<"per-op"<<"\n";
	const std::size_t nt_loops=opt.quick?8u:24u;
	bench_one("factorial",opt.quick?300u:900u,nt_loops,[&](std::size_t){
		return hash_bi(mp::factorial(opt.quick?300u:900u));
	});
	bench_one("binomial",opt.quick?300u:900u,nt_loops,[&](std::size_t){
		const std::uint64_t n=opt.quick?300u:900u;
		return hash_bi(mp::binomial(n,n/2u));
	});
	bench_one("fibonacci",opt.quick?1000u:4000u,nt_loops,[&](std::size_t){
		return hash_bi(mp::fibonacci(opt.quick?1000u:4000u));
	});
	bench_one("next_prime",64,nt_loops,[&](std::size_t i){
		return hash_bi(mp::next_prime(BI::from_u64(1000000007ull+2u*i),8));
	});

	for(std::size_t prec : (opt.quick?std::vector<std::size_t>{64,192}:
						   std::vector<std::size_t>{64,192,512})){
		BF x=BF::parse("1.234567890123456789",prec);
		BF y=BF::parse("9.876543210987654321",prec);
		const std::size_t loops=std::max<std::size_t>(1,1000000/(prec+1));
		BR rx=BR::parse("123456789/100000000");
		BR ry=BR::parse("987654321/100000000");
		bench_one("BigRat add",prec,loops,[&](std::size_t){
			return hash_br(rx+ry);
		});
		bench_one("BigRat mul",prec,loops,[&](std::size_t){
			return hash_br(rx*ry);
		});
		bench_one("BigRat div",prec,std::max<std::size_t>(1,loops/2u),
				  [&](std::size_t){
			return hash_br(rx/ry);
		});
		bench_one("BigFloat add",prec,loops,[&](std::size_t){
			return hash_bf(BF::add(x,y,prec));
		});
		bench_one("BigFloat div",prec,loops,[&](std::size_t){
			return hash_bf(BF::div(y,x,prec));
		});
		bench_one("BigFloat mul",prec,loops,[&](std::size_t){
			return hash_bf(BF::mul(x,y,prec));
		});
		bench_one("BigFloat sqrt",prec,std::max<std::size_t>(1,loops/16u),
				  [&](std::size_t){
			return hash_bf(BF::sqrt(y,prec));
		});
		bench_one("BigFloat exp",prec,std::max<std::size_t>(1,loops/64u),
				  [&](std::size_t){
			return hash_bf(mp::exp(x,prec));
		});
		bench_one("BigFloat log",prec,std::max<std::size_t>(1,loops/64u),
				  [&](std::size_t){
			return hash_bf(mp::log(y,prec));
		});
		bench_one("BigFloat sin",prec,std::max<std::size_t>(1,loops/64u),
				  [&](std::size_t){
			return hash_bf(mp::sin(x,prec));
		});
		BC zx(x,y),zy(y,x);
		bench_one("BigComplex mul",prec,std::max<std::size_t>(1,loops/4u),
				  [&](std::size_t){
			return hash_bc(zx*zy);
		});
		bench_one("BigComplex div",prec,std::max<std::size_t>(1,loops/16u),
				  [&](std::size_t){
			return hash_bc(zx/zy);
		});
	}

	std::cout<<"\nBigNT / EC timing:\n";
	std::cout<<std::left<<std::setw(18)<<"operation"
			 <<std::right<<std::setw(8)<<"size"
			 <<std::setw(10)<<"loops"
			 <<std::setw(14)<<"total"
			 <<std::setw(14)<<"per-op"<<"\n";
	for(std::size_t n : (opt.quick?std::vector<std::size_t>{16,64}:
						std::vector<std::size_t>{16,64,128})){
		std::vector<std::uint64_t> va(n),vb(n);
		for(std::size_t i=0;i<n;++i){
			va[i]=(i*i+3*i+1)%998244353u;
			vb[i]=(7*i+11)%998244353u;
		}
		mp::BigNT::ModPoly pa(va,998244353u),pb(vb,998244353u);
		const std::size_t loops=std::max<std::size_t>(1,20000/(n+1));
		bench_one("ModPoly mul",n,loops,[&](std::size_t){
			return hash_poly(mp::BigNT::ModPoly::mul(pa,pb));
		});
		mp::BigNT::ModPoly modp({1,0,1,1},998244353u);
		bench_one("ModPoly powmod",n,std::max<std::size_t>(1,loops/8u),
				  [&](std::size_t){
			return hash_poly(pa.pow_mod(BI(257),modp));
		});
		std::vector<std::uint64_t> vc(std::max<std::size_t>(2,n/4u));
		for(std::size_t i=0;i<vc.size();++i)
			vc[i]=(11*i*i+5*i+3)%998244353u;
		vc.back()=1;
		mp::BigNT::ModPoly pc(vc,998244353u);
		mp::BigNT::ModPoly ga=mp::BigNT::ModPoly::mul(pa,pc);
		mp::BigNT::ModPoly gb=mp::BigNT::ModPoly::mul(pb,pc);
		bench_one("ModPoly gcd",n,std::max<std::size_t>(1,loops/8u),
				  [&](std::size_t){
			return hash_poly(mp::BigNT::ModPoly::gcd(ga,gb));
		});
	}
	bench_one("factor small",64,opt.quick?16u:48u,[&](std::size_t){
		auto f=mp::BigNT::factor(BI(8051));
		return static_cast<std::uint64_t>(f.size());
	});
	bench_one("pollard rho",64,opt.quick?16u:48u,[&](std::size_t){
		return hash_bi(mp::BigNT::pollard_rho(BI(8051)));
	});
	bench_one("EC mul",64,opt.quick?32u:96u,[&](std::size_t i){
		mp::ec::Curve E{BI(17),BI(2),BI(2)};
		auto G=mp::ec::affine(BI(5),BI(1),E);
		auto P=mp::ec::mul(G,BI::from_u64(3u+static_cast<std::uint64_t>(i%13u)),E);
		return hash_bi(P.x)^hash_bi(P.y);
	});

#if MINI_MP_TEST_HAVE_OLD
	std::cout<<"\nOld/new timing sample:\n";
	std::cout<<std::left<<std::setw(18)<<"operation"
			 <<std::right<<std::setw(8)<<"bits"
			 <<std::setw(10)<<"loops"
			 <<std::setw(14)<<"total"
			 <<std::setw(14)<<"per-op"<<"\n";
	for(std::size_t bits : {256u,1024u,4096u}){
		BI an=positive_random_bi(rng,bits);
		BI bn=positive_random_bi(rng,bits);
		const std::string as=an.to_string(10);
		const std::string bs=bn.to_string(10);
		mini_mp_old::BigInt ao=mini_mp_old::BigInt::parse(as,10);
		mini_mp_old::BigInt bo=mini_mp_old::BigInt::parse(bs,10);
		const std::size_t loops=std::max<std::size_t>(1,100000/(bits+1));
		bench_one("new mul",bits,loops,[&](std::size_t){
			return static_cast<std::uint64_t>((an*bn).bit_length());
		});
		bench_one("old mul",bits,loops,[&](std::size_t){
			return static_cast<std::uint64_t>((ao*bo).bit_length());
		});
	}
#endif
	std::cout<<"sink "<<g_sink<<"\n";
}

Options parse_options(int argc,char**argv){
	Options opt;
	for(int i=1;i<argc;++i){
		const std::string arg=argv[i];
		if(arg=="--quick"){
			opt.quick=true;
			opt.cases=400;
		}else if(arg=="--no-bench"){
			opt.bench=false;
		}else if(arg=="--bench-only"){
			opt.bench_only=true;
			opt.bench=true;
		}else if(arg=="--verbose"){
			opt.verbose=true;
		}else if(arg=="--cases"&&i+1<argc){
			opt.cases=static_cast<std::size_t>(std::strtoull(argv[++i],nullptr,10));
		}else if(arg=="--seed"&&i+1<argc){
			opt.seed=std::strtoull(argv[++i],nullptr,0);
		}else if(arg=="--help"){
			std::cout<<"usage: test [--quick] [--cases N] [--seed N] "
						"[--no-bench] [--bench-only] [--verbose]\n";
			std::exit(0);
		}else{
			throw std::invalid_argument("unknown option: "+arg);
		}
	}
	return opt;
}

int main(int argc,char**argv){
	try{
		Options opt=parse_options(argc,argv);
		mp::autotune_fast();
		run_tests(opt);
		run_benchmarks(opt);
		std::cout<<"\nchecks "<<g_test.checks<<", failures "<<g_test.failed<<"\n";
		if(!g_test.first_failures.empty()){
			std::cout<<"first failures:\n";
			for(const std::string&f : g_test.first_failures)
				std::cout<<"  "<<f<<"\n";
		}
		return g_test.failed==0?0:1;
	}catch(const std::exception&e){
		std::cerr<<"fatal: "<<e.what()<<"\n";
		return 2;
	}
}
