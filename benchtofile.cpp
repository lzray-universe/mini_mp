#include "mini_mp.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc,char**argv){
	const std::string out_path=(argc>1)?argv[1]:"mini_mp_tuned_template.cpp";
	mini_mp::autotune_full();

	std::ofstream out(out_path,std::ios::binary);
	if(!out){
		std::cerr<<"failed to open output file: "<<out_path<<"\n";
		return 1;
	}

	const auto&s=mini_mp::detail::at_state();
	auto put_off=[&](std::size_t v){
		if(v==mini_mp::detail::kNttOff)
			out<<"mini_mp::detail::kNttOff";
		else
			out<<v<<"u";
	};

	out<<"#include \"mini_mp.hpp\"\n";
	out<<"#include <iostream>\n\n";
	out<<"namespace app_mp{\n\n";
	out<<"inline void tune_once(){\n";
	out<<"\tstatic bool done=false;\n";
	out<<"\tif(done)\n";
	out<<"\t\treturn;\n";
	out<<"\tdone=true;\n";
	out<<"\tauto&s=mini_mp::detail::at_state();\n";
	out<<"\ts.ntt_th="; put_off(s.ntt_th); out<<";\n";
	out<<"\ts.ntt_sq_th="; put_off(s.ntt_sq_th); out<<";\n";
	out<<"\ts.ntt_bits="<<s.ntt_bits<<"u;\n";
	out<<"\ts.kar_rec="<<s.kar_rec<<"u;\n";
	out<<"\ts.sqr_rec="<<s.sqr_rec<<"u;\n";
	out<<"\ts.kar_th="<<s.kar_th<<"u;\n";
	out<<"\ts.kar_imb="<<s.kar_imb<<"u;\n";
	out<<"\ts.kar_dif="; put_off(s.kar_dif); out<<";\n";
	out<<"\ts.ntt_imb="<<s.ntt_imb<<"u;\n";
	out<<"\ts.hl_min="<<s.hl_min<<"u;\n";
	out<<"\ts.hl_rnd="<<s.hl_rnd<<"u;\n";
	out<<"\ts.gcd_sm="<<s.gcd_sm<<"u;\n";
	out<<"\ts.gcd_lg="<<s.gcd_lg<<"u;\n";
	out<<"\ts.gcd_qs="<<s.gcd_qs<<"u;\n";
	out<<"\ts.bz_min="<<s.bz_min<<"u;\n";
	out<<"\ts.bz_chunk="<<s.bz_chunk<<"u;\n";
	out<<"\ts.prod_leaf="<<s.prod_leaf<<"u;\n";
	out<<"\ts.fac_tree="<<s.fac_tree<<"u;\n";
	out<<"\ts.binom_tree="<<s.binom_tree<<"u;\n";
	out<<"\ts.pow_w5="<<s.pow_w5<<"u;\n";
	out<<"\ts.pow_w6="<<s.pow_w6<<"u;\n";
	out<<"\ts.d10_dc="; put_off(s.d10_dc); out<<";\n";
	out<<"\ts.d10_prs="; put_off(s.d10_prs); out<<";\n";
	out<<"\ts.t3_th="; put_off(s.t3_th); out<<";\n";
	out<<"\tmini_mp::detail::at_done().store(2u,std::memory_order_release);\n";
	out<<"}\n\n";
	out<<"using BigInt=mini_mp::BigInt;\n";
	out<<"using BigRat=mini_mp::BigRat;\n\n";
	out<<"inline BigInt parse(std::string_view s,int base=10){\n";
	out<<"\ttune_once();\n";
	out<<"\treturn BigInt::parse(s,base);\n";
	out<<"}\n\n";
	out<<"}\n\n";
	out<<"int main(){\n";
	out<<"\tapp_mp::tune_once();\n";
	out<<"\tapp_mp::BigInt a=app_mp::parse(\"12345678901234567890\");\n";
	out<<"\tapp_mp::BigInt b=app_mp::parse(\"98765432109876543210\");\n";
	out<<"\tstd::cout<<(a*b).to_string(10)<<\"\\n\";\n";
	out<<"}\n";

	if(!out){
		std::cerr<<"failed to write output file: "<<out_path<<"\n";
		return 1;
	}
	std::cout<<"wrote "<<out_path<<"\n";
	std::cout<<"kar_th "<<s.kar_th<<"\n";
	std::cout<<"kar_rec "<<s.kar_rec<<"\n";
	std::cout<<"sqr_rec "<<s.sqr_rec<<"\n";
	std::cout<<"kar_dif "<<s.kar_dif<<"\n";
	std::cout<<"gcd_sm "<<s.gcd_sm<<"\n";
	std::cout<<"gcd_lg "<<s.gcd_lg<<"\n";
	std::cout<<"gcd_qs "<<s.gcd_qs<<"\n";
	return 0;
}
