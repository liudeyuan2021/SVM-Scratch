// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>


static std::string capitalizeString(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return s;
}

static void sanitize_name(char* name)
{
	for (std::size_t i = 0; i < strlen(name); i++)
	{
		if (!isalnum(name[i]))
		{
			name[i] = '_';
		}
		if (isdigit(name[i]))
		{
			
		}
	}
}

static std::string path_to_varname(const char* path)
{
	const char* lastslash = strrchr(path, '/');
	const char* name = lastslash == NULL ? path : lastslash + 1;

	std::string varname = name;
	sanitize_name((char*)varname.c_str());

	return varname;
}

static bool vstr_is_float(const char vstr[16])
{
	// look ahead for determine isfloat
	for (int j = 0; j < 16; j++)
	{
		if (vstr[j] == '\0')
			break;

		if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
			return true;
	}

	return false;
}


static int write_memcpp(const char* modelpath, const char* memcpppath)
{
	FILE* cppfp = fopen(memcpppath, "wb");

	// dump model
	std::string model_var = path_to_varname(modelpath);
	std::string include_guard_var = path_to_varname(memcpppath);

	fprintf(cppfp, "#ifndef SVM_INCLUDE_GUARD_%s\n", capitalizeString(include_guard_var).c_str());
	fprintf(cppfp, "#define SVM_INCLUDE_GUARD_%s\n", capitalizeString(include_guard_var).c_str());

	FILE* bp = fopen(modelpath, "rb");

	if (!bp) {
		fprintf(stderr, "fopen %s failed\n", modelpath);
		return -1;
	}

	fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
	fprintf(cppfp, "static const unsigned char %s[] = {\n", model_var.substr(3).c_str());

	int i = 0;
	while (!feof(bp))
	{
		int c = fgetc(bp);
		if (c == EOF)
			break;
		fprintf(cppfp, "0x%02x,", c);

		i++;
		if (i % 16 == 0)
		{
			fprintf(cppfp, "\n");
		}
	}

	fprintf(cppfp, "};\n");

	fprintf(cppfp, "#endif // SVM_INCLUDE_GUARD_%s\n", capitalizeString(include_guard_var).c_str());

	fclose(bp);

	fclose(cppfp);

	return 0;
}

// int main(int argc, char** argv)
// {
// 	if (argc != 3)
// 	{
// 		fprintf(stderr, "Usage: %s orginal  file name---->mem file name\n", argv[0]);
// 		return -1;
// 	}

// 	const char* modelpath = argv[1];
// 	const char* memcpppath = argv[2];

// 	write_memcpp(modelpath, memcpppath);

// 	return 0;
// }

int main(int argc, char** argv)
{
	write_memcpp("../model/bin/01_support_int32.bin", "../model/include/svm_01_support.h");
	write_memcpp("../model/bin/02_SV_float32.bin", "../model/include/svm_02_SV.h");
	write_memcpp("../model/bin/03_nSV_int32.bin", "../model/include/svm_03_nSV.h");
	write_memcpp("../model/bin/04_sv_coef_float32.bin", "../model/include/svm_04_svCoef.h");
	write_memcpp("../model/bin/05_intercept_float32.bin", "../model/include/svm_05_intercept.h");
	write_memcpp("../model/bin/06_svm_type_float32.bin", "../model/include/svm_06_svmType.h");
	write_memcpp("../model/bin/07_kernel_int32_string.bin", "../model/include/svm_07_kernel.h");
	write_memcpp("../model/bin/08_degree_int32.bin", "../model/include/svm_08_degree.h");
	write_memcpp("../model/bin/09_gamma_float32.bin", "../model/include/svm_09_gamma.h");
	write_memcpp("../model/bin/10_coef0_float32.bin", "../model/include/svm_10_coef0.h");

	return 0;
}