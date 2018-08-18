#include "stdafx.h"
#include "LoadData.h"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <regex>

int _FEATURE_NUM = 35;

// ========================= Activation Function: ELUs ========================
template<typename _Tp>
int activation_function_ELUs(const _Tp* src, _Tp* dst, int length, _Tp a = 1.)
{
	if (a < 0) {
		fprintf(stderr, "a is a hyper-parameter to be tuned and a>=0 is a constraint\n");
		return -1;
	}

	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] >= (_Tp)0. ? src[i] : (a * (exp(src[i]) - (_Tp)1.));
	}

	return 0;
}

// ========================= Activation Function: Leaky_ReLUs =================
template<typename _Tp>
int activation_function_Leaky_ReLUs(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] >(_Tp)0. ? src[i] : (_Tp)0.01 * src[i];
	}

	return 0;
}

// ========================= Activation Function: ReLU =======================
template<typename _Tp>
int activation_function_ReLU(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = std::max((_Tp)0., src[i]);
	}

	return 0;
}

// ========================= Activation Function: softplus ===================
template<typename _Tp>
int activation_function_softplus(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = log((_Tp)1. + exp(src[i]));
	}

	return 0;
}


// =============================== 计算 sigmoid函数 ==========================
template<typename _Tp>
int activation_function_sigmoid(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(1. / (1. + exp(-src[i])));
	}

	return 0;
}

template<typename _Tp>
int activation_function_sigmoid_fast(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(src[i] / (1. + fabs(src[i])));
	}

	return 0;
}

void print_matrix(std::vector<double> mat)
{
	fprintf(stderr, "[");
	for each (double var in mat)
	{
		fprintf(stderr, "%0.9f, ",var);
	}
	fprintf(stderr, "]\n");
}

void test_activation_function()
{
	std::vector<double> src{ 1.23f, 4.14f, -3.23f, -1.23f, 5.21f, 0.234f, -0.78f, 6.23f };
	int length = src.size();
	std::vector<double> dst(length);

	fprintf(stderr, "source vector: \n");
	print_matrix(src);
	fprintf(stderr, "calculate activation function:\n");
	fprintf(stderr, "type: sigmoid result: \n");
	activation_function_sigmoid(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: sigmoid fast result: \n");
	activation_function_sigmoid_fast(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: softplus result: \n");
	activation_function_softplus(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: ReLU result: \n");
	activation_function_ReLU(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky ReLUs result: \n");
	activation_function_Leaky_ReLUs(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky ELUs result: \n");
	activation_function_ELUs(src.data(), dst.data(), length);
	print_matrix(dst);
	
}





int main(int argc, char *argv[])
{
	//test_activation_function();
	vector<int> input_s;
	input_s.push_back(1);
	input_s.push_back(_FEATURE_NUM);
	vector<double> input_m;
	vector<pair<string, pair<vector<int>, vector<double>>>> params = load_data("D:\\AIIDE\\analyzer\\replay处理脚本\\sc_nn_pytorch.model");
	bool ReLU_flag = false;
	for (int i = 0; i < _FEATURE_NUM; i++)
	{
		input_m.push_back(rand() / double(RAND_MAX));
		//cout << i << ":" << input_m[i] << endl;
	}
	pair<vector<int>, vector<double>> input = make_pair(input_s, input_m);
	pair<vector<int>, vector<double>> result = input;
	for each (pair<string, pair<vector<int>, vector<double>>> param in params)
	{
		regex weight_e("(.*)\\.weight");
		smatch weight_match;
		regex bias_e("(.*)\\.bias");
		if (std::regex_search(param.first, weight_match, weight_e))
		{
			if (stoi(weight_match.format("$1")) == 1)
			{
				ReLU_flag = true;
			}
			pair<vector<int>, vector<double>> temp;
			temp = matrix_dot(result, param.second);
			if (temp.second.size() != 0)
			{
				result = temp;
			}
		}
		//加上bias
		else if (std::regex_match(param.first, bias_e))
		{
			pair<vector<int>, vector<double>> temp;
			if (result.second.size() == param.second.second.size())
			{
				for (int i = 0; i < result.second.size(); i++)
				{
					result.second[i] += param.second.second[i];
				}
			}
		}
		if (ReLU_flag)
		{
			activation_function_ReLU(result.second.data(), result.second.data(), result.second.size());
			ReLU_flag = false;
		}
	}
	//pair<vector<int>, vector<double>> result = matrix_dot(input, params[0].second);
	cout << "size:";
	for each (int var in result.first)
	{
		cout << var << ",";
	}
	cout << endl << "result:[";
	for each (double var in result.second)
	{
		cout << var << ",";
	}
	cout << "]" << endl;
	return 0;
}