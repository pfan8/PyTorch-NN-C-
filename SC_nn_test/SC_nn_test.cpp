#include "stdafx.h"
#include "LoadData.h"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <regex>

int _FEATURE_NUM = 99;

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

// ========================= Activation Function: softmax ===================
template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	_Tp denom = 0.;
	for (int i = 0; i < length; ++i) {
		denom += (_Tp)(exp(src[i]));
	}
	//printf("demon:%f", demon);
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(exp(src[i]) / denom);
	}
	return 0;
}

// ========================= Activation Function: tanh ===================
template<typename _Tp>
int activation_function_tanh(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)((exp(src[i]) - exp(-src[i])) / (exp(src[i]) + exp(-src[i])));
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
	fprintf(stderr, "type: Leaky tanh result: \n");
	activation_function_tanh(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky softmax result: \n");
	activation_function_softmax(src.data(), dst.data(), length);
	print_matrix(dst);
}





int main(int argc, char *argv[])
{
	//test_activation_function();
	vector<int> input_s;
	input_s.push_back(1);
	input_s.push_back(_FEATURE_NUM);
	vector<double> input_m;
	vector<pair<string, pair<vector<int>, vector<double>>>> params = load_data("C:\\Users\\buaal\\Desktop\\model\\net3.model");
	for (int i = 0; i < _FEATURE_NUM; i++)
	{
		input_m.push_back(rand() / double(RAND_MAX));
		//cout << i << ":" << input_m[i] << endl;
	}
	pair<vector<int>, vector<double>> input = make_pair(input_s, input_m);
	pair<vector<int>, vector<double>> result = input;
	bool tanh_flag = false;
	bool ReLU_flag = false;
	bool softmax_flag = false;
	for each (pair<string, pair<vector<int>, vector<double>>> param in params)
	{
		regex weight_e("(.*)\\.weight");
		regex bias_e("(.*)\\.bias");
		smatch bias_match;

		// calculate a linear
		if (std::regex_match(param.first, weight_e))
		{
			pair<vector<int>, vector<double>> temp;
			temp = matrix_dot(result, param.second);
			if (temp.second.size() != 0)
			{
				result = temp;
			}
		}
		// add bias
		else if (std::regex_search(param.first, bias_match, bias_e))
		{
			pair<vector<int>, vector<double>> temp;
			if (result.second.size() == param.second.second.size())
			{
				for (int i = 0; i < result.second.size(); i++)
				{
					result.second[i] += param.second.second[i];
				}
			}
			// check activate function
			switch(stoi(bias_match.format("$1")))
			{
			case 0:
				tanh_flag = true;
				break;
			case 2:
				ReLU_flag = true;
				break;
			case 4:
				softmax_flag = true;
				break;
			default:
				break;
			}
		}
		// activate function
		if (tanh_flag)
		{
			activation_function_tanh(result.second.data(), result.second.data(), result.second.size());
			tanh_flag = false;
		}
		if (ReLU_flag)
		{
			activation_function_ReLU(result.second.data(), result.second.data(), result.second.size());
			ReLU_flag = false;
		}
		if (softmax_flag)
		{
			activation_function_softmax(result.second.data(), result.second.data(), result.second.size());
			softmax_flag = false;
		}
	}
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
	// get win prob
	double win_prob = 1 / (1 + exp(result.second[0] - result.second[1]));
	cout << "win prob:" << win_prob << endl;
	return 0;
}