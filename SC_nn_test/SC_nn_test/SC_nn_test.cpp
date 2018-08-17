#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

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


// =============================== ¼ÆËã sigmoidº¯Êý ==========================
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

int test_activation_function()
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

	return 0;
}



//int main(int argc, char *argv[])
//{
//	test_activation_function();
//}