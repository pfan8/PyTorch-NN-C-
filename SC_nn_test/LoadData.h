#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>

using namespace std;


string& trim(string &s, char delim);

//读取方式: 逐行读取, 将行读入字符串, 行之间用回车换行区分
//If you want to avoid reading into character arrays, 
//you can use the C++ string getline() function to read lines into strings
void ReadDataFromFileLBLIntoString();

static void _split(const string &s, char delim, vector<string> &elems);

vector<string> split(const string &s, char delim);

// 我将分隔符默认设为空格，当然也可以设为其他字符如','
string extract(string &values, int index, char delim = ' ');

vector<int> get_iv_from_sv(const vector<string> sv);

vector<pair<string, pair<vector<int>, vector<double>>>> load_data(char* filename);

pair<vector<int>, vector<double>> matrix_dot(pair<vector<int>, vector<double>> m1, pair<vector<int>, vector<double>> m2);