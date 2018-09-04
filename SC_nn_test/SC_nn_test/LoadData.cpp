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


string& trim(string &s, char delim)
{
	if (s.empty())
	{
		return s;
	}

	s.erase(0, s.find_first_not_of(delim));
	s.erase(s.find_last_not_of(delim) + 1);
	return s;
}

//读取方式: 逐行读取, 将行读入字符串, 行之间用回车换行区分
//If you want to avoid reading into character arrays, 
//you can use the C++ string getline() function to read lines into strings
void ReadDataFromFileLBLIntoString()
{
	ifstream fin("data.txt");
	string s;
	while (getline(fin, s))
	{
		cout << "Read from file: " << s << endl;
	}
}

static void _split(const string &s, char delim,
	vector<string> &elems) {
	stringstream ss(s);
	string item;

	while (getline(ss, item, delim)) {
		item = trim(item,' ');
		elems.push_back(item);
	}
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	_split(s, delim, elems);
	return elems;
}

// 我将分隔符默认设为空格，当然也可以设为其他字符如','
string extract(string &values, int index, char delim = ' ') {
	if (values.length() == 0)
		return string("");

	vector<string> x = split(values, delim);
	try {
		return x.at(index);
	}
	catch (const out_of_range& e) {
		return string("");  // 要是访问超出范围的元素，我们就返回空串
	}
}

//string的vector转化成int的vector
vector<int> get_iv_from_sv(const vector<string> sv)
{
	vector<int> result;
	for each (string s in sv)
	{
		result.push_back(stoi(s));
	}
	return result;
}

pair<vector<int>, vector<double>> matrix_dot(pair<vector<int>, vector<double>> m1, pair<vector<int>, vector<double>> m2)
{
	pair<vector<int>, vector<double>> null_result;
	if (m1.first.size() < 1 || m2.first.size() < 1)
	{
		cerr << "matrix dim must bigger than one!";
		return null_result;
	}
	int m1_row = m1.first[m1.first.size() - 1];
	int m2_col = m2.first[m2.first.size() - 2];
	cout << "m1_row:" << m1_row << endl;
	cout << "m2_col:" << m2_col << endl;
	int dot_num = m1_row;
	int jump_num = 1;
	int temp_num = 1; //用于相乘中循环
	vector<double> result_matrix_v;
	vector<int> result_size_v;

	// 检查能否相乘
	if (m1_row != m2_col)
	{
		cerr << "the two matrix doesn't match the dot principle !";
		return null_result;
	}

	//记录相乘后矩阵的形状
	for (int i = 0; i < m1.first.size() - 1; i++)
	{
		result_size_v.push_back(m1.first[i]);
	}
	for (int i = 0; i < m2.first.size(); i++)
	{
		if (i != m2.first.size() - 2)
		{
			result_size_v.push_back(m2.first[i]);
		}
	}

	//m2每一次相乘需要跳跃的数量
	if (m2.first.size() > 2)
	{
		jump_num = m2.first[m2.first.size() - 1];
	}
	for (int i = 1; i < m2.first.size(); i++)
	{
		temp_num *= m2.first[i];
	}
	double result = 0.0;
	for (int i = 0; i < temp_num; i++)
	{
		for (int j = 0; j < m1.second.size(); j++)
		{
			result += m1.second[j] * m2.second[jump_num * j + dot_num * i];
			if (((j + 1) % dot_num) == 0)
			{
				result_matrix_v.push_back(result);
				result = 0;
			}
		}
	}
	return make_pair(result_size_v, result_matrix_v);
}

//读取方式: 逐行读取, 将行读入字符数组, 行之间用回车换行区分
//If we were interested in preserving whitespace, 
//we could read the file in Line-By-Line using the I/O getline() function.
vector<pair<string, pair<vector<int>, vector<double>>>> load_data(char* filename)
{
	ifstream fin(filename);
	if (!fin)
	{
		cout << "Error opening " << filename << " for input" << endl;
		exit(-1);
	}
	string s;
	vector<pair<string, pair<vector<int>, vector<double>>>> matrix_v;
	smatch m1;
	regex e1("dict\\[(.*)\\]");
	smatch m2;
	regex e2("torch.Size\\(\\[(.*)\\]\\)");
	regex elb("(.*)\\[(.*)");
	regex erb("(.*)\\](.*)");
	regex rb("\\]");
	string name;
	vector<string> size_v;
	vector<double> matrix_p;
	bool name_set = false;
	bool sizev_set = false;

	while (getline(fin, s))
	{
		if (s == "\n")
			continue;
		//cout << "Read from file: " << s << endl;
		if (regex_search(s, m1, e1))
		{
			name = m1.format("$1");
			name_set = true;
		}
		else if (regex_search(s, m2, e2))
		{
			string size = m2.format("$1");
			size_v = split(size, ',');
			sizev_set = true;
		}
		else
		{
			vector<string> data = split(s, ',');
			int data_size = data.size();
			int match_count = 0;
			for (int i = 0; i < data_size; i++)
			{
				while (regex_match(data[i], elb))
					data[i] = trim(data[i], '[');
				while (regex_match(data[i], erb))
				{
					match_count = std::distance(
						std::sregex_iterator(data[i].begin(), data[i].end(), rb),
						std::sregex_iterator());
					data[i] = trim(data[i], ']');
				}
				matrix_p.push_back(stod(data[i]));
				if (match_count == size_v.size())
				{
					vector<int> isv = get_iv_from_sv(size_v);
					reverse(isv.begin(), isv.end());
					matrix_v.push_back(make_pair(name, make_pair(isv, matrix_p)));
					matrix_p.clear();
					name_set = false;
					sizev_set = false;
				}
			}
		}
	}
	for each (pair<string, pair<vector<int>, vector<double>>> var in matrix_v)
	{

		cout << var.first << " length:" << var.second.second.size() << endl;
		cout << var.first << " size:(";
		for each (int size in var.second.first)
		{
			cout << size << ",";
		}
		cout << ")" << endl;
	}
	return matrix_v;
}

//int main()
//{
//
//	// regex_search example
//	//string s("this subject has a submarine as a subsequence");
//	//smatch m;
//	//regex e("\\b(sub)([^ ]*)");   // matches words beginning by "sub"
//
//	//cout << "Target sequence: " << s << endl;
//	//cout << "Regular expression: /\\b(sub)([^ ]*)/" << endl;
//	//cout << "The following matches and submatches were found:" << endl;
//
//	//while (regex_search(s, m, e)) {
//	//	for (auto x = m.begin(); x != m.end(); x++)
//	//		cout << x->str() << " " << endl;
//	//	cout << "--> ([^ ]*) match " << m.format("$2") << endl;
//	//	s = m.suffix().str();
//	//}
//
	//vector<pair<string, pair<vector<int>, vector<double>>>> v;
//	v = load_data("D:\\AIIDE\\analyzer\\replay处理脚本\\sc_nn_pytorch.model");
//}