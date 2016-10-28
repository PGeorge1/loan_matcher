#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <string.h>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include "math.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <locale>
#include <algorithm>


/// DEBUG VAR
int global_io_verbose = 0;
FILE *csv_debug_fp;
FILE *txt_debug_fp;

void print_output_to_stdout_and_file_txt(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	va_start(ap, fmt);
	vfprintf(txt_debug_fp, fmt, ap);
	va_end(ap);
}


void print_output_to_stdout_and_file_csv(const char *fmt, ...)
{
	va_list ap;
	
	va_start(ap, fmt);
	vfprintf(csv_debug_fp, fmt, ap);
	va_end(ap);
}

std::string fix_string(std::string value)
{
	std::string result = value;
	while (!result.empty() && (result[result.size() - 1] == '\r' || result[result.size() - 1] == '\n'))
		result.erase(result.size() - 1);
	return result;
}


void readCSV(std::istream &input, std::vector< std::vector<std::string> > &output)
{
	std::string csvLine;
	// read every line from the stream
	while (std::getline(input, csvLine))
	{
		std::istringstream csvStream(csvLine);
		std::vector<std::string> csvColumn;
		std::string csvElement;
		// read every element from the line that is seperated by commas
		// and put it into the vector or strings
		while (std::getline(csvStream, csvElement, ','))
		{
			csvColumn.push_back(fix_string(csvElement));
		}
		output.push_back(csvColumn);
	}
}


class loan_pack_properties
{
public:

	loan_pack_properties(double amount, int number_in_pack, int partno, std::string state) :
		amount(amount), loan_count(number_in_pack), partno(partno), state(state)
	{
	}

	double amount = 0.;
	int loan_count = 0;
	int partno = 0;
	std::string state;
};

class loan_properties
{
public:
	loan_properties(int loan_id, int security_id, double amount, int partno) : loan_id(loan_id), security_id(security_id), amount(amount), partno(partno)
	{

	}

	int loan_id;
	int security_id;
	double amount = 0.;
	int partno = 0;
};

class loan_predicted_state_properties
{
public:
	int loan_id;
	int security_id;
	std::vector<std::string> state;
	std::vector<double> probability;
};

class loans_matcher
{
	struct matrix_item
	{
		bool has_sum = false;
		std::vector<int> previous_indices;
	};

public:
	loans_matcher(int scale = 0, double sigma_error = 0.5, double epsilon = 1e-6, int use_partno = false) :
		scale(scale),
		sigma_error(sigma_error),
		epsilon(epsilon),
		use_partno(use_partno)
	{
		scale_set_by_user = scale > 0 ? true : false;
	}

	std::vector<loan_predicted_state_properties> fit_predict(
		const std::vector<loan_properties> &loan_props,
		const std::vector<loan_pack_properties> &loan_pack_props,
		bool debug_flag = false,
		bool verbose = false)
	{
		clear_all();

		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Total loans in security: %d\n\n", loan_props.size());
		}

		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Setting_params:\n");
		}


		if (verbose)
		{
			if (use_partno)
				print_output_to_stdout_and_file_txt("Use partsno-logic!!\n");
		}

		if (!use_partno)
		{
			set_sigma(loan_pack_props, verbose);
			set_scale(loan_pack_props, verbose);
		}

		set_max_possible_total_amount_by_default(loan_pack_props, verbose);

		if (verbose)
		{
			print_output_to_stdout_and_file_txt("\n");
		}

		get_possible_loans(loan_props, verbose);

		if (verbose)
			print_output_to_stdout_and_file_txt("Building internal structures\n");

		compute_matrix();

		if (verbose)
			print_output_to_stdout_and_file_txt("Total allocated memory: %d bytes\n\n", (int)allocated_memory_size);

		if (debug_flag)
			print_matrix();

		if (verbose)
			print_output_to_stdout_and_file_txt("Building solution\n");


		get_all_possible_loans_for_amount_packs(loan_pack_props);

		check_loan_amounts(loan_props, loan_pack_props, verbose);

		if (verbose)
			print_loan_amounts(loan_props, loan_pack_props);

		return create_output(loan_props, loan_pack_props);
	}

private:
	/// model properties
	unsigned int max_possible_total_amount;
	int sigma;
	int scale;
	double sigma_error;
	double epsilon;

	bool use_partno;
	bool scale_set_by_user;

	std::unique_ptr<matrix_item[]> matrix;
	std::vector<int> loan_values;
	std::vector<std::vector<std::vector<int>>> loan_sums;
	std::vector<int> loan_mapping;

	size_t allocated_memory_size = 0;

	int convert(double price)
	{
		price /= scale;
		return (int)(price + 0.5);
	}

	void get_all_possible_loans_for_amount_packs(const std::vector<loan_pack_properties> &loan_pack_props)
	{
		if (!use_partno)
		{
			for (const loan_pack_properties &loan_pack : loan_pack_props)
			{
				loan_sums.push_back(get_loans_for_total_amount(amount_range(loan_pack.amount), loan_pack.loan_count));
			}
		}
		else
		{
			for (const loan_pack_properties &loan_pack : loan_pack_props)
			{
				std::vector<int> range;
				range.push_back(loan_pack.partno);
				loan_sums.push_back(get_loans_for_total_amount(range, loan_pack.loan_count));
			}
		}
	}

	void get_possible_loans(const std::vector<loan_properties> &loan_props, int verbose)
	{
		std::vector<int> removed;

		for (unsigned int i = 0; i < loan_props.size(); ++i)
		{
			int value = use_partno ? loan_props[i].partno : convert(loan_props[i].amount);

			if (value > 0 && value < (int)max_possible_total_amount)
			{
				loan_values.push_back(value);
				loan_mapping.push_back(i);
			}
			else
			{
				removed.push_back(i);
			}
		}

		if (verbose)
		{
			if (removed.size())
			{
				print_output_to_stdout_and_file_txt("Removing impossible loans. Impossible loans: \n");
				for (auto i : removed)
				{
					if (verbose)
						print_output_to_stdout_and_file_txt("%d (%.2lf, %d); ", loan_props[i].loan_id, loan_props[i].amount, loan_props[i].partno);
				}
				print_output_to_stdout_and_file_txt("\n");
			}
		}
	}

	void set_max_possible_total_amount_by_default(const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
	{
		if (!use_partno)
		{
			double max = 0;
			for (const auto loan_pack : loan_pack_props)
			{
				if (max < loan_pack.amount)
				{
					max = loan_pack.amount;
				}
			}
			max_possible_total_amount = (max / scale) + sigma * 2 + 10;
		}
		else
		{
			max_possible_total_amount = 2000;
		}

		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Max = %d\n", (int)max_possible_total_amount);
		}
	}

	void set_sigma(const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
	{
		int max = 0;
		for (const auto loan_pack : loan_pack_props)
		{
			if (max < loan_pack.loan_count)
			{
				max = loan_pack.loan_count;
			}
		}

		sigma = (int)(max * sigma_error + 0.5);
		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Sigma = %d\n", (int)sigma);
		}
	}

	void set_scale(const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
	{
		if (!scale_set_by_user)
		{
			double max = 0;
			for (const auto loan_pack : loan_pack_props)
			{
				if (max < loan_pack.amount)
				{
					max = loan_pack.amount;
				}
			}

			scale = 1;
			if (max > 100000.)
				scale = 1000;
		}

		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Scale = %d\n", (int)scale);
		}
	}

	void clear_all()
	{
		loan_values.clear();
		loan_sums.clear();
		loan_mapping.clear();
		allocated_memory_size = 0;
		matrix.release();
	}

	void print_loan_amounts(
		const std::vector<loan_properties> &loan_props,
		const std::vector<loan_pack_properties> &loan_pack_props)
	{
		for (unsigned int i = 0; i < loan_pack_props.size(); ++i)
		{
			if (loan_sums[i].size())
			{
				print_output_to_stdout_and_file_txt("For pack %d with amount %.2lf found loans:\n", i, loan_pack_props[i].amount);
				for (const auto &item_vector : loan_sums[i])
				{
					for (auto item : item_vector)
					{
						print_output_to_stdout_and_file_txt("%d ", loan_props[loan_mapping[item]].loan_id);
					}
					print_output_to_stdout_and_file_txt("\n");
				}
			}
		}
	}


	void check_loan_amounts(
		const std::vector<loan_properties> &loan_props,
		const std::vector<loan_pack_properties> &loan_pack_props,
		int verbose)
	{
		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Check result for consistency: eps = %lf\n", epsilon);
		}
		int total_removed = 0;
		for (unsigned int i = 0; i < loan_pack_props.size(); ++i)
		{
			if (loan_sums[i].size())
			{
				auto iterator = loan_sums[i].begin();

				while (iterator != loan_sums[i].end())
				{
					double sum = 0.;
					for (auto item : *iterator)
					{
						sum += loan_props[loan_mapping[item]].amount;
					}
					if (fabs(sum - loan_pack_props[i].amount) > epsilon)
					{
						iterator = loan_sums[i].erase(iterator);
						total_removed++;
					}
					else
					{
						iterator++;
					}
				}
			}
		}
		if (verbose)
		{
			print_output_to_stdout_and_file_txt("Total removed after checking: %d\n\n", total_removed);
		}
	}

	std::vector<int> amount_range(double amount)
	{
		std::vector<int> result;
		for (int i_sigma = -sigma; i_sigma <= sigma; ++i_sigma)
		{
			int value = convert(amount) + i_sigma;
			if (value > 0 && value < (int)max_possible_total_amount)
				result.push_back(convert(amount) + i_sigma);
		}
		return result;
	}

	std::vector<loan_predicted_state_properties> normalize_probabilities(std::vector<loan_predicted_state_properties> loans)
	{
		auto result = loans;
		for (auto &loan : result)
		{
			std::map <std::string, int> m;
			unsigned int total = loan.state.size();
			for (unsigned int i = 0; i < loan.state.size(); ++i)
			{
				if (m.find(loan.state[i]) == m.end())
				{
					m[loan.state[i]] = 1;
				}
				else
				{
					m[loan.state[i]]++;
				}
			}

			loan.state.clear();
			loan.probability.clear();
			for (auto it : m)
			{
				loan.state.push_back(it.first);
				loan.probability.push_back((double)it.second / total);
			}
		}
		return result;
	}

	std::vector<loan_predicted_state_properties> create_output(
		const std::vector<loan_properties> &loan_props,
		const std::vector<loan_pack_properties> &loan_pack_props)
	{
		std::vector<loan_predicted_state_properties> result(loan_props.size());

		for (unsigned int i = 0; i < loan_props.size(); ++i)
		{
			result[i].loan_id = loan_props[i].loan_id;
			result[i].security_id = loan_props[i].security_id;
		}

		for (unsigned int i = 0; i < loan_pack_props.size(); ++i)
		{
			std::string state = loan_pack_props[i].state;

			for (const auto &item_vector : loan_sums[i])
			{
				for (auto item : item_vector)
				{
					result[loan_mapping[item]].state.push_back(state);
				}
			}
		}

		return normalize_probabilities(result);
	}

	void print_matrix()
	{
		for (unsigned int i_loan = 0; i_loan < loan_values.size(); ++i_loan)
		{
			for (unsigned int i = 0; i < max_possible_total_amount; ++i)
			{
				print_output_to_stdout_and_file_txt("(%d, %d) ", matrix[max_possible_total_amount * i_loan + i].has_sum, (int)matrix[max_possible_total_amount * i_loan + i].previous_indices.size());
			}
			print_output_to_stdout_and_file_txt("\n");
		}
		print_output_to_stdout_and_file_txt("\n");
	}


	void compute_matrix()
	{
		allocated_memory_size = loan_values.size() * max_possible_total_amount * sizeof(matrix_item);
		matrix.reset(new matrix_item[loan_values.size() * max_possible_total_amount]);

		/// Initialize
		for (unsigned int i_loan = 0; i_loan < loan_values.size(); ++i_loan)
		{
			matrix_item &item = matrix[max_possible_total_amount * i_loan + loan_values[i_loan]];
			item.has_sum = true;
		}

		/// Compute
		for (unsigned int i_loan = 1; i_loan < loan_values.size(); ++i_loan)
		{
			for (unsigned int i_value = 0; i_value < max_possible_total_amount; ++i_value)
			{
				matrix_item &current_item = matrix[i_loan * max_possible_total_amount + i_value];

				int amount_without_i_loan = i_value - loan_values[i_loan];

				if (amount_without_i_loan > 0)
				{
					for (unsigned int j_loan = 0; j_loan < i_loan; ++j_loan)
					{
						matrix_item &item = matrix[j_loan * max_possible_total_amount + amount_without_i_loan];

						if (item.has_sum)
						{
							current_item.has_sum = true;
							current_item.previous_indices.push_back(j_loan);
						}
					}
				}
			}
		}
	}

	std::vector<std::vector<int>> get_loans_for_total_amount(std::vector<int> total_amounts, int size)
	{
		std::vector<std::vector<int>> result;
		for (auto amount : total_amounts)
		{
			for (unsigned int i_loan = 0; i_loan < loan_values.size(); ++i_loan)
			{
				matrix_item &current_item = matrix[i_loan * max_possible_total_amount + amount];
				if (current_item.has_sum)
				{
					get_all_paths(result, std::vector<int>(), amount, i_loan, size);
				}
			}
		}
		return result;
	}

	void get_all_paths(std::vector<std::vector<int>> &result,
		std::vector<int> current_path,
		int current_sum,
		int i_current_loan,
		unsigned int size)
	{
		if (current_sum < 0 || current_sum >= (int)max_possible_total_amount)
			return;
		matrix_item &current_item = matrix[i_current_loan * max_possible_total_amount + current_sum];
		if (!current_item.has_sum)
			return;

		current_path.push_back(i_current_loan);

		if (current_sum - loan_values[i_current_loan] == 0 && size == current_path.size())
			result.push_back(current_path);

		for (int prev_loan_i : current_item.previous_indices)
		{
			get_all_paths(result, current_path, current_sum - loan_values[i_current_loan], prev_loan_i, size);
		}
	}

};

std::vector<std::vector<loan_pack_properties>> read_securities_as_txt(std::string datafile, bool use_partno)
{
	std::vector<std::vector<loan_pack_properties>> securities;

	FILE * fp_securities = fopen(datafile.c_str(), "r");
	if (!fp_securities)
	{
		print_output_to_stdout_and_file_txt("No such file or directory: %s\n", datafile.c_str());
		return securities;
	}

	int row_counter = 0;

	char tmp[256];
	char c;
	char *r;

	while (!feof(fp_securities))
	{
		row_counter++;
		int security_id, loan_count, partno = 0;
		double amount;
		if (!use_partno)
		{
			if (fscanf(fp_securities, "%d %d %lf %s", &security_id, &loan_count, &amount, tmp) < 4)
			{
				if (feof(fp_securities))
					break;
				if (row_counter != 1)
					print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
				r = fgets(tmp, sizeof(tmp), fp_securities);
				continue;
			}
		}
		else
		{
			if (fscanf(fp_securities, "%d %d %lf %d %s", &security_id, &loan_count, &amount, &partno, tmp) < 5)
			{
				if (feof(fp_securities))
					break;
				if (row_counter != 1)
					print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
				r = fgets(tmp, sizeof(tmp), fp_securities);
				continue;
			}
		}

		if (fscanf(fp_securities, "%c", &c) < 1 || (c != '\n' && c != '\r'))
		{
			print_output_to_stdout_and_file_txt("%s: incorrect data format at row %d, skipping! \n", datafile.c_str(), row_counter);
			r = fgets(tmp, sizeof(tmp), fp_securities);
			continue;
		}

		if (security_id >= (int)securities.size())
		{
			securities.resize(security_id + 1);
		}

		if (global_io_verbose)
		{
			if (!use_partno)
				print_output_to_stdout_and_file_txt("%d %d %lf %s\n", security_id, loan_count, amount, tmp);
			else
				print_output_to_stdout_and_file_txt("%d %d %lf %d %s\n", security_id, loan_count, amount, partno, tmp);
		}

		securities[security_id].push_back(loan_pack_properties(amount, loan_count, partno, std::string(tmp)));
	}
	fclose(fp_securities);
	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("\n");
	return securities;
}

std::vector<std::vector<loan_properties>> read_loans_as_txt(std::string datafile, bool use_partno)
{
	std::vector<std::vector<loan_properties>> loans;

	FILE * fp_loans = fopen(datafile.c_str(), "r");
	if (!fp_loans)
	{
		print_output_to_stdout_and_file_txt("No such file or directory %s:\n", datafile.c_str());
		return loans;
	}

	int row_counter = 0;

	char tmp[1024];
	char c;
	char *r;

	while (!feof(fp_loans))
	{
		row_counter++;
		int loan_id, security_id, partno = 0;
		double amount;

		if (!use_partno)
		{
			if (fscanf(fp_loans, "%s %s %lf", &loan_id, &security_id, &amount) < 3)
			{
				if (feof(fp_loans))
					break;

				if (row_counter != 1)
					print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
				r = fgets(tmp, sizeof(tmp), fp_loans);
				continue;
			}
		}
		else
		{
			if (fscanf(fp_loans, "%s %s %lf %d", &loan_id, &security_id, &amount, &partno) < 4)
			{
				if (feof(fp_loans))
					break;

				if (row_counter != 1)
					print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
				r = fgets(tmp, sizeof(tmp), fp_loans);
				continue;
			}
		}

		if (fscanf(fp_loans, "%c", &c) < 1 || (c != '\n' && c != '\r'))
		{
			print_output_to_stdout_and_file_txt("%s: incorrect data format at row %d, skipping! \n", datafile.c_str(), row_counter);
			r = fgets(tmp, sizeof(tmp), fp_loans);
			continue;
		}


		if (global_io_verbose)
		{
			if (!use_partno)
				print_output_to_stdout_and_file_txt("%d %d %lf\n", security_id, loan_id, amount);
			else
				print_output_to_stdout_and_file_txt("%d %d %lf %d\n", security_id, loan_id, amount, partno);
		}

		if (security_id >= (int)loans.size())
		{
			loans.resize(security_id + 1);
		}
		loans[security_id].push_back(loan_properties(loan_id, security_id, amount, partno));
	}

	fclose(fp_loans);
	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("\n");
	return loans;
}


std::vector<std::vector<loan_pack_properties>> read_securities_as_csv(std::string datafile, bool use_partno)
{
	std::vector<std::vector<loan_pack_properties>> securities;
	std::fstream file(datafile.c_str(), std::ios::in);

	if (!file.is_open())
	{
		print_output_to_stdout_and_file_txt("No such file or directory : %s\n", datafile.c_str());
		return securities;
	}

	typedef std::vector<std::vector<std::string>> csvVector;
	csvVector csvData;

	readCSV(file, csvData);
	// print out read data to prove reading worked
	int row_counter = 0;
	for (auto &string : csvData)
	{
		row_counter++;
		int correct_num = use_partno ? 5 : 4;
		if ((int)string.size() != correct_num)
		{
			print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}

		int security_id;
		int loan_count, partno = 0;
		double amount;
		std::string state;
		try
		{
			if (!use_partno)
			{
				security_id = std::stoi(string[0]);
				loan_count = std::stoi(string[1]);
				amount = std::stod(string[2]);
				state = string[3];
			}
			else
			{
				security_id = std::stoi(string[0]);
				loan_count = std::stoi(string[1]);
				amount = std::stod(string[2]);
				partno = std::stoi(string[3]);
				state = string[4];
			}
		}
		catch (std::invalid_argument& e)
		{
			// if no conversion could be performed
			if (row_counter != 1)
				print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}
		catch (std::out_of_range& e)
		{
			// if the converted value would fall out of the range of the result type
			// or if the underlying function (std::strtol or std::strtoull) sets errno
			// to ERANGE.
			if (row_counter != 1)
				print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}


		if (global_io_verbose)
		{
			if (!use_partno)
			{
				print_output_to_stdout_and_file_txt("%d %d %lf %s\n", security_id, loan_count, amount, state.c_str());
			}
			else
			{
				print_output_to_stdout_and_file_txt("%d %d %lf %d %s\n", security_id, loan_count, amount, partno, state.c_str());
			}
		}


		if (security_id >= (int)securities.size())
		{
			securities.resize(security_id + 1);
		}

		securities[security_id].push_back(loan_pack_properties(amount, loan_count, partno, state));
	}
	file.close();

	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("\n");
	return securities;
}


std::vector<std::vector<loan_properties>> read_loans_as_csv(std::string datafile, bool use_partno)
{
	std::vector<std::vector<loan_properties>> loans;
	std::fstream file(datafile.c_str(), std::ios::in);

	if (!file.is_open())
	{
		print_output_to_stdout_and_file_txt("No such file or directory %s:\n", datafile.c_str());
		return loans;
	}

	typedef std::vector<std::vector<std::string>> csvVector;
	csvVector csvData;

	readCSV(file, csvData);
	// print out read data to prove reading worked
	int row_counter = 0;
	for (auto &string : csvData)
	{
		row_counter++;
		int correct_num = use_partno ? 4 : 3;

		if (string.size() != correct_num)
		{
			print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}
		int loan_id;
		int security_id, partno = 0;
		double amount;
		try
		{
			loan_id = std::stoi(string[0]);
			security_id = std::stoi(string[1]);
			amount = std::stod(string[2]);

			if (use_partno)
			{
				partno = std::stoi(string[3]);
			}
		}
		catch (std::invalid_argument& e)
		{
			// if no conversion could be performed
			if (row_counter != 1)
				print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}
		catch (std::out_of_range& e)
		{
			// if the converted value would fall out of the range of the result type
			// or if the underlying function (std::strtol or std::strtoull) sets errno
			// to ERANGE.
			if (row_counter != 1)
				print_output_to_stdout_and_file_txt("%s: incorrect data at row %d, skipping! \n", datafile.c_str(), row_counter);
			continue;
		}

		if (global_io_verbose)
		{
			if (!use_partno)
				print_output_to_stdout_and_file_txt("%d %d %lf\n", loan_id,  security_id, amount);
			else
				print_output_to_stdout_and_file_txt("%d %d %lf %d\n", loan_id, security_id, amount, partno);
		}
		if (security_id >= (int)loans.size())
		{
			loans.resize(security_id + 1);
		}
		loans[security_id].push_back(loan_properties(loan_id, security_id, amount, partno));
	}
	file.close();
	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("\n");

	return loans;
}


std::string get_extension(std::string datafile)
{
	std::string ext;
	if (datafile.size() > 3)
	{
		int size = datafile.size();

		for (int i = 0; i < 3; ++i)
			ext += datafile[size - 3 + i];
	}
	return ext;
}

int count_words_in_string(const char* str)
{
	if (str == NULL)
		return 0;  // let the requirements define this...

	bool inSpaces = true;
	int numWords = 0;

	while (*str != NULL)
	{
		if (*str == ' ')
		{
			inSpaces = true;
		}
		else if (inSpaces)
		{
			numWords++;
			inSpaces = false;
		}

		++str;
	}

	return numWords;
}

int get_words_number(std::string datafile)
{
	if (get_extension(datafile) == "txt")
	{
		FILE *fp = fopen(datafile.c_str(), "r");
		if (!fp)
			return 0;

		char tmp[1024];
		if (!fgets(tmp, sizeof(tmp), fp))
			return 0;
		fclose(fp);
		return count_words_in_string(tmp);
	}
	else
		if (get_extension(datafile) == "csv")
		{
			std::fstream file(datafile.c_str(), std::ios::in);
			typedef std::vector<std::vector<std::string>> csvVector;
			csvVector csvData;
			readCSV(file, csvData);
			return csvData[0].size();
		}
	return 0;
}

std::vector<std::vector<loan_pack_properties>> read_securities(std::string datafile)
{
	std::string ext = get_extension(datafile);
	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("Reading input securities:\n");

	if (ext == "txt")
	{
		bool use_partno = get_words_number(datafile) == 5 ? true : false;
		return read_securities_as_txt(datafile, use_partno);
	}
	else
		if (ext == "csv")
		{
			bool use_partno = get_words_number(datafile) == 5 ? true : false;
			return read_securities_as_csv(datafile, use_partno);
		}
		else
		{
			print_output_to_stdout_and_file_txt("Unsupported extension\n");
			return std::vector<std::vector<loan_pack_properties>>();
		}
}

std::vector<std::vector<loan_properties>> read_loans(std::string datafile)
{
	std::string ext = get_extension(datafile);
	if (global_io_verbose)
		print_output_to_stdout_and_file_txt("Reading input loans:\n");

	if (ext == "txt")
	{
		bool use_partno = get_words_number(datafile) == 4 ? true : false;
		return read_loans_as_txt(datafile, use_partno);
	}
	else
		if (ext == "csv")
		{
			bool use_partno = get_words_number(datafile) == 4 ? true : false;
			return read_loans_as_csv(datafile, use_partno);
		}
		else
		{
			print_output_to_stdout_and_file_txt("Unsupported extension\n");
			return std::vector<std::vector<loan_properties>>();
		}
}

void print_result_to_stdout(unsigned int /*i_security*/, const std::vector<loan_predicted_state_properties> &result)
{
	print_output_to_stdout_and_file_txt("\nResult:\n");
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		print_output_to_stdout_and_file_txt("%d: ", result[i].loan_id);
		if (result[i].state.size())
		{
			for (unsigned int item = 0; item < result[i].state.size(); ++item)
			{
				print_output_to_stdout_and_file_txt("%s P = %.2lf;  ", result[i].state[item].c_str(), result[i].probability[item]);
				print_output_to_stdout_and_file_csv("%d, %d, %s, %.2lf \n  ", result[i].loan_id, result[i].security_id, result[i].state[item].c_str(), result[i].probability[item]);
			}
		}
		else
		{
			print_output_to_stdout_and_file_txt("None");
		}
		print_output_to_stdout_and_file_txt("\n");
	}
}

struct configs
{
	double sigma_error = 0.5;
	double epsilon = 1e-6;
	int scale = 70;
	int use_partno = 0;
	std::string securities = "securities.csv";
	std::string loans = "loans.csv";
	std::string csv_output = "sec_loan_out.csv";
	std::string txt_output = "sec_loan_out.txt";
	bool verbose = true;
};

configs read_config(std::string configfile)
{
	configs cfg;
	std::fstream file(configfile.c_str(), std::ios::in);
	if (!file.is_open())
	{
		printf("No such file or directory: %s\n", configfile.c_str ());
		return cfg;
	}

	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream is_line(line);
		std::string key;
		if (std::getline(is_line, key, '='))
		{
			std::string value;

			if (std::getline(is_line, value))
			{
				value = fix_string(value);
				if (key == "txt_output")
				{
					cfg.txt_output = value;
				}
				else
					if (key == "csv_output")
					{
						cfg.csv_output = value;
					}
				else
					if (key == "securities")
					{
						cfg.securities = value;
					}
					else
						if (key == "loans")
						{
							cfg.loans = value;
						}
						else
							if (key == "verbose")
							{
								if (value == "true" || value == "True")
									cfg.verbose = 1;
								else
									if (value == "false" || value == "False")
										cfg.verbose = 0;
							}
							else
								if (key == "sigma_error")
								{
									cfg.sigma_error = std::stod(value);
								}
								else
									if (key == "epsilon")
									{
										cfg.epsilon = std::stod(value);
									}
									else
										if (key == "scale")
										{
											cfg.scale = std::stoi(value);
										}
										else
											if (key == "partsno-logic")
											{
												if (value == "true" || value == "True")
													cfg.use_partno = 1;
												else
													if (value == "false" || value == "False")
														cfg.use_partno = 0;
											}
			}
		}
	}
	file.close();
	return cfg;
}

int input_consistency_check(configs cfg)
{
	if ((get_words_number(cfg.securities) != 5 || get_words_number(cfg.loans) != 4) && cfg.use_partno)
	{
		print_output_to_stdout_and_file_txt("Please specify partno data in input files to use partsno-logic\n");
		return -1;
	}

	return 0;
}

void windows_hack_prevent_closing_console()
{
	/// hack for windows. do not close terminal to early
	char c;
	if (scanf("%c", &c))
	{
	}
}

int main(int /*argc*/, char **/*argv*/)
{
	auto cfg = read_config("config.txt");

	if (input_consistency_check(cfg))
	{
		windows_hack_prevent_closing_console();
		return 0;
	}

	txt_debug_fp = fopen(cfg.txt_output.c_str(), "w");
	csv_debug_fp = fopen(cfg.csv_output.c_str(), "w");

	// set DEBUG variable
	global_io_verbose = cfg.verbose;


	std::vector<std::vector<loan_pack_properties>> securities = read_securities(cfg.securities);
	std::vector<std::vector<loan_properties>> loans = read_loans(cfg.loans);
	loans_matcher ln_mtch(cfg.scale, cfg.sigma_error, cfg.epsilon, cfg.use_partno);

	int total_securities = std::min(securities.size(), loans.size());
	for (int i_security = 0; i_security < total_securities; ++i_security)
	{
		if (!securities[i_security].size())
			continue;
		print_output_to_stdout_and_file_txt("-----------------SECURITY %d---------------------\n\n", i_security);
		print_result_to_stdout(i_security, ln_mtch.fit_predict(loans[i_security], securities[i_security], 0, cfg.verbose));
		print_output_to_stdout_and_file_txt("\n\n");
	}

	print_output_to_stdout_and_file_txt("End! Have a nice day!:)\n");
	fclose(txt_debug_fp);
	fclose(csv_debug_fp);

	windows_hack_prevent_closing_console();
	return 0;
}
