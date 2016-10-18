#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include "math.h"

/// DEBUG VAR
int global_io_verbose = 0;

std::string fix_string (std::string value)
{
  std::string result = value;
  while (!result.empty () && (result[result.size () - 1] == '\r' || result[result.size() - 1] == '\n'))
    result.erase (result.size () - 1);
  return result;
}


void readCSV (std::istream &input, std::vector< std::vector<std::string> > &output)
{
  std::string csvLine;
  // read every line from the stream
  while( std::getline(input, csvLine) )
    {
      std::istringstream csvStream(csvLine);
      std::vector<std::string> csvColumn;
      std::string csvElement;
      // read every element from the line that is seperated by commas
      // and put it into the vector or strings
      while(std::getline (csvStream, csvElement, ','))
        {
          csvColumn.push_back (fix_string (csvElement));
        }
      output.push_back (csvColumn);
    }
}


class loan_pack_properties
{
public:

  loan_pack_properties (double amount, int number_in_pack,  std::string state) :
    amount (amount), loan_count (number_in_pack), state (state)
  {
  }

  double amount = 0.;
  int loan_count = 0;
  std::string state;
};

class loan_properties
{
public:
  loan_properties (int loan_id, double amount) : loan_id (loan_id), amount (amount)
  {
  }

  int loan_id;
  double amount = 0.;
};

class loan_predicted_state_properties
{
public:
  int loan_id;
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
  loans_matcher (int scale = 0, double sigma_error = 0.3, double epsilon = 1e-6) :
    scale (scale),
    sigma_error (sigma_error),
    epsilon (epsilon)
  {
    scale_set_by_user = scale > 0 ? true : false;
  }

  std::vector<loan_predicted_state_properties> fit_predict (
      const std::vector<loan_properties> &loan_props,
      const std::vector<loan_pack_properties> &loan_pack_props,
      bool debug_flag = false,
      bool verbose = false)
  {
    clear_all ();

    if (verbose)
      {
        std::cout << "Setting params:" << std::endl;
      }
    set_sigma (loan_pack_props, verbose);
    set_scale (loan_pack_props, verbose);
    set_max_possible_total_amount_by_default (loan_pack_props, verbose);

    if (verbose)
      {
        std::cout << std::endl;
      }

    get_possible_loans (loan_props, verbose);

    if (verbose)
      std::cout << "Building internal structures" << std::endl;

    compute_matrix ();

    if (verbose)
      std::cout << "Total allocated memory: " << allocated_memory_size << " bytes\n\n";

    if (debug_flag)
      print_matrix ();

    if (verbose)
      std::cout << "Building solution\n" << std::endl;

    for (const loan_pack_properties &loan_pack : loan_pack_props)
      {
        loan_sums.push_back (get_loans_for_total_amount (amount_range (loan_pack.amount), loan_pack.loan_count));
      }

    check_loan_amounts (loan_props, loan_pack_props, verbose);

    if (verbose)
      print_loan_amounts (loan_props, loan_pack_props);

    return create_output (loan_props, loan_pack_props);
  }

private:
  /// model properties
  unsigned int max_possible_total_amount;
  int sigma;
  int scale;
  double sigma_error;
  double epsilon;

  bool scale_set_by_user;

  std::unique_ptr<matrix_item []> matrix;
  std::vector<int> loan_amounts;
  std::vector<std::vector<std::vector<int>>> loan_sums;
  std::vector<int> loan_mapping;

  size_t allocated_memory_size = 0;

  int convert (double price)
  {
    price /= scale;
    return (int) (price + 0.5);
  }

  void get_possible_loans (const std::vector<loan_properties> &loan_props, int verbose)
  {
    std::vector<int> removed;

    for (unsigned int i = 0; i < loan_props.size (); ++i)
      {
        int value = convert (loan_props[i].amount);

        if (value > 0 && value < (int)max_possible_total_amount)
          {
            loan_amounts.push_back (convert (loan_props[i].amount));
            loan_mapping.push_back (i);
          }
        else
          {
            removed.push_back (i);
          }
      }

    if (verbose)
      {
        if (removed.size ())
          {
            printf ("Removing too big loans. Impossible loans: \n");
            for (auto i : removed)
              {
                if (verbose)
                  printf ("%d (%.2lf); ", loan_props[i].loan_id, loan_props[i].amount);
              }
          }
      }
  }

  void set_max_possible_total_amount_by_default (const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
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
    if (verbose)
      {
        printf ("Max = %d\n", (int)max_possible_total_amount);
      }
  }

  void set_sigma (const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
  {
    int max = 0;
    for (const auto loan_pack : loan_pack_props)
      {
        if (max < loan_pack.loan_count)
          {
            max = loan_pack.loan_count;
          }
      }

    sigma = (int) (max * sigma_error + 0.5);
    if (verbose)
      {
        printf ("Sigma = %d\n", (int)sigma);
      }
  }

  void set_scale (const std::vector<loan_pack_properties> &loan_pack_props, int verbose)
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
        printf ("Scale = %d\n", (int)scale);
      }
  }

  void clear_all ()
  {
    loan_amounts.clear ();
    loan_sums.clear ();
    loan_mapping.clear ();
    allocated_memory_size = 0;
    matrix.release ();
  }

  void print_loan_amounts (
      const std::vector<loan_properties> &loan_props,
      const std::vector<loan_pack_properties> &loan_pack_props)
  {
    for (unsigned int i = 0; i < loan_pack_props.size (); ++i)
      {
        if (loan_sums[i].size ())
          {
            printf ("For pack %d with amount %.2lf found loans:\n", i, loan_pack_props[i].amount);
            for (const auto &item_vector : loan_sums[i])
              {
                for (auto item : item_vector)
                  {
                    printf ("%d ", loan_props[loan_mapping[item]].loan_id);
                  }
                printf ("\n");
              }
          }
      }
  }


  void check_loan_amounts (
      const std::vector<loan_properties> &loan_props,
      const std::vector<loan_pack_properties> &loan_pack_props,
      int verbose)
  {
    if (verbose)
      {
        printf ("Check result for consistency: eps = %lf\n", epsilon);
      }
    int total_removed = 0;
    for (unsigned int i = 0; i < loan_pack_props.size (); ++i)
      {
        if (loan_sums[i].size ())
          {
            auto iterator = loan_sums[i].begin ();

            while (iterator != loan_sums[i].end ())
              {
                double sum = 0.;
                for (auto item : *iterator)
                  {
                    sum += loan_props[loan_mapping[item]].amount;
                  }
                if (fabs (sum - loan_pack_props[i].amount) > epsilon)
                  {
                    iterator = loan_sums[i].erase (iterator);
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
        printf ("Total removed after checking: %d\n\n", total_removed);
      }
  }

  std::vector<int> amount_range (double amount)
  {
    std::vector<int> result;
    for (int i_sigma = -sigma; i_sigma <= sigma; ++i_sigma)
      {
        int value = convert (amount) + i_sigma;
        if (value > 0 && value < (int)max_possible_total_amount)
          result.push_back (convert (amount) + i_sigma);
      }
    return result;
  }

  std::vector<loan_predicted_state_properties> normalize_probabilities (std::vector<loan_predicted_state_properties> loans)
  {
    auto result = loans;
    for (auto &loan : result)
      {
        std::map <std::string,int> m;
        unsigned int total = loan.state.size ();
        for (unsigned int i = 0; i < loan.state.size (); ++i)
          {
            if ( m.find(loan.state[i]) == m.end() )
              {
                m[loan.state[i]] = 1;
              }
            else
              {
                m[loan.state[i]]++;
              }
          }

        loan.state.clear ();
        loan.probability.clear ();
        for (auto it : m)
          {
            loan.state.push_back (it.first);
            loan.probability.push_back ((double) it.second / total);
          }
      }
    return result;
  }

  std::vector<loan_predicted_state_properties> create_output (
      const std::vector<loan_properties> &loan_props,
      const std::vector<loan_pack_properties> &loan_pack_props)
  {
    std::vector<loan_predicted_state_properties> result (loan_props.size ());

    for (unsigned int i = 0; i < loan_props.size (); ++i)
      {
        result[i].loan_id = loan_props[i].loan_id;
      }

    for (unsigned int i = 0; i < loan_pack_props.size (); ++i)
      {
        std::string state = loan_pack_props[i].state;

        for (const auto &item_vector : loan_sums[i])
          {
            for (auto item : item_vector)
              {
                result[loan_mapping[item]].state.push_back (state);
              }
          }
      }

    return normalize_probabilities (result);
  }

  void print_matrix ()
  {
    for (unsigned int i_loan = 0; i_loan < loan_amounts.size (); ++i_loan)
      {
        for (unsigned int i = 0; i < max_possible_total_amount; ++i)
          {
            printf ("(%d, %d) ", matrix[max_possible_total_amount * i_loan + i].has_sum, (int) matrix[max_possible_total_amount * i_loan + i].previous_indices.size ());
          }
        printf ("\n");
      }
    printf ("\n");
  }


  void compute_matrix ()
  {
    allocated_memory_size = loan_amounts.size () * max_possible_total_amount * sizeof (matrix_item);
    matrix.reset (new matrix_item[loan_amounts.size () * max_possible_total_amount]);

    /// Initialize
    for (unsigned int i_loan = 0; i_loan < loan_amounts.size (); ++i_loan)
      {
        matrix_item &item = matrix[max_possible_total_amount * i_loan + loan_amounts[i_loan]];
        item.has_sum = true;
      }

    /// Compute
    for (unsigned int i_loan = 1; i_loan < loan_amounts.size (); ++i_loan)
      {
        for (unsigned int i_value = 0; i_value < max_possible_total_amount; ++i_value)
          {
            matrix_item &current_item = matrix[i_loan * max_possible_total_amount + i_value];

            int amount_without_i_loan = i_value - loan_amounts[i_loan];

            if (amount_without_i_loan > 0)
              {
                for (unsigned int j_loan = 0; j_loan < i_loan; ++j_loan)
                  {
                    matrix_item &item = matrix[j_loan * max_possible_total_amount + amount_without_i_loan];

                    if (item.has_sum)
                      {
                        current_item.has_sum = true;
                        current_item.previous_indices.push_back (j_loan);
                      }
                  }
              }
          }
      }
  }

  std::vector<std::vector<int>> get_loans_for_total_amount (std::vector<int> total_amounts, int size)
  {
    std::vector<std::vector<int>> result;
    for (auto amount : total_amounts)
      {
        for (unsigned int i_loan = 0; i_loan < loan_amounts.size (); ++i_loan)
          {
            matrix_item &current_item = matrix[i_loan * max_possible_total_amount + amount];
            if (current_item.has_sum)
              {
                get_all_paths (result, std::vector<int> (), amount, i_loan, size);
              }
          }
      }
    return result;
  }

  void get_all_paths (std::vector<std::vector<int>> &result,
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

    current_path.push_back (i_current_loan);

    if (current_sum - loan_amounts[i_current_loan] == 0 && size == current_path.size ())
      result.push_back (current_path);

    for (int prev_loan_i : current_item.previous_indices)
      {
        for (int i_sigma = -sigma; i_sigma <= sigma; ++i_sigma)
          get_all_paths (result, current_path, current_sum - loan_amounts[i_current_loan] + i_sigma, prev_loan_i, size);
      }
  }

};

std::vector<std::vector<loan_pack_properties>> read_securities_as_txt (std::string datafile)
    {
  std::vector<std::vector<loan_pack_properties>> securities;

  FILE * fp_securities = fopen (datafile.c_str (), "r");
  if (!fp_securities)
    {
      printf ("No such file or directory: %s\n", datafile.c_str ());
      return securities;
    }

  int row_counter = 0;

  char tmp[256];
  char c;
  char *r;

  while (!feof(fp_securities))
    {
      row_counter++;
      int security_id, loan_count;
      double amount;
      if (fscanf (fp_securities, "%d %d %lf %s", &security_id, &loan_count, &amount, tmp) < 4)
        {
          if (feof (fp_securities))
            break;
          if (row_counter != 1)
            printf ("%s: incorrect data at row %d, skipping! \n", datafile.c_str (), row_counter);
          r = fgets (tmp, sizeof (tmp), fp_securities);
          continue;

        }

      if (fscanf (fp_securities, "%c", &c) < 1 || (c != '\n' && c != '\r'))
        {
          printf ("%s: incorrect data format at row %d, skipping! \n", datafile.c_str (), row_counter);
          r = fgets(tmp, sizeof (tmp), fp_securities);
          continue;
        }

      if (security_id >= (int) securities.size ())
        {
          securities.resize (security_id + 1);
        }

      if (global_io_verbose)
        printf ("%d %lf %d %s\n", security_id, amount, loan_count, tmp);

      securities[security_id].push_back (loan_pack_properties (amount, loan_count, std::string (tmp)));
    }
  fclose (fp_securities);
  if (global_io_verbose)
    printf ("\n");
  return securities;
}

std::vector<std::vector<loan_properties>> read_loans_as_txt (std::string datafile)
{
  std::vector<std::vector<loan_properties>> loans;

  FILE * fp_loans = fopen (datafile.c_str (), "r");
  if (!fp_loans)
    {
      printf ("No such file or directory %s:\n", datafile.c_str ());
      return loans;
    }

  int row_counter = 0;

  char tmp[1024];
  char c;
  char *r;

  while (!feof (fp_loans))
    {
      row_counter++;
      int loan_id, security_id;
      double amount;

      if (fscanf (fp_loans, "%d %d %lf", &loan_id, &security_id, &amount) < 3)
        {
          if (feof (fp_loans))
            break;

          if (row_counter != 1)
            printf ("%s: incorrect data at row %d, skipping! \n", datafile.c_str (), row_counter);
          r = fgets (tmp, sizeof(tmp), fp_loans);
            continue;
        }

      if (fscanf (fp_loans, "%c", &c) < 1 || (c != '\n' && c != '\r'))
        {
          printf ("%s: incorrect data format at row %d, skipping! \n", datafile.c_str (), row_counter);
          r = fgets (tmp, sizeof(tmp), fp_loans);
          continue;
        }


      if (global_io_verbose)
        printf ("%d %d %lf\n", security_id, loan_id, amount);

      if (security_id >= (int)loans.size ())
        {
          loans.resize (security_id + 1);
        }
      loans[security_id].push_back (loan_properties (loan_id, amount));
    }

  fclose (fp_loans);
  if (global_io_verbose)
    printf ("\n");
  return loans;
}

std::vector<std::vector<loan_pack_properties>> read_securities_as_csv (std::string datafile)
{
  std::vector<std::vector<loan_pack_properties>> securities;
  std::fstream file (datafile.c_str (), std::ios::in);

  if (!file.is_open ())
    {
      std::cout << "No such file or directory: " << datafile << std::endl;
      return securities;
    }

  typedef std::vector<std::vector<std::string>> csvVector;
  csvVector csvData;

  readCSV (file, csvData);
  // print out read data to prove reading worked
  int row_counter = 0;
  for (auto &string : csvData)
    {
      row_counter++;
      if (string.size () != 4)
        {
          std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
          continue;
        }

  int security_id;
  int loan_count;
  double amount;
  std::string state;
  try
  {
    security_id = std::stoi (string[0]);
    loan_count = std::stoi (string[1]);
    amount = std::stod (string[2]);
    state = string[3];
  }
  catch (std::invalid_argument& e)
  {
    // if no conversion could be performed
    if (row_counter != 1)
      std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
    continue;
  }
  catch (std::out_of_range& e)
  {
    // if the converted value would fall out of the range of the result type
    // or if the underlying function (std::strtol or std::strtoull) sets errno
    // to ERANGE.
    if (row_counter != 1)
      std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
    continue;
  }


      if (global_io_verbose)
        printf ("%d %lf %d %s\n", security_id, amount, loan_count, state.c_str ());

      if (security_id >= (int) securities.size ())
        {
          securities.resize (security_id + 1);
        }

      securities[security_id].push_back (loan_pack_properties (amount, loan_count, state));
    }
  file.close ();

  if (global_io_verbose)
    printf ("\n");
  return securities;
}

std::vector<std::vector<loan_properties>> read_loans_as_csv (std::string datafile)
{
  std::vector<std::vector<loan_properties>> loans;
  std::fstream file (datafile.c_str (), std::ios::in);

  if (!file.is_open ())
    {
      std::cout << "No such file or directory: " << datafile << std::endl;
      return loans;
    }

  typedef std::vector<std::vector<std::string>> csvVector;
  csvVector csvData;

  readCSV (file, csvData);
  // print out read data to prove reading worked
  int row_counter = 0;
  for (auto &string : csvData)
    {
      row_counter++;
      if (string.size () != 3)
        {
          std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
          continue;
        }
  int loan_id;
  int security_id;
  double amount;
  try
  {
    loan_id = std::stoi (string[0]);
    security_id = std::stoi (string[1]);
    amount = std::stod (string[2]);
  }
  catch (std::invalid_argument& e)
  {
    // if no conversion could be performed
    if (row_counter != 1)
      std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
    continue;
  }
  catch (std::out_of_range& e)
  {
    // if the converted value would fall out of the range of the result type
    // or if the underlying function (std::strtol or std::strtoull) sets errno
    // to ERANGE.
    if (row_counter != 1)
      std::cout << datafile << ": incorrect data at row " << row_counter << ", skipping!\n";
    continue;
  }

      if (global_io_verbose)
        printf ("%d %d %lf\n", security_id, loan_id, amount);
      if (security_id >= (int)loans.size ())
        {
          loans.resize (security_id + 1);
        }
      loans[security_id].push_back (loan_properties (loan_id, amount));
    }
  file.close ();
  if (global_io_verbose)
    printf ("\n");

  return loans;
}


std::string get_extension (std::string datafile)
{
  std::string ext;
  if (datafile.size () > 3)
    {
      int size = datafile.size ();

      for (int i = 0; i < 3; ++i)
        ext += datafile[size - 3 + i];
    }
  return ext;
}

std::vector<std::vector<loan_pack_properties>> read_securities (std::string datafile)
{
  std::string ext = get_extension (datafile);
  if (global_io_verbose)
    printf ("Reading input securities:\n");

  if (ext == "txt")
    {
      return read_securities_as_txt (datafile);
    }
  else
  if (ext == "csv")
    {
      return read_securities_as_csv (datafile);
    }
  else
    {
      std::cout << "Unsupported extension!\n";
      return std::vector<std::vector<loan_pack_properties>> ();
    }
}

std::vector<std::vector<loan_properties>> read_loans (std::string datafile)
{
  std::string ext = get_extension (datafile);
  if (global_io_verbose)
    printf ("Reading input loans:\n");

  if (ext == "txt")
    {
      return read_loans_as_txt (datafile);
    }
  else
  if (ext == "csv")
    {
      return read_loans_as_csv (datafile);
    }
  else
    {
      std::cout << "Unsupported extension!\n";
      return std::vector<std::vector<loan_properties>> ();
    }
}

void print_result_to_stdout (unsigned int /*i_security*/, const std::vector<loan_predicted_state_properties> &result)
{
  printf ("\nResult:\n");
  for (unsigned int i = 0; i < result.size (); ++i)
    {
      printf ("%d: ", result[i].loan_id);
      if (result[i].state.size ())
        {
          for (unsigned int item  = 0; item < result[i].state.size (); ++item)
            {
              printf ("%s P = %lf;  ", result[i].state[item].c_str (), result[i].probability[item]);
            }
        }
      else
        {
          printf ("None");
        }
      printf ("\n");
    }
}

struct configs
{
  double sigma_error = 0.3;
  double epsilon = 1e-6;
  int scale = 0;
  std::string securities = "securities.txt";
  std::string loans = "loans.txt";
  bool verbose = true;
};

configs read_config (std::string configfile)
{
  configs cfg;
  std::fstream file (configfile.c_str (), std::ios::in);
  if (!file.is_open ())
    {
      std::cout << "No such file or directory: " << configfile << std::endl;
      return cfg;
    }

  std::string line;
  while( std::getline(file, line) )
  {
    std::istringstream is_line(line);
    std::string key;
    if( std::getline(is_line, key, '=') )
    {
      std::string value;

      if( std::getline(is_line, value) )
        {
          value = fix_string (value);

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
              cfg.sigma_error = std::stod (value);
            }
          else
          if (key == "epsilon")
            {
              cfg.epsilon = std::stod (value);
            }
          else
          if (key == "scale")
            {
              cfg.scale = std::stoi (value);
            }
        }
    }
  }
  file.close ();
  return cfg;
}


int main (int /*argc*/, char **/*argv*/)
{
  auto cfg = read_config ("config.txt");

  // set DEBUG variable
  global_io_verbose = cfg.verbose;

  std::vector<std::vector<loan_pack_properties>> securities = read_securities (cfg.securities);
  std::vector<std::vector<loan_properties>> loans = read_loans (cfg.loans);
  loans_matcher ln_mtch (cfg.scale, cfg.sigma_error, cfg.epsilon);

  int total_securities = std::min (securities.size (), loans.size ());
  for (int i_security = 0; i_security < total_securities; ++i_security)
    {
      if (!securities[i_security].size ())
        continue;
      printf ("-----------------SECURITY %d---------------------\n\n", i_security);
      print_result_to_stdout (i_security, ln_mtch.fit_predict (loans[i_security], securities[i_security], 0, cfg.verbose));
      printf ("\n\n");
    }


  /// hack for windows. do not close terminal to early
//  char c;
//  if (scanf ("%c", &c))
//    {
//    }

  return 0;
}
