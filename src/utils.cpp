#include "../include/utils.h"
#include <sstream>
#include <fstream>
#include <unordered_set>

namespace utils {
    matrix::CSR readCSR(const std::string &filename) {
        std::vector<double> values;
        std::vector<int> column_index;
        std::vector<int> offset_index;
        int n_rows, n_columns, nnz, max_nnz_in_row;

        std::ifstream file_stream(filename);
        std::string line;
        // read metadata
        {
            std::getline(file_stream, line);
            std::istringstream iss(line);
            iss >> n_rows >> n_columns >> nnz >> max_nnz_in_row;
        }
        // read CSR values
        {
            std::getline(file_stream, line);
            std::istringstream iss(line);
            values.resize(nnz);
            for (int i = 0; i < nnz; ++i) {
                iss >> values[i];
            }
        }
        // read CSR offset_index (row_offset)
        {
            std::getline(file_stream, line);
            std::istringstream iss(line);
            offset_index.resize(n_rows + 1);
            for (int i = 0; i < n_rows + 1; ++i) {
                iss >> offset_index[i];
            }
        }
        // read CSR column_index
        {
            std::getline(file_stream, line);
            std::istringstream iss(line);
            column_index.resize(nnz);
            for (int i = 0; i < nnz; ++i) {
                iss >> column_index[i];
            }
        }
        // Create CSRMatrix
        return matrix::CSR(n_rows, std::move(values), std::move(column_index),
                           std::move(offset_index));
    }

    void Args::parse(int argc, char **argv) {
        std::unordered_set<std::string> unary_args{"-i", "-v", "-acs"};
        std::unordered_set<std::string> binary_args{"-f", "-c", "-g", "-e", "-s", "-o"};

        int i = 1;
        while (i < argc) {
            if (binary_args.find(argv[i]) != binary_args.end()) {
                if (i + 1 == argc) {
                    throw std::runtime_error(
                            std::string("missing second argument for: ") + std::string(argv[i]));
                }
                std::string val = std::string(argv[i + 1]);
                if (argv[i] == std::string("-f")) {
                    this->sparse_matrix_file = val;
                } else if (argv[i] == std::string("-c")) {
                    this->replication_group_size = std::stoi(val);
                } else if (argv[i] == std::string("-g")) {
                    this->ge_value = std::make_unique<double>(std::stod(val));
                } else if (argv[i] == std::string("-e")) {
                    this->exponent = std::stoi(val);
                } else if (argv[i] == std::string("-s")) {
                    this->seed = std::stoi(val);
                } else if (argv[i] == std::string("-o")) {
                    this->outfile = val;
                }
                i += 2;
            } else if (unary_args.find(argv[i]) != unary_args.end()) {
                if (argv[i] == std::string("-i")) {
                    this->inner_abc = true;
                } else if (argv[i] == std::string("-v")) {
                    this->verbose = true;
                } else if (argv[i] == std::string("-acs")) {
                    this->async_computation_stage = false;
                }
                i += 1;
            } else {
                throw std::runtime_error(
                        std::string("unexpected argument: ") + std::string(argv[i]));
            }
        }

        if (this->sparse_matrix_file.empty()) {
            throw std::runtime_error("[-f sparse_matrix_file] is required");
        } else if (this->replication_group_size <= 0) {
            throw std::runtime_error(
                    "[-c replication_group_size] is required, replication_group_size > 0");
        } else if (this->exponent < 0) {
            throw std::runtime_error("[-e ge_value] is required, ge_value >= 0");
        } else if (this->seed < 0) {
            throw std::runtime_error("[-s] seed is required, s >= 0");
        } else if (this->verbose == true && this->ge_value != nullptr) {
            throw std::runtime_error("[-v] and [-g] are mutually exclusive");
        } else if (!this->outfile.empty() && this->ge_value != nullptr) {
            throw std::runtime_error("[-o] and [-g] are mutually exclusive");
        }
    }
}
