#ifndef MPI_MATRIX_MULTIPLICATION_UTILS_H
#define MPI_MATRIX_MULTIPLICATION_UTILS_H

#include "matrix.h"
#include <string>
#include <memory>

namespace utils {
    struct Args {
        // Obligatory
        std::string sparse_matrix_file;
        int replication_group_size = -1;
        int exponent = -1;
        int seed = -1;
        // Options
        bool inner_abc = false;
        std::unique_ptr<double> ge_value = nullptr;
        bool verbose = false;
        std::string outfile;
        // On by default
        bool async_computation_stage = true;

        void parse(int argc, char **argv);
    };

    /**
     * Format assumption:
     * The first row contains 4 integers: the number of rows, the number of columns,
     * the total number of non-zero elements, and the maximum number of non-zero elements in each row.
     * The following 3 rows specify entries (array A in wikipedia's description):
     * values, column indices, and row offsets.
     */
    matrix::CSR readCSR(const std::string& filename);
}

#endif //MPI_MATRIX_MULTIPLICATION_UTILS_H
