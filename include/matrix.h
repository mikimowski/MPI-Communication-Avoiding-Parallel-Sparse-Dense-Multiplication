#ifndef MPI_MATRIX_MULTIPLICATION_MATRIX_H
#define MPI_MATRIX_MULTIPLICATION_MATRIX_H

#include <vector>
#include <iostream>

namespace matrix {
    class CSR {
        /** Stores n x n matrix M in CSR format.
        * https://en.wikipedia.org/wiki/Sparse_matrix
        **/
    public:
        int n;
        // size == NNZ(M). Contains non-zero values.
        std::vector<double> values;
        // size == NNZ(M). Contains column indices of those non-zero values.
        std::vector<int> column_index;
        // size == n_rows + 1. Encodes common index in `values` && `column_index where description
        // of the given row starts (row_index).
        std::vector<int> offset_index;

        /**
         * Assumption: Square matrix.
         * Invariant: values.size() == column_index.size()
         * Invariant: offset_index.size() == n + 1
         *
         * @param n :- n_rows == n_columns
         * @param values :- values in CSR format
         * @param column_index :- column_index in CSR format
         * @param offset_index :- offset_index in CSR format
         */
        CSR(int n, std::vector<double> &&values,
            std::vector<int> &&column_index, std::vector<int> &&offset_index);

        /** Creates empty CSR with fields of a given sizes */
        CSR(int n, int values_size);

        CSR(CSR &&lhs, CSR &&rhs);

        friend std::ostream &operator<<(std::ostream &os, const CSR &matrix);

        int getNumRows() const;

        int getNumColumns() const;

        std::string to_string();
    };

    struct DenseBlock {
        /**
         * Dense matrix representation.
         * Because matrix will be distributed across multiple processes
         * additional metadata is stored about local values.
         *
         * If n_columns = n_rows/n_processes is not integer then padding is added,
         * therefore block for each process has the same shape.
         *
         * Design choice:
         * 1. 1.5D ColA algorithm does not split dense matrix B
         * 2. 1.5D InnerABC splits dense matrix B into p/c columns blocks,
         * therefore we will keep this data
         */

        // Each process stores all rows
        int _n;
        // Each process stores some part of the columns, this is local number.
        // Because matrices are squared, thus n_columns_global == n
        // We add padding consisting of zeros, so that each process stores exactly the same number of columns
        int _n_columns;
        // maximum number of columns to print. Columns of higher id are skipped
        // it's used to print final matrix
        int _max_print_columns = -1;

        // As suggested during guest lecture:
        // "It's better to schedule one big transfer, rather than multiple smaller".
        // Thus we will keep all the columns in one vector which allows us for fast and easy
        // MPI send/recv commands. For instance, assuming num rows = 2
        // [col1 col1 col2 col2]
        // Another approach would be [col1row1 col2row1 col1row2 col2row] but it would be less
        // efficient as we want to iterate through column in the algorithm.
        // Each block stores: _n * _n_columns values
        std::vector<double> values;

        // Creates DenseBlock of Dense Matrix.
        // This blocks is n x n_columns
        // column_global_offset specifies column_global_offset for densematgen generator
        DenseBlock(int n, int n_columns, int column_global_offset, int seed);

        DenseBlock(int n, int n_columns);

        friend std::ostream &operator<<(std::ostream &os, const DenseBlock &matrix);

        int getIndex(int row, int column) const;

        void setValue(int row, int column, double value);

        double getValue(int row, int column) const;

        int getNumRows() const;

        int getNumColumns() const;

        void updateValue(int row, int column, double delta);

        void mergeColumnBlocks(const DenseBlock &rhs, int column_offset);

        void setMaxPrintColumns(int max_print_columns);

        int getMaxPrintColumns() const;

        int countGE(double value, int column_global_offset) const;

        void merge(const DenseBlock &rhs);

        /** Because dense matrices are stored as vectors, they require post-processing after merging
         * if we want to keep the in natural, concise layout.
         * Input layout: [P1_vals, P2_vals, P3_vals...]
         * Output layout: [row1_vals, row2_vals, row3_vals...]
         */
        void rearrangeAfterMerge(int columns_per_process);
    };
}


#endif //MPI_MATRIX_MULTIPLICATION_MATRIX_H
