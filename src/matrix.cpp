#include "../include/matrix.h"
#include "../include/densematgen.h"
#include <cassert>
#include <iostream>
#include <algorithm>

namespace matrix {

    CSR::CSR(int n, std::vector<double> &&values,
             std::vector<int> &&column_index, std::vector<int> &&offset_index)
            : n(n), values(values), column_index(column_index), offset_index(offset_index) {}

    CSR::CSR(int n, int values_size)
            : n(n), values(values_size), column_index(values_size), offset_index(n + 1, 0) {}

    CSR::CSR(CSR &&lhs, CSR &&rhs) {
        if (lhs.n == 0) {
            *this = std::move(rhs);
        } else if (rhs.n == 0) {
            *this = std::move(lhs);
        } else {
            this->n = std::max(lhs.n, rhs.n);
            values.resize(lhs.values.size() + rhs.values.size());
            column_index.resize(lhs.column_index.size() + rhs.column_index.size());
            offset_index.resize(lhs.offset_index.size());
            offset_index[0] = 0;

            for (int row = 0; row < this->n; ++row) {
                offset_index[row + 1] = offset_index[row];
                int lhs_row_idx = lhs.offset_index[row];
                int lhs_last_row_idx_excl = lhs.offset_index[row + 1];
                int rhs_row_idx = rhs.offset_index[row];
                int rhs_last_row_idx_excl = rhs.offset_index[row + 1];

                while (lhs_row_idx < lhs_last_row_idx_excl && rhs_row_idx < rhs_last_row_idx_excl) {
                    // There should be no common columns
                    assert(lhs.column_index[lhs_row_idx] != rhs.column_index[rhs_row_idx]);
                    // LHS goes first
                    if (lhs.column_index[lhs_row_idx] < rhs.column_index[rhs_row_idx]) {
                        values[offset_index[row + 1]] = lhs.values[lhs_row_idx];
                        column_index[offset_index[row + 1]] = lhs.column_index[lhs_row_idx];
                        // New value in the row
                        offset_index[row + 1]++;
                        lhs_row_idx++;
                    } else {
                        values[offset_index[row + 1]] = rhs.values[rhs_row_idx];
                        column_index[offset_index[row + 1]] = rhs.column_index[rhs_row_idx];
                        // New value in the row
                        offset_index[row + 1]++;
                        rhs_row_idx++;
                    }
                }

                // Fill the tails, if rhs ended first
                while (lhs_row_idx < lhs_last_row_idx_excl) {
                    values[offset_index[row + 1]] = lhs.values[lhs_row_idx];
                    column_index[offset_index[row + 1]] = lhs.column_index[lhs_row_idx];
                    // New value in the row
                    offset_index[row + 1]++;
                    lhs_row_idx++;
                }
                // Fill the tails, if lhs ended first
                while (rhs_row_idx < rhs_last_row_idx_excl) {
                    values[offset_index[row + 1]] = rhs.values[rhs_row_idx];
                    column_index[offset_index[row + 1]] = rhs.column_index[rhs_row_idx];
                    // New value in the row
                    offset_index[row + 1]++;
                    rhs_row_idx++;
                }
            }
        }
    }


    std::ostream &operator<<(std::ostream &os, const CSR &matrix) {
        const std::string default_value = "0.00000";
        int offset_idx = 0;

        for (int row = 0; row < matrix.getNumRows(); ++row) {
            int last_offset_idx_excl = matrix.offset_index[row + 1];
            for (int col = 0; col < matrix.getNumColumns(); ++col) {
                if (offset_idx < last_offset_idx_excl && matrix.column_index[offset_idx] == col) {
                    os << matrix.values[offset_idx];
                    offset_idx++;
                } else {
                    os << default_value;
                }
                os << " ";
            }
            os << "\n";
        }

        return os;
    }

    std::string CSR::to_string() {
        std::string str = "n = " + std::to_string(this->n) + "\n";
        str += "values: [";
        for (auto value: this->values) {
            str += std::to_string(value) + " ";
        }
        str += "]\n";
        str += "column_index: [";
        for (auto cidx: this->column_index) {
            str += std::to_string(cidx) + " ";
        }
        str += "]\n";
        str += "offset_index: [";
        for (auto offset: this->offset_index) {
            str += std::to_string(offset) + " ";
        }
        str += "]\n";
        return str;
    }

    int CSR::getNumRows() const {
        return this->n;
    }

    int CSR::getNumColumns() const {
        return this->n;
    }

    DenseBlock::DenseBlock(int n, int n_columns, int column_global_offset, int seed)
            : _n(n), _n_columns(n_columns), values(_n * _n_columns, 0) {
        for (int row = 0; row < _n; row++) {
            for (int column = 0; column < _n_columns; column++) {
                if (column_global_offset + column < _n) {
                    setValue(row, column,
                             generate_double(seed, row, column_global_offset + column));
                } else {
                    // 0-padding to make all processes store the same number of columns
                    setValue(row, column, 0);
                }
            }
        }
    }

    DenseBlock::DenseBlock(int n, int n_columns)
        : _n(n), _n_columns(n_columns), values(_n * _n_columns, 0) {}


    /*
     * Merges lhs with rhs in the following way:
     * lhs = [a1, a2, a3], rhs = [b1, b2, b3] -> lhs = [a1, a2, a3, b1, b2, b3]
     */
    void DenseBlock::mergeColumnBlocks(const DenseBlock &rhs, int column_offset) {
        std::transform(this->values.cbegin() + column_offset * _n,
                       this->values.cbegin() + column_offset * _n + rhs.values.size(),
                       rhs.values.cbegin(),
                       this->values.begin() + column_offset * _n, std::plus<double>());
    }

    void DenseBlock::merge(const DenseBlock &rhs) {
        std::transform(this->values.cbegin(), this->values.cend(), rhs.values.cbegin(),
                       this->values.begin(), std::plus<double>());
    }


    int DenseBlock::getIndex(int row, int column) const {
        return row * _n_columns + column;
    }

    void DenseBlock::setValue(int row, int column, double value) {
        values[getIndex(row, column)] = value;
    }

    void DenseBlock::updateValue(int row, int column, double delta) {
        values[getIndex(row, column)] += delta;
    }

    double DenseBlock::getValue(int row, int column) const {
        return values[getIndex(row, column)];
    }

    int DenseBlock::getNumRows() const {
        return _n;
    }

    int DenseBlock::getNumColumns() const {
        return _n_columns;
    }

    int DenseBlock::countGE(double value, int column_global_offset) const {
        int cnt = 0;
        for (int row = 0; row < _n; row++) {
            for (int column = 0; column < _n_columns; column++) {
                // don't count 0-padding
                if (column + column_global_offset >= _n) {
                    break;
                }
                if (getValue(row, column) >= value) {
                    cnt++;
                }
            }
        }
        return cnt;
    }

    void DenseBlock::rearrangeAfterMerge(int columns_per_process) {
        std::vector<double> values_old(this->values.size(), 0);
        this->values.swap(values_old);

        int values_per_process = columns_per_process * this->_n;

        for (int row = 0; row < this->getNumRows(); ++row) {
            for (int col = 0; col < std::min(this->_n_columns, this->_n); ++col) {
                int col_owner = col / columns_per_process;
                int idx_col = col_owner * values_per_process + col % columns_per_process;
                int idx = row * columns_per_process + idx_col;
                this->setValue(row, col, values_old[idx]);
            }
        }
    }


    std::ostream &operator<<(std::ostream &os, const DenseBlock &matrix) {
        for (int row = 0; row < matrix.getNumRows(); ++row) {
            for (int col = 0; col < matrix.getMaxPrintColumns(); ++col) {
                os << matrix.getValue(row, col) << " ";
            }
            os << "\n";
        }

        return os;
    }


    void DenseBlock::setMaxPrintColumns(int max_print_columns) {
        _max_print_columns = max_print_columns;
    }

    int DenseBlock::getMaxPrintColumns() const {
        return _max_print_columns >= 0 ? _max_print_columns : _n_columns;
    }
}
