#include "../include/matmul.h"

namespace matmul {
    namespace {
        int get_dense_columns_per_process_before_replication(int n, int num_processes) {
            return (n + num_processes) / num_processes;
        }

        int get_column_owner(int column, int n_columns, int num_processes) {
            // e.g. p = 4, n = 10 then [0,1,2],[3,4,5],[6,7,8],[9,10]
            // e.g. column = 6, we get 5/4 = 1, pid=1
            return column /
                   get_dense_columns_per_process_before_replication(n_columns, num_processes);
        }

        int get_row_owner(int row, int n_rows, int num_processes) {
            return row / get_dense_columns_per_process_before_replication(n_rows, num_processes);
        }

        std::vector<matrix::CSR> splitByRow(const matrix::CSR &matrix, int num_processes) {
            std::vector<matrix::CSR> matrices(num_processes, matrix::CSR{matrix.getNumRows(), 0});

            for (int row = 0; row < matrix.getNumRows(); ++row) {
                // append 'end_of_row' to owner's offset_index. Default value equal the last values already inserted.
                // If new value is added to this then we will increment corresponding offset
                for (int pid = 0; pid < num_processes; ++pid) {
                    matrices[pid].offset_index[row + 1] = matrices[pid].offset_index[row];
                }
                // all columns from this row will go to this process
                int pid = get_row_owner(row, matrix.getNumRows(), num_processes);

                // Iterate through only non-zero columns (CSR format)
                for (int offset_idx = matrix.offset_index[row];
                     offset_idx < matrix.offset_index[row + 1]; offset_idx++) {
                    double value = matrix.values[offset_idx];
                    int column = matrix.column_index[offset_idx];

                    // Increment offset, as new value is added to current row
                    matrices[pid].offset_index[row + 1]++;
                    matrices[pid].column_index.push_back(column);
                    matrices[pid].values.push_back(value);
                }
            }

            return matrices;
        }


        std::vector<matrix::CSR> splitByColumn(const matrix::CSR &matrix, int num_processes) {
            std::vector<matrix::CSR> matrices(num_processes, matrix::CSR{matrix.getNumRows(), 0});

            // Iterate through each row
            for (int row = 0; row < matrix.getNumRows(); ++row) {
                // append 'end_of_row' to each offset_index. Default value equal the last values already inserted.
                // If new value is added to this then we will increment corresponding offset
                for (int pid = 0; pid < num_processes; ++pid) {
                    matrices[pid].offset_index[row + 1] = matrices[pid].offset_index[row];
                }

                // Iterate through only non-zero columns (CSR format)
                for (int offset_idx = matrix.offset_index[row];
                     offset_idx < matrix.offset_index[row + 1]; offset_idx++) {
                    double value = matrix.values[offset_idx];
                    int column = matrix.column_index[offset_idx];
                    int pid = get_column_owner(column, matrix.getNumColumns(), num_processes);

                    // Increment offset, as new value is added to current row
                    matrices[pid].offset_index[row + 1]++;
                    matrices[pid].column_index.push_back(column);
                    matrices[pid].values.push_back(value);
                }
            }

            return matrices;
        }


        /**
         * Divides sparse matrix into blocks that will be locally stored by processes.
         * @return
         */
        std::vector<matrix::CSR>
        splitMatrix(const matrix::CSR &matrix, int num_processes, bool by_row) {
            if (by_row) {
                return splitByRow(matrix, num_processes);
            } else {
                return splitByColumn(matrix, num_processes);
            }
        }

    }


    /******************************************* MatMul *******************************************/
    MatMul::MatMul(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c,
                   bool async_computation_stage)
            : comm_world(comm_world), A(std::move(A)), c(c),
              async_computation_stage(async_computation_stage) {}

    void MatMul::initializationStage(int seed, bool split_by_row) {
        // Master distributes Sparse A to
        if (comm_world->myRank() == mpi::MASTER_RANK) {
            // split by row - for InnerABC
            auto matrices = splitMatrix(*A, comm_world->numProcesses(), split_by_row);
            A = std::make_unique<matrix::CSR>(
                    comm_world->distributeCSRMatrices(matrices));
        } else {
            A = std::make_unique<matrix::CSR>(
                    comm_world->receiveCSRMatrix(*A, mpi::MASTER_RANK));
        }

        // Because A, B, C are squared matrices we already know `n`
        int block_size = get_dense_columns_per_process_before_replication(A->n,
                                                                          comm_world->numProcesses());

        // Each process initialize its local B independently - thanks to stateless densematgen
        // For it to work, process has to know "global_column_offset", that is what are the original
        // indices of columns it's storing.
        B = std::make_unique<matrix::DenseBlock>(A->n, block_size,
                                                 comm_world->myRank() * block_size, seed);

        // Locally each process starts with C filled with zeros, such that `shape C` == `shape B`.
        C = std::make_unique<matrix::DenseBlock>(A->n, block_size);
    }


    void MatMul::replicateA() {
        // This is were ColA and InnerABC differ, each has its own way of distributing A, which
        // depends on how processes are split into replication groups
        mpi::Communicator comm_replication = get_comm_replication_A();
        matrix::CSR accumulatedA(0, 0);
        for (int pid = 0; pid < comm_replication.num_processes; ++pid) {
            accumulatedA = matrix::CSR{std::move(accumulatedA),
                                       comm_replication.CSRBroadcastStageReplication(*A,
                                                                                     pid)};
        }
        A = std::make_unique<matrix::CSR>(accumulatedA);
    }


    /**
    * In order to leverage memory layout, we don't want to operate on pointers, thus we use
    * this helper function.
    */
    void MatMul::compute(const matrix::CSR &sparse, const matrix::DenseBlock &dense,
                         matrix::DenseBlock &result) {
        const int result_n_columns = result._n_columns;
        const int dense_n_columns = dense._n_columns;
        for (int row_global = 0; row_global < sparse.n; row_global++) {
            int last_offset_idx_excl = sparse.offset_index[row_global + 1];
            // for non-zero column in current row_global of A
            for (int offset = sparse.offset_index[row_global];
                 offset < last_offset_idx_excl; offset++) {
                const double val_a = sparse.values[offset];
                const int column_global = sparse.column_index[offset];
                const int result_row_offset = result_n_columns * row_global;
                // update all locations in C to which this (A_r, A_c) contributes
                // to do that iterate over all columns of b
                for (int column_b = 0; column_b < dense_n_columns; column_b++) {
                    const double d_val = dense.values[column_global * dense_n_columns + column_b];
                    result.values[result_row_offset + column_b] += val_a * d_val;
                }
            }
        }
    }

    void MatMul::computationRound() {
        compute(*A, *B, *C);
    }
}
