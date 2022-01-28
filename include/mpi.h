#ifndef MPI_MATRIX_MULTIPLICATION_MPI_H
#define MPI_MATRIX_MULTIPLICATION_MPI_H

#include "../include/matrix.h"
#include <memory>
#include <mpi.h>

namespace mpi {
    using rank = int;

    const rank MASTER_RANK = 0;


    namespace {
        /**
         * It's enough to send n and values_size because
         * column_index_size = values_size and offset_index_size = n + 1
         */
        struct CSRMeta {
            int n = 0;
            int values_size = 0;
            static const int n_fields = 2;

            CSRMeta() = default;

            explicit CSRMeta(const matrix::CSR &matrix)
                    : n(matrix.n),
                      values_size(matrix.values.size()) {}

            std::vector<int> as_vector() const {
                return { n, values_size };
            }

            void from_vector(const std::vector<int> &vec) {
                n = vec[0];
                values_size = vec[1];
            }
        };
    }

    struct Async {
        std::vector<MPI_Status> statuses;
        std::vector<MPI_Request> requests;

        Async(int n) : statuses(n), requests(n) {}

        void waitAll() {
            MPI_Waitall(requests.size(), requests.data(), statuses.data());
        }
    };

    struct Communicator {
        MPI_Comm mpi_comm;
        // Required for destructor.
        bool mpi_world_comm = false;
        // World size of this communicator.
        int num_processes;
        // Process's rank within this communicator
        int my_rank;

        /**
         * Creates MPI_WORLD_COMM based on program args.
         * @param argc
         * @param argv
         */
        Communicator(int argc, char **argv);

        /**
         * Creates new MPI_COMM using MPI_Comm_split, based on given color and key.
         * @param comm - MPI_Comm to split.
         * @param color
         * @param key
         */
        Communicator(Communicator &comm, int color, int key);

        ~Communicator();
        matrix::CSR CSRBroadcastStageInitialization(matrix::CSR &matrix, mpi::rank master);
        matrix::CSR CSRBroadcastStageReplication(matrix::CSR &matrix, rank master);

//        void sendCSR(matrix::CSR &matrix, const rank receiver);
        std::unique_ptr<matrix::CSR> recvCSR(const rank sender);

        mpi::Async sendCSRAsync(matrix::CSR &matrix, const rank receiver) const;

        std::unique_ptr<matrix::CSR> recvCSRNoMeta(int n, int values_size, const rank sender);
        Async sendCSRAsyncNoMeta(matrix::CSR &matrix, size_t values_size, const rank receiver);
        Async recvCSRAsyncNoMeta(matrix::CSR &matrix, size_t values_size, const rank sender);

        void sendDenseBlock(matrix::DenseBlock &C, const rank dest, int tag=0) const;
        void recvDenseBlock(matrix::DenseBlock &block, const rank source, int tag=0) const;

        int reduceInt(int n, const rank root);

        mpi::rank myRank() const;
        int numProcesses() const;

        matrix::CSR distributeCSRMatrices(std::vector<matrix::CSR> &matrices);
        matrix::CSR CSRBroadcastSendStageReplication(matrix::CSR &matrix) const;

        matrix::CSR receiveCSRMatrix(matrix::CSR &current_matrix, const rank sender);
        matrix::CSR CSRBroadcastReceiveStageReplication(const rank sender);

        void broadcastDenseBlockSend(matrix::DenseBlock &block, const rank root);
        void bcastDenseBlock(matrix::DenseBlock &block, const rank root);

        void allReduceDenseBlock(matrix::DenseBlock &block);
        void reduceDenseBlock(matrix::DenseBlock &block, const rank root);

        std::unique_ptr<matrix::DenseBlock> gatherDenseBlock(matrix::DenseBlock &block, const rank root) const;

        void broadcastInt(int *n, const rank root);
    private:
        mpi::CSRMeta receiveCSRMeta(const rank sender);
        std::tuple<MPI_Request, MPI_Status> sendAsyncCSR(matrix::CSR &matrix, const rank receiver);
        std::unique_ptr<matrix::CSR> recvAsyncCSR(matrix::CSR &matrix, const rank sender);

        void sendInt(int n, int dest, int tag=0) const;
        int recvInt(int source, int tag=0) const;

    };
}

#endif //MPI_MATRIX_MULTIPLICATION_MPI_H
