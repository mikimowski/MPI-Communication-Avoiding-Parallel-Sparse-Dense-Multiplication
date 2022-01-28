#include "../include/mpi.h"
#include <mpi.h>
#include <vector>
#include <cassert>
#include <memory>

namespace mpi {
    namespace tag {
        enum CSR {
            META,
            VALUES,
            COLUMN,
            OFFSET,
        };
    }

    namespace csr {
        const int messages_per_process = 4;
    }

    Communicator::Communicator(int argc, char **argv)
            : mpi_comm(MPI_COMM_WORLD), mpi_world_comm(true) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    }

    Communicator::Communicator(Communicator &comm, int color, int key) {
        MPI_Comm_split(comm.mpi_comm, color, key, &mpi_comm);
        MPI_Comm_size(mpi_comm, &num_processes);
        MPI_Comm_rank(mpi_comm, &my_rank);
    }

    Communicator::~Communicator() {
        if (mpi_world_comm) {
            MPI_Finalize();
        } else {
            MPI_Comm_free(&mpi_comm);
        }
    }


    CSRMeta Communicator::receiveCSRMeta(const rank sender) {
        CSRMeta matrix_meta;
        MPI_Status status;
        MPI_Request request;

        std::vector<int> matrix_meta_vec = matrix_meta.as_vector();
        MPI_Irecv(matrix_meta_vec.data(), CSRMeta::n_fields,
                  MPI_INT, sender, tag::CSR::META, mpi_comm, &request);
        MPI_Wait(&request, &status);
        matrix_meta.from_vector(matrix_meta_vec);
        return matrix_meta;
    }


    matrix::CSR
    Communicator::distributeCSRMatrices(std::vector<matrix::CSR> &matrices) {
        assert(my_rank == 0);

        const int total_messages = (num_processes - 1) * csr::messages_per_process;
        std::vector<MPI_Status> statuses(total_messages);
        std::vector<MPI_Request> requests(total_messages);
        std::vector<std::vector<int>> matrices_meta;
        int pid_idx_offset = 0;
        for (int pid = 0; pid < num_processes; ++pid) {
            if (pid == my_rank) {
                continue;
            }
            // We have to keep it in memory until we are sure that the message was sent
            // and then the buffer can be freed
            matrices_meta.emplace_back(CSRMeta(matrices[pid]).as_vector());

            MPI_Isend(matrices_meta.back().data(), CSRMeta::n_fields,
                      MPI_INT, pid, tag::CSR::META, mpi_comm,
                      &requests[pid_idx_offset]);
            MPI_Isend(matrices[pid].values.data(), matrices[pid].values.size(),
                      MPI_DOUBLE, pid, tag::CSR::VALUES, mpi_comm,
                      &requests[pid_idx_offset + 1]);
            MPI_Isend(matrices[pid].column_index.data(), matrices[pid].column_index.size(),
                      MPI_INT, pid, tag::CSR::COLUMN, mpi_comm,
                      &requests[pid_idx_offset + 2]);
            MPI_Isend(matrices[pid].offset_index.data(), matrices[pid].offset_index.size(),
                      MPI_INT, pid, tag::CSR::OFFSET, mpi_comm,
                      &requests[pid_idx_offset + 3]);
            pid_idx_offset += csr::messages_per_process;
        }

        // Wait until all the messages are processed
        MPI_Waitall(total_messages, requests.data(), statuses.data());

        return matrices[my_rank];
    }

    matrix::CSR Communicator::receiveCSRMatrix(matrix::CSR &current_matrix,
                                               const rank sender) {
        assert(my_rank > 0);

        std::vector<MPI_Status> statuses(csr::messages_per_process - 1);
        std::vector<MPI_Request> requests(csr::messages_per_process - 1);
        // First receive metadata
        CSRMeta matrix_meta = receiveCSRMeta(sender);

        std::vector<double> values(matrix_meta.values_size);
        std::vector<int> column_index(matrix_meta.values_size);
        std::vector<int> offset_index(matrix_meta.n + 1);
        MPI_Irecv(values.data(), matrix_meta.values_size,
                  MPI_DOUBLE, sender, tag::CSR::VALUES, mpi_comm, &requests[0]);
        MPI_Irecv(column_index.data(), values.size(),
                  MPI_INT, sender, tag::CSR::COLUMN, mpi_comm, &requests[1]);
        MPI_Irecv(offset_index.data(), offset_index.size(),
                  MPI_INT, sender, tag::CSR::OFFSET, mpi_comm, &requests[2]);
        MPI_Waitall(csr::messages_per_process - 1, requests.data(), statuses.data());

        return matrix::CSR{matrix_meta.n, std::move(values),
                           std::move(column_index), std::move(offset_index)};
    }


    matrix::CSR Communicator::CSRBroadcastSendStageReplication(matrix::CSR &matrix) const {
        const int total_messages = (num_processes - 1) * csr::messages_per_process;
        std::vector<MPI_Status> statuses(total_messages);
        std::vector<MPI_Request> requests(total_messages);
        auto meta = CSRMeta(matrix).as_vector();

        int pid_idx_offset = 0;
        for (int pid = 0; pid < num_processes; ++pid) {
            if (pid != my_rank) {
                MPI_Isend(meta.data(), CSRMeta::n_fields,
                          MPI_INT, pid, tag::CSR::META, mpi_comm,
                          &requests[pid_idx_offset]);
                MPI_Isend(matrix.values.data(), matrix.values.size(),
                          MPI_DOUBLE, pid, tag::CSR::VALUES, mpi_comm,
                          &requests[pid_idx_offset + 1]);
                MPI_Isend(matrix.column_index.data(), matrix.column_index.size(),
                          MPI_INT, pid, tag::CSR::COLUMN, mpi_comm,
                          &requests[pid_idx_offset + 2]);
                MPI_Isend(matrix.offset_index.data(), matrix.offset_index.size(),
                          MPI_INT, pid, tag::CSR::OFFSET, mpi_comm,
                          &requests[pid_idx_offset + 3]);
                pid_idx_offset += csr::messages_per_process;
            }
        }

        // Wait until all the messages are sent
        MPI_Waitall(total_messages, requests.data(), statuses.data());

        return matrix;
    }


    matrix::CSR Communicator::CSRBroadcastReceiveStageReplication(const rank sender) {
        std::vector<MPI_Status> statuses(csr::messages_per_process - 1);
        std::vector<MPI_Request> requests(csr::messages_per_process - 1);
        // First receive metadata
        CSRMeta matrix_meta = receiveCSRMeta(sender);

        // Then receive the rest
        std::vector<double> values(matrix_meta.values_size);
        std::vector<int> column_index(matrix_meta.values_size);
        std::vector<int> offset_index(matrix_meta.n + 1);
        MPI_Irecv(values.data(), values.size(),
                  MPI_DOUBLE, sender, tag::CSR::VALUES, mpi_comm, &requests[0]);
        MPI_Irecv(column_index.data(), column_index.size(),
                  MPI_INT, sender, tag::CSR::COLUMN, mpi_comm, &requests[1]);
        MPI_Irecv(offset_index.data(), offset_index.size(),
                  MPI_INT, sender, tag::CSR::OFFSET, mpi_comm, &requests[2]);
        MPI_Waitall(csr::messages_per_process - 1, requests.data(), statuses.data());

        matrix::CSR received_matrix{matrix_meta.n, std::move(values),
                                    std::move(column_index), std::move(offset_index)};
        return received_matrix;
    }


    matrix::CSR Communicator::CSRBroadcastStageReplication(matrix::CSR &matrix, mpi::rank master) {
        if (num_processes == 1) {
            return matrix;
        }

        if (my_rank == master) {
            return CSRBroadcastSendStageReplication(matrix);
        } else {
            return CSRBroadcastReceiveStageReplication(master);
        }
    }


    Async Communicator::sendCSRAsync(matrix::CSR &matrix, const rank receiver) const {
        Async async = Async{csr::messages_per_process};
        auto meta = CSRMeta(matrix).as_vector();

        MPI_Isend(meta.data(), CSRMeta::n_fields,
                  MPI_INT, receiver, tag::CSR::META, mpi_comm,
                  &async.requests[0]);
        MPI_Isend(matrix.values.data(), matrix.values.size(),
                  MPI_DOUBLE, receiver, tag::CSR::VALUES, mpi_comm,
                  &async.requests[1]);
        MPI_Isend(matrix.column_index.data(), matrix.column_index.size(),
                  MPI_INT, receiver, tag::CSR::COLUMN, mpi_comm,
                  &async.requests[2]);
        MPI_Isend(matrix.offset_index.data(), matrix.offset_index.size(),
                  MPI_INT, receiver, tag::CSR::OFFSET, mpi_comm,
                  &async.requests[3]);

        return async;
    }


    std::unique_ptr<matrix::CSR> Communicator::recvCSR(const rank sender) {

        std::vector<MPI_Status> statuses(csr::messages_per_process - 1);
        std::vector<MPI_Request> requests(csr::messages_per_process - 1);
        // First receive metadata
        CSRMeta matrix_meta = receiveCSRMeta(sender);
        // Then receive the rest
        std::vector<double> values(matrix_meta.values_size);
        std::vector<int> column_index(matrix_meta.values_size);
        std::vector<int> offset_index(matrix_meta.n + 1);
        MPI_Irecv(values.data(), values.size(),
                  MPI_DOUBLE, sender, tag::CSR::VALUES, mpi_comm, &requests[0]);
        MPI_Irecv(column_index.data(), column_index.size(),
                  MPI_INT, sender, tag::CSR::COLUMN, mpi_comm, &requests[1]);
        MPI_Irecv(offset_index.data(), offset_index.size(),
                  MPI_INT, sender, tag::CSR::OFFSET, mpi_comm, &requests[2]);
        MPI_Waitall(csr::messages_per_process - 1, requests.data(), statuses.data());

        return std::make_unique<matrix::CSR>(
                matrix::CSR{matrix_meta.n, std::move(values),
                            std::move(column_index), std::move(offset_index)});
    }

    Async
    Communicator::recvCSRAsyncNoMeta(matrix::CSR &matrix, size_t values_size, const rank sender) {
        Async async = Async{csr::messages_per_process - 1};
        MPI_Irecv(matrix.values.data(), values_size,
                  MPI_DOUBLE, sender, tag::CSR::VALUES, mpi_comm, &async.requests[0]);
        MPI_Irecv(matrix.column_index.data(), values_size,
                  MPI_INT, sender, tag::CSR::COLUMN, mpi_comm, &async.requests[1]);
        MPI_Irecv(matrix.offset_index.data(), matrix.offset_index.size(),
                  MPI_INT, sender, tag::CSR::OFFSET, mpi_comm, &async.requests[2]);
        return async;
    }

    Async
    Communicator::sendCSRAsyncNoMeta(matrix::CSR &matrix, size_t values_size, const rank receiver) {
        Async async = Async{csr::messages_per_process - 1};
        MPI_Isend(matrix.values.data(), values_size,
                  MPI_DOUBLE, receiver, tag::CSR::VALUES, mpi_comm,
                  &async.requests[0]);
        MPI_Isend(matrix.column_index.data(), values_size,
                  MPI_INT, receiver, tag::CSR::COLUMN, mpi_comm,
                  &async.requests[1]);
        MPI_Isend(matrix.offset_index.data(), matrix.offset_index.size(),
                  MPI_INT, receiver, tag::CSR::OFFSET, mpi_comm,
                  &async.requests[2]);
        return async;
    }


    std::unique_ptr<matrix::DenseBlock>
    Communicator::gatherDenseBlock(matrix::DenseBlock &block, const rank root) const {
        std::unique_ptr<matrix::DenseBlock> recv_block = nullptr;
        if (my_rank == root) {
            recv_block = std::make_unique<matrix::DenseBlock>(block.getNumRows(),
                                                              block.getNumColumns() *
                                                              num_processes);
        }
        MPI_Gather(block.values.data(), block.values.size(), MPI_DOUBLE,
                   recv_block == nullptr ? nullptr : recv_block->values.data(),
                   block.values.size(), MPI_DOUBLE, root, mpi_comm);
        return recv_block;
    }

    void Communicator::reduceDenseBlock(matrix::DenseBlock &block, const rank root) {
        MPI_Reduce(my_rank == root ? MPI_IN_PLACE : block.values.data(),
                   my_rank == root ? block.values.data() : nullptr,
                   block.values.size(), MPI_DOUBLE, MPI_SUM,
                   root, mpi_comm);
    }

    void Communicator::allReduceDenseBlock(matrix::DenseBlock &block) {
        MPI_Allreduce(MPI_IN_PLACE, block.values.data(), block.values.size(), MPI_DOUBLE, MPI_SUM,
                      mpi_comm);
    }

    void Communicator::bcastDenseBlock(matrix::DenseBlock &block, const rank root) {
        MPI_Bcast(block.values.data(), block.values.size(), MPI_DOUBLE, root, mpi_comm);
    }

    void Communicator::recvDenseBlock(matrix::DenseBlock &block, const rank source, int tag) const {
        MPI_Recv(block.values.data(), block.values.size(), MPI_DOUBLE, source, tag, mpi_comm,
                 MPI_STATUS_IGNORE);
    }

    void Communicator::sendDenseBlock(matrix::DenseBlock &C, const rank dest, int tag) const {
        MPI_Send(C.values.data(), C.values.size(), MPI_DOUBLE, dest, tag, mpi_comm);
    }

    int Communicator::reduceInt(int n, const rank root) {
        int recv_data = root == my_rank ? 0 : -1;
        MPI_Reduce(&n, &recv_data, 1, MPI_INT, MPI_SUM, root, mpi_comm);
        return recv_data;
    }

    void Communicator::broadcastInt(int *n, const rank root) {
        MPI_Bcast(n, 1, MPI_INT, root, mpi_comm);
    }

    void Communicator::sendInt(int n, int dest, int tag) const {
        MPI_Send(&n, 1, MPI_INT, dest, tag, mpi_comm);
    }

    int Communicator::recvInt(int source, int tag) const {
        int n;
        MPI_Recv(&n, 1, MPI_INT, source, tag, mpi_comm, MPI_STATUS_IGNORE);
        return n;
    }

    mpi::rank Communicator::myRank() const {
        return my_rank;
    }

    int Communicator::numProcesses() const {
        return num_processes;
    }
}