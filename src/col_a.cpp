#include "../include/matmul.h"
#include <algorithm>

namespace matmul {
    namespace {
        int get_columns_per_process(int n, int num_processes) {
            return (n + num_processes) / num_processes;
        }

        int get_dense_column_global_offset(int my_rank, int columns_per_process) {
            return my_rank * columns_per_process;
        }
    }

    ColA::ColA(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c,
               bool async_computation_stage)
            : MatMul(comm_world, std::move(A), c, async_computation_stage) {
        if (comm_world->numProcesses() % c != 0) {
            throw std::runtime_error("p % c != 0 as required by ColA algorithm");
        }
    }

    int get_num_replication_groups(int num_processes, int c) {
        return num_processes / c;
    }

    int ColA::get_replication_color() {
        return comm_world->myRank() / c;
    }

    /** In ColA algorithm replication group is as simple as floor(rank/c).
     * e.g. P = 8, c = 2, then replication split will be: [P0, P1] [P2, P3], [P4, P5] [P6, P7].
     */
    mpi::Communicator ColA::get_comm_replication_A() {
        int color = get_replication_color();
        return mpi::Communicator(*comm_world, color, comm_world->myRank());
    }

    /**
     * e.g. P = 8, c = 2, then rotation split will be: [P0, P2, P4, P6], [P1, P3, P5, P7].
     * and we will use "ring" communicate data between previous and next process.
     */
    mpi::Communicator ColA::get_comm_rotation_A() {
        return mpi::Communicator(*comm_world, comm_world->myRank() % c, comm_world->myRank());
    }

    void ColA::initializationStage(int seed) {
        MatMul::initializationStage(seed, false);
    }

    void ColA::bcast_A_values_sizes() {
        A_values_sizes.resize(get_num_replication_groups(comm_world->numProcesses(), c));
        // for each replication group (iterate over masters)
        for (int root = 0; root < comm_world->numProcesses(); root += c) {
            // Temporarily initialize expected size with my local size - for a sender.
            A_values_sizes[root / c] = A->values.size();
            comm_world->broadcastInt(&A_values_sizes[root / c], root);
        }
    }

    /** ColA replicates only A */
    void ColA::replicationStage() {
        replicateA();
        if (async_computation_stage) {
            bcast_A_values_sizes();
        }
    }

    /**
     * for exponent steps:
     *   for n_replication_groups steps:
     *      multiply local matrices
     *      rotate A
     *   swap(B, C);
     */
    void ColA::computationStage(int exponent) {
        if (async_computation_stage) {
            computationStageAsync(exponent);
        } else {
            throw std::runtime_error("please run async version :)");
//            computationStageSync(exponent);
        }
    }


    void ColA::computationStageAsync(int exponent) {
        // Allocate A_recv buffer ONCE, and reuse it - it provides a significant speed up
        auto A_recv = std::make_unique<matrix::CSR>(A->n, *std::max_element(A_values_sizes.begin(),
                                                                            A_values_sizes.end()));
        // Resize local_A to have common size.
        size_t max_value_size = *std::max_element(A_values_sizes.begin(), A_values_sizes.end());
        A->values.resize(max_value_size);
        A->column_index.resize(max_value_size);

        mpi::Communicator comm_rotation = get_comm_rotation_A();
        const int num_replication_groups = comm_rotation.numProcesses();
        int receiver = (comm_rotation.myRank() + 1) % comm_rotation.numProcesses();
        // First process receives from process with pid = n-1
        int sender = comm_rotation.myRank() == 0 ? comm_rotation.numProcesses() - 1 :
                     comm_rotation.myRank() - 1;
        // As we rotate A during computation current chunks_id is updated.
        // At the beginning of each round it represents which chunk we currently store,
        // and is used to get what will be the next chunk, and thus we can lookup
        // what is the values_size to receive, allowing us for async send/recv/computation
        int A_chunk_id = comm_rotation.myRank(); //get_replication_color();
        int A_chunk_id_next;
        for (int e = 0; e < exponent; e++) {
            for (int round = 0; round < num_replication_groups; round++) {
                // We will receive from previous node, thus we do a little trick to calculate it
                // It's like doing A_chunk_id--, but shifted, so that % operator works properly.
                // E.g. for num_chunks = 4, and starting chunk_id = 2, we have: [2, 1, 0, 3]
                // and shift is necessary for transition from 0 to 3: (4+0-1) % 4 = 3
                A_chunk_id_next =
                        (num_replication_groups + A_chunk_id - 1) % num_replication_groups;
                // initialize sending and receiving
                auto async_send = comm_rotation.sendCSRAsyncNoMeta(*A, A_values_sizes[A_chunk_id],
                                                                   receiver);
                auto async_recv = comm_rotation.recvCSRAsyncNoMeta(*A_recv,
                                                                   A_values_sizes[A_chunk_id_next],
                                                                   sender);
                // compute
                computationRound();
                // wait for data
                async_send.waitAll();
                async_recv.waitAll();
                // update current A
                A.swap(A_recv);
                A_chunk_id = A_chunk_id_next;
            }
            if (e + 1 < exponent) {
                moveCtoB(true);
            }
        }
    }


    /**
 * for exponent steps:
 *   for n_replication_groups steps:
 *      multiply local matrices
 *      rotate A
 *   swap(B, C);
 */
    void ColA::computationStageSync(int exponent) {
        mpi::Communicator comm_rotation = get_comm_rotation_A();
        const int num_rounds = comm_world->numProcesses() / c;
        for (int e = 0; e < exponent; e++) {
            for (int round = 0; round < num_rounds; round++) {
                computationRound();
                rotateA(comm_rotation);
            }
            if (e + 1 < exponent) {
                moveCtoB(true);
            }
        }
    }


    /** Naive implementation for rotation of A matrix - used in computationStageSync rotation */
    void ColA::rotateA(mpi::Communicator &comm_rotation) {
        // Nothing to rotate, as all the data is local
        if (comm_rotation.numProcesses() == 1) {
            return;
        }
        int receiver = (comm_rotation.myRank() + 1) % comm_rotation.num_processes;
        // First process receives from process with pid = n-1
        int sender = comm_rotation.myRank() == 0 ? comm_rotation.numProcesses() - 1 :
                     comm_rotation.myRank() - 1;

        // Start sending local A
        auto async = comm_rotation.sendCSRAsync(*A, receiver);
        // Receive next A
        auto A_recv = comm_rotation.recvCSR(sender);
        // Wait until local A is sent
        async.waitAll();
        A.swap(A_recv);
    }

    void ColA::moveCtoB(bool zero_C) {
        B.swap(C);
        if (zero_C) {
            std::fill(C->values.begin(), C->values.end(), 0);
        }
    }

    std::unique_ptr<matrix::DenseBlock> ColA::gatherC() {
        // Memorize initial number of columns, so that we can rearrange DenseBlock
        int columns_per_process = C->_n_columns;
        auto gatherC = comm_world->gatherDenseBlock(*C, mpi::MASTER_RANK);
        if (gatherC != nullptr) {
            gatherC->rearrangeAfterMerge(columns_per_process);
        }
        return gatherC;
    }


    /** Reduces explicitly to the comm_world master */
    int ColA::countGEValues(double value) {
        int column_global_offset = get_dense_column_global_offset(comm_world->my_rank, get_columns_per_process(C->_n, comm_world->num_processes));
        int res = -1;
        res = comm_world->reduceInt(C->countGE(value, column_global_offset), 0);

        return res;
    }
}