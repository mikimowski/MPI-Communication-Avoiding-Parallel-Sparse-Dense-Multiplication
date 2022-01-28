#include "../include/matmul.h"
#include <algorithm>


namespace matmul {
    namespace {
        /** A little bit of magic that calculates partition into groups */
        int get_color_replication_A(int num_processes, int rank, int c) {
            int color = ((rank / c) + (rank % c) * (num_processes / (c * c))) % (num_processes / c);
            return color;
        }

        int get_color_rotation_A(int rank, int c) {
            int color = rank % c;
            return color;
        }

        int get_color_replication_B(int rank, int c) {
            int color = rank / c;
            return color;
        }

        /** Number of replication groups within A or B **/
        int get_num_replication_groups(int num_processes, int c) {
            return num_processes / c;
        }
    }

    InnerABC::InnerABC(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c,
                       bool async_computation_stage)
            : MatMul(comm_world, std::move(A), c, async_computation_stage) {
        if (comm_world->numProcesses() % (c * c) != 0) {
            throw std::runtime_error("p % c^2 != 0 as required by InnerABC algorithm");
        }
    }


    mpi::Communicator InnerABC::get_comm_replication_A() {
        int r = comm_world->myRank();
        int p = comm_world->num_processes;
        int color = get_color_replication_A(p, r, c);
        return mpi::Communicator(*comm_world, color, comm_world->myRank());
    }

    /**
     * e.g. P = 8, c = 2, then split will be: [P0, P2, P4, P6], [P1, P3, P5, P7]
     * and we will use "ring" communicate data between communication groups.
     */
    mpi::Communicator InnerABC::get_comm_rotation_A() {
        int color = get_color_rotation_A(comm_world->my_rank, c);
        return mpi::Communicator(*comm_world, color, comm_world->myRank());
    }

    /** Replication group for B is just floor(rank/c). */
    mpi::Communicator InnerABC::get_comm_replication_B() {
        int color = get_color_replication_B(comm_world->myRank(), c);
        return mpi::Communicator(*comm_world, color, comm_world->myRank());
    }

    void InnerABC::initializationStage(int seed) {
        MatMul::initializationStage(seed, true);
    }

    void InnerABC::bcast_A_values_sizes() {
        A_values_sizes.resize(get_num_replication_groups(comm_world->numProcesses(), c));
        // for each replication group (iterate over masters)
        for (int root = 0; root < comm_world->numProcesses(); root += c) {
            // Temporarily initialize expected size with my local size - for a sender.
            int color = get_color_replication_A(comm_world->numProcesses(), root, c);
            A_values_sizes[color] = A->values.size();
            comm_world->broadcastInt(&A_values_sizes[color], root);
        }
    }

    void InnerABC::initLocalBBeforeReplication() {
        mpi::Communicator comm_replication = get_comm_replication_B();
        auto replicated_B = std::make_unique<matrix::DenseBlock>(B->getNumRows(), B->getNumColumns() * c);
        // put local data in proper location (offset)
        replicated_B->mergeColumnBlocks(*B, B->getNumColumns() * comm_replication.my_rank);
        B.swap(replicated_B);
        C = std::make_unique<matrix::DenseBlock>(B->getNumRows(), B->getNumColumns());
    }

    void InnerABC::replicateB() {
        mpi::Communicator comm_replication = get_comm_replication_B();
        comm_replication.allReduceDenseBlock(*B);
    }

    /**
     * InnerABC replicates both A and B.
     * Those are replicated within DIFFERENT GROUPS, therefore two different communicators are used.
     */
    void InnerABC::replicationStage() {
        replicateA();
        if (async_computation_stage) {
            bcast_A_values_sizes();
        }
        int columns_per_process = B->_n_columns;
        initLocalBBeforeReplication();
        replicateB();
        B->rearrangeAfterMerge(columns_per_process);
    }

    /**
     * for exponent steps:
     *   for n_replication_groups steps:
     *      multiply local matrices
     *      rotate A
     *   swap(B, C);
     *   if not last round:
     *      synchronize B
     */
    void InnerABC::computationStage(int exponent) {
        if (async_computation_stage) {
            computationStageAsync(exponent);
        } else {
            throw std::runtime_error("please run async version :)");
//            computationStageSync(exponent);
        }
    }


    void InnerABC::computationStageSync(int exponent) {
        auto comm_rotation = get_comm_rotation_A();
        const int num_rounds = comm_world->numProcesses() / (c * c);
        for (int e = 0; e < exponent; e++) {
            for (int round = 0; round < num_rounds; round++) {
                computationRound();
                rotateA(comm_rotation);
            }
            if (e + 1 < exponent) {
                swapCWithB(true);
                replicateB();
            }
        }
    }


    void InnerABC::computationStageAsync(int exponent) {
        size_t max_value_size = *std::max_element(A_values_sizes.begin(), A_values_sizes.end());
        // Allocate A_recv buffer ONCE, and reuse it - it provides a significant speed up
        // when compared to allocating A_recv in the for loop
        auto A_recv = std::make_unique<matrix::CSR>(A->n, max_value_size);
        // Resize local_A to have common size.
        A->values.resize(max_value_size);
        A->column_index.resize(max_value_size);

        auto comm_rotation = get_comm_rotation_A();
        int receiver = (comm_rotation.myRank() + 1) % comm_rotation.numProcesses();
        // First process receives from process with pid = n-1
        int sender = comm_rotation.myRank() == 0 ? comm_rotation.numProcesses() - 1 :
                     comm_rotation.myRank() - 1;
        // As we rotate A during computation current chunks_id is updated.
        // At the beginning of each round it represents which chunk we currently store,
        // and is used to get what will be the next chunk, and thus we can lookup
        // what is the values_size to receive, allowing us for async send/recv/computation
        int A_chunk_id = get_color_replication_A(comm_world->numProcesses(), comm_world->myRank(), c);
        int A_chunk_id_next;
        const int num_replication_groups = this->A_values_sizes.size();
        const int num_rounds = comm_world->numProcesses() / (c * c);

        for (int e = 0; e < exponent; e++) {
            for (int round = 0; round < num_rounds; round++) {
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
                swapCWithB(true);
                replicateB();
            }
        }
    }


    void InnerABC::swapCWithB(bool zero_C) {
        B.swap(C);
        if (zero_C) {
            std::fill(C->values.begin(), C->values.end(), 0);
        }
    }

    void InnerABC::rotateA(mpi::Communicator &comm_rotation) {
        // nothing to rotate, as all the data is local
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

    /**
    * Two phases:
    * 1. reduceC within comm_replication_B
    * 2. gatherC values within comm_world - only masters communicate
    */
    std::unique_ptr<matrix::DenseBlock> InnerABC::gatherC() {
        if (comm_world->numProcesses() == 1) {
            return std::make_unique<matrix::DenseBlock>(*C);
        }

        // Memorize initial number of columns, so that we can rearrange DenseBlock
        int columns_per_process = C->_n_columns;
        // Divide into subsets: {rep_group_1}, {rep_group_2}, ... and gather within each subset
        mpi::Communicator comm_replication = get_comm_replication_B();
        comm_replication.reduceDenseBlock(*C, mpi::MASTER_RANK);
        if (C != nullptr) {
            C->rearrangeAfterMerge(columns_per_process);
        }
        // Divide into subsets: {masters}, {not masters} and gather within masters subset
        bool is_master = mpi::MASTER_RANK == comm_replication.my_rank ? true : false;
        int color = is_master ? 0 : 1;
        mpi::Communicator masters_comm = mpi::Communicator(*comm_world, color, comm_world->my_rank);
        if (is_master) {
            // Memorize initial number of columns, so that we can rearrange DenseBlock
            columns_per_process = C->_n_columns;
            auto replication_group_C = masters_comm.gatherDenseBlock(*C, mpi::MASTER_RANK);
            if (replication_group_C != nullptr) {
                replication_group_C->rearrangeAfterMerge(columns_per_process);
            }
            return replication_group_C;
        } else {
            return nullptr;
        }
    }


    /**
     * Two phases:
     * 1. reduceC within comm_replication_B
     * 2. countGE values within comm_world - only masters communicate
    */
    int InnerABC::countGEValues(double value) {
        if (comm_world->numProcesses() == 1) {
            return C->countGE(value, 0);
        }

        // Memorize initial number of columns, so that we can rearrange DenseBlock
        int columns_per_process = C->_n_columns;
        // Divide into subsets: {rep_group_1}, {rep_group_2}, ... and gather within each subset
        mpi::Communicator comm_replication = get_comm_replication_B();
        comm_replication.reduceDenseBlock(*C, mpi::MASTER_RANK);
        if (C != nullptr) {
            C->rearrangeAfterMerge(columns_per_process);
        }

        // Divide into subsets: {masters}, {not masters} and gather within masters subset
        bool is_master = mpi::MASTER_RANK == comm_replication.my_rank ? true : false;
        int color = is_master ? 0 : 1;
        mpi::Communicator masters_comm = mpi::Communicator(*comm_world, color, comm_world->my_rank);
        if (is_master) {
            // we need to calculate column_global_offset of first process in our replication group
            // equivalently, it's column_global_offset of REPLICATED dense matrix
            int column_global_offset = get_color_replication_B(comm_world->my_rank, c) * C->_n_columns; //get_dense_column_global_offset_after_replication(C->_n, comm_world->num_processes, comm_world->my_rank, c)
            int res = C->countGE(value, column_global_offset);
            res = masters_comm.reduceInt(res, 0);
            return res;
        } else {
            return -1;
        }
    }
}