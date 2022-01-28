#ifndef MPI_MATRIX_MULTIPLICATION_MATMUL_H
#define MPI_MATRIX_MULTIPLICATION_MATMUL_H

#include "../include/mpi.h"

/**
 * Assumptions:
 * ColA
 *  1.  p % c == 0
 * InnerABC
 *  1.  p % c^2 == 0
 */
namespace matmul {
    class MatMul {
    protected:
        mpi::Communicator *comm_world;
        std::unique_ptr<matrix::CSR> A;
        std::unique_ptr<matrix::DenseBlock> B;
        std::unique_ptr<matrix::DenseBlock> C;
        int c;

        bool async_computation_stage = true;
        // values sizes for chunks of A that are rotated. Those sizes are the ones AFTER the
        // replication phase. Thanks to this at each step process knows how much values to expect
        // for CSR matrix. Therefore we can schedule async_recv and start local calculations.
        std::vector<int> A_values_sizes;

    public:
        MatMul(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c, bool async_computation_stage);

        virtual ~MatMul() = default;

        /**
         *  Master:
         *  - reads A as local matrix
         *  - distributes A to proper processes
         *  - overrides A with his fragment
         *  - initializes local matrices
         *  Not a master:
         *  - receives A from the Master
         *  - initializes local matrices
         */
        virtual void initializationStage(int seed) = 0;

        virtual void replicationStage() = 0;

        virtual void computationStage(int exponent) = 0;

        virtual std::unique_ptr<matrix::DenseBlock> gatherC() = 0;

        virtual int countGEValues(double value) = 0;

    protected:
        /**
         * Utility function to initialize matrices. The only difference between ColA and InnerABC
         * it that the first splits A by column and another splits by row.
         */
        void initializationStage(int seed, bool split_by_row);

        /**
         * Replicates matrix A within replication group. Both algorithms perform this step
         * but they differ in the way they split into replication groups.
         */
        void replicateA();

        /** Creates communicator used to replicate A **/
        virtual mpi::Communicator get_comm_replication_A() = 0;

        void computationRound();

        static void
        compute(const matrix::CSR &sparse, const matrix::DenseBlock &dense,
                matrix::DenseBlock &result);
    };

    class ColA : public MatMul {


    public:
        ColA(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c, bool async_computation_stage=true);

        void initializationStage(int seed) override;

        /** Replicates Sparse A matrix withing replication group. */
        void replicationStage() override;

        void computationStage(int exponent) override;

        std::unique_ptr<matrix::DenseBlock> gatherC() override;

    private:
        void rotateA(mpi::Communicator &comm_rotation);

        void moveCtoB(bool zero_C);

        mpi::Communicator get_comm_replication_A() override;

        /** Creates communicator used to rotate A **/
        mpi::Communicator get_comm_rotation_A();

        int countGEValues(double value) override;

        void bcast_A_values_sizes();

        int get_replication_color();

        void computationStageAsync(int exponent);

        void computationStageSync(int exponent);
    };


    class InnerABC : public MatMul {

    public:
        InnerABC(mpi::Communicator *comm_world, std::unique_ptr<matrix::CSR> &&A, int c, bool async_computation_stage);

        void initializationStage(int seed) override;

        void replicationStage() override;

        void computationStage(int exponent) override;

        int countGEValues(double value) override;

    private:
        void initLocalBBeforeReplication();

        void rotateA(mpi::Communicator &comm_rotation);

        void swapCWithB(bool zero_C);


        std::unique_ptr<matrix::DenseBlock> gatherC() override;

        mpi::Communicator get_comm_replication_A() override;

        /** Creates communicator used to rotate A **/
        mpi::Communicator get_comm_rotation_A();

        mpi::Communicator get_comm_replication_B();

        void replicateB();

        void computationStageReplicateBAllReduce();
        void computationStageReplicateBBCast();

        void bcast_A_values_sizes();

        void computationStageAsync(int exponent);

        void computationStageSync(int exponent);
    };
}

#endif //MPI_MATRIX_MULTIPLICATION_MATMUL_H
