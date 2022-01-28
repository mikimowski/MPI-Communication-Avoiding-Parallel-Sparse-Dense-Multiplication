#include <iostream>
#include "../include/utils.h"
#include "../include/mpi.h"
#include "../include/matmul.h"
#include <fstream>
#include <chrono>

int main(int argc, char **argv) {
    auto args = utils::Args{};
    args.parse(argc, argv);
    auto comm_world = std::make_unique<mpi::Communicator>(argc, argv);

    std::unique_ptr<matrix::CSR> A;
    if (comm_world->myRank() == 0) {
        A = std::make_unique<matrix::CSR>(utils::readCSR(args.sparse_matrix_file));
    } else {
        A = std::make_unique<matrix::CSR>(0, 0);
    }
    // for printing - handles zero padding
    const int N = A->getNumRows();

    // Create proper algorithm
    std::unique_ptr<matmul::MatMul> algorithm;
    if (args.inner_abc) {
        algorithm = std::make_unique<matmul::InnerABC>(comm_world.get(), std::move(A),
                                                   args.replication_group_size,
                                                   args.async_computation_stage);
    } else {
        algorithm = std::make_unique<matmul::ColA>(comm_world.get(), std::move(A),
                                                   args.replication_group_size,
                                                   args.async_computation_stage);
    }

    algorithm->initializationStage(args.seed);
    algorithm->replicationStage();
    algorithm->computationStage(args.exponent);

    if (args.verbose || !args.outfile.empty()) {
        auto finalC = algorithm->gatherC();
        if (comm_world->my_rank == mpi::MASTER_RANK) {
            // for printing - handles zero padding
            finalC->setMaxPrintColumns(N);
            if (args.verbose) {
                std::cout << finalC->_n << " " << finalC->_n << std::endl;
                std::cout << *finalC << std::endl;
            }
//            if (args.outfile != "") {
//                std::ofstream ofs(args.outfile, std::ofstream::out);
//                ofs << finalC->_n << " " << finalC->_n << std::endl;
//                ofs << *finalC << std::endl;
//                ofs.close();
//            }
        }
    }
    if (args.ge_value) {
        int ge = algorithm->countGEValues(*args.ge_value);
        if (comm_world->myRank() == 0) {
            std::cout << ge << std::endl;
        }
    }

    comm_world.reset(nullptr);
    return 0;
}
