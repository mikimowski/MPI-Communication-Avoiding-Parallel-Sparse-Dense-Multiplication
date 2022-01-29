# MPI Communication-Avoiding Parallel Sparse-Dense Multiplication

C++ implementation of `ColA` and `InnerABC` algorithms from [Communication-Avoiding Parallel Sparse-Dense Multiplication](https://people.eecs.berkeley.edu/~yelick/papers/spdmmm16.pdf) leveraging Message Passing Interface (MPI).

Abstract: "Multiplication of a sparse matrix with a dense matrix is a building block of an increasing number of applications in many areas such as machine learning and graph algorithms. However, most previous work on parallel matrix multiplication considered only both dense or both sparse matrix operands. This paper analyzes the communication lower bounds and compares the communication costs of various classic parallel algorithms in the context of sparse-dense matrix-matrix multiplication. We also present new communication-avoiding algorithms based on a 1D decomposition, called 1.5D, which — while suboptimal in dense-dense and sparse-sparse cases — outperform the 2D and 3D variants both theoretically and in practice for sparsedense multiplication. Our analysis separates one-time costs from per iteration costs in an iterative machine learning context. Experiments demonstrate speedups up to 100x over a baseline 3D SUMMA implementation and show parallel scaling over 10 thousand cores."

## Project structure

The following files contain definitions of:
* [include/matrix.h](include/matrix.h) - matrices formats: CSR and DenseBlock
* [include/mpi.h](include/mpi.h) - MPI communication used across the program. Some of them are implemented and not used in final algorithm, as they were used to provide empirical comparison of runtime results
* [include/utils.h](include/utils.h) - utils such as args parser and CSR reader
* [include/matmul.h](include/matmul.h) - common parts of both algorithms, as well as their individual parts
* [include/logger.h](include/logger.h) - logging && performance measurement utilities

In general `src/*.cpp` contains corresponding implementations. The only exceptions are:
* [src/matmul.cpp](src/matmul.cpp) - implementation of common parts
* [src/col_a.cpp](src/col_a.cpp) - implementation of ColA algorithm specific features
* [src/inner_abc.cpp](src/inner_abc.cpp) - implementation of InnerABC algorithm specific features

However, all of them were declared in [include/matmul.h](include/matmul.h).

## Overview of the solution

### Notation
```
A - sparse matrix
B - dense matrix
C - dense result matrix
p - num processes
c - replication factor
n - number of rows/columns in matrix
```

### Matrices

[matrix.h](include/matrix.h)

#### CSR format

Follows [wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix) description

#### DenseBlock format

Dense matrix is represented as a vector `[row1, row2, row3...]`, and corresponding metadata: `n :- number of local (==global) rows, n_columns :- number of local columns`. Because matrices are squared, `n :- number of global rows == number of global columns`. To ensure that each process has the same number of local columns, that is `n_columns` is equal for each process, zero padding is added. I have also considered a little different memory layout for dense block and compared their times - see [DenseBlock memory layout - simple but crucial](#denseblock-memory-layout-simple-but-crucial)

## Common

### High-level overview
High level view on [main.cpp](src/main.cpp)
```
read sparse A
initializationStage()
replicationStage()
computationStage()
optional: printing
```
### Initialization Stage
```
1. Master reads A.
2. Master splits A into p fragments. (depending on algorithm, splits by row or by column).
3. Master distributes those fragments between processes.
4. Each process creates local B - using densematgen
5. Each process creates local C - full of 0s
```

### Replication Stage
Both algorithms share common replication phase for sparse A. They only differ in processes that receive corresponding shards. Additionally InnerABC replicates B, 

``` 
1. Processes replicate A within replication groups.
2. Each replication group announces their final size of A (explained later in Computation Stage)
3. if InnerABC then processes replicate B within proper replications groups (those are different than for A)
```

### ReplicateA

Simple communication within replication group. Each process sends its fragment to all processes within this group.

### ReplicateB

Within replication group processes reduce their matrices - using `MPI_Allreduce`. See `inner_abc.cpp/replicateB()` for details.

For comparison, I have also implemented P2P replication of dense matrix, which, as expected resulted in much worse performance.

### Computation Stage

General idea for computation is to do local computation and rotate sparse A. However, it can be improved by *overlapping data transfer and computation using **asynchronous communication*** - see pseudocode below. Overlapping those stages mitigates the overhead comming from transfering sparse matrices, empirically resulting in no overhead at all, as computation takes more time. This requires additional memory of `max_size(localA)` for recv buffer. 
To achieve fully asynchronous rotation of sparse A, process needs to know the size of incoming data ahead of time. Therefore, each process stores additionally `A_values_sizes` containing sizes of corresponding *replicated* fragments of A. To populate this vector, after replication stage, each replication group (process of rank 0 within this group) announces globally this group's sparse A's size. See `col_a.cpp/bcast_A_values_sizes()` or  `inner_abc.cpp/bcast_A_values_sizes()` for details.

```
for exponent = 1,2...,e:
    for round = 1,...,num_replication_groups:
        start async_send local A
        start async_recv next local A
        compute local
        wait for async_send to finish
        wait for async_recv to finish
    if notFinalExponent:
        B := C
    if InnerABC:
        replicateB
```

#### GatherC

Implemented using `MPI_Gather`

#### ColA

1. Processes gather directly to master.

#### InnerABC

1. Within replication group: reduce C to group's master.
2. Within master's of these replication groups gather to global Master.

### CountGEValues

Implemented using `MPI_Reduce`

#### ColA

1. Processes reduce directly to master.

#### InnerABC

1. Within replication group: reduce to group's master.
2. Within master's of these replication groups reduce to global Master.

#### Comparison with synchronous computation
For comparison I implemented a synchronous computation stage, that first finishes local computation and only then starts exchanging A. As expected, this approach worked significantly slower. The bigger the matrix, the bigger the gap. The local computation time was growing similarly in both cases. The asynchronous version was able to use it to cover the communication overhead. Meanwhile, synchronous approach was just getting slower as communication overhead was growing alongside matrix size. For details see `col_a.cpp/computationStageAsync()` vs `col_a.cpp/computationStageSync()`, or corresponding for `inner_abc.cpp/*`

<!-- ## ColA overview

```
```

## InnerABC overview

```
1. Processes replicate B within replication groups.
``` -->

## Performance comparison

To measure performance of each part I used `Logger` and `Timer` classes introduced in `logger.h`. This applies as well to results described in previous paragraphs.

General conclusion is that, when using MPI, collectives, together with MPI_Communicator are crucial to achieve high performance. This empirically confirms what we were learning during the course. Collectives, blended together with async communication, and actual computation being performed in the meantime, allow us to reduce communication overhead nearly to 0, and make it negligible.

## Compute Intensive calculations

### DenseBlock memory layout - simple but crucial

I have considered two layouts for Dense Matrices. Both store values in a vector.

First of them allows for easy replication and gathering (merging matrices across processes):

* ``` [column1 column2 column3 ...] ```

Second approach requires additional post-processing after merging, if we want to keep concise layout and natural representation, but provides significant speed-up because memory access is aligned: 

* ``` [row1 row2 row3 ... ]```

In this layour after merging two vectors we have the following layout: `[P1_vals, P2_vals, P3_vals...]`. Thus, after replication/marging we transform it to ` [row1_vals, row2_vals, row3_vals...]`. See `matrix.cpp/rearrangeAfterMerge` for details.

This might seem as a trivial difference, but it's *crucial*, because `matmul.cpp/computationRound()` with second approach is empirically over **5x times** faster. It comes with the overhead of rearranging the matrix, but it's done once after replication, and does not take that much time, actually being negligible. Meanwhile, computing is done multiple times.

More formally, it comes from the memory access layout. Here we want to access consecutive memory fragments as often as possible - for example, it's similar to the "stride-memory-access" when adding vectors. 

### Sparse A representation

Another variant I have considered is Sparse (row, column, value). This would allow for faster iteration over the values, however additional row vector will be bigger row_offset index used in CSR. I've decided that it's better to do a little bit more work (computation), and transfer less data, rather than transfer more data and save not that much in terms of computation. More work comes from the fact that some rows might be empty and we will visit all of them in each iteration.

## Computation

### Operational intensity

*Operational intensity* in a single computation round is following:
```
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
```
Going through the top:
```
2 reads
for each row:            N times
    2 reads
    2 flops
    for each offset:     nnz(A) times
        2 reads
        2 flops
        for each column: N times
            2 read
            1 write
            6 flops
```
This gives us (adding from top):
```
Reads: 2 + N*2 + N*nnz(A)*2 + N*nnz(A)*N*2
Writes: N*nnz(A)*N*1
Flops: N*2 + N*nnz(A)*2 + N*nnz(A)*N*5
```
This gives us:
```
Q = N(2 + nnz(A)*(2 + 2N)) + 2
W = N(2 + nnz(A)*(2 + 6N))
I = W/Q > 1
```

## Scaling

To see scaling charts see the coresponding pdf files.

### Weak Scaling

Let `N = N_ROWS = N_COLUMNS, P = N_PROCESSES`

Amount of work per processor is constant. Problem size grows with number of processors.

* [weak-scaling.pdf](raport/weak-scaling.pdf) represents the following setup:
```
(N, P)
(512, 1),
(812, 4),
(1024, 8),
(1172, 12),
(1290, 16),
(1389, 20),
(1476, 24),
```

### Strong Scaling

Matrix size (work) is constant, and number of processors grows.

* [strong-scaling-ColA.pdf](raport/strong-scaling-ColA.pdf) represents strong scalling for `ColA`
* [strong-scaling-InnerABC.pdf](strong-scaling-InnerABC.pdf) represents strong scalling for `InnerABC`


<!-- ### Stages comparison

In all cases, exponent was `e=300`.
Times measured in milliseconds, averaged across three runs, and rounded.
Those measurements represent rank_0 process, therefore e.g. countGE stage might take a while sometimes, as it's waiting for other processes to finish - requires synchronistation. So those stages should be treated as additional information. The main insight is for computationRounds and replication / rotation. Also, keep in mind that total time is affected by -v flag. Sometimes if values varied across the runtime (e.g. various sizes of local A might affect computation time), then range is given.

* Initialization Stage 
* Replication Stage
* Computation Stage
    * Computation Round - single local computation.
    * ShiftA - time spent on asyncWait until matrix A is fully rotated, that is, after the computation is done, and we are still sending/receiving.
    * Re-ReplicateB - replication B across replication groups, before next multiplication
* GatherC
* CountGE Values

Setup: matrix of size `NxN`, where `N=1010`, `P=16`, `c=4`, `e=300`.

#### ColA
|N|e| Initialization Stage |Replication Stage| ReplicateA | ReplicateB | Computation Stage | Computation Round | ShiftA|Re-ReplicateB|GatherC| CountGE | Total Time |
|---|--- |---|---|---|---|---|---|---| ---|---|---|---|
|512|1000|7|9|8|-|489|0|0|-|3|13|679|
|1010|300|27|30|30|-|9430|8|0.5|-|161|33|9884|
|5000|50|591|16|13|-|36348|180|0-2|-|451|70|37397|


#### InnerABC

|N|e| Initialization Stage |Replication Stage| ReplicateA | ReplicateB | Computation Stage | Computation Round | ShiftA|Re-ReplicateB|GatherC| CountGE | Total Time |
|---|--- |---|---|---|---|---|---|---| ---|---|---|---|
|512|500|7|11|8|2|662|0-1|0|0-1|5|44|728|
|1010|300|52|14|3|4|10000|30-40|0.7|1-5|20|58|10648|
|5000|50|586|249|11|70|41007|740|1|33|614|5555|46073|
 -->
