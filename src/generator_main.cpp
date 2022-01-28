/*
 * Example usage for the matrix generation library, HPC, Spring 2020/2021
 *
 * Faculty of Mathematics, Informatics and Mechanics.
 * University of Warsaw, Warsaw, Poland.
 * 
 * Krzysztof Rzadca
 * LGPL, 2021
 *
 * Maciej Mikula changelog:
 * - Added writing to a file
 */

#include <iostream>
#include <fstream>
#include "../include/densematgen.h"

int main() {
//    const int seed = 42;
    const int size = 100;
    const int n_zeros = 5;
    std::string size_str = std::string(n_zeros - (std::to_string(size)).length(), '0') + std::to_string(size);
    std::ofstream file1, file2, file3;
    file1.open ("../tests_custom/dense_" + size_str + "_00042");
    file2.open ("../tests_custom/dense_" + size_str + "_00442");
    file3.open ("../tests_custom/dense_" + size_str + "_04242");

    file1 << size << " " << size << std::endl;
    file2 << size << " " << size << std::endl;
    file3 << size << " " << size << std::endl;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
//            const double entry = generate_double(seed, r, c);
//            std::cout << entry << " ";
            file1 << generate_double(42, r, c) << " ";
            file2 << generate_double(442, r, c) << " ";
            file3 << generate_double(4242, r, c) << " ";
        }
//        std::cout << std::endl;
        file1 << std::endl;
        file2 << std::endl;
        file3 << std::endl;
    }
    file1.close();
    file2.close();
    file3.close();

    return 0;
}

