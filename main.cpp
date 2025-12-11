#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <omp.h>
#include <chrono>

#include "merges_loader.h"
#include "text_to_byte_tokens.h"
#include "parallel_blockbpe.h"

int main()
{
    // 1. Load merges
    MergeTable merges = load_gpt2_merges("data/merges.txt");

    // 2. Read input lines from file
    std::ifstream infile("data/corpus.txt");
    if (!infile.is_open()) {
        std::cerr << "ERROR: Could not open .txt file\n";
        return 1;
    }

    std::vector<std::string> inputs;
    std::string line;

    while (std::getline(infile, line)) {
        if (!line.empty()) inputs.push_back(line);
    }

    std::cout << "Loaded " << inputs.size() << " input lines.\n";

    // 3. Prepare output container
    std::vector<std::vector<int>> outputs(inputs.size());

    // ----------------------------------------------------------
    // START TIMER (measure only the parallel BlockBPE section)
    // ----------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // 4. Parallel BlockBPE encoding
    #pragma omp parallel for
    for (int i = 0; i < (int)inputs.size(); i++) {
        auto bytes = text_to_byte_tokens(inputs[i]);
        auto tokens = block_bpe_encode_parallel(bytes, merges);
        outputs[i] = tokens;
    }

    // ----------------------------------------------------------
    // STOP TIMER
    // ----------------------------------------------------------
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n[Timing] BlockBPE encoding completed in "
              << elapsed_ms << " ms\n\n";

    // // 5. Print results
    // for (int i = 0; i < (int)outputs.size(); i++) {
    //     std::cout << "Input: " << inputs[i] << "\nTokens: ";
    //     for (int t : outputs[i]) std::cout << t << " ";
    //     std::cout << "\n\n";
    // }

    return 0;
}
