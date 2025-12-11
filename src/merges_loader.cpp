#include "merges_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

MergeTable load_gpt2_merges(const std::string &filename)
{
    MergeTable merges;

    // -----------------------------------------
    // 1. Initialize vocab with byte tokens 0-255
    // -----------------------------------------
    std::unordered_map<std::string, int> vocab;
    vocab.reserve(60000);

    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        vocab[s] = i;
    }

    int next_token_id = 256;

    // -----------------------------------------
    // 2. Open merges file
    // -----------------------------------------
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open merges file: " << filename << "\n";
        return merges;
    }

    std::string line;
    int rank = 0;

    // Skip GPT-2 header line if present
    std::getline(file, line);
    if (line.rfind("#version", 0) != 0) {
        // Not a version header, so this was a merge pair → process it
        std::stringstream ss(line);
        std::string A, B;
        if (ss >> A >> B) {

            // Ensure token A exists
            if (vocab.count(A) == 0)
                vocab[A] = next_token_id++;

            // Ensure token B exists
            if (vocab.count(B) == 0)
                vocab[B] = next_token_id++;

            // Create merged symbol
            std::string merged = A + B;
            if (vocab.count(merged) == 0)
                vocab[merged] = next_token_id++;

            merges[{vocab[A], vocab[B]}] = MergeRule{rank, vocab[merged]};
            rank++;
        }
    }

    // -----------------------------------------
    // 3. Read merge pairs line-by-line
    // -----------------------------------------
    while (std::getline(file, line)) {
        if (line.size() < 2) continue;

        std::stringstream ss(line);
        std::string A, B;
        if (!(ss >> A >> B)) continue;

        // Ensure token A exists
        if (vocab.count(A) == 0)
            vocab[A] = next_token_id++;

        // Ensure token B exists
        if (vocab.count(B) == 0)
            vocab[B] = next_token_id++;

        // Create merged symbol string
        std::string merged = A + B;

        // Ensure merged token exists
        if (vocab.count(merged) == 0)
            vocab[merged] = next_token_id++;

        int left_id  = vocab[A];
        int right_id = vocab[B];
        int merged_id = vocab[merged];

        merges[{left_id, right_id}] = MergeRule{rank, merged_id};
        rank++;
    }

    std::cout << "Loaded " << merges.size() << " merge rules.\n";
    std::cout << "Final vocabulary size: " << vocab.size() << "\n";

    return merges;
}
