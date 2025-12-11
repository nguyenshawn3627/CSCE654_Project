#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

struct MergeRule {
    int rank;
    int new_token_id;
};

struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const noexcept {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

using MergeTable = std::unordered_map<std::pair<int,int>, MergeRule, PairHash>;

MergeTable load_gpt2_merges(const std::string& filename);
