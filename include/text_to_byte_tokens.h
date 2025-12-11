#pragma once
#include <vector>
#include <string>

// Convert UTF-8 text to byte-level token IDs (0–255), GPT-2 style.
// Each raw byte becomes one token ID.
std::vector<int> text_to_byte_tokens(const std::string &text);
