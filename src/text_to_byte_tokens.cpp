#include "text_to_byte_tokens.h"

std::vector<int> text_to_byte_tokens(const std::string &text) {
    std::vector<int> tokens;
    tokens.reserve(text.size()); // reserve capacity for performance

    for (unsigned char c : text) {
        tokens.push_back(static_cast<int>(c));  // convert byte → int 0–255
    }

    return tokens;
}