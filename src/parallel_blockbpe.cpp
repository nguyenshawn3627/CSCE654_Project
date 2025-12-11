#include "parallel_blockbpe.h"

// BlockBPE encoder
std::vector<int> block_bpe_encode_parallel(
    const std::vector<int> &input_tokens,
    const MergeTable &merges)
{
    if (input_tokens.empty()) return input_tokens;

    std::vector<int> tokens = input_tokens;

    while (true) {
        // Phase 1: find best pair
        auto best_pair_opt = find_best_pair_parallel(tokens, merges);
        if (!best_pair_opt.has_value()) {
            break; // no more merges
        }
        std::pair<int,int> best_pair = best_pair_opt.value();

        const MergeRule &rule = merges.at(best_pair);
        int new_token_id = rule.new_token_id;

        // Phase 2: mark positions to merge
        std::vector<char> marks = mark_merges_parallel(tokens, best_pair);

        // Phase 3: apply merges to get new tokens
        std::vector<int> new_tokens = apply_merges_parallel(tokens, marks, new_token_id);

        tokens.swap(new_tokens);
    }

    return tokens;
}