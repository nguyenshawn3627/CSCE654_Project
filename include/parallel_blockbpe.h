
/*
 * parallel_blockbpe.h
 *
 * Small header for Parallel Block BPE utilities.
 * Declares parallelized helpers used to find, mark, and apply
 * merges for Block BPE encoding.
 *
 * Project: CSCE654 Parallel BlockBPE
 * Date: 2025-12-03
 */

#pragma once

#include <vector>
#include <optional>
#include <utility>

#include "merges_loader.h"

// Returns the best pair (index, token_id) to merge, if any.
std::optional<std::pair<int,int>>
find_best_pair_parallel(const std::vector<int>& tokens,
                        const MergeTable& merges);

// Produces a mark array indicating which token positions participate
// in merges for the selected pair.
std::vector<char>
mark_merges_parallel(const std::vector<int>& tokens,
                     const std::pair<int,int>& best_pair);

// Applies marked merges and returns the new token vector where
// merged pairs are replaced by `new_token_id`.
std::vector<int>
apply_merges_parallel(const std::vector<int>& tokens,
                      const std::vector<char>& marks,
                      int new_token_id);

// Top-level parallel Block BPE encode routine.
std::vector<int>
block_bpe_encode_parallel(const std::vector<int>& tokens,
                          const MergeTable& merges);

// End of file: parallel_blockbpe.h