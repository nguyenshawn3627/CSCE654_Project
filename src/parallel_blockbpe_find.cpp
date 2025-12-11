#include "parallel_blockbpe.h"
#include <omp.h>
#include <limits>       // <<< REQUIRED for std::numeric_limits
#include <optional>     // (optional but recommended)
#include <utility>      // for std::pair

// Helper: Find best pair
std::optional<std::pair<int,int>> find_best_pair_parallel(
    const std::vector<int> &tokens,
    const MergeTable &merges)
{
    int global_best_rank = std::numeric_limits<int>::max();
    std::pair<int,int> global_best_pair = {-1, -1};

    // Parallel region
    #pragma omp parallel
    {
        int local_best_rank = std::numeric_limits<int>::max();
        std::pair<int,int> local_best_pair = {-1, -1};

        // Parallel for with reduction on best rank
        #pragma omp for nowait
        for (int i = 0; i < (int)tokens.size() - 1; i++) {
            auto it = merges.find({tokens[i], tokens[i+1]});
            if (it == merges.end()) continue;

            int rank = it->second.rank;

            if (rank < local_best_rank) {
                local_best_rank = rank;
                local_best_pair = {tokens[i], tokens[i+1]};
            }
        }

        // Combine local results into global results
        #pragma omp critical
        {
            if (local_best_rank < global_best_rank) {
                global_best_rank = local_best_rank;
                global_best_pair = local_best_pair;
            }
        }
    }

    if (global_best_rank == std::numeric_limits<int>::max()) {
        return std::nullopt;  // no merge found
    }
    return global_best_pair;
}