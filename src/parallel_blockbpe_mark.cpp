#include "parallel_blockbpe.h"
#include <omp.h>

// Helper: Mark merge positions (non-overlapping)
// best_pair = (left_id, right_id)
std::vector<char> mark_merges_parallel(
    const std::vector<int> &tokens,
    const std::pair<int,int> &best_pair)
{
    std::size_t n = tokens.size();
    std::vector<char> tentative(n, 0);
    std::vector<char> marks(n, 0);

    if (n < 2) {
        return marks; // nothing to merge
    }

    int left  = best_pair.first;
    int right = best_pair.second;

    // ----- Pass A: parallel tentative marking -----
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n) - 1; ++i) {
        if (tokens[i] == left && tokens[i + 1] == right) {
            tentative[i] = 1;  // candidate merge start
        }
    }

    // ----- Pass B: sequential overlap resolution -----
    std::size_t i = 0;
    while (i + 1 < n) {
        if (tentative[i]) {
            marks[i] = 1;   // commit a merge here
            i += 2;         // skip overlapping index
        } else {
            // no merge starting here
            marks[i] = 0;
            i += 1;
        }
    }

    // last position can never start a merge (no pair), but keep it defined
    if (i < n) {
        marks[i] = 0;
    }

    return marks;
}