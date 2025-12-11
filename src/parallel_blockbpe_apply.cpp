#include "parallel_blockbpe.h"
#include <omp.h>

// Helper: Apply merges given marks
// tokens: current token sequence
// marks : marks[i] == 1 if a merge starts at i, non-overlapping (from step 3.2)
// new_token_id: merged token for best_pair
std::vector<int> apply_merges_parallel(
    const std::vector<int> &tokens,
    const std::vector<char> &marks,
    int new_token_id)
{
    std::size_t n = tokens.size();
    if (n == 0) return {};

    // Step 1: compute emit[i] in parallel
    std::vector<int> emit(n, 0);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        if (marks[i] == 1) {
            // left side of a merge: emits merged token
            emit[i] = 1;
        } else if (i > 0 && marks[i - 1] == 1) {
            // right side of a merge: emits nothing
            emit[i] = 0;
        } else {
            // normal token
            emit[i] = 1;
        }
    }

    // Step 2: parallel prefix sum on emit[]
    std::vector<int> prefix(n, 0);

    int num_threads = 1;
    std::vector<int> chunk_sums;

    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        // One thread initializes chunk_sums
        #pragma omp single
        {
            num_threads = nthr;
            chunk_sums.assign(num_threads, 0);
        }

        // Compute this thread's chunk
        std::size_t start = (n * tid) / nthr;
        std::size_t end   = (n * (tid + 1)) / nthr;

        int sum = 0;
        for (std::size_t i = start; i < end; ++i) {
            sum += emit[i];
            prefix[i] = sum;
        }
        chunk_sums[tid] = sum;

        #pragma omp barrier

        // Compute offset for this thread's chunk
        int offset = 0;
        for (int k = 0; k < tid; ++k) {
            offset += chunk_sums[k];
        }

        if (offset != 0) {
            for (std::size_t i = start; i < end; ++i) {
                prefix[i] += offset;
            }
        }
    }

    int out_len = prefix[n - 1];
    std::vector<int> out(out_len);

    // Step 3: parallel write tokens into output
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        if (emit[i] == 0) {
            // this index produces no output
            continue;
        }

        int pos = prefix[i] - 1; // 0-based index in out

        if (marks[i] == 1) {
            // merged token
            out[pos] = new_token_id;
        } else if (!(i > 0 && marks[i - 1] == 1)) {
            // normal token (not right side of a merge)
            out[pos] = tokens[i];
        }
    }

    return out;
}