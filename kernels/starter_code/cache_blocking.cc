#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include "../matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::mat_mul_cache_blocking(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    
    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int m = C->row;
    const int n = C->column;
    const int k = A->column;
    
    // Defining block sizes for each dimension
    const int BM = 32;  // Block size for rows
    const int BN = 32;  // Block size for columns
    const int BK = 32;  // Block size for reduction dimension

    // Iterate over blocks
    for (int i0 = 0; i0 < m; i0 += BM) {
        for (int j0 = 0; j0 < n; j0 += BN) {
            // Initialize accumulator block
            const int imax = std::min(i0 + BM, m);
            const int jmax = std::min(j0 + BN, n);
            
            for (int i = i0; i < imax; i++) {
                for (int j = j0; j < jmax; j++) {
                    C->data_ptr[i * n + j] = 0;
                }
            }
            
            // Process blocks along k dimension
            for (int k0 = 0; k0 < k;) {
#ifdef QM_x86
                for (int i = i0; i < imax; i++) {
                    for (int j = j0; j < jmax; j++) {
                        // Get scales for the current block
                        float s_w = params->scales[(j * k + k0) / block_size];
                        float s_a = params->A_scales[(i * k + k0) / block_size];
                        float s_w_2nd = params->scales[(j * k + k0) / block_size + 1];
                        float s_a_2nd = params->A_scales[(i * k + k0) / block_size + 1];
                        
                        // Process the block
                        uint8_t *w_int4 = &B->int4_data_ptr[(j * k + k0) / 2];
                        const signed char *a_int8 = &A->int8_data_ptr[i * k + k0];
                        
                        int intermediate_sum = 0, intermediate_sum_2nd = 0;
                        for (int qj = 0; qj < 32; qj++) {
                            uint8_t packed_int4_0 = w_int4[qj];
                            signed char w_de_0 = (packed_int4_0 & 0x0F) - 8;
                            signed char w_de_32 = (packed_int4_0 >> 4) - 8;
                            
                            intermediate_sum += a_int8[qj] * w_de_0;
                            intermediate_sum_2nd += a_int8[qj + 32] * w_de_32;
                        }
                        
                        C->data_ptr[i * n + j] += (float)intermediate_sum * s_a * s_w;
                        C->data_ptr[i * n + j] += (float)intermediate_sum_2nd * s_a_2nd * s_w_2nd;
                    }
                }
                k0 += block_size * 2;
#endif
            }
        }
    }
}
}  // namespace matmul