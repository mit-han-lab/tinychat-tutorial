#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif

#define CACHE_BLOCK_SIZE 32  // Cache block size for blocking optimization

struct w4a8_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};

static void *all_techniques_worker_func(void *args) {
    struct w4a8_thread_args *mat_args = (struct w4a8_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column;
    int block_size = params->block_size;

#ifdef QM_x86
    // Cache blocking technique integrated
    for (int row_block = 0; row_block < m; row_block += CACHE_BLOCK_SIZE) {
        for (int col_block = mat_args->start_j; col_block < mat_args->end_j; col_block += CACHE_BLOCK_SIZE) {
            for (int row = row_block; row < row_block + CACHE_BLOCK_SIZE && row < m; row++) {
                for (int col = col_block; col < col_block + CACHE_BLOCK_SIZE && col < n; col++) {
                    __m256 accumulator = _mm256_setzero_ps();
                    float *s_ptr = &params->scales[col * k / 32];
                    float *sa_ptr = &params->A_scales[row * k / 32];
                    const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
                    const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
                    const int num_block = k / block_size;

                    for (int q = 0; q < num_block; q += 4) {
                        const __m256i lowMask = _mm256_set1_epi8(0xF);
                        __m256i raw_w = _mm256_loadu_si256(w_start);
                        __m256i raw_w_next = _mm256_loadu_si256(w_start + 1);

                        __m256i w_low = _mm256_and_si256(raw_w, lowMask);
                        __m256i w_high = _mm256_srli_epi16(raw_w, 4);
                        w_high = _mm256_and_si256(w_high, lowMask);

                        __m256i w_low_next = _mm256_and_si256(raw_w_next, lowMask);
                        __m256i w_high_next = _mm256_srli_epi16(raw_w_next, 4);
                        w_high_next = _mm256_and_si256(w_high_next, lowMask);

                        const __m256i zero_point = _mm256_set1_epi8(8);
                        __m256i w_0 = _mm256_sub_epi8(w_low, zero_point);
                        __m256i w_128 = _mm256_sub_epi8(w_high, zero_point);
                        __m256i w_0_next = _mm256_sub_epi8(w_low_next, zero_point);
                        __m256i w_128_next = _mm256_sub_epi8(w_high_next, zero_point);

                        __m256i dot = _mm256_maddubs_epi16(_mm256_sign_epi8(w_0, w_0), _mm256_sign_epi8(a_start[0], w_0));
                        __m256i dot2 = _mm256_maddubs_epi16(_mm256_sign_epi8(w_128, w_128), _mm256_sign_epi8(a_start[1], w_128));
                        __m256i dot3 = _mm256_maddubs_epi16(_mm256_sign_epi8(w_0_next, w_0_next), _mm256_sign_epi8(a_start[2], w_0_next));
                        __m256i dot4 = _mm256_maddubs_epi16(_mm256_sign_epi8(w_128_next, w_128_next), _mm256_sign_epi8(a_start[3], w_128_next));

                        const __m256i ones = _mm256_set1_epi16(1);
                        __m256 intermediate = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot));
                        __m256 intermediate2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot2));
                        __m256 intermediate3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot3));
                        __m256 intermediate4 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot4));

                        __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                        __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
                        __m256 v_s3 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);
                        __m256 v_s4 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);

                        accumulator = _mm256_fmadd_ps(intermediate, v_s, accumulator);
                        accumulator = _mm256_fmadd_ps(intermediate2, v_s2, accumulator);
                        accumulator = _mm256_fmadd_ps(intermediate3, v_s3, accumulator);
                        accumulator = _mm256_fmadd_ps(intermediate4, v_s4, accumulator);

                        s_ptr += 4;
                        sa_ptr += 4;
                        w_start += 2;
                        a_start += 4;
                    }
                    float *ptr = (float *)&accumulator;
                    C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
                }
            }
        }
    }
#endif
    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    assert(params->block_size == 32);  // Ensure block size is 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, params->block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct w4a8_thread_args threads_args[num_thread];
    int cols_per_thread = C->column / num_thread;

    for (int i = 0; i < num_thread; i++) {
        threads_args[i].start_j = i * cols_per_thread;
        threads_args[i].end_j = (i == num_thread - 1) ? C->column : (i + 1) * cols_per_thread;
        threads_args[i].params = params;
        pthread_create(&thread_pool[i], NULL, all_techniques_worker_func, &threads_args[i]);
    }
    for (int i = 0; i < num_thread; i++) {
        pthread_join(thread_pool[i], NULL);
    }
};
}  // namespace matmul
