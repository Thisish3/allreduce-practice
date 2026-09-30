/*
 * 메시지 크기별로 세 알고리즘의 지연을 재고, NCCL과 같은 방식으로 busbw를 계산한다.
 *   algbw = S / t
 *   busbw = algbw * 2(N-1)/N   (알고리즘과 무관하게 링크 사용률을 비교하기 위한 정규화)
 *
 * 사용법: ./bench_allreduce [nranks=8] [reps=20]
 */
#include <stdio.h>
#include <stdlib.h>

#include "../src/allreduce.h"
#include "../src/runner.h"

static void reset(int rank, float *buf, size_t count) {
    for (size_t i = 0; i < count; i++)
        buf[i] = (float)(rank + 1);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_us(comm_t *c, allreduce_fn fn, float **bufs, size_t count, int reps) {
    double *t = malloc(reps * sizeof(double));
    run_allreduce(c, fn, bufs, count, reps, reset, t);
    qsort(t, reps, sizeof(double), cmp_double);
    double m = t[reps / 2];
    free(t);
    return m;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    int is_pow2 = (n & (n - 1)) == 0;

    size_t max_bytes = 16u << 20;
    float **bufs = malloc(n * sizeof(float *));
    for (int r = 0; r < n; r++)
        bufs[r] = malloc(max_bytes);
    comm_t *c = comm_create(n);

    struct {
        const char *name;
        allreduce_fn fn;
    } algos[] = {{"naive", allreduce_naive}, {"ring", allreduce_ring}, {"rd", allreduce_rd}};
    int n_algos = is_pow2 ? 3 : 2;

    printf("nranks=%d reps=%d (median)\n\n", n, reps);
    printf("%10s", "size");
    for (int a = 0; a < n_algos; a++)
        printf(" | %8s us %7s GB/s", algos[a].name, "bus");
    printf(" | fastest\n");

    for (size_t bytes = 256; bytes <= max_bytes; bytes *= 4) {
        size_t count = bytes / sizeof(float);
        if (bytes >= 1 << 20)
            printf("%8zuMB", bytes >> 20);
        else if (bytes >= 1 << 10)
            printf("%8zuKB", bytes >> 10);
        else
            printf("%9zuB", bytes);

        double best = 0;
        const char *best_name = "";
        for (int a = 0; a < n_algos; a++) {
            double us = median_us(c, algos[a].fn, bufs, count, reps);
            double busbw = (double)bytes / (us * 1e-6) * 2.0 * (n - 1) / n / 1e9;
            printf(" | %11.1f %11.2f", us, busbw);
            if (a == 0 || us < best) {
                best = us;
                best_name = algos[a].name;
            }
        }
        printf(" | %s\n", best_name);
    }

    comm_destroy(c);
    for (int r = 0; r < n; r++)
        free(bufs[r]);
    free(bufs);
    return 0;
}
