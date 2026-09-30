/*
 * 정확성 + 통신량 검증.
 * 입력을 정수값 float로 잡아서 더하는 순서와 무관하게 결과가 정확히 같아야 한다.
 */
#include <stdio.h>
#include <stdlib.h>

#include "../src/allreduce.h"
#include "../src/runner.h"

static float input(int rank, size_t i) { return (float)((rank + 1) * (int)(i % 7 + 1)); }

static void reset(int rank, float *buf, size_t count) {
    for (size_t i = 0; i < count; i++)
        buf[i] = input(rank, i);
}

static int failures = 0;

#define CHECK(cond, ...)                   \
    do {                                   \
        if (!(cond)) {                     \
            printf("  FAIL: " __VA_ARGS__); \
            printf("\n");                  \
            failures++;                    \
        }                                  \
    } while (0)

static void check(const char *name, allreduce_fn fn, int n, size_t count) {
    comm_t *c = comm_create(n);
    float **bufs = malloc(n * sizeof(float *));
    for (int r = 0; r < n; r++)
        bufs[r] = malloc((count ? count : 1) * sizeof(float));

    run_allreduce(c, fn, bufs, count, 1, reset, NULL);

    int ok = 1;
    for (int r = 0; r < n && ok; r++)
        for (size_t i = 0; i < count; i++) {
            float want = 0;
            for (int k = 0; k < n; k++)
                want += input(k, i);
            if (bufs[r][i] != want) {
                CHECK(0, "%s n=%d count=%zu rank=%d i=%zu got=%g want=%g", name, n, count, r, i,
                      bufs[r][i], want);
                ok = 0;
                break;
            }
        }
    if (ok)
        printf("  ok  %-5s n=%d count=%zu\n", name, n, count);

    for (int r = 0; r < n; r++)
        free(bufs[r]);
    free(bufs);
    comm_destroy(c);
}

/* 이론식대로 보내는지: ring은 rank당 2(N-1)/N*S, naive는 rank 0이 (N-1)*S */
static void check_traffic(void) {
    int n = 8;
    size_t count = 8 * 1024; /* n으로 나누어떨어지게 */
    size_t S = count * sizeof(float);
    float *bufs[8];
    for (int r = 0; r < n; r++)
        bufs[r] = malloc(count * sizeof(float));

    comm_t *c = comm_create(n);
    run_allreduce(c, allreduce_ring, bufs, count, 1, reset, NULL);
    for (int r = 0; r < n; r++)
        CHECK(comm_bytes_sent(c, r) == 2 * (n - 1) * S / n, "ring rank %d sent %zu", r,
              comm_bytes_sent(c, r));
    comm_reset_stats(c);
    run_allreduce(c, allreduce_naive, bufs, count, 1, reset, NULL);
    CHECK(comm_bytes_sent(c, 0) == (n - 1) * S, "naive root sent %zu", comm_bytes_sent(c, 0));
    comm_reset_stats(c);
    run_allreduce(c, allreduce_rd, bufs, count, 1, reset, NULL);
    CHECK(comm_msgs_sent(c, 0) == 3 && comm_bytes_sent(c, 0) == 3 * S, "rd rank0 %zu msgs",
          comm_msgs_sent(c, 0));
    comm_destroy(c);
    for (int r = 0; r < n; r++)
        free(bufs[r]);
    printf("  ok  traffic matches cost model (n=8)\n");
}

int main(void) {
    int ns[] = {1, 2, 3, 4, 5, 8};
    /* 0개, rank 수보다 적은 개수, 나누어떨어지지 않는 개수까지 */
    size_t counts[] = {0, 1, 3, 7, 1000, 4099};

    printf("correctness\n");
    for (size_t a = 0; a < sizeof(ns) / sizeof(ns[0]); a++)
        for (size_t b = 0; b < sizeof(counts) / sizeof(counts[0]); b++) {
            int n = ns[a];
            size_t cnt = counts[b];
            check("naive", allreduce_naive, n, cnt);
            check("ring", allreduce_ring, n, cnt);
            if ((n & (n - 1)) == 0)
                check("rd", allreduce_rd, n, cnt);
        }

    printf("traffic\n");
    check_traffic();

    if (failures) {
        printf("\n%d FAILED\n", failures);
        return 1;
    }
    printf("\nall passed\n");
    return 0;
}
