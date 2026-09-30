#include "allreduce.h"

#include <assert.h>

static void add_into(float *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] += src[i];
}

void allreduce_naive(comm_t *c, int rank, float *buf, float *tmp, size_t count) {
    int n = comm_size(c);
    if (rank == 0) {
        for (int src = 1; src < n; src++) {
            comm_recv(c, 0, src, tmp, count);
            add_into(buf, tmp, count);
        }
        for (int dst = 1; dst < n; dst++)
            comm_send(c, 0, dst, buf, count);
    } else {
        comm_send(c, rank, 0, buf, count);
        comm_recv(c, rank, 0, buf, count);
    }
}

/* count가 n으로 나누어떨어지지 않아도 되도록 조각 경계를 비례로 계산 */
static size_t chunk_begin(size_t count, int n, int i) {
    return count * (size_t)i / (size_t)n;
}

void allreduce_ring(comm_t *c, int rank, float *buf, float *tmp, size_t count) {
    int n = comm_size(c);
    int next = (rank + 1) % n;
    int prev = (rank - 1 + n) % n;

    /* reduce-scatter: 끝나면 rank r이 조각 (r+1)%n 의 최종 합을 가진다 */
    for (int s = 0; s < n - 1; s++) {
        int send_i = (rank - s + n) % n;
        int recv_i = (rank - s - 1 + n) % n;
        size_t sb = chunk_begin(count, n, send_i), se = chunk_begin(count, n, send_i + 1);
        size_t rb = chunk_begin(count, n, recv_i), re = chunk_begin(count, n, recv_i + 1);
        comm_send(c, rank, next, buf + sb, se - sb);
        comm_recv(c, rank, prev, tmp, re - rb);
        add_into(buf + rb, tmp, re - rb);
    }

    /* all-gather: 완성된 조각을 링으로 한 바퀴 돌려 모두에게 복사 */
    for (int s = 0; s < n - 1; s++) {
        int send_i = (rank + 1 - s + n) % n;
        int recv_i = (rank - s + n) % n;
        size_t sb = chunk_begin(count, n, send_i), se = chunk_begin(count, n, send_i + 1);
        size_t rb = chunk_begin(count, n, recv_i), re = chunk_begin(count, n, recv_i + 1);
        comm_send(c, rank, next, buf + sb, se - sb);
        comm_recv(c, rank, prev, buf + rb, re - rb);
    }
}

void allreduce_rd(comm_t *c, int rank, float *buf, float *tmp, size_t count) {
    int n = comm_size(c);
    assert((n & (n - 1)) == 0 && "recursive doubling requires power-of-two ranks");
    for (int dist = 1; dist < n; dist <<= 1) {
        int peer = rank ^ dist;
        comm_send(c, rank, peer, buf, count);
        comm_recv(c, rank, peer, tmp, count);
        add_into(buf, tmp, count);
    }
}
