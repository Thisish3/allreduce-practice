#include "runner.h"

#include <pthread.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    comm_t *c;
    allreduce_fn fn;
    int rank;
    float *buf;
    size_t count;
    int reps;
    void (*reset)(int, float *, size_t);
    double *times_us;
} worker_arg_t;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void *worker(void *p) {
    worker_arg_t *a = p;
    float *tmp = malloc((a->count ? a->count : 1) * sizeof(float));
    for (int i = 0; i < a->reps; i++) {
        if (a->reset)
            a->reset(a->rank, a->buf, a->count);
        comm_barrier(a->c);
        double t0 = now_us();
        a->fn(a->c, a->rank, a->buf, tmp, a->count);
        /* 가장 늦게 끝난 rank까지 기다려야 allreduce 한 번이 끝난 것 */
        comm_barrier(a->c);
        if (a->rank == 0 && a->times_us)
            a->times_us[i] = now_us() - t0;
    }
    free(tmp);
    return NULL;
}

void run_allreduce(comm_t *c, allreduce_fn fn, float **bufs, size_t count, int reps,
                   void (*reset)(int, float *, size_t), double *times_us) {
    int n = comm_size(c);
    pthread_t *th = malloc(n * sizeof(pthread_t));
    worker_arg_t *args = malloc(n * sizeof(worker_arg_t));
    for (int r = 0; r < n; r++) {
        args[r] = (worker_arg_t){c, fn, r, bufs[r], count, reps, reset, times_us};
        pthread_create(&th[r], NULL, worker, &args[r]);
    }
    for (int r = 0; r < n; r++)
        pthread_join(th[r], NULL);
    free(th);
    free(args);
}
