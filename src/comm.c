#include "comm.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int full;
    float *data;
    size_t count;
    size_t cap;
} mailbox_t;

struct comm {
    int n;
    mailbox_t *boxes; /* boxes[src * n + dst] */
    size_t *bytes_sent;
    size_t *msgs_sent;

    /* macOS에는 pthread_barrier가 없어서 세대(generation) 카운터로 직접 구현 */
    pthread_mutex_t bar_mu;
    pthread_cond_t bar_cv;
    int bar_waiting;
    unsigned long bar_gen;
};

static mailbox_t *box(comm_t *c, int src, int dst) {
    return &c->boxes[src * c->n + dst];
}

comm_t *comm_create(int nranks) {
    comm_t *c = calloc(1, sizeof(*c));
    c->n = nranks;
    c->boxes = calloc((size_t)nranks * nranks, sizeof(mailbox_t));
    for (int i = 0; i < nranks * nranks; i++) {
        pthread_mutex_init(&c->boxes[i].mu, NULL);
        pthread_cond_init(&c->boxes[i].cv, NULL);
    }
    c->bytes_sent = calloc(nranks, sizeof(size_t));
    c->msgs_sent = calloc(nranks, sizeof(size_t));
    pthread_mutex_init(&c->bar_mu, NULL);
    pthread_cond_init(&c->bar_cv, NULL);
    return c;
}

void comm_destroy(comm_t *c) {
    for (int i = 0; i < c->n * c->n; i++) {
        pthread_mutex_destroy(&c->boxes[i].mu);
        pthread_cond_destroy(&c->boxes[i].cv);
        free(c->boxes[i].data);
    }
    free(c->boxes);
    free(c->bytes_sent);
    free(c->msgs_sent);
    pthread_mutex_destroy(&c->bar_mu);
    pthread_cond_destroy(&c->bar_cv);
    free(c);
}

int comm_size(const comm_t *c) { return c->n; }

void comm_send(comm_t *c, int src, int dst, const float *buf, size_t count) {
    mailbox_t *b = box(c, src, dst);
    pthread_mutex_lock(&b->mu);
    while (b->full)
        pthread_cond_wait(&b->cv, &b->mu);
    if (b->cap < count) {
        b->data = realloc(b->data, count * sizeof(float));
        b->cap = count;
    }
    memcpy(b->data, buf, count * sizeof(float));
    b->count = count;
    b->full = 1;
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);

    c->bytes_sent[src] += count * sizeof(float);
    c->msgs_sent[src] += 1;
}

void comm_recv(comm_t *c, int dst, int src, float *buf, size_t count) {
    mailbox_t *b = box(c, src, dst);
    pthread_mutex_lock(&b->mu);
    while (!b->full)
        pthread_cond_wait(&b->cv, &b->mu);
    assert(b->count == count);
    memcpy(buf, b->data, count * sizeof(float));
    b->full = 0;
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
}

void comm_barrier(comm_t *c) {
    pthread_mutex_lock(&c->bar_mu);
    unsigned long gen = c->bar_gen;
    if (++c->bar_waiting == c->n) {
        c->bar_waiting = 0;
        c->bar_gen++;
        pthread_cond_broadcast(&c->bar_cv);
    } else {
        while (gen == c->bar_gen)
            pthread_cond_wait(&c->bar_cv, &c->bar_mu);
    }
    pthread_mutex_unlock(&c->bar_mu);
}

size_t comm_bytes_sent(const comm_t *c, int rank) { return c->bytes_sent[rank]; }
size_t comm_msgs_sent(const comm_t *c, int rank) { return c->msgs_sent[rank]; }

void comm_reset_stats(comm_t *c) {
    memset(c->bytes_sent, 0, c->n * sizeof(size_t));
    memset(c->msgs_sent, 0, c->n * sizeof(size_t));
}
