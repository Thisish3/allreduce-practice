/* rank마다 스레드를 띄워 같은 allreduce를 reps번 실행하는 하네스 */
#ifndef RUNNER_H
#define RUNNER_H

#include <stddef.h>

#include "comm.h"

typedef void (*allreduce_fn)(comm_t *, int, float *, float *, size_t);

/*
 * bufs[r]: rank r의 입력이자 결과 버퍼 (count개).
 * reset이 NULL이 아니면 매 반복 전에 reset(rank, buf, count)로 입력을 되돌린다.
 * times_us가 NULL이 아니면 반복마다 barrier~barrier 사이 시간(us)을 기록한다.
 */
void run_allreduce(comm_t *c, allreduce_fn fn, float **bufs, size_t count, int reps,
                   void (*reset)(int, float *, size_t), double *times_us);

#endif
