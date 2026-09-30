/*
 * 장치 N개와 장치 간 점대점 링크를 흉내내는 최소 통신 계층.
 *
 * 각 rank는 스레드 하나, 링크는 (src -> dst) 방향마다 슬롯 1개짜리 메일박스.
 * send는 메일박스가 비어 있으면 바로 반환하고(비동기 전송 버퍼 1개),
 * recv는 데이터가 올 때까지 대기한다. NCCL/MPI의 send/recv와 같은 계약이다.
 */
#ifndef COMM_H
#define COMM_H

#include <stddef.h>

typedef struct comm comm_t;

comm_t *comm_create(int nranks);
void comm_destroy(comm_t *c);
int comm_size(const comm_t *c);

void comm_send(comm_t *c, int src, int dst, const float *buf, size_t count);
void comm_recv(comm_t *c, int dst, int src, float *buf, size_t count);
void comm_barrier(comm_t *c);

/* rank별로 보낸 바이트/메시지 수. 각 카운터는 해당 rank 스레드만 쓴다. */
size_t comm_bytes_sent(const comm_t *c, int rank);
size_t comm_msgs_sent(const comm_t *c, int rank);
void comm_reset_stats(comm_t *c);

#endif
