/*
 * allreduce(sum): 모든 rank의 buf를 원소별로 더한 결과를 모든 rank가 갖게 한다.
 * 텐서 병렬 추론에서 레이어마다 호출되는 NCCL의 핵심 연산.
 *
 * 세 알고리즘의 비용 (N = rank 수, S = 버퍼 바이트, a = 메시지당 지연, b = 링크 대역폭)
 *
 *   naive  : 전부 rank 0으로 모아서 더한 뒤 다시 뿌림
 *            단계 2,        rank 0이 (N-1)*S 를 보내고 받음 -> rank 0이 병목
 *   ring   : reduce-scatter + all-gather, 버퍼를 N조각으로 나눠 링으로 돌림
 *            단계 2(N-1),   rank당 2(N-1)/N * S  -> 대역폭 최적, 지연은 N에 비례
 *   rd     : recursive doubling, 단계마다 거리 2^k인 상대와 전체 버퍼 교환
 *            단계 log2(N),  rank당 log2(N) * S   -> 지연 최적, 대역폭은 비효율
 *
 * 작은 메시지는 a(단계 수)가, 큰 메시지는 b(보내는 바이트)가 지배한다.
 * NCCL이 메시지 크기에 따라 ring과 tree를 골라 쓰는 이유.
 */
#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <stddef.h>

#include "comm.h"

/* tmp는 count개 이상의 float 작업 공간 */
void allreduce_naive(comm_t *c, int rank, float *buf, float *tmp, size_t count);
void allreduce_ring(comm_t *c, int rank, float *buf, float *tmp, size_t count);
/* N이 2의 거듭제곱일 때만 */
void allreduce_rd(comm_t *c, int rank, float *buf, float *tmp, size_t count);

#endif
