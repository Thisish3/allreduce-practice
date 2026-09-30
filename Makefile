CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2 -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lpthread

SRC = src/comm.c src/allreduce.c src/runner.c

all: test_allreduce bench_allreduce

test_allreduce: $(SRC) tests/test_allreduce.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_allreduce: $(SRC) bench/bench_allreduce.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: test_allreduce
	./test_allreduce

bench: bench_allreduce
	./bench_allreduce 8

# 스레드 간 데이터 경합 검사
tsan: $(SRC) tests/test_allreduce.c
	$(CC) -std=c11 -g -O1 -fsanitize=thread -D_POSIX_C_SOURCE=200809L -o test_tsan $^ $(LDFLAGS)
	./test_tsan

clean:
	rm -f test_allreduce bench_allreduce test_tsan

.PHONY: all test bench tsan clean
