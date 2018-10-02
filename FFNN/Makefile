CC=g++
CFLAGS=-std=c++11 -g

all: run

run: main.o
	$(CC) $^ -o $@ $(CFLAGS)

main.o: main.cpp
	$(CC) $< -c -o $@ $(CFLAGS)

