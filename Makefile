
CC = swiftc

.PHONY: clean

all: main

main: main.swift
	@$(CC) -o hw1 main.swift

release: main.swift
	@$(CC) -o hw1 -O main.swift

clean:
	@rm hw1
