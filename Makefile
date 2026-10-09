SDK := $(shell xcrun --sdk macosx --show-sdk-path)
SRC := $(wildcard src/*.swift)

.PHONY: all check overfit train stories stories-check clean

all: build/gpt

build/gpt: $(SRC) metal/kernels.metal
	mkdir -p build
	cp metal/kernels.metal build/kernels.metal
	xcrun --sdk macosx swiftc -O -sdk $(SDK) -framework Metal -framework Foundation $(SRC) -o $@

check: build/gpt
	./build/gpt check

overfit: build/gpt
	./build/gpt overfit

train: build/gpt
	./build/gpt train

stories-check: build/gpt
	./build/gpt stories-check

stories: build/gpt
	./build/gpt stories

clean:
	rm -rf build
