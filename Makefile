SDK := $(shell xcrun --sdk macosx --show-sdk-path)
SRC := $(wildcard src/*.swift)

.PHONY: all check overfit train stories stories-check stories-hi tools pictures photos photo-tokens images-check images-ar images-dit images-ncm-check images-ncm demo clean

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

stories-hi: build/gpt
	./build/gpt stories-hi

tools: build/gpt
	./build/gpt tools

pictures:
	python3 scripts/make_pictures.py

photos:
	python3 scripts/make_photos.py

photo-tokens:
	python3 scripts/make_photo_tokens.py

images-check: build/gpt
	./build/gpt images-check

images-ar: build/gpt
	./build/gpt images-ar

images-dit: build/gpt
	./build/gpt images-dit

images-ncm-check: build/gpt
	./build/gpt images-ncm-check

images-ncm: build/gpt
	./build/gpt images-ncm

demo: build/gpt
	python3 demo/serve.py

clean:
	rm -rf build
