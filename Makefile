CXX_VERSION=c++20
flags = -s --subcommands --compilation_mode=dbg --action_env=BAZEL_CXXOPTS="-std=$(CXX_VERSION)" --host_cxxopt="-std=$(CXX_VERSION)" --cxxopt="-std=$(CXX_VERSION)" --@rules_cuda//cuda:archs=sm_89 --action_env=CUDA_PATH=/usr/local/cuda-12.5 --repo_env=CC=gcc --platforms=@toolchains_llvm//platforms:linux-x86_64 --@rules_cuda//cuda:compiler=nvcc --spawn_strategy=local --sandbox_debug --strip=never --features=cuda_device_debug

clean:
	rm -f compile_commands.json
	bazelisk clean --expunge

build_hello_world:
	bazelisk build $(flags) hello_world:main

run_hello_world: build_hello_world
	./bazel-bin/hello_world/main

build_vector_addition:
	bazelisk build $(flags) vector_addition:main

run_vector_addition: build_vector_addition
	./bazel-bin/vector_addition/main

build_grayscale:
	bazelisk build $(flags) grayscale:main

run_grayscale: build_grayscale
	./bazel-bin/grayscale/main

build_blur:
	bazelisk build $(flags) blur:main

run_blur: build_blur
	./bazel-bin/blur/main

build_matrix_multiplication:
	bazelisk build $(flags) matrix_multiplication:main

run_matrix_multiplication: build_matrix_multiplication
	./bazel-bin/matrix_multiplication/main

build_matrix_multiplication_per_row:
	bazelisk build $(flags) matrix_multiplication_per_row:main

run_matrix_multiplication_per_row: build_matrix_multiplication_per_row
	./bazel-bin/matrix_multiplication_per_row/main

refresh:
	bazelisk build $(flags) //...
	bazelisk run @hedron_compile_commands//:refresh_all -- $(flags)
