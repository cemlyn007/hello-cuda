CXX_VERSION=c++20
flags = -s --subcommands --compilation_mode=opt --action_env=BAZEL_CXXOPTS="-std=$(CXX_VERSION)" --host_copt="-std=$(CXX_VERSION)" --copt="-std=$(CXX_VERSION)" --@rules_cuda//cuda:archs=sm_89 --action_env=CUDA_PATH=/usr/local/cuda-12.5 --repo_env=CC=clang --platforms=@toolchains_llvm//platforms:linux-x86_64 --@rules_cuda//cuda:compiler=clang

clean:
	rm compile_commands.json
	bazelisk clean --expunge

build_hello_world:
	bazelisk build $(flags) hello_world:main

run_hello_world: build_hello_world
	./bazel-bin/hello_world/main

build_vector_addition:
	bazelisk build $(flags) vector_addition:main

run_vector_addition: build_vector_addition
	./bazel-bin/vector_addition/main

refresh:
	bazelisk build $(flags) //...
	bazelisk run @hedron_compile_commands//:refresh_all -- $(flags)
