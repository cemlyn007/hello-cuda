CXX_VERSION=c++20
flags = -s --subcommands --compilation_mode=opt --action_env=BAZEL_CXXOPTS="-std=$(CXX_VERSION)" --host_copt="-std=$(CXX_VERSION)" --copt="-std=$(CXX_VERSION)" --@rules_cuda//cuda:archs=sm_89 --action_env=CUDA_PATH=/usr/local/cuda-12.5 --repo_env=CC=clang --platforms=@toolchains_llvm//platforms:linux-x86_64 --@rules_cuda//cuda:compiler=clang

clean:
	rm compile_commands.json
	bazelisk clean --expunge

build:
	bazelisk build $(flags) :main

run: build
	./bazel-bin/main

refresh: build
	bazelisk run @hedron_compile_commands//:refresh_all -- $(flags)
