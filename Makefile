build:
	bazelisk build -c opt --cuda_archs=sm_89 :main

run: build
	./bazel-bin/main

refresh: build
	bazelisk run -c opt --cuda_archs=sm_89 @hedron_compile_commands//:refresh_all