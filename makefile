make:
	cmake --build ./build-debug/ --parallel
	./build-debug/testbed/testbed
regen:
	rm -rf ./build-debug
	cmake --preset debug