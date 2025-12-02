.PHONY: build-ext build-ext-test clean-ext format format-cpp pre-commit-install pre-commit-run

build-ext:
	cd csrc && python setup.py build_ext --inplace

build-ext-test:
	cd csrc && PYTHONPATH=. python test/test.py

clean-ext:
	rm -rf csrc/build

format:
	ruff format

format-cpp:
	find csrc -type file \( -name '*.h' -or -name '*.cpp' \) | xargs -n1 clang-format -i

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
