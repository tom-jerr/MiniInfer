def test_bench_simple_is_import_safe():
  # Import should not run the benchmark (no top-level side effects).
  import importlib

  mod = importlib.import_module("benchmark.bench_simple")
  assert hasattr(mod, "main")
