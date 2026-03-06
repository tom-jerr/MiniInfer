import pytest


def pytest_addoption(parser):
  parser.addoption(
    "--model",
    action="store",
    default=None,
    help="Model path or HuggingFace model id for real-model tests (skipped if omitted).",
  )


@pytest.fixture(scope="session")
def model_path(request) -> str:
  model = request.config.getoption("--model")
  if not model:
    pytest.skip("No --model provided for real-model tests.")

  try:
    import huggingface_hub

    huggingface_hub.snapshot_download(model, local_files_only=True)
  except Exception as e:
    pytest.skip(f"Model not found locally: {model}. Error: {e}")

  return str(model)
