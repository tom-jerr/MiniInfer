"""MiniInfer Extension - CUDA/C++ operations for inference optimization."""

try:
    from ._ext import vector_add
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import _ext module: {e}. "
        "Please build the extension first using: cd kernels/csrc && ./build.sh"
    )

    # Provide a fallback or raise
    def vector_add(*args, **kwargs):
        raise RuntimeError(
            "miniinfer_ext C++ extension not built. "
            "Please run: cd kernels/csrc && ./build.sh"
        )


__all__ = ["vector_add"]
