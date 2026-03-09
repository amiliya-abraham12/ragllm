"""
GPU Diagnostic Script for RAG LLM
Run this to verify your GPU is being used properly.
Usage: python check_gpu.py
"""

import sys

def check_pytorch_cuda():
    """Check if PyTorch can see the GPU"""
    print("=" * 50)
    print("1. PyTorch CUDA Check")
    print("=" * 50)
    try:
        import torch  # type: ignore[import]
        print(f"   PyTorch version : {torch.__version__}")
        print(f"   CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version    : {torch.version.cuda}")
            print(f"   GPU name        : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"   Total VRAM      : {total:.2f} GB")
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"   Allocated       : {allocated:.0f} MB")
            print(f"   Reserved        : {reserved:.0f} MB")
            print("   ✅ PyTorch GPU is READY")
        else:
            print("   ❌ CUDA not available — embeddings will run on CPU")
    except ImportError:
        print("   ❌ PyTorch not installed")


def check_llama_cpp():
    """Check if llama-cpp-python has CUDA/cuBLAS support"""
    print()
    print("=" * 50)
    print("2. llama-cpp-python CUDA Check")
    print("=" * 50)
    try:
        import llama_cpp  # type: ignore[import]
        print(f"   Version: {llama_cpp.__version__}")

        # Check if built with CUDA by looking at supported backends
        has_cuda = False

        # Method 1: Check for CUBLAS/CUDA in the build info
        try:
            ggml = getattr(llama_cpp, 'llama_cpp', None) or llama_cpp
            # Try to check supported backends
            if hasattr(ggml, 'LLAMA_SUPPORTS_GPU_OFFLOAD'):
                has_cuda = ggml.LLAMA_SUPPORTS_GPU_OFFLOAD
            elif hasattr(ggml, 'llama_supports_gpu_offload'):
                has_cuda = ggml.llama_supports_gpu_offload()
        except Exception:
            pass

        # Method 2: Try loading a dummy model with GPU layers (practical check)
        if not has_cuda:
            try:
                # If n_gpu_layers > 0 doesn't raise, CUDA support exists
                import os
                model_path = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
                if os.path.exists(model_path):
                    test_llm = llama_cpp.Llama(
                        model_path=model_path,
                        n_ctx=128,
                        n_gpu_layers=1,
                        verbose=True
                    )
                    has_cuda = True
                    del test_llm
                    print("   (Verified by loading model with 1 GPU layer)")
                else:
                    print(f"   ⚠️  Model not found at {model_path}, cannot do practical GPU test")
            except Exception as e:
                err_str = str(e).lower()
                if "cuda" in err_str or "cublas" in err_str:
                    has_cuda = False
                else:
                    # Other error — might still have CUDA
                    print(f"   ⚠️  Could not verify: {e}")

        if has_cuda:
            print("   ✅ llama-cpp-python has GPU support — layers WILL offload")
        else:
            print("   ❌ llama-cpp-python likely CPU-only — GPU layers are IGNORED")
            print()
            print("   FIX: Reinstall with CUDA support:")
            print("   pip install llama-cpp-python --force-reinstall --no-cache-dir \\")
            print('        -C cmake.args="-DGGML_CUDA=on"')

    except ImportError:
        print("   ❌ llama-cpp-python not installed")


def check_sentence_transformers():
    """Check if SentenceTransformers is using GPU"""
    print()
    print("=" * 50)
    print("3. SentenceTransformer Device Check")
    print("=" * 50)
    try:
        import torch  # type: ignore[import]
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Will load on: {device.upper()}")

        if device == "cuda":
            print("   ✅ Embeddings will run on GPU")
        else:
            print("   ⚠️  Embeddings will run on CPU (slower retrieval)")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")


def main():
    print()
    print("🔍 RAG LLM — GPU Diagnostic Report")
    print()
    check_pytorch_cuda()
    check_llama_cpp()
    check_sentence_transformers()
    print()
    print("=" * 50)
    print("Done! Check results above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
