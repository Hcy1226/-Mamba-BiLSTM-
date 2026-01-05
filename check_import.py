try:
    import packaging
    print(f"Packaging version: {packaging.__version__}")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("SUCCESS: Transformers imported correctly.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
