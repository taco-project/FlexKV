# FlexKV: KV-Cache Manager for LLM Inference

FlexKV is a KVCache manager designed for large language model (LLM) inference.

## Installation

To install FlexKV, run the following command:

```bash
# For debug build (no Cython)
./build.sh --debug

# For release build (with Cython)
./build.sh --release
# or simply
./build.sh
```

## Code Submission

To ensure code quality, use pre-commit hooks. Follow these steps to install and set up pre-commit:

1. Install pre-commit:

   ```bash
   pip install pre-commit
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

By following these steps, you can ensure that code checks and formatting are automatically performed when you use `git commit`.
