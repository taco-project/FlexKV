# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-27
### Feature
Universal:
- Add op-level callback for local get/put [#13](https://github.com/taco-project/FlexKV/pull/13)
- Add support for distributed sharing of the KV Cache, to suppot KV Cache sharing between CPU and SSD, as well as distributed sharing of PCFS  ([#17](https://github.com/taco-project/FlexKV/pull/17))
- Add GDS (GPU Direct Storage) Support ([#25](https://github.com/taco-project/FlexKV/pull/25))
- TP16 support ([#26](https://github.com/taco-project/FlexKV/pull/26))
- Support more kv cache layout. Now include: vLLM, SGLang, TensorRT-LM ([#27](https://github.com/taco-project/FlexKV/pull/27))
- GDS refactor & gtensor support ([#42](https://github.com/taco-project/FlexKV/pull/42))
- Support construct TensorSharedHandle directly from CUDA IPC Handle ([#44](https://github.com/taco-project/FlexKV/pull/44))


Targeting vllm: 
- Support dp > 1 while integrated with vllm ([#18](https://github.com/taco-project/FlexKV/pull/18))
- Add launch scripts for vllm adaption ([#47](https://github.com/taco-project/FlexKV/pull/47))
- Support TP16 for vLLM+FlexKV ([#59](https://github.com/taco-project/FlexKV/pull/59))

Targeting TensorRT-LLM 
- Support using FlexKV on TensorRT-LLM ([#48](https://github.com/taco-project/FlexKV/pull/48))
- Support TP16 for TensorRT-LLM+FlexKV ([#53](https://github.com/taco-project/FlexKV/pull/53))

### Optimization
- Mla d2h transfer optimization ([#19](https://github.com/taco-project/FlexKV/pull/19))
- optimize SSD I/O ([#33](https://github.com/taco-project/FlexKV/pull/33))
- Enhance cache eviction with frequency-aware grace time mechanism ([#38](https://github.com/taco-project/FlexKV/pull/38))
- Replace std::map with std::unordered_map in RadixTree ([#41](https://github.com/taco-project/FlexKV/pull/41))

### Bugfix
- Fix wrong head number for DeepSeek for vllm integration ([#23](https://github.com/taco-project/FlexKV/pull/23))
- Fix bug, if cpu match len is bigger than ssd when put, it will cause error ([#24](https://github.com/taco-project/FlexKV/pull/24))
- Fix benchmark_worker ([#31](https://github.com/taco-project/FlexKV/pull/31))
- Fix segfault caused by radix tree array out-of-bounds access ([#39](https://github.com/taco-project/FlexKV/pull/39))
- Fix cache_info ([#40](https://github.com/taco-project/FlexKV/pull/40))
- Fix port for GPU registration ([#45](https://github.com/taco-project/FlexKV/pull/45))
- Fix SSD allocator ([#46](https://github.com/taco-project/FlexKV/pull/46))
- Fix vllm init num_kv_heads bug ([#67](https://github.com/taco-project/FlexKV/pull/67))
- Fix model_config for non-MLA models ([#68](https://github.com/taco-project/FlexKV/pull/68))

### Misc
- Add doc for: 
  FlexKV + Dynamo ([#14](https://github.com/taco-project/FlexKV/pull/14)), 
  flexkv_config.json ([#15](https://github.com/taco-project/FlexKV/pull/15)),
  FlexKV + TensorRT-LLM ([#52](https://github.com/taco-project/FlexKV/pull/52))
- For config: Simplify user configuration ([#37](https://github.com/taco-project/FlexKV/pull/37)), and other slight update ([#43](https://github.com/taco-project/FlexKV/pull/43))



## [1.0.0] - 2025-09-11

### Added
- C++ radix tree for fast match, need set "index_accel": true in cache_config
- sync kernel launch
- a huge change that move cache engine to a library for accelerator(vLLM e.g.) to use instead of server-client mode.
  This accelerate the get and put when no KVCache is matched. This version includes breaking API changes and is not backward compatible. 
- add evict_ratio, need set "evict_ratio": 0.05 in cache_config
- reducing the bubble inner the launch kernel
- add vLLM 0.10.1.1 adapter

### Fixed
- cython release package


## [0.1.0] - 2025-08-29

### Init
- init version
- add license

