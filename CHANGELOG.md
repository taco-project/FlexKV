# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

