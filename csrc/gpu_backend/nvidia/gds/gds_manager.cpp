/*
 * Vendor-scoped wrapper for GDS manager.
 *
 * The actual implementation lives at csrc/gds/gds_manager.cpp; this
 * file is the single source seen by setup.py once P3 lands. Headers
 * resolved against csrc/gpu_backend/nvidia/gds/ — i.e. the migrated
 * versions of gds_manager.h / layout_transform.cuh — keeping the build
 * deterministic.
 *
 * NOTE: GDS is NVIDIA-only.
 */
#include "gds_manager.h"
#include "layout_transform.cuh"
#include "../../../gds/gds_manager.cpp"
