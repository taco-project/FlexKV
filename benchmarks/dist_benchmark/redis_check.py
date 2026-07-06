#!/usr/bin/env python3
"""
FlexKV Redis Data Inspector

Check what data the put-only node has pushed to Redis.
This script inspects all FlexKV-related keys in Redis including:
  - global:node_id       (global node ID counter)
  - node:<id>            (registered node info)
  - meta:<id>            (node meta: mooncake engine addr, buffer ptrs)
  - buffer:<id>:*        (RDMA memory region registrations)
  - CPUB:<nid>:<hash>    (CPU KVCache block metadata - the actual cached data index)
  - SSDB:<nid>:<hash>    (SSD KVCache block metadata)
  - PCFSB:<nid>:<hash>   (PCFS remote KVCache block metadata)
  - pcfs:<id>            (PCFS file node IDs)
  - mooncake/*           (Mooncake Transfer Engine metadata)

Usage:
  python benchmarks/redis_check.py [--host HOST] [--port PORT] [--password PWD]

  # With defaults from example_dist_config.yml:
  python benchmarks/redis_check.py --host 10.135.1.175 --port 6379 --password 123456
"""

import argparse
import sys

try:
    import redis
except ImportError:
    print("ERROR: redis-py is required. Install with: pip install redis")
    sys.exit(1)


def connect_redis(host, port, password):
    """Connect to Redis and verify connectivity."""
    r = redis.Redis(
        host=host, port=port,
        password=password if password else None,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    try:
        r.ping()
        print(f"✅ Connected to Redis at {host}:{port}")
    except redis.ConnectionError as e:
        print(f"❌ Failed to connect to Redis at {host}:{port}: {e}")
        sys.exit(1)
    return r


def scan_keys(r, pattern, count=1000):
    """Scan Redis keys matching pattern (non-blocking)."""
    keys = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor=cursor, match=pattern, count=count)
        keys.extend(batch)
        if cursor == 0:
            break
    return sorted(keys)


def check_global_node_id(r):
    """Check the global node ID counter."""
    print("\n" + "=" * 60)
    print("  1. Global Node ID Counter")
    print("=" * 60)
    val = r.get("global:node_id")
    if val is not None:
        print(f"  global:node_id = {val}")
        print(f"  → {val} node(s) have been registered in total")
    else:
        print("  ⚠️  global:node_id not found (no nodes registered yet)")


def check_registered_nodes(r):
    """Check registered node information."""
    print("\n" + "=" * 60)
    print("  2. Registered Nodes (node:*)")
    print("=" * 60)
    keys = scan_keys(r, "node:*")
    if not keys:
        print("  ⚠️  No registered nodes found")
        return

    print(f"  Found {len(keys)} registered node(s):\n")
    for key in keys:
        data = r.hgetall(key)
        print(f"  📌 {key}:")
        for field, value in sorted(data.items()):
            print(f"     {field}: {value}")
        print()


def check_node_meta(r):
    """Check node meta information (mooncake engine addr, buffer ptrs)."""
    print("\n" + "=" * 60)
    print("  3. Node Meta (meta:*)")
    print("=" * 60)
    keys = scan_keys(r, "meta:*")
    if not keys:
        print("  ⚠️  No node meta found")
        print("  → This means PEER2CPUTransferWorker hasn't registered yet,")
        print("    or mooncake transfer engine initialization failed.")
        return

    print(f"  Found {len(keys)} node meta entry(ies):\n")
    for key in keys:
        data = r.hgetall(key)
        print(f"  📌 {key}:")
        for field, value in sorted(data.items()):
            # Format large integers (pointers) in hex for readability
            if field in ("cpu_buffer_ptr", "ssd_buffer_ptr"):
                try:
                    int_val = int(value)
                    print(f"     {field}: {value} (0x{int_val:x})")
                except (ValueError, TypeError):
                    print(f"     {field}: {value}")
            else:
                print(f"     {field}: {value}")
        print()


def check_buffer_registrations(r):
    """Check RDMA buffer registrations."""
    print("\n" + "=" * 60)
    print("  4. RDMA Buffer Registrations (buffer:*)")
    print("=" * 60)
    keys = scan_keys(r, "buffer:*")
    if not keys:
        print("  ⚠️  No RDMA buffer registrations found")
        return

    print(f"  Found {len(keys)} buffer registration(s):\n")
    for key in keys:
        data = r.hgetall(key)
        buf_size = data.get("buffer_size", "?")
        try:
            size_mb = int(buf_size) / (1024 * 1024)
            print(f"  📌 {key}: size={buf_size} bytes ({size_mb:.2f} MB)")
        except (ValueError, TypeError):
            print(f"  📌 {key}: size={buf_size}")


def check_block_metadata(r):
    """Check KVCache block metadata - this is the core data from put operations.

    FlexKV uses different key prefixes for different device types:
      - CPUB:<node_id>:<hash>  — CPU block metadata (P2P CPU sharing)
      - SSDB:<node_id>:<hash>  — SSD block metadata (P2P SSD sharing)
      - PCFSB:<node_id>:<hash> — PCFS remote block metadata
    Each key is a Redis hash with fields: ph, pb, nid, hash, lt, state.
    """
    print("\n" + "=" * 60)
    print("  5. KVCache Block Metadata (CPUB/SSDB/PCFSB)")
    print("=" * 60)

    # FlexKV actual block key prefixes (set in hie_cache_engine.py)
    block_prefixes = {
        "CPUB": "CPU",
        "SSDB": "SSD",
        "PCFSB": "PCFS (Remote)",
    }

    grand_total = 0
    for prefix, label in block_prefixes.items():
        keys = scan_keys(r, f"{prefix}:*")
        if not keys:
            print(f"\n  [{label}] {prefix}:* — no entries found")
            continue

        grand_total += len(keys)

        # Group by node_id: key format is PREFIX:<node_id>:<hash>
        node_blocks = {}
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 2:
                node_id = parts[1]
                if node_id not in node_blocks:
                    node_blocks[node_id] = []
                node_blocks[node_id].append(key)

        print(f"\n  [{label}] {prefix}:* — {len(keys)} block(s) across {len(node_blocks)} node(s):")

        for node_id, block_keys in sorted(node_blocks.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            print(f"    📌 Node {node_id}: {len(block_keys)} block(s)")

            # Show first few blocks as samples
            sample_count = min(3, len(block_keys))
            for key in block_keys[:sample_count]:
                data = r.hgetall(key)
                if data:
                    # BlockMeta fields: ph (physical hash), pb (physical block),
                    # nid (node id), hash, lt (lease time), state
                    ph = data.get("ph", "?")
                    pb = data.get("pb", "?")
                    nid = data.get("nid", "?")
                    hash_val = data.get("hash", "?")
                    lt = data.get("lt", "?")
                    state = data.get("state", "?")
                    print(f"       {key}: ph={ph}, pb={pb}, nid={nid}, hash={hash_val}, lt={lt}, state={state}")
                else:
                    key_type = r.type(key)
                    print(f"       {key}: type={key_type}, (empty hash)")

            if len(block_keys) > sample_count:
                print(f"       ... and {len(block_keys) - sample_count} more block(s)")

    if grand_total == 0:
        print("\n  ⚠️  No block metadata found in any prefix (CPUB/SSDB/PCFSB)")
        print("  → This means no KVCache data has been published to Redis yet.")
        print("    The put-only node may still be uploading, or the upload")
        print("    interval (rebuild_interval_ms) hasn't elapsed yet.")
    else:
        print(f"\n  ✅ Total block metadata entries: {grand_total}")


def check_pcfs_data(r):
    """Check PCFS file node IDs."""
    print("\n" + "=" * 60)
    print("  6. PCFS File Node IDs (pcfs:*)")
    print("=" * 60)
    keys = scan_keys(r, "pcfs:*")
    if not keys:
        print("  (none found - this is normal if PCFS sharing is not used)")
        return

    print(f"  Found {len(keys)} PCFS entry(ies):\n")
    for key in keys:
        values = r.lrange(key, 0, -1)
        print(f"  📌 {key}: {len(values)} file node ID(s)")
        if values:
            sample = values[:10]
            print(f"     sample: {sample}")
            if len(values) > 10:
                print(f"     ... and {len(values) - 10} more")


def check_mooncake_keys(r):
    """Check Mooncake Transfer Engine related keys."""
    print("\n" + "=" * 60)
    print("  7. Mooncake Transfer Engine Keys")
    print("=" * 60)
    # Mooncake uses Redis as metadata backend, keys may vary
    # Common patterns: segment info, endpoint info
    patterns = ["mooncake/*", "mooncake:*", "segment:*", "endpoint:*", "mc:*"]
    found_any = False
    for pattern in patterns:
        keys = scan_keys(r, pattern)
        if keys:
            found_any = True
            print(f"\n  Pattern '{pattern}': {len(keys)} key(s)")
            for key in keys[:10]:
                key_type = r.type(key)
                if key_type == "hash":
                    data = r.hgetall(key)
                    print(f"    📌 {key} (hash): {data}")
                elif key_type == "string":
                    val = r.get(key)
                    if val and len(val) > 200:
                        print(f"    📌 {key} (string): {val[:200]}...")
                    else:
                        print(f"    📌 {key} (string): {val}")
                elif key_type == "set":
                    members = r.smembers(key)
                    print(f"    📌 {key} (set): {members}")
                elif key_type == "list":
                    vals = r.lrange(key, 0, 9)
                    print(f"    📌 {key} (list): {vals}")
                else:
                    print(f"    📌 {key} (type={key_type})")
            if len(keys) > 10:
                print(f"    ... and {len(keys) - 10} more")

    if not found_any:
        print("  (no mooncake-specific keys found)")


def check_all_keys_summary(r):
    """Show a summary of ALL keys in Redis grouped by prefix."""
    print("\n" + "=" * 60)
    print("  8. All Keys Summary")
    print("=" * 60)
    all_keys = scan_keys(r, "*")
    if not all_keys:
        print("  ⚠️  Redis is completely empty!")
        return

    print(f"  Total keys in Redis: {len(all_keys)}\n")

    # Group by prefix (first part before ':')
    prefix_counts = {}
    for key in all_keys:
        prefix = key.split(":")[0] if ":" in key else key
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    print(f"  {'Prefix':<30} {'Count':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        print(f"  {prefix:<30} {count:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="FlexKV Redis Data Inspector - Check put-only node data"
    )
    parser.add_argument("--host", type=str, default="10.135.1.175",
                        help="Redis host (default: 10.135.1.175)")
    parser.add_argument("--port", type=int, default=6379,
                        help="Redis port (default: 6379)")
    parser.add_argument("--password", type=str, default="123456",
                        help="Redis password (default: 123456)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlexKV Redis Data Inspector")
    print("=" * 60)
    print(f"  Target: {args.host}:{args.port}")

    r = connect_redis(args.host, args.port, args.password)

    check_global_node_id(r)
    check_registered_nodes(r)
    check_node_meta(r)
    check_buffer_registrations(r)
    check_block_metadata(r)
    check_pcfs_data(r)
    check_mooncake_keys(r)
    check_all_keys_summary(r)

    print("\n" + "=" * 60)
    print("  Inspection Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
