# Mooncake I/O batch limits

The Mooncake store JSON config accepts optional `max_get_batch_bytes` and
`max_put_batch_bytes`, measured in payload bytes per SDK call. Both default
to `0` (unbounded), preserving existing batching. For example:

```json
{
  "max_get_batch_bytes": 67108864,
  "max_put_batch_bytes": 67108864
}
```

Add these fields to the existing store config. A positive limit must fit the
largest individual KV/SWA block: oversized blocks are rejected before any data
batch is submitted. Keys are never split and failed keys are not retried.
Limits cover the sum of requested data bytes, not allocator alignment, provider
metadata, or RDMA registration overhead. Choose them for the deployed provider.

A Get succeeds only when the SDK returns exactly the requested byte count for
that key. Short reads, zero-byte reads, oversized responses and negative return
codes fail that key. Put keeps per-key success (`0`) and skips existing keys.
Malformed result counts raise an error rather than being silently truncated.
The transfer backend reports that operation as failed; earlier completed Put
batches are not rolled back.
