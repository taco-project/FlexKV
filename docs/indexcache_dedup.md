# IndexCache deduplication for GLM5.2

DSA models can reuse an earlier layer's Index-K cache on layers marked
`skip_topk`. Registering those aliases as separate physical layers stores and
transfers the same Index-K payload repeatedly.

Set `FLEXKV_DEDUP_INDEXER_GROUP=1` in every SGLang worker to register only active
Index-K buffers. The default is off. The compact group retains the original
layer IDs, so layerwise transfer still waits on the correct model layer.
Page-packed Index-K rows remain one row per page; the page size is not applied
twice. CPU and SSD block capacities are recomputed from the compact layout.

For example, a synthetic eight-layer pool with active Index-K layers
`[0, 3, 7]` registers three indexer buffers. This illustrates the mapping and is
not a model architecture or a measured capacity result. Savings depend on the
actual active layers, KV geometry, and byte budget.

The initial implementation supports TP and ordinary CP where every cache rank
exposes the complete Index-K pool. It rejects DSA cache layer split, malformed
skip metadata, empty or aliased active buffers, inconsistent buffer geometry,
and inconsistent group membership across registrations.

Use an empty cache pool or an isolated layout/version namespace when switching
this flag. Old and compact layouts must not reuse the same FlexKV or Mooncake
cache objects. This flag does not automatically migrate or clear stored data.

SGLang's optional `FLEXKV_DEFER_DUPLICATE_RESTORES=1` is a separate optimization:
it defers concurrent restores of an identical Host prefix until the producer
publishes its GPU radix entry. It requires the corresponding SGLang adaptation
and is independent of IndexCache layout deduplication.

The regression tests use synthetic CPU tensors and cover layout, capacity, and
registration contracts. GPU transfers, GLM5.2 output accuracy, performance, and
sustained operation require separate validation of the deployed revisions.
