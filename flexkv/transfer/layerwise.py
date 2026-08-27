"""Compatibility façade over :mod:`flexkv.transfer.layer_eventfd`.

This module used to hold ``LayerwiseTransferWorker`` -- a worker class whose
whole reason to exist was per-layer notification -- plus the UDS handshake that
receives the consumer's eventfds. The worker is gone: per-layer completion is
now a *contract* an ordinary CPU<->GPU worker honours
(``CompletionContract.PER_LAYER``, via ``RegionBatchGroup::submit_layerwise``),
so there is no launcher left to be layerwise-specific. What survived is the
handshake, and it moved to ``layer_eventfd`` because that is what it is.

The rename is not free, though: sglang's FlexKV connector spells the import
``from flexkv.transfer.layerwise import build_layerwise_eventfd_socket_path``,
and it is out of this repo's tree -- an installed FlexKV that dropped this path
makes ``--enable-flexkv`` raise at import, before any transfer is attempted.
The path is part of the integration contract, so it keeps working.

New code should import from ``flexkv.transfer.layer_eventfd``.
"""

from flexkv.transfer.layer_eventfd import (
    build_layerwise_eventfd_socket_path,
    receive_layer_eventfds,
)

__all__ = [
    "build_layerwise_eventfd_socket_path",
    "receive_layer_eventfds",
]
