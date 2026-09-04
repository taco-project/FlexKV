"""What "done" means for a CPU->GPU transfer.

Today "layerwise" names three things at once: a TransferType, a launch shape
(one H2D batch per layer), and a notification protocol (one eventfd per
original layer).  They are not the same thing.  The launch shape is a
performance knob; the notification protocol is a contract with the consumer.
Fusing them is why ``layer_granularity`` is hardcoded to 1 at the one Python
call site even though cpp has supported any value since it was written.

``CompletionContract`` is the contract half, split out:

``WHOLE``
    The consumer is told once, when the whole op has landed.  No eventfds; the
    op completes through the normal finished-ops queue.  cpp is free to move
    all layers in one batch.

``PER_LAYER``
    The consumer is told once per *original* layer, as that layer lands, via
    the per-layer eventfds it handed us over the UDS socket.  This is what
    lets sglang start layer L's attention while layer L+1 is still in flight.
    It requires ``layer_granularity == 1``: a coarser batch would post layer
    L's eventfd only after L+1 had also landed, which is a correctness
    problem, not a slowdown -- the consumer would read a layer it was told
    was ready before it was.

The contract picks the granularity rather than the other way round.
"""
from enum import Enum


class CompletionContract(Enum):
    """When the consumer of a CPU->GPU transfer is told a layer is readable."""

    WHOLE = "whole"
    PER_LAYER = "per_layer"

    @classmethod
    def from_str(cls, value: str) -> "CompletionContract":
        try:
            return cls(value.strip().lower())
        except ValueError:
            raise ValueError(
                f"unknown completion contract {value!r}; "
                f"expected one of {[c.value for c in cls]}"
            ) from None

    @property
    def needs_eventfd(self) -> bool:
        """PER_LAYER has no way to signal a layer without the consumer's fds."""
        return self is CompletionContract.PER_LAYER

    def layer_granularity(self, num_layers: int) -> int:
        """Layers per launched batch.

        PER_LAYER must be 1 (see the module docstring). WHOLE has no per-layer
        milestone to hit, so it hands cpp every layer at once and lets the
        batching there be driven by bandwidth rather than by the protocol.
        """
        if self is CompletionContract.PER_LAYER:
            return 1
        return max(1, num_layers)
