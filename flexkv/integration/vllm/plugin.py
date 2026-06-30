# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point for FlexKV.

Registered under the ``vllm.general_plugins`` setuptools entry-point group
(see FlexKV's ``setup.py``).  vLLM auto-loads every plugin in this group
inside EVERY process — the async-LLM host (process 0), the EngineCore
subprocess, and worker subprocesses — at startup, before any of them
constructs Prometheus / connector machinery.  See
``vllm/plugins/__init__.py`` ("Default plugins group will be loaded in all
processes").

Why we need this plugin
-----------------------
In PD-disaggregated deployments — ``MultiConnector([FlexKVConnectorV1,
NixlConnector])`` is the typical Dynamo setup — vLLM's
``MultiKVConnectorPromMetrics.observe`` asserts that every sub-connector
present in ``transfer_stats_data`` has a corresponding entry in its
Prometheus-metrics registry::

    # vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:119
    assert connector_id in self._prom_metrics, (
        f"{connector_id} is not contained in the list of registered "
        f"connectors with Prometheus metrics support: ..."
    )

The registry is populated from each connector class's
``build_prom_metrics`` classmethod; vLLM's stock ``FlexKVConnectorV1``
inherits the base implementation which returns ``None``.  That makes
FlexKV absent from the registry, and the assertion fires on the first
scheduler step that emits FlexKV stats — engine dies with
``EngineDeadError``.

Why an import-time monkey patch inside ``vllm_v1_adapter`` is not enough:
the Prometheus registry is built inside the async-LLM host process when
``StatLoggerManager`` constructs ``KVConnectorProm`` (see
``vllm/v1/engine/async_llm.py:160``), but ``vllm_v1_adapter`` only gets
imported lazily inside ``FlexKVConnectorV1.__init__`` — which happens in
the EngineCore subprocess, never in the host process.  So an
adapter-side patch never runs where the assertion will eventually fire.
The ``vllm.general_plugins`` entry point covers both processes uniformly.

What this plugin does
---------------------
Inject a ``build_prom_metrics`` classmethod onto vLLM's stock
``FlexKVConnectorV1`` so that ``MultiConnector.build_prom_metrics`` adds
FlexKV to the per-connector Prometheus registry with a *no-op* handler.
That satisfies the assertion and unblocks PD-disaggregated serving; the
FlexKV stats themselves continue to flow through FlexKV's own logging
path (the connector logs hit ratios / failure counts to its own
``flexkv_logger``).

Follow-up improvement (out of scope for this fix): replace the no-op with
real ``Gauge`` / ``Counter`` / ``Histogram`` registration so FlexKV
hit-ratio / failure / transfer-bytes metrics surface through the standard
vLLM ``/metrics`` endpoint.

This module must remain side-effect-free at import time.  All work
happens inside ``register()`` so vLLM's plugin loader stays in control
of when the patch is applied.
"""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """vLLM plugin entry point — called by ``load_general_plugins`` in
    every vLLM process at startup.

    Idempotent: safe to call multiple times.  vLLM's loader already
    guards against duplicate loads within a single process (see
    ``vllm.plugins.plugins_loaded``), but a manual call from a launch
    script before vLLM init also works without ill effects.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorPromMetrics,
        )
    except ImportError:
        # vLLM not installed, or too old to ship the metrics module — nothing
        # to patch.  Plugin entry-point invocations must not fail loudly.
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector import (
            FlexKVConnectorV1,
        )
    except ImportError:
        # vLLM version doesn't ship FlexKVConnectorV1 — nothing to patch.
        return

    class _FlexKVConnectorPromMetrics(KVConnectorPromMetrics):
        """No-op per-connector Prometheus metrics handler for
        ``FlexKVConnectorV1``.

        Exists to satisfy ``MultiKVConnectorPromMetrics.observe``'s
        type-equality assertion.  Records nothing; the FlexKV connector
        emits its own stats through ``flexkv_logger``.
        """

        def observe(self, transfer_stats_data, engine_idx: int = 0):
            return  # explicit no-op

    @classmethod
    def _build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ):
        return _FlexKVConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    # Only patch if FlexKVConnectorV1 itself does not already define
    # ``build_prom_metrics``.  Two cases this guards against:
    #   1) A future vLLM ships its own override; we must defer to it
    #      rather than clobber.
    #   2) This plugin was already loaded earlier in the same process
    #      (vLLM's loader has a single-process guard but a launch script
    #      may also call ``register()`` manually); calling again must be
    #      a no-op.
    #
    # We check ``__dict__`` directly instead of ``cls.method is
    # base.method`` because for classmethod descriptors each attribute
    # access produces a fresh bound method, so identity comparisons
    # across two accesses are unreliable.
    if "build_prom_metrics" not in FlexKVConnectorV1.__dict__:
        FlexKVConnectorV1.build_prom_metrics = _build_prom_metrics
        logger.info(
            "[FlexKV] vLLM plugin registered: injected build_prom_metrics "
            "into FlexKVConnectorV1 (no-op handler — required for "
            "MultiConnector PD-disaggregated deployments)."
        )
    else:
        logger.debug(
            "[FlexKV] vLLM plugin: FlexKVConnectorV1.build_prom_metrics "
            "already defined on the class (upstream override or earlier "
            "plugin load); plugin patch skipped."
        )
