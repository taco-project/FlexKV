from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.flexkv_stress.config import ConversationConfig, load_config
from benchmarks.flexkv_stress.main import _should_stop
from benchmarks.flexkv_stress.metrics import (
    LogHistogram, OperationResult, Reporter, TurnResult, decimal_gb, percentile,
    throughput_gb_s,
)
from benchmarks.flexkv_stress.models import (
    DSV4_FLASH_RATIOS,
    GLM52_INDEXER_TYPES,
    get_preset,
    model_bytes_per_block,
    update_from_hf_config,
)
from benchmarks.flexkv_stress.workload import PrefixOracle, SlotAllocator, WorkloadGenerator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "benchmarks" / "flexkv_stress" / "configs"


class ModelPresetTests(unittest.TestCase):
    def test_model_presets_have_valid_physical_groups(self):
        expected_layers = {
            "dsv4_pro": 61,
            "dsv4_flash": 43,
            "glm5": 78,
            "glm5_2": 78,
        }
        for name, count in expected_layers.items():
            with self.subTest(name=name):
                preset = get_preset(name)
                self.assertEqual(preset.num_layers, count)
                self.assertGreater(model_bytes_per_block(preset), 0)
                for group in preset.groups:
                    self.assertTrue(group.layer_indices)
                    self.assertLess(max(group.layer_indices), count)
                    self.assertEqual(preset.tokens_per_block % group.compress_ratio, 0)

    def test_dsv4_geometry(self):
        preset = get_preset("dsv4_pro")
        groups = {group.name: group for group in preset.groups}
        self.assertEqual(groups["c4"].head_size, 585)
        self.assertEqual(groups["c128"].head_size, 864)
        self.assertEqual(groups["c4_indexer"].head_size, 132)
        self.assertEqual(groups["c4"].layer_indices, groups["c4_indexer"].layer_indices)

    def test_glm52_indexer_partition(self):
        self.assertEqual(len(GLM52_INDEXER_TYPES), 78)
        preset = get_preset("glm5_2")
        groups = {group.name: group for group in preset.groups}
        combined = set(groups["indexer_full"].layer_indices) | set(groups["indexer_shared"].layer_indices)
        self.assertEqual(combined, set(range(78)))
        self.assertFalse(set(groups["indexer_full"].layer_indices) & set(groups["indexer_shared"].layer_indices))

    def test_hf_config_refreshes_dsv4_service_layer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 43,
                "compress_ratios": DSV4_FLASH_RATIOS,
            }))
            preset = update_from_hf_config(get_preset("dsv4_flash"), path)
            self.assertEqual(preset.num_layers, 43)
            self.assertLess(max(layer for group in preset.groups for layer in group.layer_indices), 43)


class ConfigTests(unittest.TestCase):
    def test_all_example_configs_load(self):
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertGreater(config.required_gpus, 0)
                self.assertGreater(config.bytes_per_block, 0)

    def test_explicit_layer_group_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("""
model:
  preset: glm5
  layer_groups:
    - name: tiny
      layers: [0, 1]
      head_size: 64
      dtype: uint8
cache: {num_cpu_blocks: 8}
run: {rounds: 1}
""")
            config = load_config(path)
            self.assertEqual(config.preset.groups[0].layer_indices, (0, 1))

    def test_stop_conditions(self):
        config = load_config(CONFIG_DIR / "glm5.yaml")
        config.run.rounds = 10
        config.run.duration_seconds = 100
        config.run.stop_when = "either"
        self.assertTrue(_should_stop(config, 10, 1))
        config.run.stop_when = "both"
        self.assertFalse(_should_stop(config, 10, 1))
        self.assertTrue(_should_stop(config, 10, 100))

    def test_composite_tp_cp_overlap_matches_sglang_topology(self):
        config = load_config(CONFIG_DIR / "glm5_cpu_smoke.yaml")
        self.assertEqual(config.model.tp_size, 4)
        self.assertEqual(config.model.cp_size, 2)
        self.assertEqual(config.kv_tp_size, 2)
        self.assertEqual(config.workers_per_dp, 4)
        self.assertEqual(config.required_gpus, 4)
        self.assertEqual(config.bytes_per_block, config.bytes_per_gpu_block * 2)
        self.assertEqual(
            [config.worker_ranks(rank) for rank in range(config.workers_per_dp)],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
        )

    def test_composite_tp_must_be_divisible_by_cp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("""
model: {preset: glm5, tp_size: 3, cp_size: 2}
run: {rounds: 1}
""")
            with self.assertRaisesRegex(ValueError, "must be divisible"):
                load_config(path)


class WorkloadTests(unittest.TestCase):
    def test_multiturn_lengths_and_oracle(self):
        cfg = ConversationConfig(
            conversations_per_round=1,
            turns_min=2,
            turns_max=2,
            system_prompt_blocks=2,
            first_input_blocks=1,
            added_input_blocks=1,
            output_blocks=1,
        )
        conversation = WorkloadGenerator(cfg, 16, seed=7).generate_round(0)[0]
        first, second = conversation.turns
        self.assertEqual(first.added_input_tokens, 16)
        self.assertEqual(first.output_tokens, 16)
        self.assertEqual(second.expected_hit_tokens, len(first.put_tokens))
        oracle = PrefixOracle(16)
        self.assertEqual(oracle.match(first.get_tokens), 0)
        oracle.put(first.put_tokens)
        self.assertEqual(oracle.match(second.get_tokens), len(first.put_tokens))

    def test_shared_system_prompt_hits_across_conversations(self):
        cfg = ConversationConfig(
            conversations_per_round=2,
            turns_min=1,
            turns_max=1,
            system_prompt_blocks=2,
            first_input_blocks=1,
            output_blocks=1,
        )
        conversations = WorkloadGenerator(cfg, 8, seed=9).generate_round(0)
        oracle = PrefixOracle(8)
        oracle.put(conversations[0].turns[0].put_tokens)
        self.assertEqual(oracle.match(conversations[1].turns[0].get_tokens), 16)

    def test_oracle_drops_old_sequences_at_capacity(self):
        oracle = PrefixOracle(2, capacity_blocks=2)
        oracle.put([1, 2])
        oracle.put([3, 4])
        self.assertEqual(oracle.match([1, 2]), 2)
        oracle.put([5, 6])
        self.assertEqual(oracle.match([1, 2]), 0)
        self.assertEqual(oracle.match([5, 6]), 2)

    def test_slot_allocator_wraps_by_block(self):
        allocator = SlotAllocator(4, 8)
        blocks, slots = allocator.allocate(10)
        self.assertEqual(blocks.tolist(), [0, 1])
        self.assertEqual(slots.tolist(), list(range(10)))
        blocks, _ = allocator.allocate(24)
        self.assertEqual(blocks.tolist(), [2, 3, 0])


class MetricsAndDeviceTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.50), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 0.99), 4)

    def test_log_histogram_is_mergeable(self):
        left, right = LogHistogram(), LogHistogram()
        for value in (1, 2):
            left.record(value)
        for value in (3, 4):
            right.record(value)
        left.merge(right)
        self.assertEqual(left.count, 4)
        self.assertGreaterEqual(left.percentile(.99), 3.9)

    def test_decimal_gb_and_bandwidth_units(self):
        self.assertEqual(decimal_gb(1_000_000_000), 1)
        self.assertEqual(throughput_gb_s(2_000_000_000, 2), 1)

    def test_latency_reporter_writes_stable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_DIR / "glm5_cpu_smoke.yaml")
            config.output.directory = directory
            reporter = Reporter(config)
            result = TurnResult(
                0, 1, 0, 16, 16, 32, 12, 8, -4, 24,
                1, 2, 3, True, True, True, query_tokens=16,
            )
            reporter.write_latency_window(
                "unloaded", 0, __import__("datetime").datetime.now().astimezone(),
                1, [result], [], batch_size=1, concurrency=1,
            )
            output = reporter.directory
            reporter.close()
            for name in ("summary.csv", "metrics.csv", "summary.json"):
                self.assertTrue((output / name).exists(), name)
            self.assertFalse((output / "errors.csv").exists())
            with (output / "summary.json").open() as handle:
                summary = json.load(handle)
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertFalse(summary["run"]["performance_valid"])
            self.assertEqual(summary["scenarios"][0]["hit"]["ratio"], .5)
            self.assertEqual(summary["scenarios"][0]["hit"]["exact_rate"], 0)
            self.assertEqual(summary["scenarios"][0]["hit"]["token_accuracy"], .75)
            with (output / "summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["hit_ratio"]), summary["scenarios"][0]["hit"]["ratio"])
            self.assertNotIn("transfer_bytes", row)

    def test_bandwidth_reporter_uses_mode_specific_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_DIR / "glm5_cpu_smoke.yaml")
            config.run.mode = "bandwidth"
            config.output.directory = directory
            reporter = Reporter(config)
            reporter.write_bandwidth_window(
                "cpu_to_gpu_load", 2, 0,
                __import__("datetime").datetime.now().astimezone(), 2,
                [OperationResult("launch_get", True, 10, transfer_bytes=2_000_000_000)],
                1, 1, [],
            )
            output = reporter.directory
            reporter.close()
            with (output / "summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["throughput_gb_s"]), 1)
            self.assertNotIn("hit_ratio", row)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["scenarios"][0]["throughput_gb_s"], 1)

class RunnerLogicTests(unittest.TestCase):
    class FakeManager:
        class Status:
            name = "SUCCESS"

        class Response:
            def __init__(self):
                self.status = RunnerLogicTests.FakeManager.Status()

        def __init__(self):
            self.sequences = []
            self.tasks = {}
            self.next_task = 1

        def _match(self, tokens):
            best = 0
            for stored in self.sequences:
                common = 0
                for left, right in zip(tokens, stored):
                    if left != right:
                        break
                    common += 1
                best = max(best, common)
            return best

        def get_match(self, tokens, swa_aware=False):
            task = self.next_task
            self.next_task += 1
            hit = self._match(tokens)
            mask = np.zeros(len(tokens), dtype=np.int64)
            mask[:hit] = 1
            self.tasks[task] = ("get", tuple(tokens))
            return task, mask

        def put_match(self, tokens):
            task = self.next_task
            self.next_task += 1
            hit = self._match(tokens)
            mask = np.ones(len(tokens), dtype=np.int64)
            mask[:hit] = 0
            self.tasks[task] = ("put", tuple(tokens))
            return task, mask

        def launch(self, task_ids, slot_mappings, **kwargs):
            for task in task_ids:
                operation, tokens = self.tasks[task]
                if operation == "put":
                    self.sequences.append(tokens)
            return [10_000 + task_ids[0]] if kwargs.get("as_batch") else list(task_ids)

        def wait(self, task_ids, **kwargs):
            return {task: self.Response() for task in task_ids}

    def test_manager_driver_multiturn_flow(self):
        from benchmarks.flexkv_stress.runner import ManagerDriver

        config = load_config(CONFIG_DIR / "glm5.yaml")
        config.features.layerwise = False
        config.features.swa = False
        config.features.ssd = False
        config.features.concurrency = False
        config.model.gpu_blocks_per_rank = 256
        conversation_cfg = ConversationConfig(
            conversations_per_round=1,
            turns_min=2,
            turns_max=2,
            system_prompt_blocks=1,
            first_input_blocks=1,
            added_input_blocks=1,
            output_blocks=1,
        )
        turns = WorkloadGenerator(conversation_cfg, config.tokens_per_block, 11).generate_round(0)[0].turns
        driver = ManagerDriver(config, 0, self.FakeManager(), workers=[])
        fake_command = lambda *_args, **_kwargs: [{"ok": True}]
        with patch("benchmarks.flexkv_stress.runner.command_dp", fake_command):
            first, first_operations = driver.execute_batch([turns[0]], 0, set())
            second, second_operations = driver.execute_batch([turns[1]], 0, set())
        self.assertTrue(first[0].success)
        self.assertTrue(second[0].success)
        self.assertEqual(second[0].actual_hit_tokens, second[0].expected_hit_tokens)
        put_match = next(item for item in first_operations if item.operation == "put_match")
        launch_get = next(item for item in second_operations if item.operation == "launch_get")
        self.assertGreaterEqual(first[0].put_ms, put_match.latency_ms)
        self.assertGreaterEqual(second[0].get_ms, launch_get.latency_ms)

    def test_swa_mapping_uses_token_slots_and_rotates_pages(self):
        from benchmarks.flexkv_stress.runner import ManagerDriver

        config = load_config(CONFIG_DIR / "dsv4_pro.yaml")
        driver = ManagerDriver(config, 0, self.FakeManager(), workers=[])
        first_blocks, first_mapping = driver._swa_mapping()
        second_blocks, second_mapping = driver._swa_mapping()
        self.assertEqual(first_blocks, [0])
        self.assertEqual(second_blocks, [1])
        self.assertEqual(len(first_mapping), config.tokens_per_block)
        self.assertEqual(int(second_mapping[0] // config.tokens_per_block), 1)

    def test_swa_validation_uses_terminal_pattern_when_main_is_sampled(self):
        from benchmarks.flexkv_stress.cpu_stub import CpuTorchWorker

        config = load_config(CONFIG_DIR / "cpu_stub.yaml")
        config.features.swa = True
        config.cache.swa_num_slots = 2
        config.preset = replace(
            config.preset,
            swa_enabled=True,
            swa_num_layers=1,
            swa_head_size=2,
        )
        worker = CpuTorchWorker(config, dp_rank=0, effective_rank=0, device_id=0)
        worker.request({
            "operation": "seed",
            "block_ids": [0],
            "patterns": [11],
            "swa_block_ids": [0],
            "swa_patterns": [99],
        })
        response = worker.request({
            "operation": "verify",
            "block_ids": [0],
            "patterns": [11],
            "swa_block_ids": [0],
            "swa_patterns": [99],
            "max_layers": 1,
            "max_bytes": 8,
        })
        self.assertTrue(response["ok"], response["errors"])

    def test_cp_workers_hold_full_duplicate_kv_for_each_kv_tp_rank(self):
        from benchmarks.flexkv_stress.cpu_stub import start_cpu_workers
        from benchmarks.flexkv_stress.gpu_worker import command_dp

        config = load_config(CONFIG_DIR / "glm5_cpu_smoke.yaml")
        workers = start_cpu_workers(config)
        self.assertEqual(len(workers), 4)
        self.assertEqual(
            [(worker.kv_tp_rank, worker.cp_rank) for worker in workers],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
        )
        command_dp(workers, 0, {
            "operation": "seed", "block_ids": [0], "patterns": [1234],
        })
        for group_index in range(len(workers[0].tensors_per_group)):
            self.assertTrue(torch.equal(
                workers[0].tensors_per_group[group_index][0][0],
                workers[2].tensors_per_group[group_index][0][0],
            ))
            self.assertTrue(torch.equal(
                workers[1].tensors_per_group[group_index][0][0],
                workers[3].tensors_per_group[group_index][0][0],
            ))
            self.assertFalse(torch.equal(
                workers[0].tensors_per_group[group_index][0][0],
                workers[1].tensors_per_group[group_index][0][0],
            ))

    def test_cpu_stub_copies_real_torch_blocks(self):
        from benchmarks.flexkv_stress.cpu_stub import CpuStubManager, start_cpu_workers

        config = load_config(CONFIG_DIR / "cpu_stub.yaml")
        workers = start_cpu_workers(config)
        manager = CpuStubManager(config, 0, workers, config.cache.num_cpu_blocks // 2)
        tokens = np.arange(config.tokens_per_block * 2, dtype=np.int64)
        slots = np.arange(config.tokens_per_block * 2, dtype=np.int64)
        source = [worker for worker in workers if worker.dp_rank == 0]
        for worker in source:
            for group in worker.tensors_per_group:
                for tensor in group:
                    tensor[0].fill_(11 + worker.effective_rank)
                    tensor[1].fill_(22 + worker.effective_rank)
        put_task, put_mask = manager.put_match(tokens)
        manager.launch([put_task], [slots[put_mask]])
        for worker in source:
            for group in worker.tensors_per_group:
                for tensor in group:
                    tensor[0].zero_()
                    tensor[1].zero_()
        get_task, get_mask = manager.get_match(tokens)
        self.assertEqual(int(get_mask.sum()), len(tokens))
        manager.launch([get_task], [slots])
        for worker in source:
            for group in worker.tensors_per_group:
                for tensor in group:
                    self.assertTrue(bool(torch.all(tensor[0] == 11 + worker.effective_rank)))
                    self.assertTrue(bool(torch.all(tensor[1] == 22 + worker.effective_rank)))


if __name__ == "__main__":
    unittest.main()
