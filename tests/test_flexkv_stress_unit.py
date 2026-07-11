from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.flexkv_stress.config import ConversationConfig, load_config
from benchmarks.flexkv_stress.device import DeviceBackend
from benchmarks.flexkv_stress.main import _should_stop
from benchmarks.flexkv_stress.metrics import OperationResult, Reporter, TurnResult, percentile
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

    def test_reporter_writes_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = Reporter(directory)
            result = TurnResult(0, 1, 0, 16, 16, 32, 0, 0, 0, 32, 1, 2, 3, True, True, True)
            reporter.write_round(0, __import__("time").perf_counter(), [result], [OperationResult("put", True, 1)])
            output = reporter.directory
            reporter.close()
            for name in (
                "rounds.csv", "windows.csv", "turns.csv", "operations.csv",
                "validation_samples.csv", "resources.csv", "errors.csv", "summary.csv",
            ):
                self.assertTrue((output / name).exists(), name)
            with (output / "summary.csv").open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["turns"], "1")

    def test_backend_detection(self):
        fake_rocm = type("Torch", (), {"version": type("Version", (), {"hip": "6.0"})()})()
        fake_cuda = type("Torch", (), {"version": type("Version", (), {"hip": None})()})()
        self.assertEqual(DeviceBackend.detect(fake_rocm).name, "rocm")
        self.assertEqual(DeviceBackend.detect(fake_cuda).name, "cuda")

    def test_memory_handle_gates_direct_cuda_ipc(self):
        source = (ROOT / "flexkv" / "common" / "memory_handle.py").read_text()
        self.assertIn("IS_ROCM = getattr(torch.version, \"hip\", None) is not None", source)
        self.assertIn("Direct CUDA IPC is not available on ROCm", source)


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
            first, _ = driver.execute_batch([turns[0]], 0, set())
            second, _ = driver.execute_batch([turns[1]], 0, set())
        self.assertTrue(first[0].success)
        self.assertTrue(second[0].success)
        self.assertEqual(second[0].actual_hit_tokens, second[0].expected_hit_tokens)

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
