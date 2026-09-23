"""Representation, data-isolation and deployment-contract regression tests."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from starVLA.piper import geometry as g
from starVLA.piper import normalize
from starVLA.piper.adapter import MockBackend, Observation, PiperAdapter
from starVLA.piper.data_x import DEFAULT_TASKS, PiperXEEFDataSet, eef_states, prepare
from starVLA.piper.dataset import contract
from starVLA.piper.evaluate import physical_errors
from starVLA.piper.native_x import PiperXAdapter
from starVLA.piper.pretrained import select_backbone
from starVLA.piper.transfer import PiperXTransfer


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.joints = rng.normal(0, 0.15, (51, 14))
        self.joints[:, [6, 13]] = 0.04

    def test_s0_future_joints_and_absolute_grippers(self):
        state = self.joints[0]
        actions = g.encode_actions(state, self.joints[1:], "s0")
        for start in (0, 7):
            np.testing.assert_allclose(
                actions[:, start : start + 6] + state[start : start + 6], self.joints[1:, start : start + 6]
            )
        np.testing.assert_array_equal(actions[:, [6, 13]], self.joints[1:, [6, 13]])

    def test_s2_and_pair_share_exact_actions(self):
        s2, pair = (g.build_states(self.joints, rung) for rung in ("s2", "s2_pair"))
        np.testing.assert_array_equal(
            g.encode_actions(s2[0], s2[1:], "s2"), g.encode_actions(pair[0], pair[1:], "s2_pair")
        )
        relative = g.inverse(g.fk(self.joints[:, :6])) @ g.RIGHT_BASE @ g.fk(self.joints[:, 7:13])
        np.testing.assert_allclose(pair[:, 14:17], relative[:, :3, 3])
        np.testing.assert_allclose(pair[:, 17:], relative[:, :2, :3].reshape(-1, 6))

    def test_rotation_rows_and_roundtrip(self):
        for rung in ("s2", "s2_pair"):
            state = g.build_states(self.joints, rung)
            action = g.encode_actions(state[0], state[1:], rung)
            poses, grippers = g.decode_actions(state[0], action, rung)
            np.testing.assert_allclose(poses[:, 0], g.fk(self.joints[1:, :6]), atol=1e-10)
            np.testing.assert_allclose(poses[:, 1], g.fk(self.joints[1:, 7:13]), atol=1e-10)
            np.testing.assert_allclose(grippers, self.joints[1:, [6, 13]])

    def test_nominal_alignment_direction(self):
        data = np.zeros((3, 16))
        data[:, [6, 14]] = 1
        data[:, [7, 15]] = 0.07
        state = eef_states(data, alignment=PiperXTransfer.nominal_axes())
        expected = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_allclose(g.pose_from_vector(state[:, :6])[:, :3, :3], np.tile(expected, (3, 1, 1)), atol=1e-10)
        np.testing.assert_allclose(state[:, [6, 13]], 0.07)

    def test_measured_alignment_roundtrip(self):
        a = np.tile(np.eye(4), (2, 1, 1))
        c = a.copy()
        a[:, :3, 3] = [0.2, -0.1, 0.3]
        c[:, :3, 3] = [0, 0, 0.12]
        mapping = PiperXTransfer("measured_test", a, c)
        pose = g.fk(self.joints[:, :6])
        np.testing.assert_allclose(mapping.to_physical(mapping.to_policy(pose, 0), 0), pose, atol=1e-12)

    def test_normalization_constant_and_roundtrip(self):
        stats = normalize.NormStats(np.zeros(3), np.ones(3), np.array([0, -1, 2]), np.array([0, 1, 2]))
        values = np.array([[0, 0.4, 2], [0.5, -2, 3]])
        norm = normalize.transform(values, stats)
        np.testing.assert_allclose(norm[0], [0, 0.4, 0])
        np.testing.assert_allclose(normalize.transform(norm, stats, inverse=True), values)

    def test_deployment_owns_anchor_and_rejects_degenerate(self):
        backend = MockBackend()
        adapter = PiperAdapter("s2_pair", prompt="Fold.")
        request = adapter.prepare(backend.capture())
        actions = g.encode_actions(request.state, np.tile(request.state, (50, 1)), "s2_pair")
        backend.joints[:] = 1
        chunk = adapter.decode(request, actions)
        self.assertEqual(chunk.values.shape, (50, 14))
        actions[:, 3:9] = 0
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            adapter.decode(request, actions)
        adapter.reset(2)
        with self.assertRaisesRegex(ValueError, "another"):
            adapter.decode(request, actions)

    def test_native_x_uses_supplied_flange(self):
        backend = MockBackend()
        observation = backend.capture()
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, 0, 3] = [0.8, 0.9]
        observation = Observation(observation.joints, observation.images, time.monotonic(), flange_poses=poses)
        for rung in ("s2", "s2_pair"):
            state = PiperXAdapter(rung, prompt="Stack.").prepare(observation).state
            np.testing.assert_allclose(state[[0, 7]], [0.8, 0.9])

    def test_physical_units(self):
        state = g.build_states(self.joints, "s2")[0]
        target = g.encode_actions(state, np.tile(state, (50, 1)), "s2")
        prediction = target.copy()
        prediction[:, 0] += 0.001
        prediction[:, 9] += 0.002
        error = physical_errors(state, prediction, target, "s2")
        np.testing.assert_allclose(error[:, 0, 0], 1, atol=1e-10)
        np.testing.assert_allclose(error[:, 0, 2], 2)

    def test_variable_resolution_rgb_uses_shared_resize(self):
        from starVLA.piper.dataset import CAMERAS, prepare_images

        shapes = [(360, 640, 3), (720, 1280, 3), (480, 640, 3)]
        images = {key: np.zeros(shape, np.uint8) for key, shape in zip(CAMERAS, shapes, strict=True)}
        resized = prepare_images(images, (224, 224))
        self.assertEqual([image.size for image in resized], [(224, 224)] * 3)
        images[CAMERAS[0]] = np.zeros((360, 640, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            prepare_images(images, (224, 224))

    def test_strict_pretrain_mapping(self):
        tensor = np.zeros((2, 3))
        expected = {"model.model.visual.weight": tensor}
        selected = select_backbone(
            {"module.qwen_vl_interface.model.visual.weight": tensor, "action_model.weight": tensor}, expected
        )
        self.assertEqual(set(selected), set(expected))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            select_backbone({"action_model.weight": tensor}, expected)
        with self.assertRaisesRegex(ValueError, "shapes"):
            select_backbone({"qwen_vl_interface.model.visual.weight": np.zeros(4)}, expected)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.cache = self.root / "cache"
        self.name = DEFAULT_TASKS[0]
        source = self.source / self.name
        (source / "meta").mkdir(parents=True)
        (source / "data").mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 30,
            "chunks_size": 1000,
            "total_episodes": 4,
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}_{episode_index:06d}.mp4",
            "features": {"observation.qpos_ee": {"shape": [16]}, "observation.state": {"shape": [14]}},
        }
        for key in ("top_head", "hand_left", "hand_right"):
            info["features"]["observation.images." + key] = {"shape": [480, 640, 3]}
        (source / "meta/info.json").write_text(json.dumps(info))
        (source / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Test task."}) + "\n")
        held = int(np.random.default_rng(0).permutation(4)[0])
        episodes = []
        for index in range(4):
            length = 55 if index != 3 else 50
            episodes.append({"episode_index": index, "tasks": ["Test task."], "length": length})
            ee = np.zeros((length, 16))
            ee[:, [6, 14]] = 1
            ee[:, [7, 15]] = 0.04
            ee[:, 0] = 1000 if index == held else index * 0.01
            ee[:, 8] = 0.3
            ee[:, 1] = np.arange(length) * 0.0001
            joints = np.zeros((length, 14))
            joints[:, [6, 13]] = 0.04
            table = pa.table(
                {
                    "observation.qpos_ee": ee.tolist(),
                    "observation.state": joints.tolist(),
                    "timestamp": (np.arange(length) / 30).tolist(),
                    "task_index": [0] * length,
                }
            )
            pq.write_table(table, source / f"data/episode_{index:06d}.parquet")
            for key in ("top_head", "hand_left", "hand_right"):
                path = source / f"videos/observation.images.{key}_{index:06d}.mp4"
                path.parent.mkdir(exist_ok=True)
                path.touch()
        (source / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)) + "\n")
        self.manifest = prepare(
            self.source, self.cache, tasks=(self.name,), rungs=("s0", "s2", "s2_pair"), stats_samples=8
        )

    def test_split_tail_and_train_only_stats(self):
        task = self.manifest["tasks"][self.name]
        self.assertFalse(set(task["split"]["train"]) & set(task["split"]["held_out"]))
        self.assertEqual(task["excluded_episodes"], [{"episode_index": 3, "length": 50}])
        self.assertTrue(set(task["stats_episode_ids"]) <= set(task["split"]["train"]))
        self.assertLess(normalize.load(self.cache / "stats/s2")["state"].q99[0], 1)
        dataset = PiperXEEFDataSet(self.cache, rung="s2", images=False)
        for index in range(len(dataset)):
            name, episode, frame = dataset.locate(index)
            self.assertLessEqual(frame + 50, 54)
            self.assertEqual(dataset[index]["actions"].shape, (50, 20))

    def test_reuse_and_changed_settings_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare(self.source, self.cache, tasks=(self.name,), rungs=("s0", "s2", "s2_pair"), stats_samples=9)
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare(self.source, self.source / "bad")

    def test_native_contract_and_mismatch(self):
        metadata = contract(self.cache, "s2_pair")
        PiperXAdapter("s2_pair", prompt="Test.").validate_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "tool frame"):
            PiperAdapter("s2_pair", prompt="Test.").validate_metadata(metadata)
        metadata["pose_alignment"] = "wrong"
        with self.assertRaisesRegex(ValueError, "aligned"):
            PiperXAdapter("s2_pair", prompt="Test.").validate_metadata(metadata)

    def test_source_is_pinned(self):
        dataset = PiperXEEFDataSet(self.cache, rung="s2")
        with mock.patch.dict("os.environ", {"PIPER_X_SOURCE_ROOT": "/not/the/snapshot"}):
            with mock.patch("starVLA.piper.data_x.read_image", return_value=np.zeros((480, 640, 3), np.uint8)) as reader:
                dataset[0]
        self.assertTrue(all(str(call.args[0]).startswith(str(self.source)) for call in reader.call_args_list))

    def test_aligned_cache_and_double_conversion_guard(self):
        aligned = self.root / "aligned"
        prepare(
            self.source,
            aligned,
            tasks=(self.name,),
            rungs=("s2", "s2_pair"),
            alignment=PiperXTransfer.nominal_axes(),
            stats_samples=8,
        )
        metadata = contract(aligned, "s2_pair")
        PiperAdapter("s2_pair", prompt="Test.").validate_metadata(metadata)
        with self.assertRaises(ValueError):
            PiperXAdapter("s2_pair", prompt="Test.").validate_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "native"):
            prepare(
                self.source,
                self.root / "bad_s0",
                tasks=(self.name,),
                rungs=("s0",),
                alignment=PiperXTransfer.nominal_axes(),
            )


class MixtureTests(unittest.TestCase):
    def setUp(self):
        import shutil

        CacheTests.setUp(self)
        self.source2 = self.root / "source2"
        shutil.copytree(self.source, self.source2)
        self.cache2 = self.root / "cache2"
        prepare(self.source2, self.cache2, tasks=(self.name,), rungs=("s0", "s2", "s2_pair"), stats_samples=8)
        self.config = self.root / "mix.json"
        self.entries = [
            {"name": "first", "cache": str(self.cache), "weight": 1},
            {"name": "second", "cache": str(self.cache2), "weight": 3},
        ]
        self.config.write_text(json.dumps({"datasets": self.entries}))

    def test_weights_coverage_and_train_only_statistics(self):
        from starVLA.piper.dataset import raw_dataset
        from starVLA.piper.mixture import MixtureDataset
        from starVLA.piper.mixture import prepare as prepare_mix

        output = self.root / "mixed"
        manifest = prepare_mix(self.entries, output, data_mix="test_mix", stats_samples=16)
        self.assertAlmostEqual(manifest["effective_probabilities"]["second"], 0.75)
        dataset = raw_dataset(output, "s2", images=False)
        owners = [dataset.locate(i) for i in range(len(dataset))]
        for member, child in enumerate(dataset.datasets):
            self.assertEqual({anchor for owner, anchor in owners if owner == member}, set(range(len(child))))
        held = MixtureDataset(output, "s2", split="held_out", images=False)
        self.assertEqual(len(held), sum(held.counts))
        self.assertLess(normalize.load(output / "stats/s2")["state"].q99[0], 1)
        s2 = normalize.load(output / "stats/s2")["actions"]
        pair = normalize.load(output / "stats/s2_pair")["actions"]
        np.testing.assert_array_equal(s2.q01, pair.q01)
        np.testing.assert_array_equal(s2.q99, pair.q99)
        PiperXAdapter("s2", prompt="Test.").validate_metadata(contract(output, "s2"))

    def test_changed_cache_and_settings_fail(self):
        from starVLA.piper.mixture import MixtureDataset
        from starVLA.piper.mixture import prepare as prepare_mix

        output = self.root / "mixed"
        prepare_mix(self.entries, output, data_mix="test_mix", stats_samples=16)
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_mix(self.entries, output, data_mix="test_mix", stats_samples=32)
        manifest_path = self.cache2 / "manifest.json"
        manifest_path.write_text(manifest_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            MixtureDataset(output, "s2", images=False)

    def test_overlap_bad_weights_and_frames_fail(self):
        from starVLA.piper.mixture import prepare as prepare_mix

        self.entries[1]["cache"] = str(self.cache)
        self.config.write_text(json.dumps({"datasets": self.entries}))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            prepare_mix(self.entries, self.root / "overlap", data_mix="test_mix", stats_samples=8)
        self.entries[1]["weight"] = 0
        self.config.write_text(json.dumps({"datasets": self.entries}))
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_mix(self.entries, self.root / "zero", data_mix="test_mix", stats_samples=8)
        aligned = self.root / "aligned2"
        prepare(
            self.source2,
            aligned,
            tasks=(self.name,),
            rungs=("s2", "s2_pair"),
            alignment=PiperXTransfer.nominal_axes(),
            stats_samples=8,
        )
        self.entries[1].update(cache=str(aligned), weight=3)
        self.config.write_text(json.dumps({"datasets": self.entries}))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            prepare_mix(self.entries, self.root / "frame", data_mix="test_mix", stats_samples=8)

    def test_registered_loader_uses_weights_and_robot_transform(self):
        from omegaconf import OmegaConf

        from starVLA.dataloader.piper_datasets import DATASET_NAMED_MIXTURES, get_vla_dataset
        from starVLA.piper.mixture import prepare as prepare_mix

        output = self.root / "registered"
        entries = [{"name": self.name, "cache": str(self.cache), "weight": 1.0}]
        prepare_mix(entries, output, data_mix="test_registered", stats_samples=8)
        cfg = OmegaConf.create({"data_mix": "test_registered", "cache_dir": str(output), "representation": "s2"})
        with mock.patch.dict(DATASET_NAMED_MIXTURES, {"test_registered": [(self.name, 1.0, "piper_x")]}):
            self.assertGreater(len(get_vla_dataset(cfg)), 0)
            DATASET_NAMED_MIXTURES["test_registered"] = [(self.name, 2.0, "piper_x")]
            with self.assertRaisesRegex(ValueError, "weights differ"):
                get_vla_dataset(cfg)
            DATASET_NAMED_MIXTURES["test_registered"] = [(self.name, 1.0, "piper_x_aligned")]
            with self.assertRaisesRegex(ValueError, "transforms differ"):
                get_vla_dataset(cfg)

    def test_registry_has_independent_default_task_weights(self):
        from starVLA.dataloader.piper_datasets import resolve_mixture

        entries = resolve_mixture("piper_all_aligned")
        self.assertEqual(len(entries), 8)
        self.assertEqual(len({name for name, _, _ in entries}), 8)
        self.assertTrue(all(weight == 1.0 for _, weight, _ in entries))

    def test_size_weighted_sampling_matches_existing_codebase(self):
        from starVLA.piper.mixture import MixtureDataset
        from starVLA.piper.mixture import prepare as prepare_mix

        manifest_path = self.cache2 / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tasks"][self.name]["split"]["train"] = [0]
        manifest_path.write_text(json.dumps(manifest))
        output = self.root / "size_weighted"
        self.entries[0]["weight"] = self.entries[1]["weight"] = 1.0
        prepare_mix(self.entries, output, data_mix="size_test", balance_dataset_weights=True, stats_samples=16)
        dataset = MixtureDataset(output, "s2", images=False)
        np.testing.assert_allclose(dataset.slots / len(dataset), dataset.counts / dataset.counts.sum())

    def test_cross_split_leakage_fails(self):
        from starVLA.piper.mixture import prepare as prepare_mix

        path = self.cache2 / "manifest.json"
        manifest = json.loads(path.read_text())
        task = manifest["tasks"][self.name]
        task["source"] = str(self.source / self.name)
        task["split"]["train"], task["split"]["held_out"] = task["split"]["held_out"], task["split"]["train"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "leak"):
            prepare_mix(self.entries, self.root / "leak", data_mix="test_mix", stats_samples=8)


if __name__ == "__main__":
    unittest.main()
