import json
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as cli
import utils as helpers
from efficient_ir import EfficientIR
from utils import FEATURE_BATCH_SIZE, NOTEXISTS, ProcessCancelled, Utils, _extract_fv

MODEL_INIT = EfficientIR.init_model
REAL_POOL = multiprocessing.Pool


class BlockingExtractor:
    def get_fv(self, path):
        Path(path + ".started").touch()
        time.sleep(30)
        return np.ones(1000, dtype=np.float32)


def init_blocking_worker(img_size, model_path, stop_flag_path, num_threads):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    helpers._WORKER_ENGINE = BlockingExtractor()
    helpers._WORKER_STOP_FLAG = stop_flag_path


class ReadyResults:
    def __init__(self, results):
        self.results = iter(results)

    def next(self, timeout):
        return next(self.results)


class FakePool:
    def __init__(self, engine, stop_flag_path=None):
        self.engine = engine
        self.stop_flag_path = stop_flag_path
        self.batches = []
        self.terminated = False
        self.closed = False

    def imap_unordered(self, fn, batch):
        self.batches.append(list(batch))
        return ReadyResults(
            [
                (task[0], None, None)
                if self.stop_flag_path and os.path.exists(self.stop_flag_path)
                else _extract_fv(self.engine, task)
                for task in reversed(batch)
            ]
        )

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed = True

    def join(self):
        pass


class IndexUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.config = {
            "img_size": 260,
            "index_capacity": 8,
            "model_path": str(self.root / "unused.onnx"),
            "index_path": str(self.root / "index.bin"),
            "combined_index_path": str(self.root / "combined.json"),
            "stop_flag_path": str(self.root / "process.stop"),
            "search_dir": [str(self.images)],
        }
        model_patch = patch.object(EfficientIR, "init_model")
        model_patch.start()
        self.addCleanup(model_patch.stop)
        self.u = Utils(self.config)
        self.u.ir_engine.get_fv = Mock(return_value=np.ones(1000, dtype=np.float32))
        pool_patch = patch.object(
            helpers.multiprocessing,
            "Pool",
            side_effect=lambda **kwargs: FakePool(
                self.u.ir_engine, self.u.stop_flag_path
            ),
        )
        self.create_pool = pool_patch.start()
        self.addCleanup(pool_patch.stop)

    def image(self, name="photo.jpg"):
        path = self.images / name
        Image.new("RGB", (16, 16), "red").save(path)
        return path

    def records(self):
        return json.loads(Path(self.u.combined_index_path).read_text(encoding="utf-8"))

    def update(self, check_meta=False, max_process=1):
        with patch.object(cli, "utils", self.u):
            cli.update_index([str(self.images)], check_meta, max_process)

    def reload(self):
        return Utils(self.config)

    def test_static_suffixes_and_no_gif_or_video(self):
        names = ["a.TIF", "b.jfif", "c.jpe", "d.avif", "e.pgm", "f.gif", "g.mp4"]
        for name in names:
            (self.images / name).touch()
        found = {Path(p).name for p in self.u.get_file_list(str(self.images))}
        self.assertEqual(found, set(names[:5]))

    def test_animated_png_is_not_extracted(self):
        path = self.images / "animated.png"
        Image.new("RGB", (16, 16), "red").save(
            path, save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")]
        )
        self.assertIsNone(self.u.ir_engine.img_preprocess(path))

    def test_default_update_still_skips_successful_existing_files(self):
        path = self.image()
        self.update()
        old = self.records()
        path.write_bytes(b"changed")
        self.u.ir_engine.get_fv.reset_mock()
        self.update()
        self.u.ir_engine.get_fv.assert_not_called()
        self.assertEqual(self.records(), old)

    def test_no_changes_write_neither_index_nor_metadata(self):
        self.image()
        self.update()
        self.u = self.reload()
        with (
            patch.object(self.u.ir_engine, "save_index") as save_index,
            patch.object(self.u, "save_meta_files") as save_meta,
        ):
            self.update(check_meta=True)
        save_index.assert_not_called()
        save_meta.assert_not_called()

    def test_only_metadata_changes_do_not_rewrite_binary(self):
        self.image()
        self.update()
        binary = Path(self.config["index_path"]).read_bytes()
        self.image("failed.jpg")
        self.u.ir_engine.get_fv.return_value = None
        with patch.object(self.u.ir_engine, "save_index") as save_index:
            self.update()
        save_index.assert_not_called()
        self.assertEqual(Path(self.config["index_path"]).read_bytes(), binary)
        self.assertEqual(len(self.records()), 2)
        self.assertIsNone(self.records()[1]["size"])

    def test_failed_retry_with_unchanged_metadata_writes_nothing(self):
        self.image()
        self.u.ir_engine.get_fv.return_value = None
        self.update()
        with (
            patch.object(self.u.ir_engine, "save_index") as save_index,
            patch.object(self.u, "save_meta_files") as save_meta,
        ):
            self.update()
        save_index.assert_not_called()
        save_meta.assert_not_called()

    def test_empty_initial_index_is_saved_once(self):
        self.update()
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 0)
        with (
            patch.object(self.u.ir_engine, "save_index") as save_index,
            patch.object(self.u, "save_meta_files") as save_meta,
        ):
            self.update()
        save_index.assert_not_called()
        save_meta.assert_not_called()

    def test_empty_rebuild_persists_reset_index(self):
        self.image()
        self.update()
        with patch.object(cli, "utils", self.u):
            cli.rebuild_index({**self.config, "search_dir": []}, max_process=1)
        self.assertEqual(self.records(), [])
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 0)

    def test_deletion_is_persisted_without_new_features(self):
        path = self.image()
        self.update()
        path.unlink()
        with patch.object(
            self.u.ir_engine, "save_index", wraps=self.u.ir_engine.save_index
        ) as save_index:
            self.update()
        save_index.assert_called_once()
        self.assertEqual(
            self.records(), [{"path": NOTEXISTS, "size": None, "mtime": None}]
        )
        loaded = self.reload()
        with self.assertRaises(RuntimeError):
            loaded.ir_engine.hnsw_index.get_items([0])

    def test_legacy_json_only_tombstone_is_repaired(self):
        self.image()
        self.update()
        Path(self.u.combined_index_path).write_text(
            json.dumps([{"path": NOTEXISTS, "size": None, "mtime": None}])
        )
        self.images.joinpath("photo.jpg").unlink()
        self.u = self.reload()
        self.update()
        with self.assertRaises(RuntimeError):
            self.reload().ir_engine.hnsw_index.get_items([0])

    def test_search_after_deletion_uses_live_count(self):
        for i in range(3):
            self.image(f"{i}.jpg")
        self.update()
        self.images.joinpath("1.jpg").unlink()
        self.images.joinpath("2.jpg").unlink()
        self.update()
        self.u = self.reload()
        self.u.ir_engine.get_fv = Mock(return_value=np.ones(1000, dtype=np.float32))
        paths = self.u.get_exists_index()
        self.assertEqual(len(self.u.checkout(str(self.images / "0.jpg"), paths)), 1)
        self.assertEqual(list(self.u.get_duplicate(paths, 98.5, False)), [])
        self.images.joinpath("0.jpg").unlink()
        self.update()
        self.assertEqual(self.u.checkout("unused.jpg", self.u.get_exists_index()), [])

    def test_failed_new_file_retries_after_reload(self):
        self.image()
        self.u.ir_engine.get_fv.return_value = None
        self.update()
        self.assertIsNone(self.records()[0]["size"])
        self.u = self.reload()
        self.u.ir_engine.get_fv = Mock(return_value=np.ones(1000, dtype=np.float32))
        self.update()
        self.assertIsNotNone(self.records()[0]["size"])
        self.assertEqual(self.u.ir_engine.hnsw_index.get_ids_list(), [0])

    def test_failed_changed_file_retries_without_check_meta(self):
        path = self.image()
        self.update()
        path.write_bytes(b"changed")
        self.u.ir_engine.get_fv.return_value = None
        self.update(check_meta=True)
        self.assertIsNone(self.records()[0]["mtime"])
        self.u.ir_engine.get_fv.return_value = np.zeros(1000, dtype=np.float32)
        self.update()
        np.testing.assert_array_equal(
            self.reload().ir_engine.hnsw_index.get_items([0])[0], 0
        )
        self.assertEqual(self.records()[0]["size"], len(b"changed"))

    def test_failed_file_can_be_deleted_and_recreated(self):
        path = self.image()
        self.u.ir_engine.get_fv.return_value = None
        self.update()
        path.unlink()
        self.update()
        self.assertEqual(self.records()[0]["path"], NOTEXISTS)
        self.image()
        self.u.ir_engine.get_fv.return_value = np.ones(1000, dtype=np.float32)
        self.update()
        self.assertEqual(self.records()[1]["path"], str(path))
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_ids_list(), [1])

    def test_file_changing_during_extraction_stays_pending(self):
        path = self.image()

        def change_file(_):
            path.write_bytes(b"changed while reading")
            return np.ones(1000, dtype=np.float32)

        self.u.ir_engine.get_fv.side_effect = change_file
        self.update()
        self.assertIsNone(self.records()[0]["size"])
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 0)

    def test_nonfinite_features_stay_pending(self):
        self.image()
        self.u.ir_engine.get_fv.return_value = np.full(1000, np.nan)
        self.update()
        self.assertIsNone(self.records()[0]["size"])
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 0)

    def test_rebuild_recomputes_existing_files_and_current_directories(self):
        self.image()
        self.update()
        old_dir = self.root / "old-images"
        old_dir.mkdir()
        old = old_dir / "old.jpg"
        Image.new("RGB", (16, 16)).save(old)
        with patch.object(cli, "utils", self.u):
            cli.update_index([str(old_dir)], max_process=1)
            self.u.ir_engine.get_fv.reset_mock()
            self.u.ir_engine.get_fv.return_value = np.zeros(1000, dtype=np.float32)
            cli.rebuild_index(self.config, max_process=1)
        self.u.ir_engine.get_fv.assert_called_once()
        self.assertEqual(len(self.records()), 1)
        loaded = self.reload().ir_engine.hnsw_index
        self.assertEqual(loaded.get_current_count(), 1)
        np.testing.assert_array_equal(loaded.get_items([0])[0], 0)

    def test_cancel_before_rebuild_leaves_disk_files_unchanged(self):
        self.image()
        self.update()
        binary = Path(self.config["index_path"]).read_bytes()
        meta = Path(self.config["combined_index_path"]).read_bytes()
        Path(self.u.stop_flag_path).touch()
        with patch.object(cli, "utils", self.u), self.assertRaises(ProcessCancelled):
            cli.rebuild_index(self.config, max_process=1)
        self.assertEqual(Path(self.config["index_path"]).read_bytes(), binary)
        self.assertEqual(Path(self.config["combined_index_path"]).read_bytes(), meta)

    def test_sequential_cancel_saves_received_features_and_resumes(self):
        for i in range(4):
            self.image(f"{i}.jpg")
        calls = 0

        def cancel_after_two(_):
            nonlocal calls
            calls += 1
            if calls == 2:
                Path(self.u.stop_flag_path).touch()
            return np.ones(1000, dtype=np.float32)

        self.u.ir_engine.get_fv.side_effect = cancel_after_two
        with (
            patch.object(
                helpers.multiprocessing, "Pool", side_effect=OSError("no pool")
            ),
            self.assertRaises(ProcessCancelled),
        ):
            self.update()
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 2)
        self.assertEqual(sum(r["size"] is not None for r in self.records()), 2)
        Path(self.u.stop_flag_path).unlink()
        self.u = self.reload()
        self.u.ir_engine.get_fv = Mock(return_value=np.ones(1000, dtype=np.float32))
        self.update()
        self.assertEqual(self.u.ir_engine.get_fv.call_count, 2)
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 4)

    def test_one_pending_file_uses_single_worker_pool(self):
        self.image()
        self.update(max_process=4)
        self.assertEqual(self.create_pool.call_args.kwargs["processes"], 1)

    def test_single_worker_cancels_blocked_extraction(self):
        path = self.image()
        started = Path(str(path) + ".started")
        finished = threading.Event()

        def request_cancel_when_started():
            deadline = time.monotonic() + 10
            while not started.exists() and time.monotonic() < deadline:
                if finished.wait(0.02):
                    return
            if not finished.is_set():
                Path(self.u.stop_flag_path).touch()

        requester = threading.Thread(target=request_cancel_when_started, daemon=True)
        requester.start()
        start = time.monotonic()
        try:
            with (
                patch.object(helpers.multiprocessing, "Pool", side_effect=REAL_POOL),
                patch.object(helpers, "_init_worker", init_blocking_worker),
                self.assertRaises(ProcessCancelled),
            ):
                self.update(max_process=1)
        finally:
            finished.set()
            requester.join(timeout=2)
        self.assertTrue(started.exists())
        self.assertLess(time.monotonic() - start, 20)
        self.u.ir_engine.get_fv.assert_not_called()
        self.assertIsNone(self.records()[0]["size"])
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 0)

    def test_parallel_tasks_and_insertions_are_bounded(self):
        for i in range(FEATURE_BATCH_SIZE * 2 + 3):
            self.image(f"{i}.jpg")
        pool = FakePool(self.u.ir_engine)
        inserted_sizes = []
        original_add = self.u.ir_engine.add_fv

        def record_add(fvs, ids):
            inserted_sizes.append(len(ids))
            original_add(fvs, ids)

        with (
            patch.object(helpers.os, "cpu_count", return_value=8),
            patch.object(helpers.multiprocessing, "Pool", return_value=pool) as create,
            patch.object(self.u.ir_engine, "add_fv", side_effect=record_add),
        ):
            self.update(max_process=4)
        self.assertEqual([len(b) for b in pool.batches], [128, 128, 3])
        self.assertEqual(inserted_sizes, [128, 128, 3])
        self.assertEqual(create.call_args.kwargs["initargs"][-1], 2)
        self.assertTrue(pool.closed)
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 259)

    def test_parallel_cancel_drains_ready_results_without_waiting(self):
        for i in range(4):
            self.image(f"{i}.jpg")
        pool = FakePool(self.u.ir_engine)
        results = [
            _extract_fv(self.u.ir_engine, (i, str(self.images / f"{i}.jpg")))
            for i in range(2)
        ]
        timeouts = []

        class PartialResults:
            def next(inner, timeout):
                timeouts.append(timeout)
                if results:
                    Path(self.u.stop_flag_path).touch()
                    return results.pop(0)
                raise multiprocessing.TimeoutError

        pool.imap_unordered = Mock(return_value=PartialResults())
        with (
            patch.object(helpers.os, "cpu_count", return_value=4),
            patch.object(helpers.multiprocessing, "Pool", return_value=pool),
            self.assertRaises(ProcessCancelled),
        ):
            self.update(max_process=2)
        self.assertEqual(timeouts, [0.1, 0, 0])
        self.assertTrue(pool.terminated)
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 2)

    def test_cancelled_rebuild_saves_new_progress_and_resumes(self):
        for i in range(3):
            self.image(f"{i}.jpg")
        self.update()

        def cancel(_):
            Path(self.u.stop_flag_path).touch()
            return np.zeros(1000, dtype=np.float32)

        self.u.ir_engine.get_fv.side_effect = cancel
        with patch.object(cli, "utils", self.u), self.assertRaises(ProcessCancelled):
            cli.rebuild_index(self.config, max_process=1)
        loaded = self.reload().ir_engine.hnsw_index
        self.assertEqual(loaded.get_current_count(), 1)
        np.testing.assert_array_equal(loaded.get_items(loaded.get_ids_list())[0], 0)
        Path(self.u.stop_flag_path).unlink()
        self.u = self.reload()
        self.u.ir_engine.get_fv = Mock(return_value=np.zeros(1000, dtype=np.float32))
        self.update()
        self.assertEqual(self.u.ir_engine.get_fv.call_count, 2)
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 3)

    def test_pool_failure_falls_back_only_for_unreceived_tasks(self):
        for i in range(3):
            self.image(f"{i}.jpg")
        pool = FakePool(self.u.ir_engine)
        first = _extract_fv(self.u.ir_engine, (0, str(self.images / "0.jpg")))
        self.u.ir_engine.get_fv.reset_mock()

        class FailingResults:
            called = False

            def next(inner, timeout):
                if not inner.called:
                    inner.called = True
                    return first
                raise RuntimeError("broken pool")

        pool.imap_unordered = Mock(return_value=FailingResults())
        with (
            patch.object(helpers.os, "cpu_count", return_value=4),
            patch.object(helpers.multiprocessing, "Pool", return_value=pool),
        ):
            self.update(max_process=2)
        self.assertTrue(pool.terminated)
        self.assertEqual(self.u.ir_engine.get_fv.call_count, 2)
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 3)

    def test_ctrl_c_during_checkpoint_is_deferred(self):
        self.image()
        save = self.u.ir_engine.save_index

        def interrupt_save():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            save()

        with patch.object(self.u.ir_engine, "save_index", side_effect=interrupt_save):
            with self.assertRaises(ProcessCancelled):
                self.update()
        self.assertEqual(self.reload().ir_engine.hnsw_index.get_current_count(), 1)
        self.assertIsNotNone(self.records()[0]["size"])

    def test_metadata_write_failure_is_not_reported_as_success(self):
        self.image()
        with patch.object(self.u, "save_meta_files", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.update()

    def test_metadata_write_failure_preserves_previous_binary(self):
        path = self.image()
        self.update()
        binary = Path(self.config["index_path"]).read_bytes()
        metadata = Path(self.u.combined_index_path).read_bytes()
        path.write_bytes(b"changed")
        with (
            patch.object(self.u, "save_meta_files", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            self.update(check_meta=True)
        self.assertEqual(Path(self.config["index_path"]).read_bytes(), binary)
        self.assertEqual(Path(self.u.combined_index_path).read_bytes(), metadata)

    def test_model_session_uses_explicit_thread_budget(self):
        engine = EfficientIR.__new__(EfficientIR)
        engine.model_path = self.config["model_path"]
        with patch("efficient_ir.onnxruntime.InferenceSession"):
            MODEL_INIT(engine, num_threads=3)
        self.assertEqual(engine.session_opti.intra_op_num_threads, 3)
        self.assertEqual(engine.session_opti.inter_op_num_threads, 1)

    def test_stop_path_environment_matches_cli_and_workers(self):
        override = str(self.root / "override.stop")
        with patch.dict(os.environ, EFFICIENTIR_STOP_FLAG_PATH=override):
            self.assertEqual(cli.get_stop_flag_path(self.config), override)
            self.assertEqual(Utils(self.config).stop_flag_path, override)

    def test_cancel_request_does_not_clear_an_existing_flag(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        with patch.object(cli, "clear_cancel_flag") as clear:
            cli.main(["--config_path", str(config_path), "--cancel_process"])
        clear.assert_not_called()
        self.assertTrue(Path(self.u.stop_flag_path).exists())

    def test_gui_early_cancel_is_not_cleared_on_startup(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        Path(self.u.stop_flag_path).touch()
        with (
            patch.object(cli, "utils", self.u),
            patch.dict(os.environ, EFFICIENTIR_CANCEL_PREPARED="1"),
            self.assertRaises(ProcessCancelled),
        ):
            cli.main(["--config_path", str(config_path), "--update_index"])


if __name__ == "__main__":
    unittest.main()
