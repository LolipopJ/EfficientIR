import json
import multiprocessing
import os
import signal
import sys

import numpy as np
from tqdm import tqdm

from efficient_ir import EfficientIR, FeatureExtractor

NOTEXISTS = "NOTEXISTS"

current_file_path = os.path.dirname(os.path.abspath(__file__))
STOP_FLAG_FILENAME = "process.stop"
STOP_FLAG_PATH = os.path.join(current_file_path, STOP_FLAG_FILENAME)
FEATURE_BATCH_SIZE = 128
STATIC_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".tif",
    ".tiff",
    ".bmp",
    ".dib",
    ".webp",
    ".avif",
    ".tga",
    ".ico",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
)


class ProcessCancelled(Exception):
    """Cancellation acknowledged after completed work has been saved."""


# Global feature extractor used inside worker processes.
# Each worker loads only the ONNX model, not the HNSW index.
_WORKER_ENGINE = None
_WORKER_STOP_FLAG = None


def _init_worker(img_size, model_path, stop_flag_path=None, num_threads=1):
    """Initializer for worker processes. Creates a module-global
    FeatureExtractor instance (ONNX model only) to avoid reloading the model
    on every task call inside the same worker.
    """
    global _WORKER_ENGINE, _WORKER_STOP_FLAG
    # The parent handles Ctrl+C and saves progress before stopping workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_STOP_FLAG = stop_flag_path
    try:
        _WORKER_ENGINE = FeatureExtractor(img_size, model_path, num_threads)
    except Exception as exc:
        print(f"Failed to initialize feature worker: {exc}", file=sys.stderr)
        _WORKER_ENGINE = None


def _extract_fv(engine, task):
    """Return metadata for the file actually read, only if it stayed stable."""
    idx, fpath = task
    try:
        before = os.stat(fpath)
        fv = engine.get_fv(fpath)
        after = os.stat(fpath)
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if fv is None or before_signature != after_signature:
            return idx, None, None
        if np.shape(fv) != (1000,) or not np.isfinite(fv).all():
            return idx, None, None
        return idx, fv, [after.st_size, after.st_mtime]
    except Exception as exc:
        print(f"Failed to extract {fpath}: {exc}", file=sys.stderr)
        return idx, None, None


def _worker_get_fv(task):
    if _WORKER_ENGINE is None or (
        _WORKER_STOP_FLAG and os.path.exists(_WORKER_STOP_FLAG)
    ):
        return task[0], None, None
    return _extract_fv(_WORKER_ENGINE, task)


class Utils:
    def __init__(self, config):
        self.exists_index_path = self.get_absolute_path(
            config.get("exists_index_path", "index/name_index.json")
        )
        self.metainfo_path = self.get_absolute_path(
            config.get("metainfo_path", "index/metainfo.json")
        )
        self.combined_index_path = self.get_absolute_path(
            config.get("combined_index_path", "index/combined_index.json")
        )
        self.ir_engine = EfficientIR(
            config["img_size"],
            config["index_capacity"],
            self.get_absolute_path(config.get("index_path", "index/index.bin")),
            self.get_absolute_path(
                config.get("model_path", "models/imagenet-b2-opti.onnx")
            ),
        )
        # Save worker init args: only img_size and model_path are needed
        # since workers use FeatureExtractor (no HNSW index loaded).
        self._worker_init_args = (
            config["img_size"],
            self.get_absolute_path(
                config.get("model_path", "models/imagenet-b2-opti.onnx")
            ),
        )
        # Stop flag path used to request cancellation across processes.
        self.stop_flag_path = self.get_absolute_path(
            os.environ.get("EFFICIENTIR_STOP_FLAG_PATH")
            or config.get("stop_flag_path", STOP_FLAG_PATH)
        )
        self.check_env()

    def check_env(self):
        if not os.path.exists(self.combined_index_path):
            parent_path = os.path.join(self.combined_index_path, os.pardir)
            os.makedirs(os.path.abspath(parent_path), exist_ok=True)
            # 自动迁移旧版两个独立文件（name_index.json + metainfo.json）
            tmp_path = self.combined_index_path + ".tmp"
            if os.path.exists(self.exists_index_path):
                with open(self.exists_index_path, "rb") as f:
                    exists_list = json.loads(f.read())
                meta_list = []
                if os.path.exists(self.metainfo_path):
                    with open(self.metainfo_path, "rb") as f:
                        meta_list = json.loads(f.read())
                combined = []
                for i, path in enumerate(exists_list):
                    size = meta_list[i][0] if i < len(meta_list) else None
                    mtime = meta_list[i][1] if i < len(meta_list) else None
                    combined.append({"path": path, "size": size, "mtime": mtime})
                with open(tmp_path, "w", encoding="UTF-8") as wp:
                    wp.write(self.dumps(combined))
                    os.remove(self.exists_index_path)
                    if os.path.exists(self.metainfo_path):
                        os.remove(self.metainfo_path)
            else:
                with open(tmp_path, "w", encoding="UTF-8") as wp:
                    wp.write("[]")
            os.replace(tmp_path, self.combined_index_path)

    def get_exists_index(self):
        with open(self.combined_index_path, "rb") as f:
            combined = json.loads(f.read())
        return [entry["path"] for entry in combined]

    def cancel_requested(self):
        return os.path.exists(self.stop_flag_path)

    def check_cancelled(self):
        if self.cancel_requested():
            raise ProcessCancelled

    def get_file_list(self, target_dir):
        file_path_list = []
        for root, dirs, files in os.walk(target_dir):
            self.check_cancelled()
            # Show progress per-directory when scanning files
            for name in tqdm(
                files, ascii=True, desc=f"Scanning files in {root}", unit="file"
            ):
                self.check_cancelled()
                if name.lower().endswith(STATIC_IMAGE_EXTENSIONS):
                    file_path_list.append(os.path.join(root, name))
        return file_path_list

    def get_need_index(self, target_dirs=None, check_meta=False, rebuild=False):
        # 从合并索引文件加载路径列表与元数据
        combined = []
        if not rebuild and os.path.exists(self.combined_index_path):
            with open(self.combined_index_path, "rb") as f:
                combined = json.loads(f.read())
        exists_index = [entry["path"] for entry in combined]
        metainfo = [[entry["size"], entry["mtime"]] for entry in combined]
        # 枚举指定目录（或目录列表）的所有文件全路径
        if target_dirs is None:
            target_dirs = []
        elif not isinstance(target_dirs, list | tuple):
            target_dirs = [target_dirs]
        this_index = []
        for d in tqdm(target_dirs, ascii=True, desc="Scanning directories"):
            self.check_cancelled()
            try:
                files = self.get_file_list(d)
            except OSError as exc:
                print(f"Failed to scan {d}: {exc}", file=sys.stderr)
                files = []
            this_index.extend(files)
        # 将新增的文件路径加入已有文件路径列表
        exists_index_set = set(exists_index)
        for i in tqdm(this_index, ascii=True, desc="Scanning new-added files"):
            self.check_cancelled()
            if i not in exists_index_set:
                exists_index.append(i)
                exists_index_set.add(i)
        # 获取待更新索引的文件列表
        need_index = []
        index_ids = set(self.ir_engine.hnsw_index.get_ids_list())
        for i in tqdm(
            range(len(exists_index)),
            ascii=True,
            desc="Gathering meta information",
        ):
            self.check_cancelled()
            if exists_index[i] == NOTEXISTS:
                continue
            if i >= len(metainfo):
                metainfo.append([None, None])
            pending = rebuild or i not in index_ids or None in metainfo[i]
            if pending or check_meta:
                try:
                    st = os.stat(exists_index[i])
                except OSError as exc:
                    print(f"Failed to stat {exists_index[i]}: {exc}", file=sys.stderr)
                    metainfo[i] = [None, None]
                    continue
                if pending or metainfo[i] != [st.st_size, st.st_mtime]:
                    # Only successful feature insertion can set current metadata.
                    metainfo[i] = [None, None]
                    need_index.append(i)
        return ([(i, exists_index[i]) for i in need_index], exists_index, metainfo)

    def _combined_records(self, exists_index, metainfo):
        combined = []
        for i, path in enumerate(exists_index):
            size = metainfo[i][0] if i < len(metainfo) else None
            mtime = metainfo[i][1] if i < len(metainfo) else None
            combined.append({"path": path, "size": size, "mtime": mtime})
        return combined

    def save_meta_files(self, exists_index, metainfo, replace=True):
        """Persist metadata for successfully inserted feature vectors."""
        combined = self._combined_records(exists_index, metainfo)
        tmp_path = self.combined_index_path + ".tmp"
        with open(tmp_path, "wb") as wp:
            wp.write(self.dumps(combined).encode("UTF-8"))
            wp.flush()
            os.fsync(wp.fileno())
        if replace:
            os.replace(tmp_path, self.combined_index_path)

    def save_checkpoint(self, exists_index, metainfo, save_index=True):
        if not save_index and os.path.exists(self.combined_index_path):
            with open(self.combined_index_path, "rb") as f:
                if json.load(f) == self._combined_records(exists_index, metainfo):
                    return
        # Defer Ctrl+C until both files are saved, just like the stop flag.
        interrupted = False

        def defer_interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        previous_handler = signal.signal(signal.SIGINT, defer_interrupt)
        try:
            # Prepare metadata first so serialization/disk-full errors do not
            # publish a binary index whose path table could not be written.
            self.save_meta_files(exists_index, metainfo, replace=False)
            if save_index:
                self.ir_engine.save_index()
            os.replace(self.combined_index_path + ".tmp", self.combined_index_path)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        if interrupted:
            raise ProcessCancelled

    def update_ir_index(
        self, need_index, max_process, exists_index, metainfo, rebuild=False
    ):
        """Bound outstanding tasks and preserve received vectors on cancellation."""
        cpu_count = os.cpu_count() or 1
        num_workers = max(
            1, min(cpu_count, len(need_index) or 1, max_process, FEATURE_BATCH_SIZE)
        )
        threads_per_worker = max(1, cpu_count // num_workers)
        pool = None
        completed = []
        cancelled = False
        index_changed = rebuild or not os.path.exists(self.ir_engine.index_path)

        def accept(result):
            if result[1] is not None:
                completed.append(result)
            progress.update(1)

        def flush_batch():
            nonlocal index_changed
            if completed:
                self.ir_engine.add_fv(
                    np.asarray([item[1] for item in completed], dtype="float32"),
                    [item[0] for item in completed],
                )
                index_changed = True
                for idx, fv, meta in completed:
                    metainfo[idx] = meta
                completed.clear()

        with tqdm(
            total=len(need_index), ascii=True, desc="Computing feature vectors"
        ) as progress:
            try:
                if need_index and not self.cancel_requested():
                    try:
                        pool = multiprocessing.Pool(
                            processes=num_workers,
                            initializer=_init_worker,
                            initargs=self._worker_init_args
                            + (self.stop_flag_path, threads_per_worker),
                        )
                    except Exception as exc:
                        print(f"Using sequential extraction: {exc}", file=sys.stderr)

                for offset in range(0, len(need_index), FEATURE_BATCH_SIZE):
                    if self.cancel_requested():
                        cancelled = True
                        break
                    batch = need_index[offset : offset + FEATURE_BATCH_SIZE]
                    pending = dict(batch)
                    if pool is not None:
                        try:
                            # Only this bounded batch is submitted to the pool.
                            results = pool.imap_unordered(_worker_get_fv, batch)
                            while pending:
                                try:
                                    # On cancellation drain results already received,
                                    # without waiting for unfinished inference.
                                    timeout = 0 if self.cancel_requested() else 0.1
                                    result = results.next(timeout=timeout)
                                except multiprocessing.TimeoutError:
                                    if self.cancel_requested():
                                        cancelled = True
                                        break
                                    continue
                                except StopIteration:
                                    break
                                pending.pop(result[0])
                                accept(result)
                        except Exception as exc:
                            pool.terminate()
                            pool.join()
                            pool = None
                            print(
                                f"Using sequential extraction: {exc}", file=sys.stderr
                            )
                    if pool is None:
                        for task in pending.items():
                            if self.cancel_requested():
                                cancelled = True
                                break
                            accept(_extract_fv(self.ir_engine, task))
                    flush_batch()
                    if cancelled or self.cancel_requested():
                        cancelled = True
                        break
            except KeyboardInterrupt:
                cancelled = True
                flush_batch()
            finally:
                if pool is not None:
                    if cancelled or self.cancel_requested() or sys.exc_info()[0]:
                        pool.terminate()
                    else:
                        pool.close()
                    pool.join()

        # New/rebuilt indexes must be saved even when empty. Otherwise only
        # publish changed vectors or metadata for retryable failures.
        self.save_checkpoint(exists_index, metainfo, save_index=index_changed)
        if cancelled or self.cancel_requested():
            raise ProcessCancelled

    def remove_nonexists(self):
        """Mark none-existent files in the combined index file."""
        combined = []
        if os.path.exists(self.combined_index_path):
            with open(self.combined_index_path, "rb") as f:
                combined = json.loads(f.read())
        index_ids = set(self.ir_engine.hnsw_index.get_ids_list())
        changed = False
        index_changed = False

        def mark_deleted(idx):
            nonlocal index_changed
            if idx not in index_ids:
                return False
            try:
                self.ir_engine.hnsw_index.mark_deleted(idx)
                index_changed = True
                return True
            except RuntimeError as exc:
                if "already deleted" not in str(exc).lower():
                    raise
                return False

        for idx in tqdm(
            range(len(combined)), ascii=True, desc="Removing non-existent records"
        ):
            if self.cancel_requested():
                break
            filePath = combined[idx]["path"]
            if filePath == NOTEXISTS:
                # Repair tombstones written by older versions without saving
                # their binary deletion marks.
                changed = mark_deleted(idx) or changed
                continue
            try:
                os.stat(filePath)
            except (FileNotFoundError, NotADirectoryError):
                mark_deleted(idx)
                combined[idx] = {"path": NOTEXISTS, "size": None, "mtime": None}
                changed = True
            except OSError as exc:
                print(f"Cannot check {filePath}: {exc}", file=sys.stderr)
        if changed:
            self.save_checkpoint(
                [entry["path"] for entry in combined],
                [[entry["size"], entry["mtime"]] for entry in combined],
                save_index=index_changed,
            )
        self.check_cancelled()

    def checkout(self, image_path, exists_index, match_n=5):
        fv = self.ir_engine.get_fv(image_path)
        total_count = self.get_active_count(exists_index)
        if total_count == 0 or fv is None:
            return []
        match_n = min(match_n, total_count)
        sim, ids = self.ir_engine.match(fv, match_n)
        return [(sim[i], exists_index[ids[i]]) for i in range(len(ids))]

    def get_duplicate(self, exists_index, threshold, same_folder):
        matched = set()
        total_count = self.get_active_count(exists_index)
        if total_count == 0:
            return

        for idx in tqdm(
            range(len(exists_index)), ascii=True, desc="Retrieving duplicate records"
        ):
            if exists_index[idx] == NOTEXISTS:
                continue

            try:
                fv = self.ir_engine.hnsw_index.get_items([idx])[0]
            except RuntimeError:
                continue

            match_n = min(5, total_count)
            sim, ids = self.ir_engine.match(fv, match_n)
            while sim[-1] > threshold and match_n < total_count:
                next_match_n = min(round(match_n * 1.5) + 1, total_count)
                if next_match_n <= match_n:
                    break
                match_n = next_match_n
                sim, ids = self.ir_engine.match(fv, match_n)
            for i in range(len(ids)):
                if ids[i] == idx:
                    continue
                if sim[i] < threshold:
                    continue
                if ids[i] in matched:
                    continue
                if exists_index[ids[i]] == NOTEXISTS:
                    continue
                if idx not in matched:
                    matched.add(idx)
                path_a = os.path.normpath(exists_index[idx])
                path_b = os.path.normpath(exists_index[ids[i]])
                if same_folder:
                    if os.path.dirname(path_a) != os.path.dirname(path_b):
                        continue
                yield (path_a, path_b, sim[i])

    def get_active_count(self, exists_index):
        # HNSW's current_count includes deleted nodes and is not a safe k limit.
        ids = set(self.ir_engine.hnsw_index.get_ids_list())
        return sum(
            path != NOTEXISTS and idx in ids for idx, path in enumerate(exists_index)
        )

    def get_absolute_path(self, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(current_file_path, path)

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, ensure_ascii=False, **kwargs)
