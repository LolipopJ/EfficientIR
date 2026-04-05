import json
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

from efficient_ir import EfficientIR, FeatureExtractor

NOTEXISTS = "NOTEXISTS"

current_file_path = os.path.dirname(os.path.abspath(__file__))
STOP_FLAG_FILENAME = "process.stop"
STOP_FLAG_PATH = os.path.join(current_file_path, STOP_FLAG_FILENAME)

# Global feature extractor used inside worker processes.
# Each worker loads only the ONNX model, not the HNSW index.
_WORKER_ENGINE = None


def _init_worker(img_size, model_path, stop_flag_path=None):
    """Initializer for worker processes. Creates a module-global
    FeatureExtractor instance (ONNX model only) to avoid reloading the model
    on every task call inside the same worker.
    """
    global _WORKER_ENGINE
    try:
        _WORKER_ENGINE = FeatureExtractor(img_size, model_path)
        global _WORKER_STOP_FLAG
        _WORKER_STOP_FLAG = stop_flag_path
    except Exception:
        _WORKER_ENGINE = None


def _worker_get_fv(task):
    """Worker function. Receives a tuple (idx, fpath) and returns (idx, fv)
    or None on failure.
    """
    global _WORKER_ENGINE
    try:
        # If a stop flag path was provided to the worker initializer and the
        # stop file exists, allow worker to exit early.
        try:
            if (
                "_WORKER_STOP_FLAG" in globals()
                and _WORKER_STOP_FLAG
                and os.path.exists(_WORKER_STOP_FLAG)
            ):
                return None
        except Exception:
            pass

        if _WORKER_ENGINE is None:
            return None
        idx, fpath = task
        fv = _WORKER_ENGINE.get_fv(fpath)
        return (idx, fv)
    except Exception:
        return None


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
            config.get("stop_flag_path", STOP_FLAG_PATH)
        )
        self.check_env()

    def check_env(self):
        if not os.path.exists(self.combined_index_path):
            parent_path = os.path.join(self.combined_index_path, os.pardir)
            os.makedirs(os.path.abspath(parent_path), exist_ok=True)
            # 自动迁移旧版两个独立文件（name_index.json + metainfo.json）
            tmp_path = self.combined_index_path + ".tmp"
            if os.path.exists(self.exists_index_path):
                exists_list = json.loads(open(self.exists_index_path, "rb").read())
                meta_list = []
                if os.path.exists(self.metainfo_path):
                    meta_list = json.loads(open(self.metainfo_path, "rb").read())
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
        combined = json.loads(open(self.combined_index_path, "rb").read())
        return [entry["path"] for entry in combined]

    def get_file_list(self, target_dir):
        accepted_exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
        file_path_list = []
        for root, dirs, files in os.walk(target_dir):
            # Show progress per-directory when scanning files
            for name in tqdm(
                files, ascii=True, desc=f"Scanning files in {root}", unit="file"
            ):
                if name.lower().endswith(tuple(accepted_exts)):
                    file_path_list.append(os.path.join(root, name))
        return file_path_list

    def get_need_index(self, target_dirs=[], check_meta=False):
        # 从合并索引文件加载路径列表与元数据
        combined = []
        if os.path.exists(self.combined_index_path):
            combined = json.loads(open(self.combined_index_path, "rb").read())
        exists_index = [entry["path"] for entry in combined]
        metainfo = [[entry["size"], entry["mtime"]] for entry in combined]
        # 枚举指定目录（或目录列表）的所有文件全路径
        if not isinstance(target_dirs, (list, tuple)):
            target_dirs = [target_dirs]
        this_index = []
        for d in tqdm(target_dirs, ascii=True, desc="Scanning directories"):
            try:
                files = self.get_file_list(d)
            except Exception:
                files = []
            this_index.extend(files)
        # 将新增的文件路径加入已有文件路径列表
        for i in tqdm(this_index, ascii=True, desc="Scanning new-added files"):
            if i not in exists_index:
                exists_index.append(i)
        # 获取待更新索引的文件列表
        need_index = []
        for i in tqdm(
            range(len(exists_index)),
            ascii=True,
            desc="Gathering meta information",
        ):
            if NOTEXISTS in exists_index[i]:
                continue
            if i >= len(metainfo) or check_meta:
                file_stat = os.stat(exists_index[i])
                file_size = file_stat.st_size
                file_mtime = file_stat.st_mtime
                if i >= len(metainfo):
                    # 索引新文件
                    metainfo.append([file_size, file_mtime])
                    need_index.append(i)
                elif check_meta:
                    if metainfo[i][0] != file_size or metainfo[i][1] != file_mtime:
                        # 重新索引元数据发生变化的文件
                        metainfo[i] = [file_size, file_mtime]
                        need_index.append(i)
        return ([(i, exists_index[i]) for i in need_index], exists_index, metainfo)

    def save_meta_files(self, exists_index, metainfo):
        """Persist exists index and metainfo to the combined index file.

        This should be called after update_ir_index finishes so that the
        on-disk index reflects completed updates.
        """
        combined = []
        for i, path in enumerate(exists_index):
            size = metainfo[i][0] if i < len(metainfo) else None
            mtime = metainfo[i][1] if i < len(metainfo) else None
            combined.append({"path": path, "size": size, "mtime": mtime})
        tmp_path = self.combined_index_path + ".tmp"
        try:
            with open(tmp_path, "wb") as wp:
                wp.write(self.dumps(combined).encode("UTF-8"))
            os.replace(tmp_path, self.combined_index_path)
        except Exception:
            pass

    def update_ir_index(self, need_index, max_process):
        # If no items, nothing to do
        if not need_index:
            return

        # Determine number of workers: at most cpu_count and len(need_index)
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, min(cpu_count, len(need_index), max_process))
        results = []

        # Use a multiprocessing Pool. Each worker will initialize its own
        # EfficientIR instance (via _init_worker) and compute feature vectors.
        try:
            with multiprocessing.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=self._worker_init_args + (self.stop_flag_path,),
            ) as pool:
                # imap keeps memory usage lower for large lists
                results_iter = pool.imap(_worker_get_fv, need_index)
                for r in tqdm(
                    results_iter,
                    total=len(need_index),
                    ascii=True,
                    desc="Computing feature vectors",
                ):
                    results.append(r)
        except Exception:
            # If multiprocessing fails for any reason, fall back to
            # sequential processing to keep behavior correct.
            for idx, fpath in tqdm(
                need_index, ascii=True, desc="Computing feature vectors (sequential)"
            ):
                try:
                    fv = self.ir_engine.get_fv(fpath)
                except Exception:
                    fv = None
                results.append((idx, fv) if fv is not None else None)

        # Collect valid (idx, fv) pairs then add to the index in one batch
        # call, which is significantly faster than adding one vector at a time.
        valid_ids = []
        valid_fvs = []
        for item in results:
            if not item:
                continue
            idx, fv = item
            if fv is None:
                continue
            valid_ids.append(idx)
            valid_fvs.append(fv)

        if valid_ids:
            self.ir_engine.add_fv(np.array(valid_fvs), valid_ids)

        # Persist index
        self.ir_engine.save_index()

    def remove_nonexists(self):
        """Mark none-existent files in the combined index file."""
        combined = []
        if os.path.exists(self.combined_index_path):
            combined = json.loads(open(self.combined_index_path, "rb").read())
        for idx in tqdm(
            range(len(combined)), ascii=True, desc="Removing non-existent records"
        ):
            if not os.path.exists(combined[idx]["path"]):
                try:
                    self.ir_engine.hnsw_index.mark_deleted(idx)
                    combined[idx] = {"path": NOTEXISTS, "size": None, "mtime": None}
                except Exception:
                    pass
        tmp_path = self.combined_index_path + ".tmp"
        with open(tmp_path, "wb") as wp:
            wp.write(self.dumps(combined).encode("UTF-8"))
        os.replace(tmp_path, self.combined_index_path)

    def checkout(self, image_path, exists_index, match_n=5):
        fv = self.ir_engine.get_fv(image_path)
        sim, ids = self.ir_engine.match(fv, match_n)
        return [(sim[i], exists_index[ids[i]]) for i in range(len(ids))]

    def get_duplicate(self, exists_index, threshold, same_folder):
        matched = set()
        for idx in tqdm(
            range(len(exists_index)), ascii=True, desc="Retrieving duplicate records"
        ):
            match_n = 5
            try:
                fv = self.ir_engine.hnsw_index.get_items([idx])[0]
            except RuntimeError:
                continue
            sim, ids = self.ir_engine.match(fv, match_n)
            while sim[-1] > threshold:
                match_n = round(match_n * 1.5)
                sim, ids = self.ir_engine.match(fv, match_n)
            for i in range(len(ids)):
                if ids[i] == idx:
                    continue
                if sim[i] < threshold:
                    continue
                if ids[i] in matched:
                    continue
                if idx not in matched:
                    matched.add(idx)
                path_a = os.path.normpath(exists_index[idx])
                path_b = os.path.normpath(exists_index[ids[i]])
                if same_folder:
                    if os.path.dirname(path_a) != os.path.dirname(path_b):
                        continue
                yield (path_a, path_b, sim[i])

    def get_absolute_path(self, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(current_file_path, path)

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, ensure_ascii=False, **kwargs)
