import os
import json
import multiprocessing
from tqdm import tqdm
from efficient_ir import EfficientIR

NOTEXISTS = 'NOTEXISTS'

current_file_path = os.path.dirname(os.path.abspath(__file__))
STOP_FLAG_FILENAME = 'process.stop'
STOP_FLAG_PATH = os.path.join(current_file_path, STOP_FLAG_FILENAME)

# Global engine used inside worker processes. Each worker will initialize
# its own EfficientIR instance to compute feature vectors (get_fv).
_WORKER_ENGINE = None


def _init_worker(img_size,
                 index_capacity,
                 index_path,
                 model_path,
                 stop_flag_path=None):
    """Initializer for worker processes. Creates a module-global EfficientIR
    instance to avoid reloading model on every task call inside the same
    worker.
    """
    global _WORKER_ENGINE
    try:
        _WORKER_ENGINE = EfficientIR(img_size, index_capacity, index_path,
                                     model_path)
        # Worker-visible stop flag path (optional)
        global _WORKER_STOP_FLAG
        _WORKER_STOP_FLAG = stop_flag_path
    except Exception:
        # If worker init fails, ensure _WORKER_ENGINE is None so worker
        # tasks will return None.
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
            if '_WORKER_STOP_FLAG' in globals() and _WORKER_STOP_FLAG and \
                    os.path.exists(_WORKER_STOP_FLAG):
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
        self.metainfo_path = self.get_absolute_path(config['metainfo_path'])
        self.exists_index_path = self.get_absolute_path(
            config['exists_index_path'])
        self.ir_engine = EfficientIR(
            config['img_size'], config['index_capacity'],
            self.get_absolute_path(config['index_path']),
            self.get_absolute_path(config['model_path']))
        # Save worker init args so child processes can create their own
        # EfficientIR instances for get_fv computation.
        self._worker_init_args = (config['img_size'], config['index_capacity'],
                                  self.get_absolute_path(config['index_path']),
                                  self.get_absolute_path(config['model_path']))
        # Stop flag path used to request cancellation across processes.
        self.stop_flag_path = self.get_absolute_path(
            config.get('stop_flag_path', STOP_FLAG_PATH))
        self.check_env()

    def check_env(self):
        if not os.path.exists(self.exists_index_path):
            parent_path = os.path.join(self.exists_index_path, os.pardir)
            os.makedirs(os.path.abspath(parent_path), exist_ok=True)
            with open(self.exists_index_path, 'w') as wp:
                wp.write("[]")

    def get_exists_index(self):
        return json.loads(open(self.exists_index_path, 'rb').read())

    def get_file_list(self, target_dir):
        accepted_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']
        file_path_list = []
        for root, dirs, files in os.walk(target_dir):
            for name in files:
                if name.lower().endswith(tuple(accepted_exts)):
                    file_path_list.append(os.path.join(root, name))
        return file_path_list

    def get_need_index(self, target_dir):
        # 如果已有文件索引就加载
        exists_index = []
        if os.path.exists(self.exists_index_path):
            exists_index = json.loads(
                open(self.exists_index_path, 'rb').read())
        # 如果已有元信息索引就加载
        metainfo = []
        if os.path.exists(self.metainfo_path):
            metainfo = json.loads(open(self.metainfo_path, 'rb').read())
        # 枚举当前指定目录的所有文件全路径
        this_index = self.get_file_list(target_dir)
        # 需要特征索引的文件
        need_index = []
        # 更新文件索引
        for i in tqdm(this_index, ascii=True, desc='Scanning new-added files'):
            if i not in exists_index:
                exists_index.append(i)
        # 更新元信息索引
        for i in tqdm(
                range(len(exists_index)),
                ascii=True,
                desc='Gathering metainfo',
        ):
            if NOTEXISTS in exists_index[i]:
                continue
            # 采集元信息
            file_size = os.path.getsize(exists_index[i])
            file_mtime = os.path.getmtime(exists_index[i])
            # 新增元信息
            if i >= len(metainfo):
                metainfo.append([file_size, file_mtime])
                need_index.append(i)
                continue
            # 检查元信息更新
            if metainfo[i][0] != file_size or metainfo[i][1] != file_mtime:
                metainfo[i] = [file_size, file_mtime]
                need_index.append(i)
        return ([(i, exists_index[i])
                 for i in need_index], exists_index, metainfo)

    def save_meta_files(self, exists_index, metainfo):
        """Persist exists index and metainfo to disk.

        This should be called after update_ir_index finishes so that the
        on-disk index reflects completed updates.
        """
        try:
            with open(self.exists_index_path, 'wb') as wp:
                wp.write(self.dumps(exists_index).encode('UTF-8'))
        except Exception:
            pass
        try:
            with open(self.metainfo_path, 'wb') as wp:
                wp.write(self.dumps(metainfo).encode('UTF-8'))
        except Exception:
            pass

    def update_ir_index(self, need_index):
        # If no items, nothing to do
        if not need_index:
            return

        # Determine number of workers: at most cpu_count and len(need_index)
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, min(cpu_count, len(need_index)))
        results = []

        # Use a multiprocessing Pool. Each worker will initialize its own
        # EfficientIR instance (via _init_worker) and compute feature vectors.
        try:
            with multiprocessing.Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=self._worker_init_args + (self.stop_flag_path, ),
            ) as pool:
                # imap keeps memory usage lower for large lists
                results_iter = pool.imap(_worker_get_fv, need_index)
                for r in tqdm(results_iter,
                              total=len(need_index),
                              ascii=True,
                              desc='Computing feature vectors'):
                    results.append(r)
        except Exception:
            # If multiprocessing fails for any reason, fall back to
            # sequential processing to keep behavior correct.
            for idx, fpath in tqdm(
                    need_index,
                    ascii=True,
                    desc='Computing feature vectors (sequential)'):
                try:
                    fv = self.ir_engine.get_fv(fpath)
                except Exception:
                    fv = None
                results.append((idx, fv) if fv is not None else None)

        # Add computed feature vectors to the main ir_engine in the main
        # process to avoid concurrent writes to the index structure.
        for item in results:
            if not item:
                continue
            idx, fv = item
            if fv is None:
                continue
            self.ir_engine.add_fv(fv, idx)

        # Persist index
        self.ir_engine.save_index()

    def remove_nonexists(self):
        exists_index = []
        if os.path.exists(self.exists_index_path):
            exists_index = json.loads(
                open(self.exists_index_path, 'rb').read())
        for idx in tqdm(range(len(exists_index)),
                        ascii=True,
                        desc='Removing non-existent records'):
            if not os.path.exists(exists_index[idx]):
                exists_index[idx] = NOTEXISTS
                try:
                    self.ir_engine.hnsw_index.mark_deleted(idx)
                except Exception:
                    pass
        with open(self.exists_index_path, 'wb') as wp:
            wp.write(self.dumps(exists_index).encode('UTF-8'))

    def checkout(self, image_path, exists_index, match_n=5):
        fv = self.ir_engine.get_fv(image_path)
        sim, ids = self.ir_engine.match(fv, match_n)
        return [(sim[i], exists_index[ids[i]]) for i in range(len(ids))]

    def get_duplicate(self, exists_index, threshold, same_folder):
        matched = set()
        for idx in tqdm(range(len(exists_index)),
                        ascii=True,
                        desc='Retrieving duplicate records'):
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
