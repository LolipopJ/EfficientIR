import json
import multiprocessing
import os
import sys
from getopt import GetoptError, getopt

from utils import STOP_FLAG_PATH, ProcessCancelled, Utils

current_file_path = os.path.dirname(os.path.abspath(__file__))
utils: Utils | None = None


def get_utils(config=None) -> Utils:
    global utils
    if utils is None:
        if config is None:
            config_path = os.path.join(current_file_path, "./config.json")
            with open(config_path, "rb") as f:
                config = json.loads(f.read())
        utils = Utils(config)
    return utils


def main(argv):
    # When multiprocessing spawns child processes on Windows (spawn start
    # method), the child may start the script with internal args like
    # '--multiprocessing-fork'. Avoid running CLI parsing in such child
    # processes.
    if any(str(a).startswith("--multiprocessing") for a in sys.argv):
        return

    config_path = os.path.join(current_file_path, "./config.json")
    add_index_dir_list = []
    remove_index_dir_list = []
    is_get_index_dir = False
    is_update_all_index = False  # update all existed index dir
    update_index_dir_list = []
    is_check_meta = False  # check file meta info for re-indexing
    is_search_all_index = False  # search all existed index dir
    search_target = ""  # search for similar images to the image
    similarity_threshold = 98.5  # 70 <= threshold <= 100
    same_dir = False  # search images of same dir
    match_n = 5
    max_process = 4
    is_rebuild_index = False
    is_cancel_process = False

    argv = normalize_argv(argv)
    try:
        opts, args = getopt(
            argv,
            "",
            [
                "config_path=",
                "add_index_dir=",
                "remove_index_dir=",
                "get_index_dir",
                "update_index",
                "update_index_dir=",
                "check_meta",
                "search_index",
                "search_target=",
                "similarity_threshold=",
                "same_dir",
                "match_n=",
                "max_process=",
                "rebuild_index",
                "cancel_process",
            ],
        )
    except GetoptError:
        sys.stderr.write("Wrong parameters.\n")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--config_path":
            config_path = arg
        elif opt == "--add_index_dir":
            add_index_dir_list.append(arg)
        elif opt == "--remove_index_dir":
            remove_index_dir_list.append(arg)
        elif opt == "--get_index_dir":
            is_get_index_dir = True
        elif opt == "--update_index":
            is_update_all_index = True
        elif opt == "--update_index_dir":
            update_index_dir_list.append(arg)
        elif opt == "--check_meta":
            is_check_meta = True
        elif opt == "--search_index":
            is_search_all_index = True
        elif opt == "--search_target":
            search_target = arg
        elif opt == "--similarity_threshold":
            threshold = float(arg)
            if (threshold > 100) or (threshold < 70):
                sys.stderr.write("similarity_threshold should between 70 and 100\n")
                sys.exit(2)
            similarity_threshold = threshold
        elif opt == "--same_dir":
            same_dir = True
        elif opt == "--match_n":
            match_n = int(arg)
        elif opt == "--max_process":
            max_process = int(arg)
        elif opt == "--rebuild_index":
            is_rebuild_index = True
        elif opt == "--cancel_process":
            is_cancel_process = True

    with open(config_path, "rb") as f:
        config = json.loads(f.read())

    stop_flag_path = get_stop_flag_path(config)

    if is_cancel_process:
        # Signal-only request: avoid loading the ONNX model / HNSW index.
        request_cancel_process(stop_flag_path)
    else:
        is_index_update = (
            is_rebuild_index or is_update_all_index or bool(update_index_dir_list)
        )
        # Electron clears this before spawning, so an early cancel is not lost
        # while Python imports dependencies. CLI callers clear stale flags here.
        if is_index_update and os.environ.get("EFFICIENTIR_CANCEL_PREPARED") != "1":
            clear_cancel_flag(stop_flag_path)
        get_utils(config)
        if is_index_update:
            get_utils().check_cancelled()

        if is_rebuild_index:
            rebuild_index(config, max_process)
        elif len(add_index_dir_list):
            add_index_dir(config_path, config, add_index_dir_list)
        elif len(remove_index_dir_list):
            remove_index_dir(config_path, config, remove_index_dir_list)
        elif is_get_index_dir:
            get_index_dir(config)
        elif is_update_all_index:
            update_index(
                dirs=config["search_dir"],
                check_meta=is_check_meta,
                max_process=max_process,
            )
        elif len(update_index_dir_list):
            update_index(
                dirs=update_index_dir_list,
                check_meta=is_check_meta,
                max_process=max_process,
            )
        elif is_search_all_index:
            search_index_dir(similarity_threshold, same_dir)
        elif search_target:
            search_index_dir_target(search_target, match_n)


def dumps(obj, **kwargs):
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def add_index_dir(config_path, config, dirs):
    config["search_dir"].extend(dirs)
    config["search_dir"] = list(set(config["search_dir"]))
    save_settings(config_path, config)


def remove_index_dir(config_path, config, dirs):
    for dir in dirs:
        try:
            config["search_dir"].remove(dir)
        except ValueError:
            sys.stderr.write("Path `" + dir + "` not exists in index dir list\n")
    save_settings(config_path, config)


def get_index_dir(config):
    sys.stdout.write(dumps(config["search_dir"]))


def update_index(dirs=None, check_meta=False, max_process=4):
    u = get_utils()
    u.remove_nonexists()
    need_index, exists_index, metainfo = u.get_need_index(
        target_dirs=dirs, check_meta=check_meta
    )
    u.update_ir_index(need_index, max_process, exists_index, metainfo)


def rebuild_index(config, max_process=4):
    """Build a fully initialized fresh index without deleting disk state first."""
    u = get_utils()
    need_index, exists_index, metainfo = u.get_need_index(
        config.get("search_dir", []), rebuild=True
    )
    u.check_cancelled()
    u.ir_engine.init_index()
    u.update_ir_index(need_index, max_process, exists_index, metainfo, rebuild=True)


def search_index_dir(threshold, same_dir):
    u = get_utils()
    if not os.path.exists(u.combined_index_path):
        sys.stderr.write("You should update index before searching")
        sys.exit(2)
    get_duplicate_res = u.get_duplicate(u.get_exists_index(), threshold, same_dir)
    res = []
    for item in get_duplicate_res:
        res.append({"path_a": item[0], "path_b": item[1], "sim": str(item[2])})
    sys.stdout.write(dumps(res))


def search_index_dir_target(target_file_path, match_n):
    u = get_utils()
    if not os.path.exists(u.combined_index_path):
        sys.stderr.write("You should update index before searching")
        sys.exit(2)
    get_duplicate_res = u.checkout(target_file_path, u.get_exists_index(), match_n)
    res = []
    for item in get_duplicate_res:
        res.append({"path": item[1], "sim": str(item[0])})
    sys.stdout.write(dumps(res))


def save_settings(config_path, config):
    tmp_path = config_path + ".tmp"
    with open(tmp_path, "wb") as wp:
        wp.write(dumps(config, indent=2).encode("UTF-8"))
    os.replace(tmp_path, config_path)


def get_stop_flag_path(config):
    """Resolve the stop-flag path from config without constructing Utils,
    so a cancel request doesn't need to load the ONNX model / HNSW index.
    """
    path = os.environ.get("EFFICIENTIR_STOP_FLAG_PATH") or config.get(
        "stop_flag_path", STOP_FLAG_PATH
    )
    return path if os.path.isabs(path) else os.path.join(current_file_path, path)


def request_cancel_process(stop_flag_path):
    """Request a cooperative stop; the update loop saves completed work."""
    try:
        with open(stop_flag_path, "w") as wp:
            wp.write("1")
    except Exception as e:
        sys.stderr.write(f"Failed to request cancellation: {e}\n")
        sys.exit(2)


def clear_cancel_flag(stop_flag_path):
    """Remove the stop-flag file if present."""
    try:
        if os.path.exists(stop_flag_path):
            os.remove(stop_flag_path)
    except Exception as e:
        sys.stderr.write(f"Failed to clear cancel flag: {e}\n")


def normalize_argv(argv):
    """If an argv element looks like "--opt value" packed into one string
    (no '=' present), split it into two elements. This helps when callers
    (e.g. Node spawn) pass option+value as a single argument containing
    spaces.
    """
    out = []
    for a in argv:
        if isinstance(a, str) and a.startswith("--") and " " in a and "=" not in a:
            opt, val = a.split(" ", 1)
            out.append(opt)
            out.append(val)
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    # On Windows, ensure multiprocessing freeze support is enabled for
    # spawn-based child processes (pyinstaller/ frozen apps compatibility).
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    try:
        main(sys.argv[1:])
    except (ProcessCancelled, KeyboardInterrupt):
        sys.stderr.write("Cancelled. Completed feature vectors have been saved.\n")
        sys.exit(130)
