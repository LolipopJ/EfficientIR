import sys
import os
import json
from getopt import getopt, GetoptError
from utils import Utils

current_file_path = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    config_path = os.path.join(current_file_path, 'nogui/config.json')
    config = {}
    add_index_dir_list = []
    remove_index_dir_list = []
    is_get_index_dir = False
    is_update_all_index = False  # update all existed index dir
    update_index_dir_list = []
    is_search_all_index = False  # search all existed index dir
    search_target = ''  # Search for similar images to the image
    similarity_threshold = 98.5  # 70 <= threshold <= 100
    same_dir = False  # search images of same dir
    match_n = 5

    try:
        opts, args = getopt(argv, "", [
            "config_path=", "add_index_dir=", "remove_index_dir=",
            "get_index_dir", "update_index", "update_index_dir=",
            "search_index", "search_target=", "similarity_threshold=",
            "same_dir", "match_n="
        ])
    except GetoptError:
        sys.stderr('Wrong parameters.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--config_path':
            config_path = arg
        elif opt == '--add_index_dir':
            add_index_dir_list.append(arg)
        elif opt == '--remove_index_dir':
            remove_index_dir_list.append(arg)
        elif opt == '--get_index_dir':
            is_get_index_dir = True
        elif opt == '--update_index':
            is_update_all_index = True
        elif opt == '--update_index_dir':
            update_index_dir_list.append(arg)
        elif opt == '--search_index':
            is_search_all_index = True
        elif opt == '--search_target':
            search_target = arg
        elif opt == '--similarity_threshold':
            threshold = float(arg)
            if (threshold > 100) or (threshold < 70):
                sys.stderr('similarity_threshold should ' +
                           'between 70 and 100')
                sys.exit(2)
            similarity_threshold = threshold
        elif opt == '--same_dir':
            same_dir = True
        elif opt == '--match_n':
            match_n = int(arg)

    config = json.loads(open(config_path, 'rb').read())

    if len(add_index_dir_list):
        add_index_dir(config_path, config, add_index_dir_list)
    elif len(remove_index_dir_list):
        remove_index_dir(config_path, config, remove_index_dir_list)
    elif is_get_index_dir:
        get_index_dir(config)
    elif is_update_all_index:
        update_all_index(config)
    elif len(update_index_dir_list):
        update_index(config, update_index_dir_list)
    elif is_search_all_index:
        search_index_dir(config, similarity_threshold, same_dir)
    elif search_target:
        search_index_dir_target(config, search_target, match_n)


def dumps(obj, **kwargs):
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def add_index_dir(config_path, config, dirs):
    config['search_dir'].extend(dirs)
    config['search_dir'] = list(set(config['search_dir']))
    save_settings(config_path, config)


def remove_index_dir(config_path, config, dirs):
    for dir in dirs:
        try:
            config['search_dir'].remove(dir)
        except ValueError:
            sys.stderr('Path `' + dir + '` not exists in index dir list')
    save_settings(config_path, config)


def get_index_dir(config):
    sys.stdout.write(dumps(config['search_dir']))


def update_index(config, dirs):
    utils = Utils(config)
    utils.remove_nonexists()
    for index_dir in dirs:
        index_dir_to_update = utils.index_target_dir(index_dir)
        utils.update_ir_index(index_dir_to_update)


def update_all_index(config):
    update_index(config, config['search_dir'])


def search_index_dir(config, threshold, same_dir):
    utils = Utils(config)
    if not os.path.exists(utils.exists_index_path):
        sys.stderr('You should update index before searching')
        sys.exit(2)
    get_duplicate_res = utils.get_duplicate(utils.get_exists_index(),
                                            threshold, same_dir)
    res = []
    for item in get_duplicate_res:
        res.append({'path_a': item[0], 'path_b': item[1], 'sim': str(item[2])})
    sys.stdout.write(dumps(res))


def search_index_dir_target(config, target_file_path, match_n):
    utils = Utils(config)
    if not os.path.exists(utils.exists_index_path):
        sys.stderr('You should update index before searching')
        sys.exit(2)
    get_duplicate_res = utils.checkout(target_file_path,
                                       utils.get_exists_index(), match_n)
    res = []
    for item in get_duplicate_res:
        res.append({'path': item[1], 'sim': str(item[0])})
    sys.stdout.write(dumps(res))


def save_settings(config_path, config):
    with open(config_path, 'wb') as wp:
        wp.write(dumps(config, indent=2).encode('UTF-8'))


if __name__ == "__main__":
    main(sys.argv[1:])
