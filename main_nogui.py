import sys
import os
import json
from getopt import getopt, GetoptError
from utils import Utils

current_path = os.path.dirname(__file__)
config_path = os.path.join(current_path, 'nogui/config.json')
config = json.loads(open(config_path, 'rb').read())
utils = Utils(config)


def main(argv):
    add_index_dir_list = []
    remove_index_dir_list = []
    is_get_index_dir = False
    is_update_all_index = False  # update all existed index dir
    update_index_dir_list = []
    is_search_all_index = False  # Search all existed index dir
    search_index_similarity_threshold = 98.5  # 70 <= threshold <= 100
    search_index_same_dir = False

    try:
        opts, args = getopt(argv, "", [
            "add_index_dir=", "remove_index_dir=", "get_index_dir",
            "update_index", "update_index_dir=", "search_index",
            "search_index_similarity_threshold=", "search_index_same_dir"
        ])
    except GetoptError:
        sys.stderr('Wrong parameters.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--add_index_dir':
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
        elif opt == '--search_index_similarity_threshold':
            threshold = float(arg)
            if (threshold > 100) or (threshold < 70):
                sys.stderr('search_index_similarity_threshold should ' +
                           'between 70 and 100')
                sys.exit(2)
            search_index_similarity_threshold = threshold
        elif opt == '--search_index_same_dir':
            search_index_same_dir = True

    if len(add_index_dir_list):
        add_index_dir(add_index_dir_list)
    elif len(remove_index_dir_list):
        remove_index_dir(remove_index_dir_list)
    elif is_get_index_dir:
        get_index_dir()
    elif is_update_all_index:
        update_all_index()
    elif len(update_index_dir_list):
        update_index(update_index_dir_list)
    elif is_search_all_index:
        search_index_dir(search_index_similarity_threshold,
                         search_index_same_dir)


def add_index_dir(dirs):
    config['search_dir'].extend(dirs)
    config['search_dir'] = list(set(config['search_dir']))
    save_settings()


def remove_index_dir(dirs):
    for dir in dirs:
        try:
            config['search_dir'].remove(dir)
        except ValueError:
            sys.stderr('Path `' + dir + '` not exists in index dir list')
    save_settings()


def get_index_dir():
    sys.stdout.write(utils.dumps(config['search_dir']))


def update_index(dirs):
    utils.remove_nonexists()
    for index_dir in dirs:
        index_dir_to_update = utils.index_target_dir(index_dir)
        utils.update_ir_index(index_dir_to_update)


def update_all_index():
    update_index(config['search_dir'])


def search_index_dir(threshold, same_dir):
    if not os.path.exists(utils.exists_index_path):
        sys.stderr('You should update index before searching')
        sys.exit(2)
    get_duplicate_res = utils.get_duplicate(utils.get_exists_index(),
                                            threshold, same_dir)
    res = []
    for item in get_duplicate_res:
        res.append({'path_a': item[0], 'path_b': item[1], 'sim': item[2]})
    sys.stdout.write(utils.dumps(res))


def save_settings():
    with open(config_path, 'wb') as wp:
        wp.write(utils.dumps(config, indent=2).encode('UTF-8'))


if __name__ == "__main__":
    main(sys.argv[1:])
