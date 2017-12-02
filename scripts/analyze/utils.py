from collections import defaultdict
import json


def collect_autologistic_results(file_paths):
    results = defaultdict(list)
    for file_path in file_paths:
        jdata = None
        with open(file_path, 'r') as f:
            jdata = json.load(f)
        if jdata is None:
            continue
        feature = jdata['feature']
        results[feature].append(jdata)
    return results
