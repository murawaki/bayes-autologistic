from collections import defaultdict
from itertools import combinations
import glob
import os
from collections import OrderedDict
import numpy as np

from .utils import collect_autologistic_results

AREA2SPEC = [
    # area, start, end (inclusive), color, marker
    ("Phonology", range(1, 19+1), "r", "o"),
    ("Morphology", range(20, 29+1), "g", "v"),
    ("Nominal Categories", range(30, 57+1), "b", "^"),
    ("Nominal Syntax", range(58, 64+1), "olive", "<"),
    ("Verbal Categories", range(65, 80+1), "purple", ">"),
    ("Word Order", range(81, 97+1), "pink", "D"),
    ("Simple Clause", range(98, 121+1), "cyan", "s"),
    ("Complex Sentences", range(122, 128+1), "magenta", "p"),
    ("Lexicon", range(129, 138+1), "black", "*"),
    # Sign Languages
    ("Other", range(141, 142+1), "grey", "h"),
    ("Word Order", range(143, 144+1), "pink", "D"),
]


def analyze_param_results(dir_results_pah):
    path = os.path.join(dir_results_pah, 'param/**/*.json')
    jsons = glob.glob(path, recursive=True)
    results = collect_autologistic_results(jsons)
    vs = np.array(list(map(lambda x: results[x][0]['parameters']['v'], results)))
    hs = np.array(list(map(lambda x: results[x][0]['parameters']['h'], results)))
    print("v max: ", np.max(vs))
    print("h max: ", np.max(hs))

    results = _group_params_per_areas(results)
    if len(results) == 0:
        return

    _print_param_results(results)
    _plot_param_results(results)


def _group_params_per_areas(results):
    results_per_cat = defaultdict(list)
    for feature in results:
        params = results[feature][0]['parameters']
        v = params['v']
        h = params['h']

        area_n = int(feature.split(' ')[0][0:-1])
        cat, c, m = _search_category(area_n)
        results_per_cat[cat].append(
            OrderedDict({
                'feature': feature,
                'values': {
                    'v': v,
                    'h': h,
                    'vh': v/h
                },
                'color': c,
                'mark': m,
                'sampled_params': {
                    'v': np.array(list(map(lambda x: x["v"], results[feature][0]["sampled_params"]))),
                    'h': np.array(list(map(lambda x: x["h"], results[feature][0]["sampled_params"]))),
                },
                "estimate_results": results[feature][0]["estimate_results"]
            })
        )
    return results_per_cat


def _print_param_results(results_per_cat):
    for category in results_per_cat:
        for param in results_per_cat[category]:
            feature = "{0:<120}".format(param['feature'])
            print(feature, end="\t", sep="")
            # 2sd: 95 percent confidence interval (frequentist) assuming a Normal distribution (not used)
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            ss.fit(np.expand_dims(param['sampled_params']['v'], axis=1))
            v_std = ss.scale_[0]
            ss = StandardScaler()
            ss.fit(np.expand_dims(param['sampled_params']['h'], axis=1))
            h_std = ss.scale_[0]
            print(_format_value(param['values']['v']), "\t+_", _format_value(2.0 * v_std), "\t",
                  _format_value(param['values']['h']), "\t+_", _format_value(2.0 * h_std),
            )

def _format_value(value):
    return ('{0:9.8f}').format(value)


def _search_category(area_n):
    found = False
    for cat, n_range, c, m in AREA2SPEC:
        if area_n in n_range:
            found = True
            break

    if found:
        return (cat, c, m)
    print(area_n)
    return (False, None, 'o')


def _plot_param_results(results_per_cat):
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rc('pdf', fonttype=42)
    matplotlib.rcParams["font.family"] = 'serif'
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    catlegend = {}

    plt.xlabel(r"$h_i$", size="24")
    plt.ylabel(r"$v_i$", size="24")
    plt.xticks(np.linspace(0.0, 0.030, num=7), size="18")
    plt.yticks(np.linspace(0.0, 0.040, num=9), size="18")
    plt.xlim((0.0, 0.025))
    plt.ylim((0.0, 0.035))
    # plt.xticks(np.linspace(-0.01, 0.020, num=4), size="18")
    # plt.yticks(np.linspace(-0.03, 0.020, num=6), size="18")
    # plt.xlim((-0.01, 0.02))
    # plt.ylim((-0.035, 0.028))
    plt.axhline(color="black")
    plt.axvline(color="black")

    for category in results_per_cat:
        for param in results_per_cat[category]:
            # feature_id = param['feature'].split(' ')
            dot = plt.scatter(param['values']['h'],
                              param['values']['v'],
                              s=15,
                              c=param['color'], marker=param['mark'],
                              label=category)
            if not category:
                continue
            if category not in catlegend:
                catlegend[category] = dot

    catlegend_sorted = []
    for cat in sorted(catlegend.keys()):
        catlegend_sorted.append(catlegend[cat])
    plt.legend(
        handles=catlegend_sorted,
        loc='center left', bbox_to_anchor=(1.0, 0.5),
        scatterpoints=1, fontsize=15)
    plt.savefig('autologistic_params.pdf', bbox_inches="tight")
