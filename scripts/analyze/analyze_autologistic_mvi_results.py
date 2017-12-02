import sys
import glob
import os
import numpy as np

from .utils import collect_autologistic_results

def analyze_mvi_results(dir_results_pah):
    path = os.path.join(dir_results_pah, 'mvi/**/*.json')
    jsons = glob.glob(path, recursive=True)
    results = collect_autologistic_results(jsons)
    if len(results) == 0:
        return

    accuracies = _calc_accuracies(results)
    _print_accuracies(accuracies)
    _plot_accuracies(accuracies)

def _calc_accuracies(results):
    p_avg_ma = 0.0
    gm_avg_ma = 0.0
    n_avg_ma = 0.0

    p_avg_mi = 0.0
    gm_avg_mi = 0.0
    n_avg_mi = 0.0

    c_all = 0
    vs_all = []
    hs_all = []

    accuracies = {}
    accuracies['macro_per_features'] = {}
    accuracy_micro = {
        'proposed': (0, 0),
        'global_majority': (0, 0),
        'neighborhood': (0, 0),
        'universal': (0, 0)
    }
    for feature in sorted(results.keys()):
        c = len(results[feature])
        c_all += c
        accuracy_per_feature = {
            'proposed': (0, 0),
            'global_majority': (0, 0),
            'neighborhood': (0, 0),
            'universal': (0, 0)
        }
        gm_avg = 0.0
        n_avg = 0.0
        vs = []
        hs = []

        for i in range(c):
            for model_name in ['proposed', 'global_majority', 'neighborhood', 'universal']:
                current_value = accuracy_per_feature[model_name]
                m = model_name
                segment = results[feature][i]['accuracy'][m]
                if m != 'global_majority':
                    segment = segment[0]
                accuracy_per_feature[model_name] = (
                    current_value[0] + segment['correct'],
                    current_value[1] + segment['total']
                )

                current_value = accuracy_micro[model_name]
                accuracy_micro[model_name] = (
                    current_value[0] + segment['correct'],
                    current_value[1] + segment['total']
                )

        fn = "{0:<120}".format(feature)
        accuracies['macro_per_features'][fn] = {
            'num': c
        }

        for model_name in ['proposed', 'global_majority', 'neighborhood', 'universal']:
            current_value = accuracy_per_feature[model_name]
            accuracy = current_value[0] / current_value[1]
            accuracies['macro_per_features'][fn][model_name] = accuracy

    accuracies['macro'] = {'proposed': 0.0, 'global_majority': 0.0, 'neighborhood': 0.0, 'universal': 0.0}

    feature_num = len(results)
    for feature_name in accuracies['macro_per_features']:
        for model_name in ['proposed', 'global_majority', 'neighborhood', 'universal']:
            per_feature = accuracies['macro_per_features'][feature_name]
            accuracies['macro'][model_name] += per_feature[model_name]

    accuracies['micro'] = {'proposed': 0.0, 'global_majority': 0.0, 'neighborhood': 0.0, 'universal': 0.0} 
    for model_name in ['proposed', 'global_majority', 'neighborhood', 'universal']:
        accuracies['macro'][model_name] /= feature_num
        macro_per_model = accuracy_micro[model_name]
        accuracies['micro'][model_name] = macro_per_model[0] / macro_per_model[1]
        print("{}\t{}\t{}".format(model_name, macro_per_model[0], macro_per_model[1]))
    return accuracies


def _print_accuracies(accuracies):
    print('Feature Name', 'Num', 'P', 'GM', 'N', 'U', 'P-GM', 'P-N', sep="\t")
    for feature in sorted(accuracies['macro_per_features']):
        accuracy = accuracies['macro_per_features'][feature]
        print(
            feature,
            accuracy['num'],
            _format_percentage(accuracy['proposed']),
            _format_percentage(accuracy['global_majority']),
            _format_percentage(accuracy['neighborhood']),
            _format_percentage(accuracy['universal']),
            _format_percentage(accuracy['proposed'] - accuracy['global_majority'], True),
            _format_percentage(accuracy['proposed'] - accuracy['neighborhood'], True),
            sep="\t"
        )
    for model in ['proposed', 'global_majority', 'neighborhood', 'universal']:
        print(model, _format_percentage(accuracies['macro'][model]),\
              _format_percentage(accuracies['micro'][model]))

def _plot_accuracies(accuracies):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rc('pdf', fonttype=42)
    matplotlib.rcParams["font.family"] = 'serif'

    plt.xlabel(r"Features", size="18")
    plt.ylabel(u"Accuracy (\%)", size="18")
    plt.xticks([], size="18")
    plt.yticks(np.linspace(0.0, 100.0, num=6), size="18")
    plt.xlim((0, len(accuracies['macro_per_features']) + 2))
    plt.ylim((0.0, 100.0))

    legend_dic = {}
    for i, feature in enumerate(sorted(accuracies['macro_per_features'].keys(), key=lambda x: accuracies['macro_per_features'][x]['proposed'])):
        accuracy = accuracies['macro_per_features'][feature]
        x = i + 1
        for model, c, m, l in reversed((('proposed', 'g', 'o', 'Autologistic-Basic'),
                                        ('neighborhood', 'r', 'x', 'Neighborhood'),
                                        ('global_majority', 'b', '+', 'Global majority'))):
            dot = plt.scatter(x, accuracy[model] * 100, s=15, c=c, marker=m, label=l)
            if model not in legend_dic:
                legend_dic[model] = dot
    legends = [legend_dic['proposed'], legend_dic['neighborhood'], legend_dic['global_majority']]
    plt.legend(
        handles=legends,
        loc='lower right',
        # loc='upper left', # bbox_to_anchor=(1.0, 0.5),
        scatterpoints=1, fontsize=16)
    # plt.show()
    plt.savefig('autologistic_mvi.pdf', bbox_inches="tight")

def _format_percentage(value, with_plus=False):
    plus = '+' if with_plus else ''
    return ('{0:'+plus+'6.2f}').format(value*100)


if __name__ == '__main__':
    analyze_mvi_results(sys.argv[1])
