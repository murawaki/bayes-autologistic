import sys
import glob
import os
from scipy.stats import gaussian_kde
import numpy as np

from .utils import collect_autologistic_results
from .analyze_autologistic_param_results import _group_params_per_areas

def analyze_vh(dir_results_pah, mode="param"):
    path = os.path.join(dir_results_pah, mode + '/**/*.json')
    jsons = glob.glob(path, recursive=True)
    results = collect_autologistic_results(jsons)
    if len(results) == 0:
        return
    results = _group_params_per_areas(results)
    print_HDP(results)
    plot_vh(results)

def print_HDP(results):
    feats = {}
    for cat, data in results.items():
        for data_each in data:
            name = data_each["feature"]
            name = name.replace(" ", "_")

            vs = np.zeros(len(data_each["sampled_params"]))
            hs = np.zeros(len(data_each["sampled_params"]))
            vmin, vmax = calc_HDP_from_samples(data_each["sampled_params"]["v"])
            hmin, hmax = calc_HDP_from_samples(data_each["sampled_params"]["h"])

            missing, total = 0, len(data_each["estimate_results"])
            valsize = 0
            for lang in data_each["estimate_results"]:
                if lang["is_original"] is False:
                    missing += 1
                    valsize = len(lang["sampled_values_distributions"])
            feats[name] = {
                "v": data_each["values"]["v"],
                "h": data_each["values"]["h"],
                "vmin": vmin,
                "vmax": vmax,
                "hmin": hmin,
                "hmax": hmax,
                "vint": vmax - vmin,
                "hint": hmax - hmin,
                "coverage": 1.0 - (float(missing) / total),
                "valsize": valsize,
                "param": data_each,
            }
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(name,
                                                      _format_value(vmin),
                                                      _format_value(data_each["values"]["v"]),
                                                      _format_value(vmax),
                                                      _format_value(hmin),
                                                      _format_value(data_each["values"]["h"]),
                                                      _format_value(hmax)))
    print("Spearman's rho: <0.2: very weak, <0.4: weak, <0.6: moderate, <0.8: strong, <1.0: very strong")
    _plot_correlation(feats,
                      "vh",
                      "v",
                      "h",
                      r"$v_i$",
                      r"$h_i$",
                      xmax=0.03, ymax=0.6,
                      xticks=7, yticks=4,
                      textx=0.005, texty=0.04)
    _plot_correlation(feats,
                      "v-coverage",
                      "v",
                      "coverage",
                      r"$v_i$",
                      "coverage",
                      xmax=0.03, ymax=0.6,
                      xticks=7, yticks=4,
                      textx=0.005, texty=0.04)
    _plot_correlation(feats,
                      "h-coverage",
                      "h",
                      "coverage",
                      r"$h_i$",
                      "coverage",
                      xmax=0.025, ymax=0.6,
                      xticks=6, yticks=4,
                      textx=0.005, texty=0.04)
    _plot_correlation(feats,
                      "vint-coverage",
                      "vint",
                      "coverage",
                      r"95\% HPD interval for $v_i$",
                      "coverage",
                      xmax=0.050, ymax=0.6,
                      xticks=6, yticks=4,
                      textx=0.03, texty=0.4)
    _plot_correlation(feats,
                      "hint-coverage",
                      "hint",
                      "coverage",
                      r"95\% HPD interval for $h_i$",
                      "coverage",
                      xmax=0.020, ymax=0.6,
                      xticks=5, yticks=4,
                      textx=0.012, texty=0.4)
    _plot_correlation(feats,
                      "v-valsize",
                      "v",
                      "valsize",
                      r"$v_i$",
                      "\# of unique values",
                      xmax=0.03, ymax=20,
                      xticks=7, yticks=5,
                      textx=0.005, texty=15)
    _plot_correlation(feats,
                      "h-valsize",
                      "h",
                      "valsize",
                      r"$h_i$",
                      "\# of unique values",
                      xmax=0.025, ymax=20,
                      xticks=6, yticks=5,
                      textx=0.005, texty=15)
    _plot_correlation(feats,
                      "vint-valsize",
                      "vint",
                      "valsize",
                      r"95\% HPD interval for $v_i$",
                      "\# of unique values",
                      xmax=0.050, ymax=20,
                      xticks=6, yticks=5,
                      textx=0.03, texty=15)
    _plot_correlation(feats,
                      "hint-valsize",
                      "hint",
                      "valsize",
                      r"95\% HPD interval for $h_i$",
                      "\# of unique values",
                      xmax=0.020, ymax=20,
                      xticks=6, yticks=5,
                      textx=0.012, texty=15)

def _plot_correlation(feats, name, xkey, ykey, xlabel, ylabel,
                      xmax=0.03, ymax=0.6,
                      xticks=7, yticks=7,
                      textx=0.01, texty=0.05):
    from scipy.stats import spearmanr

    rho = spearmanr(_attr_list(feats, xkey), _attr_list(feats, ykey))
    print("{}\t{}\t{}".format(name, rho[0], rho[1]))
    
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rc('pdf', fonttype=42)
    matplotlib.rcParams["font.family"] = 'serif'
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    catlegend = {}

    plt.xlabel(xlabel, size="24")
    plt.ylabel(ylabel, size="24")
    plt.xticks(np.linspace(0.0, xmax, num=xticks), size="18")
    plt.yticks(np.linspace(0.0, ymax, num=yticks), size="18")
    plt.xlim((0.0, xmax))
    plt.ylim((0.0, ymax))
    plt.text(textx, texty, "$\\rho={:.3f}$".format(rho[0]), size="24")
    for feat in feats.values():
        dot = plt.scatter(feat[xkey], feat[ykey],
                          s=15,
                          c=feat["param"]['color'], marker=feat["param"]['mark'])
    plt.savefig(name + '.pdf', bbox_inches="tight")
    plt.clf()

def _attr_list(feats, attr):
    return np.array(list(map(lambda x: x[attr], feats.values())))
    
def _format_value(value):
    return ('{0:9.8f}').format(value)

def calc_HDP_from_samples(posterior_samples, credible_mass=0.95):
    # taken from https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval (assuming unimodality)
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)

def plot_vh(results, scatter=False):
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rc('pdf', fonttype=42)
    matplotlib.rcParams["font.family"] = 'serif'
    matplotlib.rc('pdf', fonttype=42)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    zm = []
    for cat, data in results.items():
        for data_each in data:
            name = data_each["feature"]
            name = name.replace(" ", "_")

            vs = data_each["sampled_params"]["v"]
            hs = data_each["sampled_params"]["h"]

            # fig, ax = plt.subplots()
            # plt.figure()
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1]) #fill the entire axis
            plt.xlabel(r"$h_i$", size="24")
            plt.ylabel(u"$v_i$", size="24")

            plt.xlim((0.0, 0.030))
            plt.ylim((0.0, 0.050))
            plt.xticks([0.0, 0.01, 0.02, 0.03], size="18")
            plt.yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05], size="18")
            val = np.vstack((hs, vs))
            kernel = gaussian_kde(val)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            _X, _Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
            positions = np.vstack([_X.ravel(), _Y.ravel()])
            Z = np.reshape(kernel(positions).T, _X.shape)
            a = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], vmax=80000)

            if scatter:
                ax.scatter(hs, vs, s=0.5, c="black", marker=",")

            # fig = plt.figure()
            # fig_coord = [0.2, 0.8, 0.25, 0.05]
            # cbar_ax = fig.add_axes(fig_coord)
            # plt.colorbar(a, cax=cbar_ax, ticks=[0.,Z.max()])
            cbaxes = fig.add_axes([0.6, 0.6, 0.01, 0.2]) #add axis for colorbar, define size
            ticker = matplotlib.ticker.FixedLocator([0, 40000, 80000])
            cb = fig.colorbar(a, cax=cbaxes, ticks=ticker) #make colorbar
            cb.ax.tick_params(labelsize=18)

            plt.legend()
            # plt.show()
            plt.savefig(name + '.pdf', bbox_inches="tight")
            plt.clf()
            zm.append(Z.max())
    print(max(zm))

if __name__ == '__main__':
    analyze_vh(sys.argv[1])
