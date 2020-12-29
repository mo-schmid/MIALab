import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class ResultParam():
    """Result Parameter"""
    def __init__(self, path: Path, param_str: str):
        """Initializes a new instance of the Result Parameter

        Args:
            path (Path): path to the desired result file
            param_str (str): string containing the parameters used in the postprocessing

        """
        self.path = path
        self.param_str = param_str



def set_box_format(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['caps'], linewidth=1)
    plt.setp(bp['medians'], color='red')
    plt.setp(bp['medians'], linewidth=1.5)
    plt.setp(bp['fliers'], marker='.')
    plt.setp(bp['fliers'], markerfacecolor='black')
    plt.setp(bp['fliers'], alpha=1)


def boxplot(file_path: str, data: list, title: str, x_label: str, y_label: str, x_ticks: tuple,
            min_: float = None, max_: float = None):
    if len(data) != len(x_ticks):
        raise ValueError('arguments data and x_ticks need to have same length')

    fig = plt.figure(
        figsize=( 2 *1.5, 5*1.5))  # figsize defaults to (width, height) =(6.4, 4.8),

    # for boxplots, we want the ratio to be inversed
    ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
    bp = ax.boxplot(data, widths=0.6)
    set_box_format(bp, '000')

    # set and format litle, labels, and ticks
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=18)
    # ax.set_xlabel(x_label, fontweight='bold', fontsize=9.5)  # we don't use the x-label since it should be clear from the x-ticks
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticklabels(x_ticks, fontdict={'fontsize': 18, 'fontweight': 'bold'}, rotation=45)


    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # adjust min and max if provided
    if min_ is not None or max_ is not None:
        min_original, max_original = ax.get_ylim()
        min_ = min_ if min_ is not None and min_ < min_original else min_original
        max_ = max_ if max_ is not None and max_ > max_original else max_original
        ax.set_ylim(min_, max_)

    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def format_data(data, label: str, metric: str):
    return data[data['LABEL'] == label][metric].values


def metric_to_readable_text(metric: str):
    if metric == 'DICE':
        return 'Dice coefficient'
    elif metric == 'HDRFDST':
        return 'Hausdorff distance (mm)'
    else:
        raise ValueError('Metric "{}" unknown'.format(metric))


def main(results: [ResultParam], plot_dir: Path):
    metrics = ('DICE', 'HDRFDST')  # the metrics we want to plot the results for
    metrics_yaxis_limits = ((0.0, 1.0), (0.0, 18))  # tuples of y-axis limits (min, max) for each metric. Use None if unknown
    labels = ('WhiteMatter','GreyMatter', 'Hippocampus','Amygdala','Thalamus')  # the brain structures/tissues you are interested in


    # load the CSVs. We usually want to compare different methods (e.g. a set of different features), therefore,
    # we load two CSV (for simplicity, it is the same here)
    # todo: adapt to your needs to compare different methods (e.g. load different CSVs)
    dfs = []
    methods = []
    for res in results:
        dfs.append(pd.read_csv(res.path, sep=';'))
        methods.append(res.param_str)

    # todo: read parameter values from text file, use them to plot the information about the paramter


    # some parameters to improve the plot's readability
    title = '{}'

    for label in labels:
        for metric, (min_, max_) in zip(metrics, metrics_yaxis_limits):
            boxplot(os.path.join(plot_dir, '{}_{}.png'.format(label, metric)),
                    [format_data(df, label, metric) for df in dfs],
                    title.format(label),
                    'Method', metric_to_readable_text(metric),
                    methods,
                    min_, max_
                    )


if __name__ == '__main__':

    results = []

    results.append(ResultParam(Path(Path.cwd() / "mia-result\gridsearch_PKF/2020-12-11-09-51-54/no_PP/results.csv"),
                               "no pp"))
    results.append(ResultParam(Path(Path.cwd() /"mia-result/gridsearch_PKF/2020-12-11-09-51-54/with_PP/PP-V-20_0-BG-True/results.csv"),
                               "with pp"))



    main(results, Path(Path.cwd() / 'mia-result/plot_results'))
