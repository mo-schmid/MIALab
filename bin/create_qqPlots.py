import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg


from pathlib import Path


def main(result_dir_ref: str, result_dir_pp: str, plot_dir: str):
    """Generates qq-Plots of the testing results

    Args:
        result_dir_ref (Path): path to the reference data folder (without post-processing)
        result_dir_pp (Path): path to the data folder with post-processing
        plot_dir (Path): path to the desired result folder to store the qq-plots

    """

    # get absolut path of the result directory
    result_dir_ref = Path(Path.cwd() / result_dir_ref)
    result_dir_pp = Path(Path.cwd() / result_dir_pp)
    plot_dir = Path(Path.cwd() / plot_dir)

    # load the data into pandas
    ref = pd.read_csv(Path(result_dir_ref / 'results.csv'), sep=';')
    pp =  pd.read_csv(Path(result_dir_pp / 'results.csv'), sep=';')

    results = pd.concat([ref, pp])

    # get two dataframes with reference an post-processed values
    ref = results[['SUBJECT','LABEL','DICE', 'HDRFDST']][~results['SUBJECT'].str.contains('PP', na= False)].sort_values(by=['SUBJECT','LABEL'])
    pp = results[['SUBJECT','LABEL','DICE', 'HDRFDST']][results['SUBJECT'].str.contains('PP', na= False)].sort_values(by=['SUBJECT','LABEL'])

    # build data frame with differences in the metrics
    data = ref[['SUBJECT', 'LABEL']]
    data[['DIF_DICE', 'DIF_HDRFDST']] = ref[['DICE', 'HDRFDST']] - pp[['DICE', 'HDRFDST']]

    for label in data['LABEL'].unique():

        #create a subfigure per label wit the two qq plots
        # fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
        fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
        fig.suptitle(f"Q-Q plots of {label}")

        # create q-q plot DICE
        pg.qqplot(ref['DICE'][ref.LABEL == label], ax=ax[0, 0])
        ax[0, 0].set_title(f"Dice coefficient before post-processing")

        pg.qqplot(pp['DICE'][pp.LABEL == label], ax=ax[0, 1])
        ax[0, 1].set_title(f"Dice coefficient after post-processing")

        pg.qqplot(pp['DICE'][pp.LABEL == label] - ref['DICE'][ref.LABEL == label], ax=ax[0, 2])
        ax[0, 2].set_title(f"Difference in Dice coefficient")

        # create q-q plot HDRFDST
        pg.qqplot(ref['HDRFDST'][ref.LABEL == label], ax=ax[1, 0])
        ax[1, 0].set_title(f"Hausdorff distance before post-processing")

        pg.qqplot(pp['HDRFDST'][pp.LABEL == label], ax=ax[1, 1])
        ax[1, 1].set_title(f"Hausdorff distance after post-processing")

        pg.qqplot(pp['HDRFDST'][pp.LABEL == label] - ref['HDRFDST'][ref.LABEL == label], ax=ax[1, 2])
        ax[1, 2].set_title(f"Difference in Hausdorff distance")

        # modify appearance of plot
        plt.subplots_adjust( hspace = 0.5, wspace = 0.5)

        for axis in ax.flatten():
            axis.set_xlim([-1.6, 1.6])
            axis.set_ylim([-1.6, 1.6])
            axis.texts = []
            lines = axis.get_lines()
            lines[0].set_color('black')
            lines[0].set_markerfacecolor('None')
            lines[1].set_color('black')
            lines[1].set_linestyle('--')

            lines[2].set_color('black')
            lines[3].set_color('grey')
            lines[4].set_color('grey')

        plt.savefig(Path(plot_dir/ label))
        plt.close()

    # plt.show()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--result_dir_ref',
        type=str,
        default='./mia-result/gridsearch_PKF/2020-12-11-09-51-54/no_PP/',
        help='Path to the result dir without post-processing.'
    )

    parser.add_argument(
        '--result_dir_pp',
        type=str,
        default='./mia-result/gridsearch_PKF/2020-12-11-09-51-54/with_PP/PP-V-20_0-BG-True/',
        help='Path to the result dir with post-processing.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./mia-result/plot_results/QQ_plots/',
        help='Path to the plot directory.'
    )



    args = parser.parse_args()
    main(args.result_dir_ref, args.result_dir_pp, args.plot_dir)