import argparse
import os
import json

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pingouin as pg


from pathlib import Path


def main(result_dir_ref: str, result_dir_pp: str):


    # get absolut path of the result directory
    result_dir_ref = Path(Path.cwd() / result_dir_ref)
    result_dir_pp = Path(Path.cwd() / result_dir_pp)

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
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharey=True, sharex=True)
        fig.suptitle(f"Q-Q plots of {label}")

        # create q-q plot DICE
        pg.qqplot(ref['DICE'][ref.LABEL == label], ax=ax[0, 0])
        ax[0, 0].set_title(f"Dice coefficient before post-processing")


        pg.qqplot(pp['DICE'][pp.LABEL == label], ax=ax[0, 1])
        ax[0, 1].set_title(f"Dice coefficient after post-processing")

        # create q-q plot HDRFDST
        pg.qqplot(ref['HDRFDST'][ref.LABEL == label], ax=ax[1, 0])
        ax[1, 0].set_title(f"Hausdorff distance before post-processing")

        pg.qqplot(pp['HDRFDST'][pp.LABEL == label], ax=ax[1, 1])
        ax[1, 1].set_title(f"Hausdorff distance after post-processing")

        # modify appearance of plot
        plt.subplots_adjust( hspace = 0.5, wspace = 0.5)

        for axis in ax.flatten():
            axis.texts = []
            lines = axis.get_lines()
            lines[0].set_color('black')
            lines[0].set_markerfacecolor('None')
            lines[1].set_color('black')
            lines[1].set_linestyle('--')

            lines[2].set_color('black')
            lines[3].set_color('grey')
            lines[4].set_color('grey')

    # plt.show()


    # for label in data['LABEL'].unique():


    # conduct t-test
    stat_test = pd.DataFrame(columns=['LABEL','mean_diff_DICE','p_DICE','mean_diff_HDRFDST','p_HDRFDST'])

    for label in data['LABEL'].unique():
        # statistical test of Dice coefficient
        t_DICE, p_DICE = stats.ttest_rel(ref['DICE'][ref.LABEL == label], pp['DICE'][pp.LABEL == label])

        mean_DICE = np.mean(pp['DICE'][pp.LABEL == label] - ref['DICE'][ref.LABEL == label])
        mean_HDRFDST = np.mean(pp['HDRFDST'][pp.LABEL == label] - ref['HDRFDST'][ref.LABEL == label])

        # statistical test of Hausdorff distance
        # t_HDRFDST, p_HDRFDST = stats.ttest_rel(ref['HDRFDST'][ref.LABEL == label], pp['HDRFDST'][pp.LABEL == label])
        t_HDRFDST, p_HDRFDST = stats.wilcoxon(ref['HDRFDST'][ref.LABEL == label], pp['HDRFDST'][pp.LABEL == label])

        stat_test = stat_test.append({'LABEL': label,'mean_diff_DICE':mean_DICE, 'p_DICE': p_DICE, 'mean_diff_HDRFDST':mean_HDRFDST,  'p_HDRFDST': p_HDRFDST}, ignore_index=True)

    print(stat_test)
    print('end main')



if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--result_dir_ref',
        type=str,
        default='./mia-result/Best_Values/no_PP_best/',
        help='Path to the result dir without post-processing.'
    )

    parser.add_argument(
        '--result_dir_pp',
        type=str,
        default='./mia-result/Best_Values/with_PP_best/',
        help='Path to the result dir with post-processing.'
    )


    args = parser.parse_args()
    main(args.result_dir_ref, args.result_dir_pp)