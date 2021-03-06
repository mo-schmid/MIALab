import argparse
import os
import json

import numpy as np
import pandas as pd
from pathlib import Path


def main(result_dir: str):
    """evaluates the a gridsearch

    Finds the best parameter set by ranking all the parameter sets by the DICE coefficient and by the Hausdorff
    distance. Computes a mean rank per parameter set.

    Args:
        result_dir (Path): Path to gridsearch folder containing subfolders with results per parameter set
    """

    # get absolut path of the result directory
    result_dir = Path(Path.cwd() / result_dir)

    # create a list with the path to all subdirectories
    dir_list = [x for x in result_dir.iterdir() if x.is_dir()]

    # load all data in a data frame
    dfs = pd.DataFrame()
    methods = []
    for dir in dir_list:

        new_data = pd.read_csv(Path(dir / 'results_summary.csv'), sep=';')

        f = open(Path(dir / 'parameter.txt'))
        parameter = json.load(f)

        for key, value in parameter.items():
            new_data.insert(new_data.shape[1], key, np.repeat(value, new_data.shape[0]), allow_duplicates=True)

        dfs = dfs.append(new_data, ignore_index=True)


    # rank the data by dice coefficient
    group_DICE = dfs[dfs.STATISTIC == 'MEAN'][dfs.METRIC == 'DICE'].groupby(list(parameter.keys()))[
        'VALUE'].mean()
    group_DICE = group_DICE.sort_values(ascending=False)
    group_DICE = pd.DataFrame(group_DICE)
    group_DICE.insert(group_DICE.shape[1], 'RANK_DICE', np.arange(group_DICE.shape[0]))

    # rank data by hausdorff distance
    group_HDRFDST = dfs[dfs.STATISTIC == 'MEAN'][dfs.METRIC == 'HDRFDST'].groupby(list(parameter.keys()))['VALUE'].mean()
    group_HDRFDST  = group_HDRFDST.sort_values(ascending=True)
    group_HDRFDST = pd.DataFrame(group_HDRFDST)
    group_HDRFDST.insert(group_HDRFDST.shape[1], 'RANK_HDRFDST', np.arange(group_HDRFDST.shape[0]))

    # compute mean rank
    combined = pd.concat([group_DICE, group_HDRFDST], axis=1)
    combined['MEAN_RANK'] = combined[['RANK_DICE', 'RANK_HDRFDST']].mean(axis=1)
    combined = combined.sort_values(by='MEAN_RANK', ascending=True)

    print("Best parameters")
    for i, string in enumerate(combined.iloc[0].name):
        print(f"{list(parameter.keys())[i]} : {string}")




if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')


    # # result folder of gridsearch of random forest
    # parser.add_argument(
    #     '--result_dir',
    #     type=str,
    #     default='./mia-result/gridsearch_randomForest/combined_results',
    #     help='Path to the result dir.'
    # )

    # result folder of gridsearch of PKF
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./mia-result/gridsearch_PKF/2020-12-11-09-51-54/with_PP',
        help='Path to the result dir.'
    )


    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./mia-result/plot_results',
        help='Path to the plot directory.'
    )

    args = parser.parse_args()
    main(args.result_dir,)