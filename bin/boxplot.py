import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass is just a placeholder if there is no other code


    result_folders = Path(Path.cwd() / 'mia-result')
    result_folders = [x for x in result_folders.iterdir() if x.is_dir()]

    # opens most recent results
    # results_pd = pd.read_csv(result_folders[-1] / 'results.csv', delimiter=';')
    results_pd = pd.read_csv(result_folders[-1] / 'results_composed.csv', delimiter=';')


    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot
    # results_pd.boxplot(by='LABEL', column='DICE')

    results_pd.boxplot(by='LABEL', column='DICE')

    results_pd.boxplot(by='LABEL', column='HDRFDST')
    plt.show()


if __name__ == '__main__':
    main()
