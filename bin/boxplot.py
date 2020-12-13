import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, etc) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass is just a placeholder if there is no other code

    os.chdir('mia-result/Ubelix/gauss_dims')
    result_path = glob.glob('*/results.csv')[0]
    print(result_path)
    #seg_path = glob.glob('*/117122_SEG-PP.mha')[0]

    result = pd.read_csv(result_path, ";")
    boxplot = result.boxplot(column='DICE', by="LABEL")
    plt.show()
    boxplot = result.boxplot(column='HDRFDST', by="LABEL")
    plt.show()

if __name__ == '__main__':
    main()
