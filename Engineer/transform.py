import pandas as pd
import numpy as np
import matplotlib.pyplot as ptl
import scipy.stats as stats
import pylab


# diagnostic
def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.show()