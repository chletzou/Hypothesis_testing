import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.weightstats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("./data.csv")
    # Subset the Data
    NTU = df.query('group == "NTU"')['score']
    NTNU = df.query('group == "NTNU"')['score']

    #Descriptive Statistics
    print(df.groupby('group').describe())

    #Checking the Normality of the Data
    print(stats.shapiro(NTU))
    print(stats.shapiro(NTNU))

    #Checking the Homogeneity of Variances Assumption
    print(stats.levene(NTU, NTNU))

    #Visualize the Data
    plt.title("boxplot")
    sns.boxplot(x='group', y='score', data=df)
    plt.show()
    #Result in Scipy
    res1 = stats.ttest_ind(NTU, NTNU, equal_var=True)
    print(res1)

    #Result in Pingouin
    res2 = pg.ttest(NTU, NTNU, correction=False)
    print(res2)

    #Result in Statsmodels
    res3 = ttest_ind(NTU, NTNU)
    print(res3)