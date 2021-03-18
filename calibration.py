import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant

vaisala = pd.read_csv("vaisala.txt", 
                      skiprows=1,
                      names=["day", "time", "average"],
                      usecols=[0,1,4],
                      parse_dates={'date': ['day', 'time']},
                      index_col="date",
                      sep='\t')
vaisala.index = vaisala.index.round("1min")
vaisala = vaisala.tz_localize("EET").tz_convert('UTC')
vaisala["vaisala_millivolts"] = vaisala["average"]/100

cmp11 = pd.read_csv("Solar_1min_db_store.txt", 
                    names=["datetime", "CMP11_millivolts"], 
                    usecols=[0,6],
                    parse_dates=True,
                    index_col="datetime")

cmp11 = cmp11.tz_localize("UTC")
merged = pd.concat([vaisala, cmp11], axis=1).dropna()

# vaisala["average_millivolts"].plot(label="vaisala")
# cmp11["GHI_volts"].plot(label="cmp11")
# plt.legend()
# plt.show()



def plot_all(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    plt.plot(x,y,".")
    plt.xlabel("CMP11")
    plt.ylabel("Vaisala")
    plt.plot(x, intercept + slope*x, 'r', label=f'{slope:.2f} x + {intercept:.2f}')
    plt.title("Pyranometer voltage (mV)")
    plt.legend()
    plt.show()



def calc_outliers(df):
    resids = df["vaisala_millivolts"] - df["CMP11_millivolts"]
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    df["good"] = (resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr))
    good = df[df["good"] == True]
    bad = df[df["good"] == False]
    return good, bad


def plot_regression_and_residuals(good, bad):
    y = good["vaisala_millivolts"]
    x = good["CMP11_millivolts"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    plt.plot(x,y, ".")
    plt.plot(bad["CMP11_millivolts"],bad["vaisala_millivolts"], ".", label="Outliers")
    plt.plot(x, intercept + slope*x, 'r', label=f'{slope:.2f} x + {intercept:.2f}')
    plt.title("Pyranometer voltage (mV)")
    plt.xlabel("CMP11")
    plt.ylabel("Vaisala")
    plt.legend()
    plt.show()


def regression_results(x,y):  
    X = add_constant(x)  # include constant (intercept) in ols model
    mod = sm.OLS(y, X)
    return mod.fit()

def regression_results_to_df(results):
    reg_df = pd.concat([results.params, results.bse, results.pvalues], axis=1)
    reg_df.columns = ["Value", "Std. Error", "P-value"]
    reg_df.index = ["intercept", "slope"]
    return reg_df

plot_all(merged["CMP11_millivolts"], merged["vaisala_millivolts"])
good, bad = calc_outliers(merged)
plot_regression_and_residuals(good, bad)

y = good["vaisala_millivolts"]
x = good["CMP11_millivolts"]

regresults = regression_results(x, y)
regresults_df = regression_results_to_df(regresults)

print(regresults_df)
print("R-squared: ", regresults.rsquared)
