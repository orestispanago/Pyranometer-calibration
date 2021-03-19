import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant

cmp11_sensitivity = 8.63 # μV
cmp11_constant = 1000 / cmp11_sensitivity # mV/(W/m2)
meteon_sensitivity = 10
meteon_constant = 1000 / meteon_sensitivity

friedrichs = pd.read_csv("friedrichs.txt", 
                      skiprows=1,
                      names=["day", "time", "average"],
                      usecols=[0,1,4],
                      parse_dates={'date': ['day', 'time']},
                      index_col="date",
                      sep='\t')
friedrichs.index = friedrichs.index.round("1min")
friedrichs = friedrichs.tz_localize("EET").tz_convert('UTC')
friedrichs["friedrichs_mV"] = friedrichs["average"]/meteon_constant

cmp11 = pd.read_csv("Solar_1min_db_store.txt",
                    names=["datetime", "CMP11_mV"],
                    usecols=[0,6],
                    parse_dates=True,
                    index_col="datetime")

cmp11 = cmp11.tz_localize("UTC")
merged = pd.concat([friedrichs, cmp11], axis=1).dropna()



def plot_scatter(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    plt.plot(x,y,".")
    plt.xlabel("CMP11")
    plt.ylabel("Friedrichs")
    plt.plot(x, intercept + slope*x, 'r', label=f'{slope:.2f} x + {intercept:.2f}')
    plt.title("Pyranometer voltage (mV)")
    plt.legend()
    plt.show()



def calc_outliers(df):
    resids = df["friedrichs_mV"] - df["CMP11_mV"]
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    df["good"] = (resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr))
    good = df[df["good"] == True]
    bad = df[df["good"] == False]
    return good, bad


def plot_regression_and_residuals(good, bad):
    y = good["friedrichs_mV"]
    x = good["CMP11_mV"]
    slope = regression_results(x, y).params[0]
    plt.plot(x,y, ".")
    plt.plot(bad["CMP11_mV"],bad["friedrichs_mV"], ".", label="Outliers")
    plt.plot(x, slope*x, 'r', label=f'y = {slope:.2f} x')
    plt.title("Pyranometer voltage (mV)")
    plt.xlabel("CMP11")
    plt.ylabel("Friedrichs")
    plt.legend()
    plt.show()


def regression_results(x,y):
    # X = add_constant(x)  # include constant (intercept) in ols model
    mod = sm.OLS(y, x)
    return mod.fit()

# plot_scatter(merged["CMP11_mV"], merged["friedrichs_mV"])
good, bad = calc_outliers(merged)
plot_regression_and_residuals(good, bad)

y = good["friedrichs_mV"]
x = good["CMP11_mV"]

regresults = regression_results(x, y)

slope = regresults.params[0]
slope_error = regresults.bse[0]
pvalue = regresults.pvalues[0]
r2 = regresults.rsquared

print(f"Slope: {slope} ± {slope_error}")
print(r'$\pm$')
print(f"P-value: {pvalue}")
print("R-squared: ", r2)



friedrichs_sensitivity = cmp11_sensitivity * regresults.params[0]
friedrichs_constant = 1000 / friedrichs_sensitivity

print(f"Friedrichs sensitivity: {friedrichs_sensitivity:.2f} μV/(W/m2)")
print(f"Irradiance in W/m2 = mV * 1000 / {friedrichs_sensitivity:.2f}")

merged["CMP11_W/m2"] = merged["CMP11_mV"] * cmp11_constant
merged["friedrichs_W/m2_cor"] = merged["friedrichs_mV"] * friedrichs_constant
merged["CMP11_W/m2"].plot(label="CMP11")
merged["friedrichs_W/m2_cor"].plot(label="Friedrichs")
plt.ylabel("Irradiance $(W/m^2)$")
plt.legend()
plt.show()
