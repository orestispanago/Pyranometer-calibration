import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import statsmodels.api as sm
import mysql.connector


def select(start_date, end_date):  
    mydb = mysql.connector.connect(
        host="173.249.63.213",
        port="3360",
        user="ReadOnlyUser",
        passwd="ReadOnlyPassword",
        database="collector"
        )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT time, irradiance FROM pyranometer \
                     WHERE Time BETWEEN %s AND %s", 
                     (start_date, end_date))
    data = pd.DataFrame(mycursor.fetchall())
    data.columns = mycursor.column_names
    mydb.close()
    return data


def get_data(day):
    friedrichs = select(day, day+ pd.Timedelta(days=1))
    friedrichs = friedrichs.rename(columns={"irradiance":"Friedrichs_mV"})
    friedrichs.set_index("time", inplace=True)
    friedrichs.index = friedrichs.index.round("1min")
    friedrichs = friedrichs.tz_localize("UTC")
    cmp11 = pd.read_csv("Solar_1min_db_store.txt",
                        names=["datetime", "CMP11_mV"],
                        usecols=[0,6],
                        parse_dates=True,
                        index_col="datetime")
        
    cmp11 = cmp11.loc[day: day+ pd.Timedelta(days=1)-pd.Timedelta(minutes=1)]
    cmp11 = cmp11.tz_localize("UTC")
    merged = pd.concat([friedrichs, cmp11], axis=1).dropna()
    return merged    

def plot_raw_mV():
    merged['CMP11_mV'].plot(label="CMP11")
    merged["Friedrichs_mV"].plot(label="Friedrichs")
    plt.ylabel("Voltage $(mV)$")
    plt.legend()
    plt.savefig(f"results/{date}/raw_mV.png")

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
    resids = df["Friedrichs_mV"] - df["CMP11_mV"]
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    df["good"] = (resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr))
    good = df[df["good"] == True]
    bad = df[df["good"] == False]
    return good, bad

def plot_regression_and_outliers(good, bad):
    y = good["Friedrichs_mV"]
    x = good["CMP11_mV"]
    slope = regression_results(x, y).params[0]
    plt.plot(x,y, ".")
    plt.plot(bad["CMP11_mV"],bad["Friedrichs_mV"], ".", label="Outliers")
    plt.plot(x, slope*x, 'r', label=f'y = {slope:.2f} x')
    plt.title("Pyranometer voltage (mV)")
    plt.xlabel("CMP11")
    plt.ylabel("Friedrichs")
    plt.legend()
    plt.savefig(f"results/{date}/regression.png")
    plt.show()

def regression_results(x,y):
    mod = sm.OLS(y, x)
    return mod.fit()

def results_to_df(date, regresults):    
    results = pd.DataFrame(index=[date])
    results["Slope"] = regresults.params[0]
    results["Slope-err"] = regresults.bse[0]
    results["P-Value"] = regresults.pvalues[0]
    results["R-Squared"] = regresults.rsquared
    results["Outliers"] = len(bad)
    results["Sensitivity ??V/(W/m2)"] = cmp11_sensitivity * regresults.params[0]
    return results.T

def calc_wm2_cor(regresults, print_them=True):
    friedrichs_sensitivity = cmp11_sensitivity * regresults.params[0]
    friedrichs_constant = 1000 / friedrichs_sensitivity
    merged["CMP11_W/m2"] = merged["CMP11_mV"] * cmp11_constant
    merged["friedrichs_W/m2_cor"] = merged["Friedrichs_mV"] * friedrichs_constant
    if print_them:        
        print(f"Friedrichs sensitivity: {friedrichs_sensitivity:.2f} ??V/(W/m2)")
        # print(f"Irradiance in W/m2 = mV * 1000 / {friedrichs_sensitivity:.2f}")

def plot_wm2_cor():
    merged['CMP11_W/m2'].plot(label="CMP11")
    merged["friedrichs_W/m2_cor"].plot(label="Friedrichs")
    plt.ylabel("Irradiance $(W/m^2)$")
    plt.legend()
    plt.savefig(f"results/{date}/wm2_cor.png")
    plt.show()

def mkdir_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def report_to_html(date, regresults, merged):        
    results = results_to_df(date, regresults)
    statistics = pd.concat([results, merged.describe()], axis=1)
    statistics.fillna('', inplace=True)
    statistics.to_html(f"results/{date}.html")


cmp11_sensitivity = 8.63 # ??V
cmp11_constant = 1000 / cmp11_sensitivity # mV/(W/m2)

dates = ["2021-04-21", "2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25"]


for date in dates:
    print(date)
    mkdir_if_not_exists(f"results/{date}")
    date_time = pd.to_datetime(date, format='%Y-%m-%d')
    
    merged = get_data(date_time)
    plot_raw_mV()
    
    plot_scatter(merged["CMP11_mV"], merged["Friedrichs_mV"])
    good, bad = calc_outliers(merged)
    plot_regression_and_outliers(good, bad)
    
    regresults = regression_results(good["CMP11_mV"], good["Friedrichs_mV"])
    
    calc_wm2_cor(regresults)
    plot_wm2_cor()
    
    report_to_html(date, regresults, merged)
