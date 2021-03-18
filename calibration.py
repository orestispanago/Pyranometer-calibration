import pandas as pd
import matplotlib.pyplot as plt

vaisala = pd.read_csv("vaisala.txt", 
                      skiprows=1,
                      names=["day", "time", "min", "max", "average"],
                      parse_dates={'date': ['day', 'time']},
                      index_col="date",
                      sep='\t')
vaisala.index = vaisala.index.round("1min")
vaisala = vaisala.tz_localize("EET").tz_convert('UTC')
vaisala["average_volts"] = vaisala["average"]/100



cmp11 = pd.read_csv("Solar_1min_db_store.txt", 
                    names=["datetime", "GHI_volts"], 
                    usecols=[0,6],
                    parse_dates=True,
                    index_col="datetime")

cmp11 = cmp11.tz_localize("UTC")
cmp11["GHI_volts"] = cmp11["GHI_volts"]
merged = pd.concat([vaisala, cmp11], axis=1)


vaisala["average_volts"].plot(label="vaisala")
cmp11["GHI_volts"].plot(label="cmp11")
plt.legend()
plt.show()