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


cmp11 = pd.read_csv("cmp11.txt", 
                     names=["TIMESTAMP","RECORD","GHI_final","DNI_final",
                            "DHI_final","Sensor_Temp_Avg","LoggerVoltage_Min",
                            "LoggerTemp_C_Max","Error_Max"],
                     parse_dates=True,
                     index_col="TIMESTAMP")
cmp11 = cmp11.tz_localize("UTC")

merged = pd.concat([vaisala, cmp11], axis=1)


# plt.plot(vaisala.index ,vaisala["average"], label="vaisala")
vaisala["average"].plot(label="vaisala")
cmp11["GHI_final"].plot(label="cmp11")
# plt.xticks(rotation=45)
plt.legend()
plt.show()