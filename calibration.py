import pandas as pd
import matplotlib.pyplot as plt

vaisala = pd.read_csv("vaisala.txt", 
                      skiprows=1,
                      names=["day", "time", "min", "max", "average"],
                      parse_dates={'date': ['day', 'time']},
                      index_col="date",
                      sep='\t')


reference = pd.read_csv("reference.txt", sep=" ", 
                        names=["day", "time", "a", "b", "c", "d", "e", "f", "g"],
                        parse_dates={'date': ['day', 'time']},
                        index_col="date")


# plt.plot(vaisala.index ,vaisala["average"], label="vaisala")
vaisala["average"].plot(label="vaisala")
# plt.xticks(rotation=45)
plt.legend()
plt.show()