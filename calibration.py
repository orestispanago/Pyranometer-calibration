#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:24:36 2021

@author: orestis
"""

import pandas as pd
import matplotlib.pyplot as plt

vaisala = pd.read_csv("vaisala.txt", 
                      skiprows=1,
                      names=["day", "time", "min", "max", "average"],
                      parse_dates={'date': ['day', 'time']},
                      index_col="date",
                      sep='\t')


# plt.plot(vaisala.index ,vaisala["average"], label="vaisala")
vaisala["average"].plot(label="vaisala")
# plt.xticks(rotation=45)
plt.legend()
plt.show()