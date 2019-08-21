# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:16:47 2019

@author: d6290
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_csv('perrin-freres-monthly-champagne.csv')
df.drop(105, axis=0, inplace=True)
df.drop(106, axis=0, inplace=True)
df.columns=['Month','Sales Per Month']
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.plot()


model=sm.tsa.statespace.SARIMAX(df['Sales Per Month'], order=(1,0,0), seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales Per Month', 'forecast']].plot(figsize=(12,8))



from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
print(future_datest_df)

future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales Per Month', 'forecast']].plot(figsize=(12, 8))

