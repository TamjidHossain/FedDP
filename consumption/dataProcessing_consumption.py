# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:57:14 2021

@author: tamji
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:12:41 2021

@author: mdtamjidhossain
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
sns.set()

from matplotlib import pyplot as plt
import datetime


data = pd.read_csv('D:/codeForPaper/consumption/household_power_consumption.csv', delimiter=',')
data.Date = pd.to_datetime(data.Date)
data.Time = pd.to_timedelta(data.Time)
data['DateTime']  = data.Date + data.Time
data = data[['DateTime', 'Global_active_power', 'Global_reactive_power',
       'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3']]
data.replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
data = data.dropna()

data["Global_active_power"] = pd.to_numeric(data["Global_active_power"], downcast="float")
data["Global_reactive_power"] = pd.to_numeric(data["Global_reactive_power"], downcast="float")
data["Voltage"] = pd.to_numeric(data["Voltage"], downcast="float")
data["Global_intensity"] = pd.to_numeric(data["Global_intensity"], downcast="float")
data["Sub_metering_1"] = pd.to_numeric(data["Sub_metering_1"], downcast="float")
data["Sub_metering_2"] = pd.to_numeric(data["Sub_metering_2"], downcast="float")
data["Sub_metering_3"] = pd.to_numeric(data["Sub_metering_3"], downcast="float")

# cols = list(data.columns)[1:]
# data[cols] = data[cols].apply(pd.to_numeric, errors='coerce', axis=1)

df_main = pd.DataFrame()
df_main = data.copy()

#%%

# extract month feature
months = data.DateTime.dt.month

# extract day of month feature
day_of_months = data.DateTime.dt.day

# extract hour feature
years = data.DateTime.dt.year

# extract hour feature
hours = data.DateTime.dt.hour

# first: extract the day name literal
to_one_hot = data.DateTime.dt.day_name()
# second: one hot encode to 7 columns
days = pd.get_dummies(to_one_hot)

# daypart function
def daypart(hour):
    if hour in [2,3,4,5]:
        return "dawn"
    elif hour in [6,7,8,9]:
        return "morning"
    elif hour in [10,11,12,13]:
        return "noon"
    elif hour in [14,15,16,17]:
        return "afternoon"
    elif hour in [18,19,20,21]:
        return "evening"
    else: return "midnight"
# utilize it along with apply method
dataframe_dayparts = hours.apply(daypart)
# one hot encoding
dayparts = pd.get_dummies(dataframe_dayparts)
# re-arrange columns for convenience
dayparts = dayparts[['dawn','morning','noon','afternoon','evening','midnight']]

# is_weekend flag 
day_names = data.DateTime.dt.day_name()
is_weekend = day_names.apply(lambda x : 1 if x in ['Saturday','Sunday'] else 0)



# features table
#first step: include features with single column nature
features = pd.DataFrame({
    'Global_reactive_power' : data.Global_reactive_power,
    'Voltage' : data.Voltage,
    'Global_intensity' : data.Global_intensity,
    'Sub_metering_1' : data.Sub_metering_1,
    'Sub_metering_2' : data.Sub_metering_2,
    'Sub_metering_3' : data.Sub_metering_3,
    'years' : years,
    'month' : months,
    'day_of_month' : day_of_months,
    'hour' : hours,
    'is_weekend' : is_weekend
})
#second step: concat with one-hot encode typed features
features = pd.concat([features, days, dayparts], axis = 1)
# target column
target = data.Global_active_power

#%%
# ##save it to csv

total =  pd.concat([features, data.Global_active_power], axis = 1)
total.to_csv('D:/codeForPaper/consumption/processed_household_power_consumption.csv',index = False)


