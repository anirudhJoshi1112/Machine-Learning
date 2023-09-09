import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv("Book1.csv")    #excel file import
df

#   area	age	bedroom	price
# 0	2600	20	3.0	550000
# 1	3000	15	4.0	565000
# 2	3200	18	NaN	610000
# 3	3600	30	3.0	595000
# 4	4000	 8	5.0	760000

import math
median_bedroom=math.floor(df.bedroom.median())
median_bedroom

# incomplete csv file (2.bedroom) we need to preprocess our data because data is always messy we need to handle all the errors before operation

df.bedroom=df.bedroom.fillna(median_bedroom)
df
# all set to train our model

reg=linear_model.LinearRegression()     #in this line we have created a class object with the name LinearRegression
reg.fit(df[['area','age','bedroom']],df.price)       #fit is used to train your data sets we can use required formla and perform the calculations

reg.coef_ #with the help of training data method we will check the coefficient 

reg.intercept_   #with the help of training data method we will check the intercept

#price = m1*area + m2*bedroom + m3*age + b
# here m1,m2,m3 are coefficient and b is intercept

reg.predict([[3000,40,3]])     # now predict the price with the help of independent variables (area,age,price)
