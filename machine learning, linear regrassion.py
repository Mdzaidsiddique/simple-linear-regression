# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:02:46 2021

@author: masoo
"""
#######################################
''' simple linear Regrassion'''
#######################################

import pandas as pd
wc_ac = pd.read_csv('C:/Users/masoo/Downloads/wc.at.csv')
wc_ac.head()
#eda
from sklearn.model_selection import train_test_split

train, test = train_test_split(wc_ac, test_size = 0.20)

import numpy as np
x = np.array(wc_ac.Waist).reshape(-1,1)
y = np.array(wc_ac.AT).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2,
                                                    random_state = 123)

from sklearn.linear_model import LinearRegression
#to crete the model
model = LinearRegression() #y = mx+c

#to train model / fitting the model

model.fit(X = x_train, y = y_train)

#to coefficient of x
model.coef_
# to intercept or 
model.intercept_

# equation of line, [y = 3.32x + (-206.07)]
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x = 'Waist',
           y = 'AT',
           data = wc_ac,
           kind = 'scatter')
sns.lineplot( x = 'Waist', y = 'AT', data = wc_ac)
plt.plot(x = 'Waist', y = 'AT')

y_pred = model.predict(x_test)

#to find accuracy of the model
model.score(X = x_test, y = y_test)

############################
############################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
auto = pd.read_csv('C:/Users/masoo/Downloads/auto-mpgdata.csv', header = None)
auto.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']

auto.isnull().sum()
auto.horsepower.fillna(auto.horsepower.median(), inplace = True)
auto = auto[['mpg','displacement','horsepower','weight','acceleration']]
auto.columns
auto.isnull().sum()


# y = sales
# x = all continious data
y = auto.mpg
x = auto.iloc[:,1:]
sns.pairplot(auto)

#polynomial transformation needed
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
x_transform = pd.DataFrame(poly.fit_transform(x.iloc[:,:-1]))
x_all = pd.concat([x_transform, x.iloc[:,3]],axis = 1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_all,y,test_size=0.2,random_state =123)
from sklearn.linear_model import LinearRegression
model = LinearRegression() #y = mx+c

#to train model / fitting the model

model.fit(X = x_train, y = y_train)
model.coef_
# to intercept or 
model.intercept_

y_pred = model.predict(x_test)

#to find accuracy of the model
model.score(X = x_test, y = y_test)

plt.scatter(y = auto.mpg, x = auto.iloc[:,1:])



#######################################
#######################################

#RMSC # root mean square

train_pred = mul_reg.predict(x_train)
train_actual = y_train.copy()
residuals = train_actual - train_pred
print(residuals)
# root mean square
pred_p_train = poly_model.predict(X_train_transform)
pred_p_test = poly_model.predict(X_train_transform)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, train_pred, squared = False)

#######################################
#######################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

reg = pd.read_excel('C:/Users/masoo/Downloads/Regression.xlsx')
reg.shape
reg.isnull().sum()
reg.columns
plt.boxplot(reg)
plt.boxplot(reg.Expansion)
reg.Expansion.describe()
#good to go
sns.pairplot(reg)
# from pairplot we can see that the graph is curved thta means parabolic equation, polynomial equation, y = ax^2 +bx+ c, 
#we will have to preprocess data
x = np.array(reg.Kelvin).reshape(-1,1)
y = np.array(reg.Expansion).reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X=x_train, y = y_train)
model.coef_
model.intercept_
# y = 0.02168 x + 0.6624
#accuracy
model.score(x_test, y_test)
#prediction
predict = model.predict(x)

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.plot(reg.Kelvin,predict, "r")

#RMSC | root mean square for polynomial regaression
pred_l_train = model.predict(x_train)
pred_l_test = model.predict(x_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, pred_l_train, squared = False) # 3.2025338453612773

# y = ax^2 +bx+ c, parabolic equation, polynomial equation
#convert our train x to polynomial x
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
x_train_transform = poly.fit_transform(x_train)
from sklearn.linear_model import LinearRegression
poly_model = LinearRegression()
poly_model.fit(x_train_transform, y_train)

#score
x_test_transform = poly.fit_transform(x_test)
poly_model.score(x_test_transform, y_test)
#to get prediction for graph line
x_transform = poly.fit_transform(x)
pred = poly_model.predict(x_transform)

#ploting or graph
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.plot(reg.Kelvin, pred, "g_")

pred_p_train = poly_model.predict(x_train_transform)
pred_p_test = poly_model.predict(x_test_transform)

#RMSC | root mean square for polynomial regaression
from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, pred_p_train, squared= False)  #3.2025338453612773


########################################
########################################

#########################################
'''multy linear regaression'''
#########################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

auto_mpg = pd.read_csv('C:/Users/masoo/Downloads/auto-mpgdata.csv',
                       names=['mpg','cylinder','displacement','horsepower',
                              'weight','acceleration','model year','origin','car name']
, header=None)
auto_mpg.head()
auto_mpg.columns
#columns
1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)

auto_mpg.describe()
auto_mpg.info()
auto_mpg.isnull().sum()
auto_mpg.horsepower.median()
auto_mpg.horsepower.unique()
auto_mpg.horsepower.fillna(auto_mpg.horsepower.median(), inplace = True)
auto_mpg.isnull().sum()
auto_mpg.drop('car name', inplace = True, axis = 1)

sns.boxplot(data = auto_mpg)
plt.xticks( rotation = 40)
plt.show()
# y = mpg
# x = 'mpg','cylinder','displacement','horsepower','weight','acceleration','model year','origin','car name

from sklearn.model_selection import train_test_split
#for multylinear regression we dont need to reshape the data
y = np.array(auto_mpg.mpg)
x = np.array(auto_mpg.iloc[:,1:])

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.20, random_state = 123)

#model creation
from sklearn.linear_model import LinearRegression
mul_reg = LinearRegression()

mul_reg.fit(X = train_x, y = train_y)
#model equation

mul_reg.coef_
mul_reg.intercept_
#equation, y = [-0.3989072 ,  0.01918989,  0.00491233, -0.00700489,  0.24086977,0.75980713,  1.63987083]x + (-21.81)

#model accuracy
mul_reg.score(test_x, test_y) # 0.788143


 cylinders = [8,6,4,6]
 displacement= [500,390,440,414]
 horsepower= [90,110,56,70]
 weight= [3555,4321,3456,2345]
 acceleration= [10.5,11.2,5.6,7.9]
 model year= [70,73,75,70]
 origin= [1,1,2,3]
 
to_pred = {'cylinders':[8,6,4,6], 'displacement': [500,390,440,414], 'horsepower' : [90,110,56,70], 'weight': [3555,4321,3456,2345],
          'acceleration': [10.5,11.2,5.6,7.9], 'model year': [70,73,75,70], 'origin' : [1,1,2,3]}

to_predict = pd.DataFrame(to_pred)

#to predict new mpg value
mul_reg.predict(to_predict)


#######################################
'''AUTO REGERESSION'''
#######################################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
auto = pd.read_csv('C:/Users/masoo/Downloads/imports-85.data', header = None, na_values='?')

auto.columns = ['symboling','normalized_losses','make','fuel_types','aspiration','num_of_doors',
                'body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight',
                'engine_type','no_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio',
                'horsepower','peak_rpm','city_mpg','highway_mpg','price']
auto.columns
auto.shape
auto.nunique()
auto.isnull().sum()
auto.drop(['fuel_types','aspiration','body_style', 'drive_wheels','symboling', 'normalized_losses',
           'engine_location','engine_type','no_of_cylinders','fuel_system','num_of_doors','make'], inplace = True, axis = 1)
auto.info()
auto.price.isnull().sum()
auto.dropna(inplace = True)
auto.price.isnull().sum()
df = auto.select_dtypes(exclude = [object])
df.info()
auto.shape
df.boxplot()
plt.xticks(rotation = 40)
plt.show()
​
# y = sales
# x = all continious data
y = df.price
x = df.iloc[:,0:-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =30)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(X = x_train, y = y_train)
#to coefficient of x
mlr.coef_
# to intercept or 
mlr.intercept_
mlr.score(x_test, y_test)

#now, to predict

to_pred = {'wheel_base' : [88.5,99.5,102.6,105.5], 
           'length': [169.8,177.5,140.6,153.6], 
           'width' : [64.8,66.5,70.5,68.5],
           'height' : [49.5,50.2,60.2,48.5], 
           'curb_weight' : [2452,2645,3752,3125],
           'engine_size' : [130,140,144,156],
           'bore' : [3.47,2.5,3.6,4.7], 
           'stroke':[2.6,3.4,4.2,2.9],
           'compression_ratio' : [9,10,12,14],
           'horsepower' : [111,123,143,156],
           'peak_rpm' : [5000,4500,5500,6000],
           'city_mpg': [21,34,22,45],
           'highway_mpg':[33,44,39,50]}

To_pred = pd.DataFrame(to_pred)
To_pred.shape
type(To_pred)
mlr.predict(To_pred)

#residuals
train_pred = mlr.predict(x_train)
train_actual = y_train.copy()
residuals = train_actual - train_pred

#RMSE Root Mean Square
#basic method
from math import sqrt
sqrt(np.mean(residuals**2))
#Alternate method
from sklearn.metrics import mean_squared_error
mean_squared_error(train_actual, train_pred, squared = False)
sns.pairplot(df)


#######################################
'''bigmart data'''
#######################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
bigmart = pd.read_csv('C:/Users/masoo/Downloads/Bigmart.csv')
bigmart.shape
bigmart.head()
bigmart.columns
bigmart.isnull().sum()
bigmart.drop(['Item_Identifier', 'Item_Weight','Outlet_Identifier','Outlet_Size',
              'Outlet_Location_Type','Outlet_Type'], inplace = True, axis = 1)
bigmart.Item_Fat_Content.unique()
bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace('low fat','LF').replace('LF', 'Low Fat').replace('reg', 'Regular')
bigmart.Item_Type.unique()
bigmart.Item_Visibility = bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median())

dummy_variables = pd.get_dummies(bigmart, columns = ['Item_Fat_Content','Item_Type'])
dummy_variables.shape
dummy_variables.info()
Sbigmart = pd.concat()
bigmart.info()

sns.pairplot(bigmart)

y = bigmart.Item_Outlet_Sales
x = bigmart.iloc[:,0:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 123)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X= x_train, y = y_train)


#######################################################
'''AVOCADO REGRESSION'''
#####################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/masoo/Downloads/Avocado.csv', na_values = '?')
data.columns
data.head()
data.isnull().sum()
data.info()
data.drop(['Unnamed: 0','Date'], axis = 1, inplace = True)
data.drop(['year'], inplace = True, axis = 1)
data['region'].unique()
data['type'].unique()
data.rename(columns = {'4046':'sales_s/m','4225':'sales_l' , '4770':'sales_xl'}, inplace = True)
data.drop(['type','region'], inplace = True, axis = 1)
data.info()
data.describe()
plt.boxplot(data.AveragePrice)
y = data['AveragePrice']
x = data.iloc[:,1:]
x.columns
x.shape
y.shape
sns.pairplot(data = data)

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,Y_test = train_test_split(x,y,test_size = 0.20, random_state= 123)
from sklearn.linear_model import LinearRegression
mul_reg = LinearRegression()
mul_reg.fit(X= x_train, y = y_train)
mul_reg.coef_
mul_reg.intercept_
#equation , y = (0.00015442, -0.00015453, -0.00015434, -0.00015488, -0.03541024,0.03525586,  0.03525571,  0.03525704)x + .4222889485182275

mul_reg.score(x_test,Y_test) #.4222889485182275
pred = mul_reg.predict(x_test)
print(pred)

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(Y_test, pred))
metrics.mean_absolute_error(Y_test, pred)

#The RMSE is low so we can say that we do have a good model, but lets check to be more sure.
#Lets plot the y_test vs the predictions

plt.scatter(x = Y_test, y = pred)

#As we can see that we don't have a straight line so I am not sure that this is the best model we can apply on our data

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X = x_train, y = y_train)
predd = dtr.predict(x_test)
plt.scatter(x = Y_test, y = pred)
sns.lmplot(x = Y_test, y = predd,data=data,palette='rainbow')



#########################################################
'''CANCER CLASSIFIER'''
########################################################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

cancer = pd.read_csv('C:/Users/masoo/Downloads/breast-cancer-wisconsin.data', header=None, na_values='?')
cancer.head()
cancer.columns
cancer.columns   =["sample_code","thick","uni_cell_size","uni_cell_shape","mar_adh","Epi_cell_size","Bare_nuclei","bland_chro","normal_nuc","mitoses","Cancer_class"]  
cancer.shape
cancer.isnull().sum()
cancer.Bare_nuclei.fillna(cancer.Bare_nuclei.mean(),inplace = True)
cancer.info()
plt.boxplot(cancer)
cancer.sample_code.describe()
cancer.Cancer_class.unique()
cancer.Cancer_class= cancer.Cancer_class.replace(2, 'benign').replace(4, 'malignant')

x = cancer.iloc[:,:-1]
x.shape
x.columns
y = cancer.Cancer_class
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state = 123)

#KNeighborsClassifier
knmodel = KNeighborsClassifier(18)
knmodel.fit(X = x_train, y = y_train)
knmodel.score(x_test,y_test) #0.5785714285714286
pred = knmodel.predict(x_test)
#confusion metrics
confusion_matrix(y_test, pred)

# max k value = no of obs of train data
# k = sqrt(n/2) best practice
from math import sqrt
print(sqrt(699/2))

#for loop for best k value
result = pd.DataFrame(columns = [ "k", "score_test", "score_train"])
for k in range(1, 699):
    knmodel = KNeighborsClassifier(k)
    knmodel.fit(x_train,y_train)
    knmodel.score(x_test,y_test)
    result = result.append({ "k" : k, "score_test" : knmodel.score(x_test,y_test) , "score_train" :knmodel.score(x_train,y_train)  },ignore_index=True)
plt.plot(result.k,result.score_test)
plt.plot(result.k,result.score_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
Dtmodel = DecisionTreeClassifier()
Dtmodel.fit(X = x_train, y = y_train)
Dtmodel.score(x_test, y_test)
Dtmodel.score(x_train, y_train)
pred = Dtmodel.predict(x_test)
confusion_matrix(y_test, pred)

#displaying the decision tree
from sklearn import tree
tree.plot_tree(Dtmodel)


#ensemble learning - Together of many trees
from sklearn.ensemble import RandomForestClassifier
#random forest
rfmodel = RandomForestClassifier()
rfmodel.fit(X = x_train, y = y_train)
rfmodel.score(x_test, y_test)
pred = rfmodel.predict(x_test)
confusion_matrix(y_test, pred)

from sklearn.ensemble import GradientBoostingClassifier as GB
gbmodel = GB()
gbmodel.fit(x_train,y_train)
gbmodel.score(x_test,y_test) #testing accuracy
gbmodel.score(x_train,y_train) #training accuracy

#xtreme Gredient Boosting 
from xgboost import XGBClassifier
xgmodel = XGBClassifier()
xgmodel.fit(X_train,y_train)
xgmodel.score(X_test,y_test) #testing accuracy
xgmodel.score(X_train,y_train)

help(XGBClassifier)

#adaboost - adaptive boosting

from sklearn.ensemble import AdaBoostClassifier
abmodel = AdaBoostClassifier()
abmodel.fit(x_train,y_train)
abmodel.score(x_test,y_test) #testing accuracy
abmodel.score(x_train,y_train) #training accuracy





################################################
##############################################




























