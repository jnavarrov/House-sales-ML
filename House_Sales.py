import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)

print(df.describe())
print(df.head())
print(df.dtypes)

df.drop(["id","Unnamed: 0"],axis=1,inplace=True)
print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# EXPLORATORY DATA ANALYSIS 3.0

print(df["floors"].value_counts().to_frame())
# BOX PLOT
sns.boxplot(df["waterfront"], df["price"])
plt.ylim(0)
# SCATTER PLOT
sns.regplot(df["sqft_above"], df["price"])
plt.ylim(0)
# CORRELATION
df.corr()['price'].sort_values()

# MODEL DEVELOPMENT 4.0
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

# LINEAR REGRESSION
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
print ("The LR R^2 is: ", lm.score(X, Y))

# MULTIPLE LINEAR REGRESSION
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X = df[features]
Y = df["price"]
lm1 = LinearRegression()
lm1
lm1.fit(X, Y)
print ("The MLR R^2 is: ", lm1.score(X, Y))

# POLYNOMIAL REGRESSION & PIPELINES
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X, Y)
pipe.score(X, Y)

# MODEL EVALUATION AND REFINEMENT
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print("done")

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# RIDGE REGRESSION
from sklearn.linear_model import Ridge
rr=Ridge(alpha=0.1)
rr.fit(x_train, y_train)
print("Ridge R^2 Result is:", rr.score(x_test, y_test))

# RIDGE REGRESSION USING TRANSFORM METHOD
Poly = PolynomialFeatures(degree=2)

x_train_poly = Poly.fit_transform(x_train)
x_test_poly = Poly.fit_transform(x_test)

Rige = Ridge(alpha=0.1)
Rige.fit(x_train_poly, y_train)
print("Ridge + Transform Result is:", Rige.score(x_test_poly, y_test))







