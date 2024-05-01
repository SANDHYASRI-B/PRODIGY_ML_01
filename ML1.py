# Predicting the prices of housees using LINEAR REGRESSION model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('/content/House Price India.csv')
df.columns
df.rename(columns={"number of bedrooms":"No.of Bedrooms",'number of bathrooms':'No.of Bathrooms',
                   'Area of the house(excluding basement)':'Area of the house',
                   'number of floors':'No.of Floors','Number of schools nearby':'No.of Schools nearby'}
          ,inplace=True)

df1=df[['No.of Bedrooms', 'No.of Bathrooms','living area', 'lot area', 'No.of Floors',
       'condition of the house','Area of the house','No.of Schools nearby', 'Price']]
df1=df1.iloc[:2000,:]
print(df1.head())

# Summary statistics of the dataset
print(df1.info())

# Correlation matrix for feature extraction
correlation_matrix = df1.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Selecting features and target variable
X = df[['No.of Bedrooms', 'No.of Bathrooms', 'lot area', 'No.of Floors',
       'condition of the house','Area of the house','No.of Schools nearby','Price']]
y = df[['living area']]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Building the Linear Regression Model
model = LinearRegression()

# Fitting the model on the training data
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
# Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Accuracy:", r2*100)

# Predictions and Visualization
# To visualize the predictions against actual prices using scatter plot
plt.scatter(y_test, y_pred,marker="*",c='mediumturquoise',s=12)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices VS Predicted Prices")
plt.show()
