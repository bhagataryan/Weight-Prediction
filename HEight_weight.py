import os 
os.getcwd()

#importing eda libraries
import numpy as np  #math
import pandas as pd #excellent for data manuplation

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing 
from sklearn.preprocessing import StandardScaler

#spliting the data
from sklearn.model_selection import train_test_split

# importing Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#evalution matrics
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"D:\FILES\NIT\24JAN-MYSQL conector, project\HEIGHT & WEIGHT\SOCR-HeightWeight.csv") #reading csv file


df.head()


#converting weight pounds to kg
df['Weight_kg']=df['Weight(Pounds)']*0.453592  


#Convert inches to the desired format (feet.inches) 
df['Height(Feet.Inches)'] = df['Height(Inches)'] // 12 + (df['Height(Inches)'] % 12) / 10


df.describe()

drop_col=['Index','Height(Inches)','Weight(Pounds)'] # selecting columns to del it

#droping columns
df = df.drop(columns=drop_col,axis=1)


df.sample(3) #it will give random row information


df.shape  #checking shape of the data

df.isna().any() #checking null values

df.dtypes #checking dtypes for our dataframe

df.corr() #correlation

x=df['Height(Feet.Inches)']
y=df['Weight_kg']

sns.scatterplot(x=x,y=y)
plt.title('Height vs Weight')
plt.xlabel('Height(Feet.Inches)')
plt.ylabel('Weight_kg')
plt.show()


sns.boxplot(x=df['Weight_kg']) #checking outliers for 

sns.boxplot(x=df['Height(Feet.Inches)'])

# split the data into dependent & independent variable 
X=df.iloc[:,1]
y=df.iloc[:,0]

X

df.columns[1]

df.columns[0]


#Data scaling(preprocessing data)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X.values.reshape(-1,1))


scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))


#spliting data into 80%-20% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print('Shape of trining data')
print(X_train.shape)
print(y_train.shape)

print('Shpae of testing data')
print(X_test.shape)
print(y_test.shape)


# Reshape training data
X_train_2d = X_train.values.reshape(-1, 1)
y_train_2d = y_train.values.reshape(-1, 1)

# Reshape testing data
X_test_2d = X_test.values.reshape(-1, 1)
y_test_2d = y_test.values.reshape(-1, 1)

print("Shape of training data (X):", X_train_2d.shape)
print("Shape of training data (y):", y_train_2d.shape)
print("Shape of testing data (X):", X_test_2d.shape)
print("Shape of testing data (y):", y_test_2d.shape)



 #linear Regression
lr=LinearRegression() 
lr

lr.fit(X_train_2d,y_train_2d)

y_pred=lr.predict(X_test_2d)
y_pred[:10]

y_test_2d[:10]

mean_squared_error(y_pred,y_test_2d)



#decision tree
model_dtr=DecisionTreeRegressor()
model_dtr

model_dtr.fit(X_train_2d, y_train_2d)

y_pred_dtr=model_dtr.predict(X_test_2d)
y_pred_dtr[:5]

mean_squared_error(y_pred_dtr,y_test_2d)



#random forest
model_rfr=RandomForestRegressor()
model_rfr.fit(X_train_2d,y_train_2d)

y_pred_rfr=(X_test_2d)
y_pred_rfr[:10]

mean_squared_error(y_pred_rfr,y_test_2d)


#hyperparameter tunning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

# Define hyperparameters to tune
param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}

# Create a Linear Regression model
model_lr = LinearRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(model_lr, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train_2d, y_train_2d)

# Print the best parameters and best MSE score
print("Best Parameters:", grid_search.best_params_)
print("Best Negative MSE Score:", grid_search.best_score_)



from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model_lr = LinearRegression()

# Perform 10-fold cross-validation
accuracy_scores = cross_val_score(model_lr, X_train_2d, y_train_2d, cv=10, scoring='neg_mean_squared_error')

# Convert negative mean squared error to positive
mse_scores = -accuracy_scores

# Print the MSE scores
print("MSE Scores:", mse_scores)



#final model
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model with the best parameters
final_model = LinearRegression(fit_intercept=False, copy_X=True)

# Fit the model to the entire training data
final_model.fit(X_train_2d, y_train_2d)


# Now you can use final_model to make predictions on new data


#Converting to pickle file
import pickle

# Define the filename for the pickle file
filename = 'final_model.pkl'

# Save the final_model to a pickle file
with open(filename, 'wb') as file:
    pickle.dump(final_model, file)



os.path.abspath('final_model.pkl')



import pickle
import numpy as np

# Load the saved model from the file
filename = 'final_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Input height for prediction
height_input = 6.0

# Reshape the input height to match the shape expected by the model (2D array)
height_input_2d = np.array(height_input).reshape(1, -1)

# Use the loaded model to make predictions
predicted_weight = loaded_model.predict(height_input_2d)

# Print the predicted weight
print("Predicted weight:", predicted_weight[0, 0])





