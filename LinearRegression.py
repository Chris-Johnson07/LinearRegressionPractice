import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

num_samples = 500

years_of_experience = np.random.randint(2,21,size=num_samples)

slope = (200000 - 60000)/18
intercept = 60000

salaries = slope * years_of_experience + intercept + np.random.normal(0,10000,size=num_samples)

data = {'Years_of_Experience':years_of_experience, 'Salary':salaries}

df = pd.DataFrame(data)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Years_of_Experience',y='Salary',data=df)
sns.regplot(x='Years_of_Experience',y='Salary',data=df,scatter=False)

plt.xlabel('Years_of_Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Salary')
plt.show()


X = df[['Years_of_Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=19)

lr = LinearRegression()

lr.fit(X_train,y_train)
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

y_pred = lr.predict(X_test)
mean_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)
lr.coef_
lr.intercept_

coefficients = lr.coef_
intercept = lr.intercept_

X = np.linspace(0,20,100)
y = coefficients*X + intercept

plt.clf()
plt.scatter(X,y, label = f'y = {coefficients[0]}x+ {intercept}', color = 'blue')
plt.xlabel('Years_of_Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Salary')
plt.grid()
plt.show()