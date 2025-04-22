import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
