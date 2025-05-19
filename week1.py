#intro to ML 

#sklearn - data preprocessing (imputation, encoding, scaling)
#matplotlib + seaborn -  data visualization

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, header=None, names=columns)      #header- colm. used to easily retrieve some row, names- colm names as desired

np.random.seed(42)
for col in ['sepal_length', 'sepal_width']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

imputer = SimpleImputer(strategy='mean')
df[['sepal_length', 'sepal_width']] = imputer.fit_transform(df[['sepal_length', 'sepal_width']])

label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])    #converts category colm to numerical labels
0

scaler = StandardScaler()        #standardizes colm values to have 0 mean and unit variance
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(   
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]      
)

plt.figure(figsize=(8, 6))
plt.hist(df['sepal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='petal_length', hue='class', data=df, palette='viridis')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Petal Length (Standardized)')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()
