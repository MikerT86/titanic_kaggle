import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('./train.csv')

df.drop(['Survived'], 1)

train_df = df.copy()
X = train_df.drop(['Survived'], 1)
Y = train_df['Survived']

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

