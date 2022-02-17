import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import sys

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3]}
df =pd.DataFrame(data)
print(df)
#df = df.drop(df.index[2], axis=0)
#df.drop(['Cochice', 'Pima'])
#print(df)

'''for index, row in df.iterrows():
        print(df.loc[index, 'year'])'''

print(df['year'].mode()[0])
