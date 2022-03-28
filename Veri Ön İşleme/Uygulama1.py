
#1 Adım Kütüphaneleri Yükle

import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

#2. Adım Veri Setini Yükle

dataset= pd.read_csv('Data.csv')

X=dataset.iloc[:,:-1].values

print(X)

y=dataset.iloc[:,-1]

print(y)

#3. Adım Eksik Verileri

from sklearn.preprocessing import SimpleImputer

imputer=SimpleImputer(missing_values='np.nan', strategy='mean',axis=0)

imputer=imputer.fit(X[:,1:3])

X[:,1:3]=imputer.transform(X[:,1:3])

print(X)

#4. Adım kategorik Verilerin Düzenlenmesi

from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])


print(X)


labelEncoder_y= LabelEncoder()

y=labelEncoder_y.fit_transform(y)


#5. Adım Veri Setini Egitim ve Test olarak böl

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=26)

#6. Adım Özellik Ölçeklendirme

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

print(X_train)

X_test=sc_X.transform(X_test)

print(X_test)
















