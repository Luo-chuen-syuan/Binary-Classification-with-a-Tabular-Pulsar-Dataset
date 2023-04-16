#處理資料集
import pandas as pd
import numpy as np
from keras.utils  import to_categorical
from sklearn.model_selection import KFold

train=pd.read_csv('./train.csv')

train_x=train.drop(['label'],axis=1)
train_y=train['label']
test_x=pd.read_csv('./test.csv')

kf=KFold(n_splits=4,shuffle=True,random_state=123)

tr_idx,va_idx=list(kf.split(train_x))[0]

tr_x,va_x=train_x.iloc[tr_idx],train_x.iloc[va_idx]
tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]

tr_x,va_x=np.array(tr_x/255.0),np.array(va_x/255.0)

tr_y=to_categorical(tr_y,10)
va_y=to_categorical(va_y,10)

print(tr_x.shape)
print(tr_y.shape)
print(va_x.shape)
print(va_y.shape)

from collections import Counter
count=Counter(train['label'])
count

import seaborn as sns
sns.countplot(train['label'])

print(tr_x[0])

import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
x,y=10,5
for i in range(50):
    plt.subplot(y,x,i+1)
    plt.imshow(tr_x[i].reshape((28,28)),interpolation='nearest')
plt.show()

from keras.models import Sequential

from keras.layers import Dense,Activation

from keras.utils import custom_object_scope

model=Sequential()

model.add(Dense(128,
                input_dim=tr_x.shape[1],
                activation='sigmoid'
                ))
model.add(Dense(10,
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

result=model.fit(tr_x,tr_y,
                 epochs=5,
                 batch_size=100,
                 validation_data=(va_x,va_y),verbose=1)

model.predict(train_x)

print(result[:5])
print([x.argmax() for x in result[:5]])
y_test=[x.argmax()for x in result]