# NOTE: this script runs under IPython.  

import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import datetime



def normalize (df):
    data = df.copy()
    for col in data.columns:
        mn = data[col].min()
        w  = data[col].max() - mn
        data[col] = (data[col] - mn)/w
    return data

## os.chdir("/home/gleb/Desktop/KAGGLE/Titanic")


df_gender = pd.read_csv("gender_submission.csv", index_col = 0)
df_train  = pd.read_csv("train.csv", index_col = 0)
df_test   = pd.read_csv("test.csv", index_col = 0)

df_gender.head()

df_train.head()
df_test.head()

pd.unique(df_train["Embarked"])
pd.unique(df_train["Sex"])


columns_to_drop =["Name", "Ticket", "Embarked", "Cabin", "Parch", "Age"] 
columns_to_drop =["Name", "Ticket", "Embarked", "Cabin", "Parch", "Fare"] 
columns_to_drop =["Name", "Ticket", "Embarked", "Cabin", "Parch"] 

df_trn = df_train.drop(columns_to_drop, axis=1)
df_tst = df_test.drop (columns_to_drop, axis=1)

df_trn["Sex"]=df_trn["Sex"].apply(lambda x: 1 if x == 'male' else 0)


df_trn.dropna(inplace=True)


# normalizing colums
df_trn = normalize (df_trn)
print df_trn.head(40)
print df_trn.shape


# input data
Y=df_trn["Survived"].as_matrix()
X=df_trn.drop(["Survived"], axis=1).as_matrix()


# create model
model = Sequential()
model.add(Dense(X.shape[1]**3, input_dim = X.shape[1], activation="sigmoid"))
#model.add(Dense(32, activation="sigmoid"))
#model.add(Dense(32, activation="sigmoid"))
#model.add(Dense(32, activation="sigmoid"))
#model.add(Dense(32, activation="sigmoid"))
#model.add(Dense(32, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

# compile model

#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

# fitting
t1=datetime.datetime.now()

model.fit(X, Y, nb_epoch = 100000, batch_size = X.shape[0], verbose=0)

elapsed_time = datetime.datetime.now()-t1
print ("Elapsed time %.2f"%elapsed_time.total_seconds())

# evaluating the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

"""
Model                   Epochs        Acc

5-5^3s-1s   loss=binarcy_crossentropy, optimizer=adam    100000     93.84%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^3s-1s   loss=binarcy_crossentropy, optimizer=nadam    100000    92.16%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^3s-1s   loss=binarcy_crossentropy, optimizer=sgd      100000    0.00%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized

5-5^3s-1s   float64           100000      94.12%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^3s-1s float32 eps=1e-9    100000  0.00%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^3s-1s float64 eps=1e-9    100000  94.12%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^3s-1s float32 eps=1e-8    100000  0.00%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized

5-5^3t-1s                100000      93.14%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized

5-5^2s-1s                100000        92.72%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5^4s-1s                100000        92.02%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-250s-1s                100000        93.70%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-125s-1s                100000        94.12%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
5-5s-1s                100000        88.52%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized

5-64t-1s                100000        94.26%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized
4-64t-1s                100000        94.26%  # columns: Survived, Pclass, Sex, Age, SibSp, Fare; normalized

4-64t-1s                100000        89.92%  # columns: Survived, Pclass, Sex, Age, Fare, SibSp; normalized
4-64t-1s                100000        89.92%  # columns: Survived, Pclass, Sex, Age, SibSp; normalized
4-64s-1s                100000        89.64%  # columns: Survived, Pclass, Sex, Age, SibSp; normalized
4-64s-1s                200000        89.0%
4-64s-32s-1s            100000        86.42%
4-64s-1s                 20000        84.74%
4-64s-1s                 10000        84.74%
4-128s-1s                10000        84.62%
4-256s-1s                10000        83.84%
4-512s-1s                10000        83.50%
4-1024s-1s               10000        83.28%
4-32s-32s-1s             10000        83.73%
4-16s-32s-16s-1s         10000        82.72%
4-32s-32s-32s-1s         10000        83.05%
4-32s-32s-32s-32s-32s-1s 10000        83.16%

"""

#model.history.
#model.test_on_batch(X, Y, sample_weight=None)

#Y_pred = df_tst["Survived"].as_matrix()




