
# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# importing data
data = pd.read_csv(r'data\cleaned_data.csv')

# data into dataframe
df = pd.DataFrame(data)

# selecting columns with categorical features
categorical_features = ['AccountStatus','CreditHistory','Purpose','Savings',
                        'EmploymentDuration','PersonalStatusAndSex','OtherDebtors',
                        'Property','OtherInstallment','Housing','Job','Telephone','Foreign']

# creating empty dataframe
df_inter = pd.DataFrame()

# transforming categorical features to dummy variables
for col in categorical_features:
    dummies = pd.get_dummies(df[col],prefix=col)
    df_inter = pd.concat([df_inter,dummies],axis=1)
    
testdf = pd.concat([df,df_inter],axis=1)
testdf = testdf.drop(categorical_features,axis=1)

# selecting columns with numerical features
numerical_features = ['Duration','CreditAmount','InstallmentRatePercent','ResidenceDuration',
                      'Age','CreditsInThisBank','NumberOfPeopleLiableToMaintenance']
for col in numerical_features:
    column = np.array(testdf[col])
    column = column.reshape(len(column),1)
    sc = StandardScaler()
    sc.fit(column)
    testdf[col] = sc.transform(column)

# setting up label for output
le = LabelEncoder()
testdf['Label']=le.fit_transform(testdf['Decision'])
X = testdf.loc[:,testdf.columns !='Unnamed: 0']
X = X.loc[:,X.columns !='Decision']
X = X.loc[:,X.columns !='Label']
Y = testdf.loc[:,'Label']

# subsetting data into test and train 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=2024)

X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
Y_test_categorical = to_categorical(Y_test,num_classes=2)
Y_train_categorical = to_categorical(Y_train,num_classes=2)

# creating neural network function
def nn_model(learning_rate):
    model = Sequential()
    
    model.add(Dense(128,kernel_initializer = 'normal',input_dim = X_train.shape[1],
                       activation='relu'))
    model.add(Dense(256,kernel_initializer = 'normal',activation='relu'))
    model.add(Dense(256,kernel_initializer = 'normal',activation='relu'))
    model.add(Dense(256,kernel_initializer = 'normal',activation='relu'))
    
    model.add(Dense(2,kernel_initializer='normal',activation='sigmoid'))
    
    optimizer = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    model.summary()
    return model

# neural network function call, I choosed learning rate and number of epochs with trial and error method
model = nn_model(1e-5)
nb_epochs=200
history = model.fit(X_train,Y_train_categorical,epochs=nb_epochs,batch_size=32)

# predictions and accuracy score for test subset
predictions = model.predict(X_test)
prediction = list()
for i in range(len(predictions)):
    prediction.append(np.argmax(predictions[i]))
    
print('\n')
print('accuracy:',accuracy_score(Y_test,prediction))
print('\n')
print('confusion matrix:')
print(confusion_matrix(Y_test,prediction))

# ROC curve with ROC AUC score
fpr, tpr, thresholds = roc_curve(Y_test,predictions[:,1])

lr_auc = roc_auc_score(Y_test,predictions[:,1])

plt.plot([0,1],[0,1],'k--',)
plt.plot(fpr,tpr,label="ROC AUC=%.3f" %(lr_auc))
plt.xlabel("True positive rate")
plt.ylabel("False positive rate")
plt.title("ROC curve")
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.show()

# Conclusion:
#   In the last epoch, the model achieved an accuracy of 88% on the training set
#   The model on the test set achieved an accuracy of 75%
#   ROC AUC was 0.808