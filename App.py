import numpy as np
import pandas as pd
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

start='2015-05-27'
end='2020-05-22'

st.title('STOCK MARKET PREDICTION')

user_input=st.text_input("Enter stock ticker",'AAPL')
#df=data.DataReader(user_input,'yahoo',start,end)
df=pd.read_csv('AAPL.csv')

st.subheader('Data from 2015 - 2020')
st.write(df.describe())


#Visualizations
st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)

df1=df.reset_index()['close']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM (3 dim )
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)  #(716 , 100 , 1) reason for converting into 3dim is that 
                                                                 # we give (X_train.shape[1], 1) as input into lstm
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


model=load_model('keras_model.h5')


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=100 #timestamp for predicting next day we consider previous 100 days 
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
st.subheader('Performance of train data and test data on the complete data')
fig=plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
st.pyplot(fig)



x_input=test_data[len(test_data)-look_back:].reshape(1,-1)    #to predict next day we r considering prev 100 days ie. 441-100=341 in 1dim
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


#predicting next 30 days stock
from numpy import array

lst_output=[]
n_steps=100 #timestamp for predicting next day we consider previous 100 days 
  #considering previous 100 days stock to predict next 30 days
i=0
while(i<30):  #till 30 days
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)

day_new=np.arange(1,101)  #prev 100 days
day_pred=np.arange(101,131)  #next 30 days to predict


#plotting next 30 days stock
st.subheader('Next 30 days stock')
fig=plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-n_steps:]))  # 1258-100 coz for considering previous 100 days to predict next 30 days
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig)

st.subheader('Total Graph combing the whole dataset with next 30 days')
fig=plt.figure(figsize=(12,6))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
st.pyplot(fig)

st.subheader("Final Output")
fig=plt.figure(figsize=(12,6))
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
st.pyplot(fig)

