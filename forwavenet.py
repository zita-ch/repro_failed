#!/usr/bin/env python
# coding: utf-8

## corresponding paper: IMU-based Deep Neural Networks for Locomotor Intention Prediction

import numpy as np
import pandas as pd
import os

subject = 'AB194'
path = subject+'/'+subject+'/raw/'
for f in os.listdir(path):
    file = path + f
    try:
        data = pd.concat([data,pd.read_csv(file,usecols=list(range(12))+[48]) ])
    except:
        data = pd.read_csv(file,usecols=list(range(12))+[48]) 
    print(file,end='\r')




data = data.reset_index(drop=True)


data = pd.get_dummies(data, prefix=[data.columns[-1]],columns = ['Mode'])


def gyro2quadr(gyros):
    axs,ays,azs,gys,gzs,gxs = gyros[:,0],gyros[:,1],gyros[:,2],gyros[:,3],gyros[:,4],gyros[:,5]
    Kp = 0 * 100 
    Ki = 0 * 0.002 
    halfT = 0.001 
    qs = np.zeros((len(axs),4))
    q0 = 1
    q1 = 0
    q2 = 0
    q3 = 0
    exInt = 0
    eyInt = 0
    ezInt = 0    
    for i in range(len(axs)):
        ax, ay, az = axs[i], ays[i], azs[i]
        norm = np.sqrt(ax*ax+ay*ay+az*az)
        ax = ax/norm
        ay = ay/norm
        az = az/norm
        vx = 2*(q1*q3 - q0*q2)
        vy = 2*(q0*q1 + q2*q3)
        vz = q0*q0 - q1*q1 - q2*q2 + q3*q3
        ex = (ay*vz - az*vy)
        ey = (az*vx - ax*vz)
        ez = (ax*vy - ay*vx)
        exInt += ex*Ki
        eyInt += ey*Ki
        ezInt += ez*Ki
        gx = gxs[i] + Kp*ex + exInt
        gy = gys[i] + Kp*ey + eyInt
        gz = gzs[i] + Kp*ez + ezInt
    
        q0 += (-q1*gx - q2*gy - q3*gz)*halfT
        q1 += (q0*gx + q2*gz - q3*gy)*halfT
        q2 += (q0*gy - q1*gz + q3*gx)*halfT
        q3 += (q0*gz + q1*gy - q2*gx)*halfT
        norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        q0 /= norm
        q1 /= norm
        q2 /= norm
        q3 /= norm
        qs[i,0]=q0
        qs[i,1]=q1
        qs[i,2]=q2
        qs[i,3]=q3
    
    return qs




Xs = []
Ys = []
for i in range(0,data.shape[0]-10,3):
    if i%10000 ==0:
        print(i,end='\r')
    y1 = data.iloc[i,-7:]
    y2 = data.iloc[i+9,-7:]
    if ( False in (y1==y2).tolist()):
        continue
    Ys.append(np.array(y1))
    x_thigh = np.array(data.iloc[i:i+10,6:-7])
    x_shank = np.array(data.iloc[i:i+10,0:6])
    Xs.append(np.concatenate([x_thigh,gyro2quadr(x_thigh),x_shank,gyro2quadr(x_shank)],axis=1))





del data
Xs = np.array(Xs)
Ys = np.array(Ys)
print(Xs.shape)
print(Ys.shape)
np.save("./Xs"+subject[-3:]+".npy", Xs)
np.save("./Ys"+subject[-3:]+".npy", Ys)



#standardization
mean_ = np.mean(np.reshape(Xs,(-1,20)),axis=0)
std_ = np.std(np.reshape(Xs,(-1,20)),axis=0)
Xs = (Xs-mean_)/std_



import keras
from keras.layers import Layer,Dense,Dropout,Input,Activation,SimpleRNN,GRU,LSTM,multiply,add,Reshape,Dot,dot,Concatenate
from keras import Model,activations
from keras.optimizers import Adam
from keras.layers import Layer,Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,TimeDistributed
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras import regularizers
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    

def f1(true, pred): #shapes (batch, 7)

    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 7) 

    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
        #all with shape (7,)
    
    precision = true_positives / (pred_positives+K.epsilon()) 
    recall = true_positives / (ground_positives+K.epsilon())
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (7,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (7,)

    weighted_f1 = f1 * ground_positives / (K.sum(ground_positives) +K.epsilon())
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 


input_layer = keras.Input(shape=(10,20))
cc1d_left = Conv1D(64 , 2, strides=1, padding='causal', dilation_rate=1, activation='relu')
concat = Concatenate()
dc1d_left = Conv1D(64 , 3, padding='same', dilation_rate=2, activation=None)
dc1d_middle = Conv1D(64 , 3, padding='same', dilation_rate=2, activation=None)
dc1d_right = Conv1D(64 , 3, padding='same', dilation_rate=2, activation=None)
tanh_ = Activation(K.tanh)
sigmoid_= Activation(K.sigmoid)
relu_ = Activation(K.relu)
reshape_ = Reshape((-1,))
dense1 = Dense(50, activation=None)
dp = Dropout(0.25)
dense2 = Dense(7, activation='softmax')

left_1 = cc1d_left(input_layer)
left_2 = dc1d_left(left_1)
left_3_o = multiply([tanh_(left_2),sigmoid_(left_2)])
o1 = add([left_1 , left_3_o])
middle_1 = dc1d_middle(o1)
middle_2_o = multiply([tanh_(middle_1),sigmoid_(middle_1)])
o2 = add([middle_2_o,o1])
right_1 = dc1d_right(o2)
right_2_o = multiply([tanh_(right_1),sigmoid_(right_1)])
o3 = relu_(add([left_3_o,middle_2_o,right_2_o]))
o3_2 = dense1(reshape_(o3))
o3_3 = dp(o3_2)

output_layer = dense2(o3_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), ##
             loss='categorical_crossentropy',metrics=[f1])

weight_path = 'testmodel.h5'
callbacks_list = [
  keras.callbacks.EarlyStopping(
  monitor='val_f1',mode='max',
  patience=10,
  ),
  keras.callbacks.ModelCheckpoint(
  filepath=weight_path,
  monitor='val_f1',mode='max',
  save_best_only=True,
  save_weights_only=True,
  )
]
class_w = {}
for i in range(7):
    class_w[i]=100000/np.sum(Ys[:,i])
np.random.seed(2333)
h=model.fit(Xs,Ys,class_weight=class_w,
            batch_size=64, epochs=150, verbose=1, callbacks=callbacks_list,validation_split=0.2,
            shuffle=True)




model.load_weights(weight_path)
# y_true = np.argmax(Ys,axis=-1)
# y_pred = np.argmax(model.predict(Xs),axis=-1)
# print(confusion_matrix(y_true,y_pred))


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(211)
plt.plot(h.history['loss'],color='r')
plt.plot(h.history['val_loss'],color='g')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'])

plt.subplot(212)
plt.plot(h.history['f1'],color='b')
plt.plot(h.history['val_f1'],color='k')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train_f1', 'test_f1'])






