# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:00:19 2018

@author: Rotem Goren
"""
import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from threading import Thread
from keras.utils import multi_gpu_model
from queue import Queue
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.layers import Input,Dense, Conv2D, MaxPooling2D,Flatten,ZeroPadding2D, Reshape, Permute, Activation,UpSampling2D,Dropout,Concatenate
from keras.layers.normalization import BatchNormalization
from collections import OrderedDict
from keras.utils.np_utils import to_categorical
from keras import backend as K

import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"vgg16_weights.h5"
weights_file_name="weights_VGGsegnet.h5"
WIDTH=480
HIGHT=672

def VGGSegnet( n_classes ,  input_height=416, input_width=608 ):
    #model = Sequential()
    
    img_input = Input(shape=(3,input_height,input_width))
    
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(img_input)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(conv1)

    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first' )(conv1)
    f1 = pool1
    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first' )(conv2)
    f2 = pool2
    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(conv3)

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first' )(conv3)
    f3 = pool3
    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(drop4)
    f4 = pool4
    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(conv5)
    drop5=Dropout(0.5)(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(drop5)
    f5 = pool5
    
    x = Flatten(name='flatten')(pool5)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)
    vgg = Model(img_input, x)
    
    if (os.path.isfile(VGG_Weights_path)==True and os.path.isfile(weights_file_name)==False):    
        vgg.load_weights(VGG_Weights_path)
    
    levels = [f1 , f2 , f3 , f4 , f5 ]
    
    o = levels[4]
    	
    #o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
    conv6 = ( Conv2D(512, (3, 3), padding='same', data_format='channels_first'))(o)
    bn6 = ( BatchNormalization())(conv6)
    drop6 = Dropout(0.5)(bn6)

    up7 = ( UpSampling2D( (2,2), data_format='channels_first'))(drop6)
    merge7 = Concatenate([drop5, up7], axis=3)
    #o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    conv7 = ( Conv2D( 256, (3, 3), padding='same', data_format='channels_first'))(merge7)
    bn7 = ( BatchNormalization())(conv7)
    drop7 = Dropout(0.5)(bn7)

    up8 = ( UpSampling2D((2,2) , data_format='channels_first' ) )(drop7)
    merge8 = Concatenate([drop4, up8], axis=3)
    #o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    conv8 = ( Conv2D( 128 , (3, 3), padding='same' , data_format='channels_first' ))(merge8)
    bn8 = ( BatchNormalization())(conv8)
    drop8 = Dropout(0.5)(bn8)

    up9 = ( UpSampling2D((2,2) , data_format='channels_first' ))(drop8)
    #o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    merge9 = Concatenate([conv3, up9], axis=3)
    conv9 = ( Conv2D( 64 , (3, 3), padding='same'  , data_format='channels_first' ))(merge9)
    bn9 = ( BatchNormalization())(conv9)
    drop9 = Dropout(0.5)(bn9)

    up10 = ( UpSampling2D((2,2) , data_format='channels_first' ))(drop9)
    #o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    merge10 = Concatenate([conv2, up10], axis=3)
    conv10 = ( Conv2D( 64 , (3, 3), padding='same'  , data_format='channels_first' ))(merge10)
    bn10 = ( BatchNormalization())(conv10)
    drop10 = Dropout(0.5)(bn10)

    up11 = ( UpSampling2D((2,2) , data_format='channels_first' ))(drop10)
    merge11 = Concatenate([conv2, up11], axis=3)
    #o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    conv11 = ( Conv2D( 64 , (3, 3), padding='same'  , data_format='channels_first' ))(merge11)
    bn11 = ( BatchNormalization())(conv11)
    drop11 = Dropout(0.5)(bn11)

    output =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )(drop11)
    
 
   
    o_shape = Model(img_input , output ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    #o=(Flatten())(o)
    o = (Reshape((  -1  , outputHeight*outputWidth   )))(output)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    #o=(Dense(outputHeight*outputWidth,activation='softmax'))(o)
   
    model = Model( img_input , o )
    #model.outputWidth = outputWidth
    #model.outputHeight = outputHeight
    
    return model

#model=VGGSegnet( n_classes=12 ,input_height=416, input_width=608 ,Train=False)




camvid_colors = OrderedDict([
    ("Animal", np.array([64, 128, 64], dtype=np.uint8)),
    ("Archway", np.array([192, 0, 128], dtype=np.uint8)),
    ("Bicyclist", np.array([0, 128, 192], dtype=np.uint8)),
    ("Bridge", np.array([0, 128, 64], dtype=np.uint8)),
    ("Building", np.array([128, 0, 0], dtype=np.uint8)),
    ("Car", np.array([64, 0, 128], dtype=np.uint8)),
    ("CartLuggagePram", np.array([64, 0, 192], dtype=np.uint8)),
    ("Child", np.array([192, 128, 64], dtype=np.uint8)),
    ("Column_Pole", np.array([192, 192, 128], dtype=np.uint8)),
    ("Fence", np.array([64, 64, 128], dtype=np.uint8)),
    ("LaneMkgsDriv", np.array([128, 0, 192], dtype=np.uint8)),
    ("LaneMkgsNonDriv", np.array([192, 0, 64], dtype=np.uint8)),
    ("Misc_Text", np.array([128, 128, 64], dtype=np.uint8)),
    ("MotorcycleScooter", np.array([192, 0, 192], dtype=np.uint8)),
    ("OtherMoving", np.array([128, 64, 64], dtype=np.uint8)),
    ("ParkingBlock", np.array([64, 192, 128], dtype=np.uint8)),
    ("Pedestrian", np.array([64, 64, 0], dtype=np.uint8)),
    ("Road", np.array([128, 64, 128], dtype=np.uint8)),
    ("RoadShoulder", np.array([128, 128, 192], dtype=np.uint8)),
    ("Sidewalk", np.array([0, 0, 192], dtype=np.uint8)),
    ("SignSymbol", np.array([192, 128, 128], dtype=np.uint8)),
    ("Sky", np.array([128, 128, 128], dtype=np.uint8)),
    ("SUVPickupTruck", np.array([64, 128, 192], dtype=np.uint8)),
    ("TrafficCone", np.array([0, 0, 64], dtype=np.uint8)),
    ("TrafficLight", np.array([0, 64, 64], dtype=np.uint8)),
    ("Train", np.array([192, 64, 128], dtype=np.uint8)),
    ("Tree", np.array([128, 128, 0], dtype=np.uint8)),
    ("Truck_Bus", np.array([192, 128, 192], dtype=np.uint8)),
    ("Tunnel", np.array([64, 0, 64], dtype=np.uint8)),
    ("VegetationMisc", np.array([192, 192, 0], dtype=np.uint8)),
    ("Wall", np.array([64, 192, 0], dtype=np.uint8)),
    ("Void", np.array([0, 0, 0], dtype=np.uint8))
])



def convert_label_to_grayscale(im):
    out = (np.ones(im.shape[:2]) * 0).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = gray_val
    #assert (out != 255).all(), "rounding errors or missing classes in camvid_colors"
    return out.astype(np.uint8)

def label_to_color(im):
    out = (np.ones((im.shape[0],im.shape[1],3)) * 255).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls=np.where((im == gray_val))
        out[match_pxls]=rgb
    
    return out

def prepar_data(batch_file):
    #PATH="C:/Users/owner/Desktop/segmentaion/data/"

    #C=color_to_catagory()
    if (os.path.isfile('labeled_images_{}.npy'.format(batch_file))):
        y=np.load('labeled_images_{}.npy'.format(batch_file))
    else:
        PATH = os.path.join(file_path,'Data\\Labeled\\','*.png')
        y=[]
        count=0
        i=0
        for image in glob.glob(PATH):
            im=cv2.imread(image)
            
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im=cv2.resize(im,(HIGHT,WIDTH))
            labeled=convert_label_to_grayscale(im)
            #labeled=cv2.resize(labeled,(480,320))
            #labeled=one_hot_it(labeled)
            labeled = to_categorical(labeled, num_classes=32)
            #labeled=labeled.flatten()
            labeled=labeled.reshape(labeled.shape[0]*labeled.shape[1],-1)
            
            y.append(np.array(labeled,dtype=np.uint8))
            #count+=1
            #if(count==200):
        np.save('labeled_images_{}.npy'.format(i),y)
                #i+=1
                #count=0
                #y=[]
            

    if (os.path.isfile('input_images_{}.npy'.format(batch_file))):
        x=np.load('input_images_{}.npy'.format(batch_file))
    else:
        x=[]   
        PATH="C:/Users/owner/Desktop/segmentaion/data/"
        PATH = os.path.join(PATH,"Raw/","*.png")
        count=0
        i=0
        for image in glob.glob(PATH):
            im=cv2.imread(image)
            im=cv2.resize(im,(HIGHT,WIDTH))
            
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            
            x.append(np.transpose(im, (2, 0, 1)))
            #count+=1
            #if(count==200):
        np.save('input_images_{}.npy'.format(i),x)
                #i+=1
                #count=0
                #x=[]
   
    return x,y
               
    
def train_model(x=[],y=[]):   

        val_ratio=0.8
        
        perm=np.random.permutation(x.shape[0])
        x=x[perm,:,:,:]
        y=y[perm,:,:]
        
        x_train=x[:int(len(x)*val_ratio),:,:,:]
        y_train=y[:int(len(y)*val_ratio),:,:]
        
        x_valid=x[int(len(x)*val_ratio):,:,:,:]
        y_valid=y[int(len(y)*val_ratio):,:,:]
        del x
        del y
        x_train=np.array(x_train,dtype=np.float32)
        y_train=np.array(y_train,dtype=np.float32)
        
        x_valid=np.array(x_valid,dtype=np.float32)
        y_valid=np.array(y_valid,dtype=np.float32)

        
        model=VGGSegnet( n_classes=32 ,  input_height=WIDTH, input_width=HIGHT )
        model.summary()
        if os.path.isfile(weights_file_name):
            
            model.load_weights(weights_file_name)
           
        
        
        model_gpu=model
        #model_gpu=multi_gpu_model(model,gpus=2)
        #model_yaw=multi_gpu_model(model_yaw,gpus=2)
        
        optimizer=Adam(lr=0.001)


        
        #'sparse_categorical_crossentropy'
        model_gpu.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics=['accuracy'])
        
     
                
        model_gpu.fit(x_train,y_train,epochs=50,shuffle=False,batch_size=3,validation_data=(x_valid,y_valid),callbacks=[])      
            


        model_gpu.save_weights(weights_file_name, overwrite=True)
        del model_gpu
        K.clear_session()
        
def test_model(x_test):
    model=VGGSegnet( n_classes=32 ,  input_height=WIDTH, input_width=HIGHT)
    model.summary()
    
    if os.path.isfile(weights_file_name):
        model.load_weights(weights_file_name)
    
    x_test1=np.expand_dims(x_test[50], axis=0)
    x_predict=model.predict(x_test1,batch_size=1)

    x_predict=np.reshape(x_predict[0],(WIDTH,HIGHT,32))

    x_predict=np.argmax(x_predict,axis=2)
    x_predict=label_to_color(x_predict)
    
    del model
    K.clear_session()
        
    return x_predict

def one_hot_it(labels):
    x = np.zeros([labels.shape[0],labels.shape[1],32])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            x[i,j,labels[i][j]]=1
    return x

if __name__=='__main__':
    x_train,y_train=prepar_data(0)
    '''
    for _ in range(5):
    
            train_model(x_train,y_train)
    '''
    
    x_predict=test_model(x_train)
    
    
    plt.figure()
    plt.imshow(x_predict)
    plt.figure()
    y=y_train[50,:,:]
    y=np.reshape(y,(WIDTH,HIGHT,32))
    
    y=np.argmax(y,axis=2)
    y=label_to_color(y)
    plt.imshow(y)
    
    #prepar_data()    
