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

PYTORCH=True
from threading import Thread
from queue import Queue
if (PYTORCH):
    from Model import *
    import torch
else:
    from keras.utils import multi_gpu_model
    from keras.models import Model,Sequential
    from keras.optimizers import Adam
    from keras.layers import Input,Dense, Conv2D, MaxPooling2D,Flatten,ZeroPadding2D, Reshape, Permute, Activation,UpSampling2D,Dropout,concatenate
    from keras.layers.normalization import BatchNormalization
    from keras import backend as K
    from keras.applications.vgg16 import VGG16

from keras.utils.np_utils import to_categorical
from collections import OrderedDict
import tqdm
import os


def VGGSegnet(n_classes, input_height=416, input_width=608):
    # model = Sequential()

    img_input = Input(shape=(input_height, input_width, 3))

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)

    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(conv3)

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(conv4)

    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(conv5)

    # drop5=Dropout(0.5)(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    vgg = Model(img_input, pool5)

    if (os.path.isfile(VGG_Weights_path) == True and os.path.isfile(weights_file_name) == False):
        vgg.load_weights(VGG_Weights_path)

    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(pool5)
    bn6 = BatchNormalization()(conv6)
    drop6 = Dropout(0.5)(bn6)

    up7 = UpSampling2D((2, 2), name='block7_up1')(drop6)
    merge7 = concatenate([conv5, up7], name='block7_merge')
    # o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv1')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv2')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv3')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv4')(conv7)

    bn7 = BatchNormalization()(conv7)
    drop7 = Dropout(0.5)(bn7)

    up8 = UpSampling2D((2, 2), name='block8_up1')(drop7)
    merge8 = concatenate([conv4, up8], name='block8_merge')
    # o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv1')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv2')(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv3')(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv4')(conv8)

    bn8 = BatchNormalization()(conv8)
    drop8 = Dropout(0.5)(bn8)

    up9 = UpSampling2D((2, 2), name='block9_up1')(drop8)
    # o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    merge9 = concatenate([conv3, up9], name='block9_merge')
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv2')(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv3')(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv4')(conv9)

    bn9 = BatchNormalization()(conv9)
    drop9 = Dropout(0.5)(bn9)

    up10 = UpSampling2D((2, 2), name='block10_up1')(drop9)
    # o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    merge10 = concatenate([conv2, up10], name='block10_merge')
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv1')(merge10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv2')(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv3')(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv4')(conv10)

    bn10 = BatchNormalization()(conv10)
    drop10 = Dropout(0.5)(bn10)

    up11 = UpSampling2D((2, 2), name='block11_up1')(drop10)
    merge11 = concatenate([conv1, up11], name='block11_merge')
    # o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block11_conv1')(merge11)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block11_conv2')(conv11)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block11_conv3')(conv11)
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block11_conv4')(conv11)

    bn11 = BatchNormalization()(conv11)
    drop11 = Dropout(0.5)(bn11)

    output = Conv2D(n_classes, (3, 3), activation='softmax', padding='same', name='block12_conv')(drop11)

    # o_shape = Model(img_input , output ).output_shape
    # outputHeight = o_shape[1]
    # outputWidth = o_shape[2]
    # #o=(Flatten())(o)
    # o = (Reshape((  -1  , outputHeight*outputWidth   )))(output)
    # o = (Permute((2, 1)))(o)
    # o = (Activation('softmax'))(o)
    # #o=(Dense(outputHeight*outputWidth,activation='softmax'))(o)
    #
    model = Model(img_input, output)
    # model.outputWidth = outputWidth
    # model.outputHeight = outputHeight

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



def prepar_data(batch_file,file_path):
    # PATH="C:/Users/owner/Desktop/segmentaion/data/"
    # C=color_to_catagory()
    file_name = os.path.join(file_path, 'labeled_images_{}.npy'.format(batch_file))
    if (os.path.isfile(file_name)):
        y = np.load(file_name)
    else:
        PATH = os.path.join(file_path, 'data\\Labeled\\', '*.png')
        y = []
        count = 0
        i = 0

        for image in glob.glob(PATH):
            im = cv2.imread(image)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (WIDTH, HEIGHT))
            labeled = convert_label_to_grayscale(im)
            # labeled=cv2.resize(labeled,(480,320))
            # labeled=one_hot_it(labeled)
            labeled = to_categorical(labeled, num_classes=len(camvid_colors))
            # labeled=labeled.flatten()
            # labeled=labeled.reshape(labeled.shape[0]*labeled.shape[1],-1)

            y.append(np.array(labeled, dtype=np.uint8))

            # count+=1
            # if(count==200):
        y = np.array(y)
        np.save(os.path.join(file_path, 'labeled_images_{}.npy'.format(i)), y)
        # i+=1
        # count=0
        # y=[]

    file_name = os.path.join(file_path, 'input_images_{}.npy'.format(batch_file))
    if (os.path.isfile(file_name)):
        x = np.load(file_name)
    else:
        x = []
        PATH = os.path.join(file_path, 'data\\Raw\\', '*.png')
        count = 0
        i = 0
        for image in glob.glob(PATH):
            im = cv2.imread(image)
            im = cv2.resize(im, (WIDTH, HEIGHT))

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255

            # x.append(np.transpose(im, (2, 0, 1)))
            x.append(im)
            # count+=1
            # if(count==200):
        x = np.array(x)
        np.save(os.path.join(file_path, 'input_images_{}.npy'.format(i)), x)
        # i+=1
        # count=0
        # x=[]

    return x, y


def train_model(x=[],y=[],PYTORCH=False):

        BATCH_SIZE=16
        EPOCHS=500
        ngpu=1

        perm = np.random.permutation(x.shape[0])
        x = x[perm, :, :, :]
        y = y[perm, :, :, :]

        x_train = x[:int(len(x) * val_ratio), :, :, :]
        y_train = y[:int(len(y) * val_ratio), :, :, :]

        x_valid = x[int(len(x) * val_ratio):, :, :, :]
        y_valid = y[int(len(y) * val_ratio):, :, :, :]

        del x
        del y

        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        x_valid = np.array(x_valid, dtype=np.float32)
        y_valid = np.array(y_valid, dtype=np.float32)

        if(PYTORCH):
            train_data = Dataset(x_train, y_train)
            valid_data = Dataset(x_valid, y_valid)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                                           shuffle=True, num_workers=4,
                                                           pin_memory=True)  # divided into batches
            valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE,
                                                           shuffle=True, num_workers=4,
                                                           pin_memory=True)  # divided into batches
            model=UNetWithResNet50()
            if (os.path.isfile(model_file)):
                state_dict = torch.load(model_file, map_location=device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if (k[7:] == 'module.'):
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                # load params
                model.load_state_dict(new_state_dict)
            model=model.to(device)
            if (device.type == 'cuda') and (ngpu > 1):
                model = nn.DataParallel(model, list(range(ngpu)), dim=1)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

            train_loss=0
            valid_loss=0
            model.train()
            for epoch in range(EPOCHS):
                for (x, y) in tqdm(train_dataloader):
                    model.zero_grad()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    o=model(x)
                    loss=nn.CrossEntropyLoss()(o,y)

                    train_loss += loss

                    loss.backward()
                    optimizer.step()
                train_loss=train_loss/len(train_dataloader)
                for (x, y) in tqdm(valid_dataloader):
                    model.zero_grad()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    o = model(x)
                    loss = nn.CrossEntropyLoss()(o, y)

                    valid_loss += loss
                valid_loss = valid_loss / len(train_dataloader)
            print('[{}/{}]\tTrain loss: {}\tValid loss:{}'.format(epoch, EPOCHS,train_loss.item(), valid_loss.item()))
            torch.save(model.state_dict(), '%s' % model_file)




        else:
            model = VGGSegnet(n_classes=32, input_height=HEIGHT, input_width=WIDTH)
            model.summary()
            if os.path.isfile(weights_file_name):

                model.load_weights(weights_file_name)

            model_gpu=model
            #model_gpu=multi_gpu_model(model,gpus=2)

            optimizer=Adam(lr=0.0001)
            #'sparse_categorical_crossentropy'
            model_gpu.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics=['accuracy'])
            model_gpu.fit(x_train,y_train,epochs=EPOCHS,shuffle=True,batch_size=BATCH_SIZE,validation_data=(x_valid,y_valid),callbacks=[])
            model_gpu.save_weights(weights_file_name, overwrite=True)
            del model_gpu
            K.clear_session()


def test_model(x_test):
    model = VGGSegnet(n_classes=32, input_height=HEIGHT, input_width=WIDTH)
    model.summary()

    if os.path.isfile(weights_file_name):
        model.load_weights(weights_file_name)

    x_test1 = np.expand_dims(x_test[30], axis=0)
    x_predict = model.predict(x_test1, batch_size=1)[0, :, :, :]

    # x_predict=np.reshape(x_predict[0],(WIDTH,HIGHT,32))

    x_predict = np.argmax(x_predict, axis=2)
    x_predict = label_to_color(x_predict)

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
    file_path = 'D:\\segmentaion\\'
    VGG_Weights_path = file_path + "vgg16_weights.h5"
    weights_file_name = "weights_VGGsegnet.h5"
    model_file="Unet_resnet50.pth"
    HEIGHT = 224#480
    WIDTH  = 384#672

    # vgg=VGG16(weights='imagenet',include_top=False)
    # vgg.summary()
    # vgg.save_weights('vgg16_weights.h5')
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='/Graph', histogram_freq=100,
    #                                          write_graph=True, write_images=True)

    #for _ in range(5):
    x,y=prepar_data(0,file_path)

    val_ratio = 0.7
    test_ratio = 0.1

    x_test = x[len(x)-int(len(x)*test_ratio):, :, :, :]
    y_test = y[len(y)-int(len(y)*test_ratio):, :, :, :]

# for _ in range(5):
    x=x[:len(x)-int(len(x)*test_ratio), :, :, :]
    y=y[:len(y)-int(len(y)*test_ratio), :, : ,:]



    train_model(x,y,PYTORCH=PYTORCH)

    
    x_predict=test_model(x_test)
    #print(x_predict)

    # y = y_test[20:30, :, :]
    # y = np.argmax(y, axis=2)
    # y = label_to_color(y)


    plt.imshow(x_predict)
    plt.show()
    # plt.imshow(y)
    # plt.show()


    #plt.figure()
    # y=y_test[50,:,:]
    # #y=np.reshape(y,(WIDTH,HIGHT,32))
    #
    # y=np.argmax(y,axis=2)
    # y=label_to_color(y)
    # plt.imshow(y)
    #
    #prepar_data()    
