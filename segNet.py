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

PYTORCH = True
from threading import Thread
from queue import Queue
from Model import *

if (PYTORCH):
    import torch
else:
    from keras.utils import multi_gpu_model
    from keras.optimizers import Adam

#from keras.utils.np_utils import to_categorical
from collections import OrderedDict
import tqdm
import os

# model=VGGSegnet( n_classes=12 ,input_height=416, input_width=608 ,Train=False)


camvid_colors = OrderedDict([
    ('unlabeled', np.array([0, 0, 0], dtype=np.uint8)),
    ('dynamic', np.array([111, 74, 0], dtype=np.uint8)),
    ('ground', np.array([81, 0, 81], dtype=np.uint8)),
    ('road', np.array([128, 64, 128], dtype=np.uint8)),
    ('sidewalk', np.array([244, 35, 232], dtype=np.uint8)),
    ('parking', np.array([250, 170, 160], dtype=np.uint8)),
    ('rail track', np.array([230, 150, 140], dtype=np.uint8)),
    ('building', np.array([70, 70, 70], dtype=np.uint8)),
    ('wall', np.array([102, 102, 156], dtype=np.uint8)),
    ('fence', np.array([190, 153, 153], dtype=np.uint8)),
    ('guard rail', np.array([180, 165, 180], dtype=np.uint8)),
    ('bridge', np.array([150, 100, 100], dtype=np.uint8)),
    ('tunnel', np.array([150, 120, 90], dtype=np.uint8)),
    ('pole', np.array([153, 153, 153], dtype=np.uint8)),
    ('traffic light', np.array([250, 170, 30], dtype=np.uint8)),
    ('traffic sign', np.array([220, 220, 0], dtype=np.uint8)),
    ('vegetation', np.array([107, 142, 35], dtype=np.uint8)),
    ('terrain', np.array([152, 251, 152], dtype=np.uint8)),
    ('sky', np.array([70, 130, 180], dtype=np.uint8)),
    ('person', np.array([220, 20, 60], dtype=np.uint8)),
    ('rider', np.array([255, 0, 0], dtype=np.uint8)),
    ('car', np.array([0, 0, 142], dtype=np.uint8)),
    ('truck', np.array([0, 0, 70], dtype=np.uint8)),
    ('bus', np.array([0, 60, 100], dtype=np.uint8)),
    ('caravan', np.array([0, 0, 90], dtype=np.uint8)),
    ('trailer', np.array([0, 0, 110], dtype=np.uint8)),
    ('train', np.array([0, 80, 100], dtype=np.uint8)),
    ('motorcycle', np.array([0, 0, 230], dtype=np.uint8)),
    ('bicycle', np.array([119, 11, 32], dtype=np.uint8))])


# camvid_colors = OrderedDict([
#     ("Animal", np.array([64, 128, 64], dtype=np.uint8)),
#     ("Archway", np.array([192, 0, 128], dtype=np.uint8)),
#     ("Bicyclist", np.array([0, 128, 192], dtype=np.uint8)),
#     ("Bridge", np.array([0, 128, 64], dtype=np.uint8)),
#     ("Building", np.array([128, 0, 0], dtype=np.uint8)),
#     ("Car", np.array([64, 0, 128], dtype=np.uint8)),
#     ("CartLuggagePram", np.array([64, 0, 192], dtype=np.uint8)),
#     ("Child", np.array([192, 128, 64], dtype=np.uint8)),
#     ("Column_Pole", np.array([192, 192, 128], dtype=np.uint8)),
#     ("Fence", np.array([64, 64, 128], dtype=np.uint8)),
#     ("LaneMkgsDriv", np.array([128, 0, 192], dtype=np.uint8)),
#     ("LaneMkgsNonDriv", np.array([192, 0, 64], dtype=np.uint8)),
#     ("Misc_Text", np.array([128, 128, 64], dtype=np.uint8)),
#     ("MotorcycleScooter", np.array([192, 0, 192], dtype=np.uint8)),
#     ("OtherMoving", np.array([128, 64, 64], dtype=np.uint8)),
#     ("ParkingBlock", np.array([64, 192, 128], dtype=np.uint8)),
#     ("Pedestrian", np.array([64, 64, 0], dtype=np.uint8)),
#     ("Road", np.array([128, 64, 128], dtype=np.uint8)),
#     ("RoadShoulder", np.array([128, 128, 192], dtype=np.uint8)),
#     ("Sidewalk", np.array([0, 0, 192], dtype=np.uint8)),
#     ("SignSymbol", np.array([192, 128, 128], dtype=np.uint8)),
#     ("Sky", np.array([128, 128, 128], dtype=np.uint8)),
#     ("SUVPickupTruck", np.array([64, 128, 192], dtype=np.uint8)),
#     ("TrafficCone", np.array([0, 0, 64], dtype=np.uint8)),
#     ("TrafficLight", np.array([0, 64, 64], dtype=np.uint8)),
#     ("Train", np.array([192, 64, 128], dtype=np.uint8)),
#     ("Tree", np.array([128, 128, 0], dtype=np.uint8)),
#     ("Truck_Bus", np.array([192, 128, 192], dtype=np.uint8)),
#     ("Tunnel", np.array([64, 0, 64], dtype=np.uint8)),
#     ("VegetationMisc", np.array([192, 192, 0], dtype=np.uint8)),
#     ("Wall", np.array([64, 192, 0], dtype=np.uint8)),
#     ("Void", np.array([0, 0, 0], dtype=np.uint8))
# ])


def convert_label_to_grayscale(im):
    out = (np.ones(im.shape[:2]) * 0).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = gray_val
    # assert (out != 255).all(), "rounding errors or missing classes in camvid_colors"
    return out.astype(np.uint8)


def label_to_color(im):
    out = (np.ones((im.shape[0], im.shape[1], 3)) * 255).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = np.where((im == gray_val))
        out[match_pxls] = rgb

    return out


def prepar_data(batch_file):
    # PATH="C:/Users/owner/Desktop/segmentaion/data/"
    # C=color_to_catagory()
    file_path = 'C:\\Users\\STL\\Downloads\\data_semantics\\training'

    file_name = os.path.join(file_path, 'labeled_images_{}.npy'.format(batch_file))
    if (os.path.isfile(file_name)):
        y = np.load(file_name)
    else:
        PATH = os.path.join(file_path, 'semantic_rgb\\', '*.png')
        y = []

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
        np.save(os.path.join(file_path, 'labeled_images_{}.npy'.format(batch_file)), y)
        # i+=1
        # count=0
        # y=[]

    file_name = os.path.join(file_path, 'input_images_{}.npy'.format(batch_file))
    if (os.path.isfile(file_name)):
        x = np.load(file_name)
    else:
        x = []
        PATH = os.path.join(file_path, 'image_2\\', '*.png')

        for image in glob.glob(PATH):
            im = cv2.imread(image)
            im = cv2.resize(im, (WIDTH, HEIGHT))
            for i in range(3):
                im[:, :, i] = cv2.equalizeHist(im[:, :, i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255

            # x.append(np.transpose(im, (2, 0, 1)))
            x.append(im)
            # count+=1
            # if(count==200):
        x = np.array(x)
        np.save(os.path.join(file_path, 'input_images_{}.npy'.format(batch_file)), x)
        # i+=1
        # count=0
        # x=[]

    return x, y


def train_model(x=[], y=[], PYTORCH=False):
    BATCH_SIZE = 16
    EPOCHS = 300
    ngpu = 1
    val_ratio = 0.7

    inx = np.random.permutation(range(x.shape[0]))
    x = x[inx, :, :, :]
    y = y[inx, :, :, :]
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
    best_valid_loss=1000
    if (PYTORCH):
        train_data = Dataset(x_train, y_train)
        valid_data = Dataset(x_valid, y_valid)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE * ngpu,
                                                       shuffle=True, num_workers=4,
                                                       pin_memory=True)  # divided into batches
        valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=ngpu,
                                                       shuffle=True, num_workers=4,
                                                       pin_memory=True)  # divided into batches
        model = UNetWithResNet(n_classes=len(camvid_colors))
        if (os.path.isfile(model_file)):
            state_dict = torch.load(model_file, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if (k[:7] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            model.load_state_dict(new_state_dict)
        model = model.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            model = nn.DataParallel(model, list(range(ngpu)), dim=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

        for epoch in range(EPOCHS):
            train_loss = 0
            valid_loss = 0
            total_train = 0
            correct_train = 0
            total_valid = 0
            correct_valid = 0
            model.train()

            for i, (x, y) in (enumerate(train_dataloader)):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x = x.transpose(3, 1).transpose(3, 2)
                y = y.transpose(3, 1).transpose(3, 2)
                o = model(x)

                loss = nn.BCELoss()(o, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # accuracy

                _, predicted = torch.max(o.data, 1)
                _, true = torch.max(y.data, 1)
                total_train += true.size(0) * true.size(1) * true.size(2)
                correct_train += predicted.eq(true.data).sum().item()

            train_accuracy = correct_train / total_train
            train_loss = train_loss / len(train_dataloader)

            torch.cuda.empty_cache()
            model.eval()
            for i, (x, y) in (enumerate(valid_dataloader)):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x = x.transpose(3, 1).transpose(3, 2)
                y = y.transpose(3, 1).transpose(3, 2)
                o = model(x)

                loss = nn.BCELoss()(o, y)

                valid_loss += loss.item()

                _, predicted = torch.max(o.data, 1)
                _, true = torch.max(y.data, 1)
                total_valid += true.size(0) * true.size(1) * true.size(2)
                correct_valid += predicted.eq(true.data).sum().item()

            valid_loss = valid_loss / len(train_dataloader)
            valid_accuracy = correct_valid / total_valid
            torch.cuda.empty_cache()
            print('[{}/{}]\tTrain loss: {}\t Train Acc: {}\tValid loss:{}\t Valid Acc:{}'.format(epoch, EPOCHS,
                                                                                                 train_loss,
                                                                                                 train_accuracy,
                                                                                                 valid_loss,
                                                                                                 valid_accuracy))
            if (best_valid_loss > np.abs(train_loss*valid_loss)):
                torch.save(model.state_dict(), '%s' % model_file)
                best_valid_loss = np.abs(train_loss*valid_loss)
                print("model saved")



    else:
        model = VGGSegnet(n_classes=len(camvid_colors), input_height=HEIGHT, input_width=WIDTH)
        model.summary()
        if os.path.isfile(weights_file_name):
            model.load_weights(weights_file_name)

        model_gpu = model
        # model_gpu=multi_gpu_model(model,gpus=2)

        optimizer = Adam(lr=0.001)
        # 'sparse_categorical_crossentropy'
        model_gpu.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model_gpu.fit(x_train, y_train, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                      validation_data=(x_valid, y_valid), callbacks=[])
        model_gpu.save_weights(weights_file_name, overwrite=True)
        del model_gpu
        K.clear_session()

    return valid_loss, valid_accuracy


def test_model(PYTORCH=PYTORCH):
    file_path = "C:\\Users\\STL\\Downloads\\data_semantics\\testing"
    # file_path="D:\\segmentaion\\data"
    ngpu = 1
    if (PYTORCH):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNetWithResNet(n_classes=len(camvid_colors))
        if (os.path.isfile(model_file)):
            state_dict = torch.load(model_file, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if (k[:7] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            model.load_state_dict(new_state_dict)
        model = model.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            model = nn.DataParallel(model, list(range(ngpu)), dim=1)

        PATH = os.path.join(file_path, 'image_2\\', '*.png')
        count = 0
        i = 0
        x_predict = []
        for image in glob.glob(PATH):
            # y=np.zeros((HEIGHT,WIDTH,len(camvid_colors)))

            im = cv2.imread(image)
            im = cv2.resize(im, (WIDTH, HEIGHT))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for i in range(3):
                im[:, :, i] = cv2.equalizeHist(im[:, :, i])
            im1 = im / 255

            im1 = torch.FloatTensor(im1).to(device).unsqueeze(0)
            im1 = im1.transpose(3, 1).transpose(3, 2)
            # for _ in range(3):

            y_est = model(im1)
            y_est = y_est.transpose(3, 1).transpose(1, 2)
            y_est = y_est.cpu().detach().numpy()

            y = y_est[0, :, :, :]
            y = np.argmax(y, axis=2)
            y = label_to_color(y)

            alpha = 0.7
            image_new = np.array(alpha * cv2.cvtColor(y, cv2.COLOR_BGR2RGB) + (1 - alpha) * im, dtype=np.uint8)
            cv2.imshow('im', image_new)
            cv2.waitKey(100)
            x_predict.append(y)

    else:
        model = VGGSegnet(n_classes=32, input_height=HEIGHT, input_width=WIDTH)
        model.summary()

        if os.path.isfile(weights_file_name):
            model.load_weights(weights_file_name)

        # x_test1 = np.expand_dims(x_test[20:30], axis=0)

        x = []
        PATH = os.path.join(file_path, 'Labeled\\', '*.png')
        count = 0
        i = 0
        x_predict = []
        for image in glob.glob(PATH):
            im = cv2.imread(image)
            im = cv2.resize(im, (WIDTH, HEIGHT))

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255

            y_est = model.predict(im.unsqueeze(0))

            y = y_est[i, :, :, :]
            y = np.argmax(y, axis=2)
            y = label_to_color(y)
            x_predict.append(y)

        del model
        K.clear_session()

    # PATH = os.path.join(file_path, 'semantic_rgb\\', '*.png')
    # count = 0
    # i = 0
    # y_test = []
    # for image in glob.glob(PATH):
    #     im = cv2.imread(image)
    #     im = cv2.resize(im, (WIDTH, HEIGHT))
    #
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     #plt.imshow(im)
    #     #plt.show()
    #     y_test.append(im)

    return x_predict


def one_hot_it(labels):
    x = np.zeros([labels.shape[0], labels.shape[1], 32])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            x[i, j, labels[i][j]] = 1
    return x


if __name__ == '__main__':
    VGG_Weights_path = "vgg19_weights.h5"
    weights_file_name = "weights_VGGsegnet.h5"
    model_file = "Unet_Resnet.pth"
    HEIGHT = 224  # 192#224#480
    WIDTH = 736  # 640 #736#672

    # vgg=VGG19(weights='imagenet',include_top=False)
    # vgg.summary()
    # vgg.save_weights('vgg19_weights.h5')
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='/Graph', histogram_freq=100,
    #                                          write_graph=True, write_images=True)

    x, y = prepar_data(1)
    #valid_accuracy_avg = []
    #for _ in range(5):
    #    valid_loss,valid_accuracy=train_model(x,y,PYTORCH=PYTORCH)
       #valid_accuracy_avg.append(valid_accuracy)

    x_predict = test_model()
    print(x_predict)

    # plt.figure()
    # y=y_test[50,:,:]
    # #y=np.reshape(y,(WIDTH,HIGHT,32))
    #
    # y=np.argmax(y,axis=2)
    # y=label_to_color(y)
    # plt.imshow(y)
    #
    # prepar_data()
