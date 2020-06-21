import numpy as np
PYTORCH=True
if (PYTORCH):
    import torch
    import torchvision
    from torchsummary import summary
    from torch import nn
    import torch.utils.data as data
    import torch.nn.functional as F

else:
    from keras.utils import multi_gpu_model
    from keras.models import Model,Sequential
    from keras.layers import Input,Dense, Conv2D, MaxPooling2D,Flatten,ZeroPadding2D, Reshape, Permute, Activation,UpSampling2D,Dropout,concatenate
    from keras.layers.normalization import BatchNormalization
    from keras import backend as K
    from keras.applications.vgg19 import VGG19



class Dataset(data.Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)


        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)


    def __getitem__(self, index):
        return self.x[index,:, :, :], self.y[index,:,:,:]

    def __len__(self):
        return self.x.size(0)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super(Bridge,self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super(UpBlockForUNet,self).__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithMobileNet(nn.Module):

    def __init__(self, n_classes=2):
        super(UNetWithMobileNet, self).__init__()

        base = torchvision.models.mobilenet_v2(pretrained=True)
        #summary(base,(3,224,224))
        down_blocks = []
        up_blocks = []
        self.input_block = list(base.children())[0]

        for bottleneck in self.input_block:
            down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bridge = Bridge(1280, 1280)

        up_blocks.append(UpBlockForUNet(in_channels=1024+96, out_channels=1024,
                                        up_conv_in_channels=1280, up_conv_out_channels=1024))

        up_blocks.append(UpBlockForUNet(in_channels=512 + 32, out_channels=512,
                                        up_conv_in_channels=1024, up_conv_out_channels=512))

        up_blocks.append(UpBlockForUNet(in_channels=256 + 24, out_channels=256,
                                        up_conv_in_channels=512, up_conv_out_channels=256))

        up_blocks.append(UpBlockForUNet(in_channels=128 + 16, out_channels=128,
                                        up_conv_in_channels=256, up_conv_out_channels=128))

        up_blocks.append(UpBlockForUNet(in_channels=64 + 3, out_channels=64,
                                        up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()

        downsamples_layers=[0,2,3,7,14]
        pre_pools[f"layer_{0}"] = x
        for i, block in enumerate(self.down_blocks,1):
            x = block(x)
            pre_pools[f"layer_{i}"] = x


        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks):
            x = block(x, pre_pools[f"layer_{downsamples_layers[len(downsamples_layers)-1-i]}"])
        output_feature_map = x
        x = self.out(x)
        x = F.sigmoid(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


class UNetWithResNet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super(UNetWithResNet,self).__init__()

        base = torchvision.models.resnet18(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(base.children()))[:3]
        self.input_pool = list(base.children())[3]
        for bottleneck in list(base.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        #self.bridge = Bridge(2048, 2048)

        up_blocks.append(UpBlockForUNet(512, 256))
        up_blocks.append(UpBlockForUNet(256, 128))
        up_blocks.append(UpBlockForUNet(128, 64))
        up_blocks.append(UpBlockForUNet(in_channels=32 + 64, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks.append(UpBlockForUNet(in_channels=16 + 3, out_channels=16,
                                                    up_conv_in_channels=32, up_conv_out_channels=16))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(16, n_classes,padding=1,kernel_size=3, stride=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        #x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        x = self.Softmax(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


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




