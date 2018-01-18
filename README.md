# kaggle_dog_breed_identification

## 1. 题目简介
- [Kaggel - Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)


- `train`：10222 张训练图片
- `label.csv`：10222 张训练图片对应的标签，标签共120种 size: 10223x2
- `test`：10357 张测试图片

## 2. 数据集预处理

- `mxnet.gluon.data.vision.ImageFolderDataset` 的 [API](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=imagefolderdataset#mxnet.gluon.data.vision.ImageFolderDataset)

```python
import pandas as pd
import shutil
import os
#-----------------------------------------------------#   
## 在for_train文件夹对应的类别下存放训练图片的软链接
df = pd.read_csv('labels.csv')
path = 'for_train'
if os.path.exists(path): shutil.rmtree(path)
for i, (fname, breed) in df.iterrows():
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2): os.makedirs(path2)
    # e.g. for_train/boston_bull/000bec180eb18c7604dcecc8fe0dba07.jpg
    os.symlink('../../train/%s.jpg' % fname, '%s/%s.jpg' % (path2, fname))
#-----------------------------------------------------#   
## 在for_test文件夹下存放测试图片的软链接
df = pd.read_csv('sample_submission.csv')
path = 'for_test'
breed = '0'
if os.path.exists(path): shutil.rmtree(path)
for fname in df['id']:
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2): os.makedirs(path2)
    ## e.g. for_test/0/00a3edd22dc7859c487a64777fc8d093.jpg
    os.symlink('../../test/%s.jpg' % fname, '%s/%s.jpg' % (path2, fname))
```

##3. 获得预训练模型的特征

- `mxnet.image` 的 [API](https://mxnet.incubator.apache.org/api/python/image/image.html)、
- `mxnet.gluon.model_zoo` 的 [API](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html)

```python
import numpy as np
import h5py
from tqdm import tqdm
 
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
#-----------------------------------------------------#   
## 数据预处理函数
def transform(data, label):
    data = data.astype('float32') / 255
    for pre in preprocessing: data = pre(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')
## 导出特征向量的函数
def get_features(net, data):
    features = []
    labels = []
    for X, y in tqdm(data):
        feature = net.features(X.as_in_context(ctx)) #只前向传播特征层
        features.append(feature.asnumpy())
        labels.append(y.asnumpy())   
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
#-----------------------------------------------------#   
## 导出训练集和测试集的特征向量
ctx = mx.gpu()
preprocessing = [image.ForceResizeAug((224,224)),
                 image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]),                                                               std=nd.array([0.229, 0.224, 0.225]))]
batch_size = 64
## 训练集：vgg16_bn, resnet152_v1, densenet161
preprocessing[0] = image.ForceResizeAug((224,224))
imgs = vision.ImageFolderDataset('for_train', transform=transform)
data = gluon.data.DataLoader(imgs, batch_size)
features_vgg, labels = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet, _   = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)
## 训练集：inception_v3
preprocessing[0] = image.ForceResizeAug((299,299))
imgs_299 = vision.ImageFolderDataset('for_train', transform=transform)
data_299 = gluon.data.DataLoader(imgs_299, 64)
features_inception, _ = get_features(models.inception_v3(pretrained=True, ctx=ctx), data)
## 训练集：保存特征向量
with h5py.File('features.h5', 'w') as f:
    f['vgg'] = features_vgg
    f['resnet'] = features_resnet
    f['densenet'] = features_densenet
    f['inception'] = features_inception
    f['labels'] = labels
#-----------------------------------------------------#   
## 测试集：vgg16_bn, resnet152_v1, densenet161
preprocessing[0] = image.ForceResizeAug((224,224))
imgs = vision.ImageFolderDataset('for_test', transform=transform)
data = gluon.data.DataLoader(imgs, 64)
features_vgg, _ = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet, _ = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)
## 测试集：inception_v3
preprocessing[0] = image.ForceResizeAug((299,299))
imgs_299 = vision.ImageFolderDataset('for_test', transform=transform)
data_299 = gluon.data.DataLoader(imgs_299, 64)
## 测试集：保存特征向量
with h5py.File('features_test.h5', 'w') as f:
    f['vgg'] = features_vgg
    f['resnet'] = features_resnet
    f['densenet'] = features_densenet
    f['inception'] = features_inception
```

## 4. 拼接特征，训练

```python
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
#-----------------------------------------------------#
## 定义计算精度的函数
def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()
def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps
#-----------------------------------------------------#
## 载入训练集的特征向量
with h5py.File('features.h5', 'r') as f:
    features_vgg = np.array(f['vgg'])
    features_resnet = np.array(f['resnet'])
    features_densenet = np.array(f['densenet'])
    features_inception = np.array(f['inception'])
    labels = np.array(f['labels'])
features_resnet = features_resnet.reshape(features_resnet.shape[:2])
features_inception = features_inception.reshape(features_inception.shape[:2])
features = np.concatenate([features_resnet, features_densenet, features_inception], axis=-1)
## 构造读取数据的迭代器
ctx = mx.gpu()
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
dataset_train = gluon.data.ArrayDataset(nd.array(X_train), nd.array(y_train))
dataset_val = gluon.data.ArrayDataset(nd.array(X_val), nd.array(y_val))
batch_size = 128
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)
## 定义模型
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))
net.initialize(ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainejur(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
## 训练
epochs = 50
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    val_loss, val_acc = evaluate(net, data_iter_val)
    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))
#-----------------------------------------------------#  
## 载入测试集的特征向量
with h5py.File('features_test.h5', 'r') as f:
    features_vgg_test = np.array(f['vgg'])
    features_resnet_test = np.array(f['resnet'])
    features_densenet_test = np.array(f['densenet'])
    features_inception_test = np.array(f['inception'])
features_resnet_test = features_resnet_test.reshape(features_resnet_test.shape[:2])
features_inception_test = features_inception_test.reshape(features_inception_test.shape[:2])
features_test = np.concatenate([features_resnet_test, features_densenet_test,                                                 features_inception_test], axis=-1)
## 预测，输出csv文件
output = nd.softmax(net(nd.array(features_test).as_in_context(ctx))).asnumpy()
df = pd.read_csv('sample_submission.csv')
for i, c in enumerate(df.columns[1:]): df[c] = output[:,i]
df.to_csv('pred.csv', index=None)
```

 

## 参考文章：
- [使用Gluon识别120种狗 (ImageNet Dogs)](http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-dog.html)
- [ypwhs/DogBreed_gluon](https://github.com/ypwhs/DogBreed_gluon/blob/master/README.md)