import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import init

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
with h5py.File('features_train_stanford.h5', 'r') as f:
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
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.001)
dataset_train = gluon.data.ArrayDataset(nd.array(X_train), nd.array(y_train))
dataset_val = gluon.data.ArrayDataset(nd.array(X_val), nd.array(y_val))
batch_size = 128
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

## 定义模型
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(512, activation='relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))
net.initialize(init=init.Xavier(), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
lr = 1e-4
wd = 1e-5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

## 训练
max_acc = 0.
epochs = 100
for epoch in range(epochs):
    if epoch > 70:
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4*0.5, 'wd': wd*2})
    if epoch > 90:
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4*0.5*0.5, 'wd': wd*2})


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
    '''
    if val_acc > max_acc:
        max_acc = val_acc
        net.save_params("net_best.params")
    else:
        net = nn.Sequential()
        with net.name_scope():
            net.add(nn.Dense(256, activation='relu'))
            net.add(nn.Dropout(0.5))
            net.add(nn.Dense(120))
        net.load_params("net_best.params", ctx=mx.gpu())
        lr = lr * 0.9
        wd = wd / 0.9
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    
    '''
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
df = pd.read_csv('../data/sample_submission.csv')
for i, c in enumerate(df.columns[1:]): df[c] = output[:,i]
df.to_csv('pred.csv', index=None)