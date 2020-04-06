# KerasでCNNを構築しCIFAR-10の画像分類
# フレームワーク: Keras
# １０クラス： 飛行機 自動車 鳥 猫 鹿 犬 カエル うま 船 トラック

import random
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

import os
from keras.models import model_from_yaml

# for suppression of deprecation warning
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

### ステップ１：CIFAR10データの取得
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

### ステップ２：データの加工
#データを正規化しよう
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
#one-hotベクトルに変換しよう
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

### ステップ３：ネットワークの定義
model = Sequential()

# 1層目
model.add(Conv2D(filters=32, kernel_size=(5, 5), 
                 activation='relu', strides=(1, 1), 
                 padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 2層目
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 出力層
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax')) 

### ステップ４：学習の設定・準備  ## ステップ５：学習の実行
# エポック数：50
# ミニバッチ数：128
# 損失関数：交差エントロピー
# パラメータ更新(最適化)の手法：Adam
# 学習率：0.0003

max_epoch = 30
batchsize = 128
learning_rate = 0.0003

model.compile(
    loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=["accuracy"])

stack = model.fit(x_train, y_train,
              epochs=max_epoch, batch_size=batchsize,
              verbose=1,
              validation_split=0.1)

### テストデータの損失値
score = model.evaluate(x_test, y_test,verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save model and weights
save_dir = os.path.join(os.getcwd(), 'trained_param')
model_name = 'cifar10_model_w.h5'
model_yaml= 'cifar10_model.yaml'   # 重みを含まないアーキテクチャのみ

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

model_path = os.path.join(save_dir, model_yaml)
yaml_string = model.to_yaml()
open(model_path, 'w').write(yaml_string)





