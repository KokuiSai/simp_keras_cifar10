# 2020/4/7 10:32
# KerasでCIFAR-10の画像分類
# １０クラス： 飛行機 自動車 鳥 猫 鹿 犬 カエル うま 船 トラック

import os
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import load_model

### ステップ１：# Load model and weights
save_dir = os.path.join(os.getcwd(), 'trained_param')
model_name = 'cifar10_model_w.h5'
model_path = os.path.join(save_dir, model_name)
print(model_path)
model = load_model(model_path)
if model is None:
    print("No model")
#model = model_from_yaml(yaml_string)

model.summary()

### ステップ２：CIFAR10データの取得
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'test samples')

### ステップ３：データの加工
#データを正規化しよう
x_test = x_test.astype('float32') / 255.0
#one-hotベクトルに変換しよう
#y_test = to_categorical(y_test, 10)

### ステップ４：クラス分類のテスト
# ラベル番号とファッション種類の対応辞書
cifar10_dict = {
    0: "飛行機", 1: "自動車", 2: "鳥", 3: "猫", 4: "鹿",
    5: "犬", 6: "カエル", 7: "うま", 8: "船", 9: "トラック"}

acc = 0    # テスト画像の認識率
for cnt, img in enumerate(x_test):
    prob_dist = model.predict( img[None, ])[0]
    pred_label = prob_dist.argmax()
    if pred_label == y_test[cnt]:
        acc += 1
#    else:
#        print("==", cnt, pred_label, y_test[cnt])

acc = acc * 100.0 / x_test.shape[0]
print("CIFAR10 test image accuracy:  ", acc, "%")




