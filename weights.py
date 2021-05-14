import os
import tensorflow as tf
from effnetv2 import EffNetV2

def load_and_save(model_name):
    base_name = model_name.strip('-ft1k').strip('-21k')
    save_path = f'{model_name}.h5'
    
    if '-21k' in model_name:
        num_classes = 21843

    # build keras model
    model = EffNetV2(base_name, num_classes=num_classes)
    inputs = tf.random.uniform([1, 224, 224, 3])
    endpoints = model(inputs)

    ckpt = tf.train.latest_checkpoint(model_name)
    model.load_weights(ckpt)
    model.save_weights(save_path)
    
model_names = [
    'efficientnetv2-s-21k',
    'efficientnetv2-m-21k',
    'efficientnetv2-l-21k',
    'efficientnetv2-s-21k-ft1k',
    'efficientnetv2-m-21k-ft1k',
    'efficientnetv2-l-21k-ft1k',
]

for model_name in model_names:
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/{model_name}.tgz
    !tar -zxvf {model_name}.tgz
    load_and_save(model_name)