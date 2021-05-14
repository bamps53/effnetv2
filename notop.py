import tensorflow as tf
from effnetv2 import EffNetV2

model_names = [
    'efficientnetv2-s-21k',
    'efficientnetv2-m-21k',
    'efficientnetv2-l-21k',
    'efficientnetv2-s-21k-ft1k',
    'efficientnetv2-m-21k-ft1k',
    'efficientnetv2-l-21k-ft1k',
]

for model_name in model_names:
    base_name = model_name.strip('-ft1k').strip('-21k')
    save_path = f'{model_name}_notop.h5'
    model = EffNetV2.from_pretrained(model_name, include_top=True)
    base_model = EffNetV2(base_name, include_top=False)
    _ = base_model(tf.random.uniform([1, 224, 224, 3]))
  
    for var1, var2 in zip(model.trainable_variables, base_model.trainable_variables):
        var2.assign(var1.numpy())
    base_model.save_weights(save_path)
    print(save_path)
    