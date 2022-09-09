import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import numpy as np
import pandas as pd
import pickle

def rescale_column(df,key):
    max = np.max(df[key])
    min = np.min(df[key])
    df[key] = (df[key] - min) / (max - min)
    return min,max

def unscale_data(key,scale_dict,value):
    return value*(scale_dict[key][1]-scale_dict[key][0])+scale_dict[key][0]

def rescale_data(df):
    scale_dict={}
    for key in df.keys():
        min,max = rescale_column(df,key)
        scale_dict[key]=[min,max]
    return scale_dict

def rescale_datapoint(key,scale_dict,value):
    return (value - scale_dict[key][0])/(scale_dict[key][1]-scale_dict[key][0])


class PFW_Controller():
    def __init__(self):
        self.normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        model = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='linear'),
            tf.keras.layers.Dense(5)
        ])
        model.build(input_shape=(None,5))
        self.model = model

    def fit(self, input_data, lr, epochs, is_train=True):
         input = input_data['INPUTS']
         output = input_data['TARGETS']
         data_length = len(input)
         validation_fraction = 0.25
         split_index = int(data_length * validation_fraction)

         validation_input = input[:split_index]
         validation_output = output[:split_index]

         train_input = input[split_index:]
         train_output = output[split_index:]

         dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
         dataset_val = tf.data.Dataset.from_tensor_slices((validation_input, validation_output))

         dataset = dataset.shuffle(10000).batch(2000)
         dataset_val = dataset_val.batch(500)

         callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=200)

         self.normalizer.adapt(train_input)

         self.model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                            loss=tf.keras.losses.MSE,
                            metrics=['mse']
                            )
         if is_train:
            history = self.model.fit(dataset,
                                      epochs=epochs,
                                      validation_data=dataset_val,
                                      callbacks=[callback],
                                      )

            loss = history.history['loss']
            val_loss = history.history['val_loss']
         
         return self.model
        
    def load_parameters(self):
        self.model.load_weights('PFW_Control.hd5')

    def predict(self, vals):
        return self.model.predict(vals)
        
