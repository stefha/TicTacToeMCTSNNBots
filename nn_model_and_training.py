import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

import ast


def load_data():
    csv_file = 'data/6/all_data.csv'
    # action_df = pd.read_csv(csv_file, index_col=None)
    # data_slice = tf.data.Dataset.from_tensor_slices(action_df)

    tf.enable_eager_execution()
    input_data_pipeline = tf.data.experimental.make_csv_dataset(csv_file, batch_size=4, label_name='Actions',
                                                                select_columns=['Actions', 'States'])
    # input_data_pipeline = input_data_pipeline[2].eval()
    for feature_batch, label_batch in input_data_pipeline.take(1):
        print('Actions: {}'.format(label_batch))
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))


def load_data_constants_test():
    labels = tf.constant([4, 0, 2, 6, 3, 5])
    features = tf.constant([[0, 0, 0, 0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [-1, 0, 1, 0, 1, 0, -1, 0, 0], [-1, 0, 1, 1, 1, 0, -1, 0, 0],
                            [-1, 0, 1, 1, 1, -1, -1, 0, 0]])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    for element in dataset:
        print(element)


def load_data_pandas():
    csv_file = 'data/7/all_data.csv'
    action_df = pd.read_csv(csv_file, index_col=None, converters={2: converterer})

    data_slice = tf.data.Dataset.from_tensor_slices(action_df)


def converterer(string):
    new = ast.literal_eval(string)
    # new = np.asarray(new)
    return new


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(
        16,
        activation="relu",
        input_shape=(9,)
    ))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    pred_label = model.predict(np.array([[-1, 0, 1, 0, 1, 0, -1, 0, 0], [-1, 0, 1, 0, 1, 0, 0, 0, 0]]))
    print(pred_label[0][0])
    print(pred_label[1][0])


if __name__ == '__main__':
    #   test = eval('[0 1 0 1]')  #ast.literal_
    #  print(test)
    #    load_data()
    # create_model()
    load_data_constants_test()
