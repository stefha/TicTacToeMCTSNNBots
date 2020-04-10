import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers


def create_input_data_pipeline():
    csv_file = 'data/8/all_data.csv'
    input_data_pipeline = tf.data.experimental.make_csv_dataset(csv_file, batch_size=4, label_name='Action',
                                                                select_columns=['Action', 'State0', 'State1', 'State2',
                                                                                'State3', 'State4', 'State5', 'State6',
                                                                                'State7', 'State8'])

    return input_data_pipeline


def test_load_data():
    input_data_pipeline = create_input_data_pipeline()
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


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(
        16,
        activation="relu",
        input_shape=(9,)
    ))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def use_model():
    input_data_pipeline = create_input_data_pipeline()
    data_batch = input_data_pipeline.take(1)
    model = create_model()
    for feature_batch, label_batch in data_batch:
        values = feature_batch.values()
        value0 = list(values)[0]
        print(value0)
        pred_label = model.predict(feature_batch)
        print('Actions: {}'.format(label_batch))
        print('Label predicted:' + pred_label[0][0])


if __name__ == '__main__':
    use_model()
