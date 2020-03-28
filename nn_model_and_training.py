import pandas as pd
import numpy as np
import tensorflow as tf


def load_data():
    csv_file = 'data/6/all_data.csv'
    # action_df = pd.read_csv(csv_file, index_col=None)
    # data_slice = tf.data.Dataset.from_tensor_slices(action_df)
    tf.enable_eager_execution()
    input_data_pipeline = tf.data.experimental.make_csv_dataset(csv_file, batch_size=4, label_name='Actions',
                                                                select_columns=['Actions', 'States'])

    for feature_batch, label_batch in input_data_pipeline.take(1):
        print('Actions: {}'.format(label_batch))
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))


if __name__ == '__main__':
    load_data()
