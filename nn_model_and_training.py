import pandas as pd
import numpy as np
import tensorflow as tf


def load_data():
    action_df = pd.read_csv('data/1/actions.csv')
    data_slice = tf.data.Dataset.from_tensor_slices(action_df)
