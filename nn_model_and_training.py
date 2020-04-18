import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_input_data_pipeline():
    csv_file = 'data/8/all_data.csv'

    input_data_pipeline = tf.data.experimental.make_csv_dataset(csv_file, batch_size=9, label_name='Action',
                                                                select_columns=['Action', 'State0', 'State1', 'State2',
                                                                                'State3', 'State4', 'State5', 'State6',
                                                                                'State7', 'State8'])
    input_batches = (input_data_pipeline.cache().repeat().shuffle(500)).prefetch(tf.data.experimental.AUTOTUNE)
    return input_batches


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
    # features_dataset = tf.data.Dataset.from_tensor_slices(features)
    # labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    # dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    for element in dataset:
        print(element)


def load_data_winners_pandas():
    csv_folder = 'data/9/'
    states = pd.read_csv(csv_folder + 'states.csv')
    numpy_states = states.to_numpy()
    reformatted_states = numpy_states
    print(reformatted_states.shape)

    actions = pd.read_csv(csv_folder + 'winners.csv')
    actions_reformatted = actions.to_numpy().flatten()
    print(actions_reformatted.shape)

    dataset = tf.data.Dataset.from_tensor_slices((reformatted_states, actions_reformatted))
    dataset = dataset.shuffle(1000).batch(32)
    return dataset


def load_data_pandas(folder_number=1, output_type='winners', batch_size=50, shuffle_cache=1000):
    csv_folder = 'data/' + str(folder_number) + '/'

    states = pd.read_csv(csv_folder + 'states.csv').to_numpy()
    outputs = pd.read_csv(csv_folder + output_type + '.csv').to_numpy().flatten()
    if output_type == 'actions':
        outputs = tf.keras.utils.to_categorical(outputs)

    states_train, states_test, output_train, output_test = train_test_split(states, outputs, test_size=0.33,
                                                                            random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((states_train, output_train))
    train_dataset = train_dataset.shuffle(shuffle_cache).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((states_test, output_test))
    test_dataset = test_dataset.shuffle(shuffle_cache).batch(batch_size)
    return train_dataset, test_dataset


def create_model(output_type='winners'):
    model = models.Sequential()
    model.add(layers.Dense(
        16,
        activation="relu",
        input_shape=(9,)
    ))
    model.add(layers.Dense(16, activation="relu"))

    if output_type == 'winners':
        model.add(layers.Dense(1, activation='tanh'))
        model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",  # or try 'hinge' loss function
            metrics=["accuracy"]
        )

    elif output_type == 'actions':
        model.add(layers.Dense(9, activation='softmax'))
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    return model


def create_callbacks_array(folder_number=12):
    folder = './data/' + str(folder_number) + '/'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=folder + 'mymodel',  # _{epoch}
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
    log_dir = folder + 'tensorboard_log'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [
        early_stopping,
        model_checkpoint,
        tensorboard_callback
    ]
    return callbacks


def visualize_history(history, folder_number):
    folder = './data/' + str(folder_number) + '/'

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(folder + 'acc_plot.png')
    # plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(folder + 'loss_plot.png')
    # plt.show()


def train_model():
    folder_number = 12
    batch_size = 50
    epochs = 30
    output = 'actions'

    train_dataset, test_dataset = load_data_pandas(folder_number=folder_number, output_type=output,
                                                   batch_size=batch_size)
    model = create_model(output_type=output)
    callbacks = create_callbacks_array(folder_number=folder_number)
    history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks,
                        validation_data=test_dataset)
    visualize_history(history=history, folder_number=folder_number)


if __name__ == '__main__':
    train_model()
