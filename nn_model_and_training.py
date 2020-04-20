import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier
from definitions import WINNERS, ACTIONS


# def create_input_data_pipeline():
#     csv_file = 'data/8/all_data.csv'
#
#     input_data_pipeline = tf.data.experimental.make_csv_dataset(csv_file, batch_size=9, label_name='Action',
#                                                                 select_columns=['Action', 'State0', 'State1', 'State2',
#                                                                                 'State3', 'State4', 'State5', 'State6',
#                                                                                 'State7', 'State8'])
#     input_batches = (input_data_pipeline.cache().repeat().shuffle(500)).prefetch(tf.data.experimental.AUTOTUNE)
#     return input_batches
#
#
# def test_load_data():
#     input_data_pipeline = create_input_data_pipeline()
#     for feature_batch, label_batch in input_data_pipeline.take(1):
#         print('Actions: {}'.format(label_batch))
#         for key, value in feature_batch.items():
#             print("  {!r:20s}: {}".format(key, value))
#
#
# def load_data_constants_test():
#     labels = tf.constant([4, 0, 2, 6, 3, 5])
#     features = tf.constant([[0, 0, 0, 0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 1, 0, 0, 0, 0],
#                             [-1, 0, 1, 0, 1, 0, -1, 0, 0], [-1, 0, 1, 1, 1, 0, -1, 0, 0],
#                             [-1, 0, 1, 1, 1, -1, -1, 0, 0]])
#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     # features_dataset = tf.data.Dataset.from_tensor_slices(features)
#     # labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
#     # dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
#     for element in dataset:
#         print(element)
#
#
# def load_data_winners_pandas():
#     csv_folder = 'data/9/'
#     states = pd.read_csv(csv_folder + 'states.csv')
#     numpy_states = states.to_numpy()
#     reformatted_states = numpy_states
#     print(reformatted_states.shape)
#
#     actions = pd.read_csv(csv_folder + 'winners.csv')
#     actions_reformatted = actions.to_numpy().flatten()
#     print(actions_reformatted.shape)
#
#     dataset = tf.data.Dataset.from_tensor_slices((reformatted_states, actions_reformatted))
#     dataset = dataset.shuffle(1000).batch(32)
#     return dataset


def load_data_to_array(folder_number, output_type):
    csv_folder = 'data/' + str(folder_number) + '/'

    states = pd.read_csv(csv_folder + 'states.csv').to_numpy()
    outputs = pd.read_csv(csv_folder + output_type + '.csv').to_numpy().flatten()
    if output_type == 'actions':
        outputs = tf.keras.utils.to_categorical(outputs)

    states_train, states_test, output_train, output_test = train_test_split(states, outputs, test_size=0.33,
                                                                            random_state=42)

    return states_train, states_test, output_train, output_test


def create_dataset_from_arrays(states_train, states_test, output_train, output_test, batch_size=32, shuffle_cache=1000):
    train_dataset = tf.data.Dataset.from_tensor_slices((states_train, output_train))
    train_dataset = train_dataset.shuffle(shuffle_cache).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((states_test, output_test))
    test_dataset = test_dataset.shuffle(shuffle_cache).batch(batch_size)
    return train_dataset, test_dataset


def load_data_pandas(folder_number, output_type, batch_size=32, shuffle_cache=1000):
    states_train, states_test, output_train, output_test = load_data_to_array(folder_number, output_type)
    train_dataset, test_dataset = create_dataset_from_arrays(states_train, states_test, output_train, output_test,
                                                             batch_size, shuffle_cache)

    return train_dataset, test_dataset


def create_model(output_type, num_inner_nodes=32, inner_act_ft='relu'):
    model = models.Sequential()
    model.add(layers.Dense(
        num_inner_nodes,
        activation=inner_act_ft,
        input_shape=(9,)
    ))
    model.add(layers.Dense(num_inner_nodes, activation=inner_act_ft))

    if output_type == WINNERS:
        model.add(layers.Dense(1, activation='tanh'))
        model.compile(
            optimizer="rmsprop",
            loss="mse",  # binary_crossentropy does not work because cross entropy works in range 0 .. 1
            metrics=["accuracy"]
        )

    elif output_type == ACTIONS:
        model.add(layers.Dense(9, activation='softmax'))
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",  # or try 'hinge' loss function
            metrics=["accuracy"]
        )

    return model


def create_callbacks_array(folder_number, output):
    folder = './data/' + str(folder_number) + '/'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,

        # Should we set restore best values = True here to set model to best values ?

        verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=folder + 'mymodel_' + output,  # _{epoch}
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


def train_model(folder_number, output, batch_size=32, epochs=30, num_inner_nodes=32, inner_act_ft='relu'):
    train_dataset, test_dataset = load_data_pandas(folder_number=folder_number, output_type=output,
                                                   batch_size=batch_size)
    model = create_model(output_type=output, num_inner_nodes=num_inner_nodes, inner_act_ft=inner_act_ft)
    callbacks = create_callbacks_array(folder_number=folder_number, output=output)
    history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks,
                        validation_data=test_dataset)
    visualize_history(history=history, folder_number=folder_number)


def load_model(folder_number, output):
    folder = 'data/' + str(folder_number) + '/mymodel_' + output
    model = tf.keras.models.load_model(folder)
    return model


def load_and_reuse_model(folder_number, output):
    model = load_model(folder_number, output)

    test_input = np.asarray([np.zeros(9, dtype=np.int32)])
    label = model.predict(test_input)
    print(label)
    # Check its architecture
    # new_model.summary()


def hyperparam_tuning_sklearn(folder_number=14, output_type='actions', epochs=15, batch_size=32):
    states_train, states_test, output_train, output_test = load_data_to_array(folder_number=folder_number,
                                                                              output_type=output_type)

    poss_values_num_inner_nodes = [16, 32, 64]
    poss_values_batch_size = [32, 64, 128]
    params = dict(num_inner_nodes=poss_values_num_inner_nodes, batch_size=poss_values_batch_size)
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size)

    np.random.seed(42)

    rscv = RandomizedSearchCV(model, param_distributions=params, cv=3, n_iter=15)
    rscv_results = rscv.fit(states_train, output_train)

    print('Best score is: {} using {}'.format(rscv_results.best_score_,
                                              rscv_results.best_params_))


if __name__ == '__main__':
    # load_and_reuse_model(13,'actions')
    train_model(folder_number=14, output=ACTIONS, batch_size=64, epochs=30, num_inner_nodes=32, inner_act_ft='relu')
    # train_model(folder_number=14, output='winners', batch_size=32, epochs=30, num_inner_nodes=16, inner_act_ft='relu')
    # hyperparam_tuning_sklearn(folder_number=14, output_type='actions', epochs=20)
