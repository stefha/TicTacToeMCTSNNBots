import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_action_data(folder_number=1, output='actions'):
    folder_path = './data/' + str(folder_number) + '/' + output + '.csv'
    data = pd.read_csv(folder_path)
    flattened = data.values.flatten()
    min = flattened.min()
    max = flattened.max() + 2

    plt.hist(flattened, bins=np.arange(min, max), align='left')
    plt.show()


if __name__ == '__main__':
    plot_action_data(folder_number=14, output='winners')
