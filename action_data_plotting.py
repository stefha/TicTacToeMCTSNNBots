import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_action_data():
    action_data = pd.read_csv('data/1/actions.csv')
    flattened = action_data.values.flatten()
    min = flattened.min()
    max = flattened.max() + 2

    plt.hist(flattened, bins=np.arange(min, max), align='left')
    plt.show()


if __name__ == '__main__':
    plot_action_data()
