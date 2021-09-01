
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model-path', type=str, default="logs/ppo2/Omnibase-v0_1", help='Relative path to model of directory')

    args = parser.parse_args()

    path_to_model_directory = args.model_path
    print("Found model monitor csv. Plotting...")

    x, y = ts2xy(load_results(path_to_model_directory), 'timesteps') # 'timesteps' must be in lower case!!!
    x = x[len(x) - len(y):]
    fig = plt.figure(path_to_model_directory)
    plt.plot(x,y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(path_to_model_directory)
    # plt.ion()
    plt.show()
