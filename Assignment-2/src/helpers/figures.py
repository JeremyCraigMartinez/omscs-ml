#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt
import numpy as np
from helpers.read_csvs import average, read_fitness_function_data

plot_colors = ('r', 'g', 'b', 'c', 'darkorange', 'black')

def construct_fitness(curves, title):
    fig = Figures(title, "Iterations", "Evaluation Values")
    fig.start()

    for c in curves:
        x, y, _ = average(read_fitness_function_data, c["curve_file"])
        fig.plot_curve(c["curve_title"], x, y)

    fig.finish()

def get_compare_figure(title, label, accuracy, times):

    plt.figure()

    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel(label, color=(0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)', color=(1, 0, 0, opacity))
    ax1.plot(color=(0, 0, 1, opacity))
    ax2.plot(color=(1, 0, 0, opacity))
    index = np.arange(4)+1
    bar_width = 0.25

    bar11 = ax1.bar(index, accuracy, bar_width, alpha=opacity, color='b')

    bar21 = ax2.bar(index + bar_width, times, bar_width, alpha=opacity, color='r')

    for rect in bar11:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height, '%.1f%%' % height, ha='center', va='bottom')

    for rect in bar21:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height, '%.2fs' % height, ha='center', va='bottom')

    name = ('BackProp', 'RHC', 'SA', 'GA')

    plt.xticks(index + bar_width/2, name)

    plt.legend((bar11, bar21), (label, 'Running time'), loc="best")
    plt.tight_layout()

    plt.savefig(title + ".png")

    return plt

def get_comparison(title, label1, label2, name, accuracy, times):

    plt.figure()

    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel(label1, color=(0, 0, 1, opacity))
    ax2.set_ylabel(label2, color=(1, 0, 0, opacity))
    ax1.plot(color=(0, 0, 1, opacity))
    ax2.plot(color=(1, 0, 0, opacity))
    index = np.arange(6) + 1
    bar_width = 0.3

    bar11 = ax1.bar(index - 0.1, accuracy, bar_width, alpha=opacity, color='b')

    bar21 = ax2.bar(index + bar_width + 0.1, times, bar_width, alpha=opacity, color='r')

    for rect in bar11:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height, '%.1f%%' % height, ha='center', va='bottom')

    for rect in bar21:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height, '%.1f%%' % height, ha='center', va='bottom')


    plt.xticks(index + bar_width/2, name)

    #plt.legend((bar11, bar21), (label1, label2), loc="best")
    plt.tight_layout()

    plt.savefig("plots/" + title + ".png")

    return plt

class Figures:
    def __init__(self, title, x_label, y_label):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.num = 0

    def start(self):
        plt.figure()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

    def finish(self):
        plt.legend(loc="best")
        plt.savefig("plots/" + self.title + ".png")
        return plt

    def plot_curve(self, label, param_range, values, plot=True):
        if (plot):
            plt.plot(param_range, values, label=label, color=plot_colors[self.num])
        else:
            plt.semilogx(param_range, values, label=label, color=plot_colors[self.num])

        self.num += 1
