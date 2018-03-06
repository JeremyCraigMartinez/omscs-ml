#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt

from helpers.read_csvs import average, read_fitness_function_data
from helpers.figures import Figures

if __name__ == '__main__':
    x1, y1, t1 = average(read_fitness_function_data, "csv/TSP/MIMIC-55-20-0.5-NNN")
    x2, y2, t2 = average(read_fitness_function_data, "csv/TSP/MIMIC-55-20-0.6-NNN")
    x3, y3, t3 = average(read_fitness_function_data, "csv/TSP/MIMIC-55-20-0.7-NNN")
    x4, y4, t4 = average(read_fitness_function_data, "csv/TSP/MIMIC-55-20-0.8-NNN")
    x5, y5, t5 = average(read_fitness_function_data, "csv/TSP/MIMIC-55-20-0.9-NNN")

    f = Figures("TSP Evaluation Curves", "Iterations", "Evaluation Values")
    f.start()
    f.plot_curve("MIMIC-0.5-AVG", x1, y1)
    f.plot_curve("MIMIC-0.6-AVG", x2, y2)
    f.plot_curve("MIMIC-0.7-AVG", x3, y3)
    f.plot_curve("MIMIC-0.8-AVG", x4, y4)
    f.plot_curve("MIMIC-0.9-AVG", x5, y5)
    f.finish()
    plt.show()
