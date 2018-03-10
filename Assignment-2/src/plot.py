#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt

from helpers.read_csvs import read_optimization_function_data
from helpers.figures import construct_fitness, Figures

if __name__ == '__main__':
    # accuracy of all
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/BP/BP.tsv")
    x3, e3_tr, e3_tst, a3_tr, a3_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-10-20.tsv")
    x5, e5_tr, e5_tst, a5_tr, a5_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.1.tsv")
    x7, e7_tr, e7_tst, a7_tr, a7_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/RHC/RHC.tsv")
    f = Figures("Accuracy (best) Curves", "Iterations", "Accuracy (percentage)")
    f.start()
    f.plot_curve("BP", x1, a1_tst)
    f.plot_curve("GA-55-10-20", x3, a3_tst)
    f.plot_curve("SA-1E6-0.1 Accuracy", x5, a5_tst)
    f.plot_curve("RHC Accuracy", x7, a7_tst)
    f.finish()

    # GA Accuracy
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-10-10.tsv")
    x3, e3_tr, e3_tst, a3_tr, a3_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-10-20.tsv")
    x5, e5_tr, e5_tst, a5_tr, a5_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-20-10.tsv")
    x7, e7_tr, e7_tst, a7_tr, a7_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-20-20.tsv")
    f = Figures("GA Accuracy - Population of 55", "Iterations", "Accuracy (Percentage)")
    f.start()
    f.plot_curve("10-10 Accuracy", x1, a1_tst)
    f.plot_curve("10-20 Accuracy", x3, a3_tst)
    f.plot_curve("20-10 Accuracy", x5, a5_tst)
    f.plot_curve("20-20 Accuracy", x7, a7_tst)
    f.finish()

    # GA Error
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-10-10.tsv")
    x3, e3_tr, e3_tst, a3_tr, a3_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-10-20.tsv")
    x5, e5_tr, e5_tst, a5_tr, a5_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-20-10.tsv")
    x7, e7_tr, e7_tst, a7_tr, a7_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/GA/GA-55-20-20.tsv")
    f = Figures("GA Error Curves - Population of 55", "Iterations", "Error (Percentage)")
    f.start()
    f.plot_curve("10-10 Error", x1, e1_tst)
    f.plot_curve("10-20 Error", x3, e3_tst)
    f.plot_curve("20-10 Error", x5, e5_tst)
    f.plot_curve("20-20 Error", x7, e7_tst)
    f.finish()

    # SA Error
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.1.tsv")
    x3, e3_tr, e3_tst, a3_tr, a3_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.3.tsv")
    x5, e5_tr, e5_tst, a5_tr, a5_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.5.tsv")
    x7, e7_tr, e7_tst, a7_tr, a7_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.7.tsv")
    x9, e9_tr, e9_tst, a9_tr, a9_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.9.tsv")
    f = Figures("SA Error Curves - Temperature of 1E6", "Iterations", "Error (Percentage)")
    f.start()
    f.plot_curve("SA-0.1 Error", x1, e1_tst)
    f.plot_curve("SA-0.3 Error", x3, e3_tst)
    f.plot_curve("SA-0.5 Error", x5, e5_tst)
    f.plot_curve("SA-0.7 Error", x7, e7_tst)
    f.plot_curve("SA-0.9 Error", x9, e9_tst)
    f.finish()

    # SA Accuracy
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.1.tsv")
    x3, e3_tr, e3_tst, a3_tr, a3_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.3.tsv")
    x5, e5_tr, e5_tst, a5_tr, a5_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.5.tsv")
    x7, e7_tr, e7_tst, a7_tr, a7_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.7.tsv")
    x9, e9_tr, e9_tst, a9_tr, a9_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/SA/SA-0.9.tsv")
    f = Figures("SA Accuracy Curves - Temperature of 1E6", "Iterations", "Accuracy (Percentage)")
    f.start()
    f.plot_curve("SA-0.1 Accuracy", x1, a1_tst)
    f.plot_curve("SA-0.3 Accuracy", x3, a3_tst)
    f.plot_curve("SA-0.5 Accuracy", x5, a5_tst)
    f.plot_curve("SA-0.7 Accuracy", x7, a7_tst)
    f.plot_curve("SA-0.9 Accuracy", x9, a9_tst)
    f.finish()

    # RHC Error
    x1, e1_tr, e1_tst, a1_tr, a1_tst, _ = read_optimization_function_data("csv/OPTIMIZATION_FUNCTIONS/RHC/RHC.tsv")
    f = Figures("RHC Error Curve", "Iterations", "Error (Percentage)")
    f.start()
    f.plot_curve("Training Error", x1, e1_tr)
    f.plot_curve("Validation Error", x1, e1_tst)
    f.finish()

    # RHC Accuracy
    f = Figures("RHC Accuracy", "Iterations", "Accuracy (Percentage)")
    f.start()
    f.plot_curve("Training Accuracy", x1, a1_tr)
    f.plot_curve("Validation Accuracy", x1, a1_tst)
    f.finish()

    plt.show()
