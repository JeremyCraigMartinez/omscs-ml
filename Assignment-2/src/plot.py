#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt

from helpers.figures import construct_fitness

if __name__ == '__main__':
    # Randomized Search Optimizations
    # Backpropagation (controlled variable for comparison)
    ###############################################################################
    '''x1, y1, t1 = average(read_fitness_function_data, "csv/OPTIMIZATION_FUNCTIONS/BP/BP.TSV")

    f = Figures("TSP MIMIC Evaluation Curves", "Iterations", "Evaluation Values")
    f.start()
    f.plot_curve("MIMIC-0.1-AVG", x1, y1)
    f.finish()'''

    # Fitness Functions
    # TSP
    ###############################################################################
    construct_fitness(
        [
            {"curve_file": "csv/TSP/MIMIC-55-20-0.1-NNN", "curve_title": "MIMIC-0.1-AVG"},
            {"curve_file": "csv/TSP/MIMIC-55-20-0.3-NNN", "curve_title": "MIMIC-0.3-AVG"},
            {"curve_file": "csv/TSP/MIMIC-55-20-0.5-NNN", "curve_title": "MIMIC-0.5-AVG"},
            {"curve_file": "csv/TSP/MIMIC-55-20-0.7-NNN", "curve_title": "MIMIC-0.7-AVG"},
            {"curve_file": "csv/TSP/MIMIC-55-20-0.9-NNN", "curve_title": "MIMIC-0.9-AVG"},
        ],
        "TSP MIMIC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/TSP/GA-55-10-10-NNN", "curve_title": "GA-55-10-10-AVG"},
            {"curve_file": "csv/TSP/GA-55-10-20-NNN", "curve_title": "GA-55-10-20-AVG"},
            {"curve_file": "csv/TSP/GA-55-20-10-NNN", "curve_title": "GA-55-20-10-AVG"},
            {"curve_file": "csv/TSP/GA-55-20-20-NNN", "curve_title": "GA-55-20-20-AVG"},
        ],
        "TSP GA Evaluation Curves")

    construct_fitness(
        [{"curve_file": "csv/TSP/RHC-NNN", "curve_title": "RHC-AVG"}],
        "TSP RHC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/TSP/SA-0.1-NNN", "curve_title": "SA-0.1-AVG"},
            {"curve_file": "csv/TSP/SA-0.3-NNN", "curve_title": "SA-0.3-AVG"},
            {"curve_file": "csv/TSP/SA-0.5-NNN", "curve_title": "SA-0.5-AVG"},
            {"curve_file": "csv/TSP/SA-0.7-NNN", "curve_title": "SA-0.7-AVG"},
            {"curve_file": "csv/TSP/SA-0.9-NNN", "curve_title": "SA-0.9-AVG"},
        ],
        "TSP SA Evaluation Curves")


    construct_fitness(
        [
            {"curve_file": "csv/TSP/RHC-NNN", "curve_title": "RHC-AVG"},
            {"curve_file": "csv/TSP/GA-55-10-10-NNN", "curve_title": "GA-55-10-10-AVG"},
            {"curve_file": "csv/TSP/MIMIC-55-20-0.5-NNN", "curve_title": "MIMIC-0.5-AVG"},
            {"curve_file": "csv/TSP/SA-0.7-NNN", "curve_title": "SA-0.7-AVG"},
        ],
        "TSP (BEST) Evaluation Curves")

    # FLIPFLOP
    ###############################################################################
    construct_fitness(
        [
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.1-NNN", "curve_title": "MIMIC-0.1-AVG"},
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.3-NNN", "curve_title": "MIMIC-0.3-AVG"},
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.5-NNN", "curve_title": "MIMIC-0.5-AVG"},
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.7-NNN", "curve_title": "MIMIC-0.7-AVG"},
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.9-NNN", "curve_title": "MIMIC-0.9-AVG"},
        ],
        "FLIPFLOP MIMIC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/FLIPFLOP/GA-55-10-10-NNN", "curve_title": "GA-55-10-10-AVG"},
            {"curve_file": "csv/FLIPFLOP/GA-55-10-20-NNN", "curve_title": "GA-55-10-20-AVG"},
            {"curve_file": "csv/FLIPFLOP/GA-55-20-10-NNN", "curve_title": "GA-55-20-10-AVG"},
            {"curve_file": "csv/FLIPFLOP/GA-55-20-20-NNN", "curve_title": "GA-55-20-20-AVG"},
        ],
        "FLIPFLOP GA Evaluation Curves")

    construct_fitness(
        [{"curve_file": "csv/FLIPFLOP/RHC-NNN", "curve_title": "RHC-AVG"}],
        "FLIPFLOP RHC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/FLIPFLOP/SA-0.1-NNN", "curve_title": "SA-0.1-AVG"},
            {"curve_file": "csv/FLIPFLOP/SA-0.3-NNN", "curve_title": "SA-0.3-AVG"},
            {"curve_file": "csv/FLIPFLOP/SA-0.5-NNN", "curve_title": "SA-0.5-AVG"},
            {"curve_file": "csv/FLIPFLOP/SA-0.7-NNN", "curve_title": "SA-0.7-AVG"},
            {"curve_file": "csv/FLIPFLOP/SA-0.9-NNN", "curve_title": "SA-0.9-AVG"},
        ],
        "FLIPFLOP SA Evaluation Curves")


    construct_fitness(
        [
            {"curve_file": "csv/FLIPFLOP/RHC-NNN", "curve_title": "RHC-AVG"},
            {"curve_file": "csv/FLIPFLOP/GA-55-20-10-NNN", "curve_title": "GA-55-20-10-AVG"},
            {"curve_file": "csv/FLIPFLOP/MIMIC-55-20-0.3-NNN", "curve_title": "MIMIC-0.3-AVG"},
            {"curve_file": "csv/FLIPFLOP/SA-0.7-NNN", "curve_title": "SA-0.7-AVG"},
        ],
        "FLIPFLOP (BEST) Evaluation Curves")

    # PEAKS
    ###############################################################################
    construct_fitness(
        [
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.1-NNN", "curve_title": "MIMIC-0.1-AVG"},
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.3-NNN", "curve_title": "MIMIC-0.3-AVG"},
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.5-NNN", "curve_title": "MIMIC-0.5-AVG"},
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.7-NNN", "curve_title": "MIMIC-0.7-AVG"},
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.9-NNN", "curve_title": "MIMIC-0.9-AVG"},
        ],
        "PEAKS MIMIC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/PEAKS/GA-55-10-10-NNN", "curve_title": "GA-55-10-10-AVG"},
            {"curve_file": "csv/PEAKS/GA-55-10-20-NNN", "curve_title": "GA-55-10-20-AVG"},
            {"curve_file": "csv/PEAKS/GA-55-20-10-NNN", "curve_title": "GA-55-20-10-AVG"},
            {"curve_file": "csv/PEAKS/GA-55-20-20-NNN", "curve_title": "GA-55-20-20-AVG"},
        ],
        "PEAKS GA Evaluation Curves")

    construct_fitness(
        [{"curve_file": "csv/PEAKS/RHC-NNN", "curve_title": "RHC-AVG"}],
        "PEAKS RHC Evaluation Curves")

    construct_fitness(
        [
            {"curve_file": "csv/PEAKS/SA-0.1-NNN", "curve_title": "SA-0.1-AVG"},
            {"curve_file": "csv/PEAKS/SA-0.3-NNN", "curve_title": "SA-0.3-AVG"},
            {"curve_file": "csv/PEAKS/SA-0.5-NNN", "curve_title": "SA-0.5-AVG"},
            {"curve_file": "csv/PEAKS/SA-0.7-NNN", "curve_title": "SA-0.7-AVG"},
            {"curve_file": "csv/PEAKS/SA-0.9-NNN", "curve_title": "SA-0.9-AVG"},
        ],
        "PEAKS SA Evaluation Curves")


    construct_fitness(
        [
            {"curve_file": "csv/PEAKS/RHC-NNN", "curve_title": "RHC-AVG"},
            {"curve_file": "csv/PEAKS/GA-55-20-20-NNN", "curve_title": "GA-55-20-20-AVG"},
            {"curve_file": "csv/PEAKS/MIMIC-55-20-0.1-NNN", "curve_title": "MIMIC-0.1-AVG"},
            {"curve_file": "csv/PEAKS/SA-0.1-NNN", "curve_title": "SA-0.1-AVG"},
        ],
        "PEAKS (BEST) Evaluation Curves")

    plt.show()
