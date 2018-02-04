import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_line(mplot, classifier, label, X, y, cv=None,
              n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10),
              scoring=None, color=None):
    learn_curve_res = learning_curve(classifier, X, y,
                                     cv=cv, n_jobs=n_jobs,
                                     train_sizes=train_sizes,
                                     scoring=scoring)
    train_size, train_scores, _ = learn_curve_res

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    mplot.grid()

    mplot.fill_between(train_size,
                       1 - train_scores_mean + train_scores_std,
                       1 - train_scores_mean - train_scores_std,
                       alpha=0.1, color=color)
    mplot.plot(train_size, 1 - train_scores_mean, 'o-',
               color=color, label=label)


def plot(line, X, y, cv=None):
    plt.figure()
    plt.title('All algorithms')
    plt.xlabel("Training examples")
    plt.ylabel("Cost = - 1 Score")

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for (classifier, label) in line:
        plot_line(plt, classifier, label, X, y, cv=cv, color=colors.pop())

    plt.legend(loc='best')
    plt.show()
