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
    train_size, train_scores, test_scores = learn_curve_res

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    mplot.grid()

    mplot.fill_between(train_size,
                       1 - train_scores_mean + train_scores_std,
                       1 - train_scores_mean - train_scores_std,
                       alpha=0.1, color=color)
    mplot.plot(train_size, 1 - train_scores_mean, 'o-',
               color=color[0], label=label)

    plt.fill_between(train_size,
                     1 - test_scores_mean + test_scores_std,
                     1 - test_scores_mean - test_scores_std,
                     alpha=0.1, color=color)
    plt.plot(train_size, 1 - test_scores_mean, 'o-',
             color=color[1], label='Test score')

def plot(line, X, y, title='', cv=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Error Rate")

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for (classifier, label) in line:
        try:
            plot_line(plt, classifier, label, X, y, cv=cv, color=[colors.pop(), colors.pop()])
        except:
            print('unable to plot classifier {}'.format(label))

    plt.legend(loc='best')
    plt.show()
