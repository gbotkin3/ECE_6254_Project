# Methods for Visualizing Data

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Function that creates a 3D scatterplot using the first 3 features (first 3 columns for all rows)
# of the data that is passed in (may need to reduce dimensionality to 3 components with PCA or other methods)
# All of our datasets have more than 3 features, so we use PCA to reduce the data to 3 principal components to use
# for the scatterplot.

# data_labels is the label of each sample (row) of the data used to color the points according to their labels

# show_plot is a flag that is used to specify whether to show a plot or not.

# dataset_name is used to distinguish between different plots


def scatter_plot_all(data, data_labels, show_plot, dataset_name):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X = np.zeros(len(data))
    Y = np.zeros(len(data))
    Z = np.zeros(len(data))

    for i in range(len(data)):
        X[i] = data[i][0]
        Y[i] = data[i][1]
        Z[i] = data[i][2]

    ax.scatter(X, Y, Z, c=data_labels)

    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')

    # If we want to add the title to the 3D scatterplot.
    #plt.title(dataset_name + " Dataset 3D Scatterplot")

    # Save this scatterplot figure
    plt.savefig("../results/figures/" + dataset_name + "_3D_scatterplot.png")

    if show_plot:
        plt.show()

    return


# This function creates a KDE plots for a passed in pandas dataframe
# dataframe is the pandas dataframe of the data we want to use to generate pairplots
# show_plot is the flag that specifies whether to show the plot or not
# also include the name associated with the dataset so that we can save the plots and title it.
def kde_map_all(dataframe, show_plot, dataset_name, xaxis_label=None):  # Need to Fix Labels

    #data_coc = np.concatenate((data, data_labels), axis = 1)
    #dataframe = pd.DataFrame(data_coc, columns=["Component 0", "Component 1", "Component 2", "Label"])
    sb.set_style(rc={'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})

    g = sb.FacetGrid(dataframe, col=dataframe.columns.values.tolist()[-1])
    #g.map(sb.kdeplot, "Component 0", color = 'r', legend = True)
    #g.map(sb.kdeplot, "Component 1", color = 'g', legend = True)
    #g.map(sb.kdeplot, "Component 2", color = 'b', legend = True)
    colors = sb.color_palette()
    for i, col in enumerate(dataframe.columns.values.tolist()[:-1]):
        # if col == dataframe.columns.values.tolist()[-1]
        g.map(sb.kdeplot, col, color=colors[i], legend=True, label=col)
    # print(dataframe.columns.values.tolist()[:-1])
    for ax in g.axes.flat:
        #print(ax)
        sb.despine(left=False, right=False, bottom=False, top=False, ax=ax)
        #ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_ticks_position('both')
        #ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_ticks_position('both')
        plt.setp(ax.yaxis.get_ticklabels(), visible=ax.is_first_col())
        plt.setp(ax.xaxis.get_ticklabels(), visible=ax.is_last_row())
    # g.map_dataframe(sb.kdeplot, dataframe.columns.values.tolist()[:-1], legend=True)

    if xaxis_label is not None:
        g.set_xlabels(xaxis_label)

    g.add_legend()

    plt.savefig("../results/figures/" + dataset_name + "_kde.png")

    if show_plot:
        plt.show()

    return

# Instead of passing in numpy rows and columns, let's pass in the pandas dataframe so that we can use Seaborn for pairplots.
# def pairplot_all(data, data_labels, show_plot):

# This function creates a plot of all possible pair plots
# The diagonal plots show a plot of the univariate distribution (marginal distribution of the data in each column)
# dataframe is the pandas dataframe of the data we want to use to generate pairplots
# show_plot is the flag that specifies whether to show the plot or not
# also include the name associated with the dataset so that we can save the plots and title it.


def pairplot_all(dataframe, show_plot, dataset_name):

    #data_coc = np.concatenate((data, data_labels), axis = 1)
    #dataframe = pd.DataFrame(data_coc, columns=["Component 0", "Component 1", "Component 2", "Component 3", "Label"])

    sb.set_style(rc={'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})

    # Grab the last column, which is the label column to color the samples (points) (distributions) based on their label
    # Use seaborn to generate all the pairplots for any 2 features and the univariate distributions for all features
    plots = sb.pairplot(dataframe, hue=dataframe.columns.values.tolist()[-1], grid_kws=dict(diag_sharey=False))
    # plots.box(on=True)
    # plt.box(on=True)
    for ax in plots.axes.flat:
        #print(ax)
        sb.despine(left=False, right=False, bottom=False, top=False, ax=ax)
        #ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_ticks_position('both')
        #ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_ticks_position('both')
        plt.setp(ax.yaxis.get_ticklabels(), visible=ax.is_first_col())
        plt.setp(ax.xaxis.get_ticklabels(), visible=ax.is_last_row())
    # save the figure of pair plots.
    plt.savefig("../results/figures/" + dataset_name + "_pairplots.png")

    # Title the plot based on the name of the dataset
    #plt.title(dataset_name + " Pair Plots")
    #plots.fig.suptitle(dataset_name + " Pair Plots", y=1.08)

    if show_plot:
        plt.show()

    return


def heatmap():  # ASK PROF

    return
