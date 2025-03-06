
# Reutilizar no main.ipynb para visualização ????

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataFrameDisplayer:
    def __init__(self, df):
        self.df = df

    def plot_column(self, column: str, plot_type: str) -> None:
        """
        This method plots a column data with one of the plot types
        :param column: column name from dataframe
        :param plot_type: should be in ['hist', 'violin', 'box', 'scatter', 'lines', 'bar']
        :return:
        """

        plot_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar']
        if plot_type not in plot_types:
            raise ValueError('plot_type must be in {}'.format(plot_types))

        if column not in self.df.columns:
            raise ValueError('column must be in {}'.format(column))

        plt.figure(figsize = (12, 8))

        if plot_type == 'hist':
            sns.histplot(self.df[column], kde=True, bins=30)
        elif plot_type == 'violin':
            sns.violinplot(y=self.df[column])
        elif plot_type == 'box':
            sns.boxplot(y=self.df[column])
        elif plot_type == 'scatter':
            if self.df.shape[1] < 2:
                raise ValueError("Scatter plots require at least two columns")
            sns.scatterplot(x=self.df.index, y=self.df[column])
        elif plot_type == 'line':
            sns.lineplot(x=self.df.index, y=self.df[column])
        elif plot_type == 'bar':
            sns.barplot(x=self.df.index, y=self.df[column])

        plt.title(f'{plot_type.capitalize()} plot of {column}')
        plt.xlabel(column)
        plt.ylabel("Values")
        plt.show()