import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualization:
    """
    A class responsible for data visualization.

    Attributes:
        data_analizer (DataAnalizer): An object of the DataAnalizer class containing the dataset.

    Methods:
        perform_visualization(): Performs data visualization.
    """

    def __init__(self, data_analizer):
        """
        Initializes the DataVisualization class with a DataAnalizer object.
        """
        self.data_analizer = data_analizer

    def perform_visualization(self):
        """
        Performs data visualization.
        """
        print("Data Visualization Plots:")
        print("-------------------------")

        # Pairplot
        print("\nPairplot:")
        self.plot_pairplot()

        # Boxplot
        print("\nBoxplot:")
        self.plot_boxplot()

        # Ridgeplot
        print("\nRidgeplot:")
        self.plot_ridgeplot()

    def plot_pairplot(self):
        """
        Plots pairplot for all features.
        """
        sns.pairplot(self.data_analizer.data_train, diag_kind='kde')
        plt.title("Pairplot of Features")
        plt.show()

    def plot_boxplot(self):
        """
        Plots boxplot for all features.
        """
        # Create a single figure and axis for all boxplots
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot boxplots for each feature
        sns.boxplot(data=self.data_analizer.data_train, ax=ax)
        ax.set_title("Boxplot of all Features")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.set_xticks(range(len(self.data_analizer.data_train.columns)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_ridgeplot(self):
        """
        Plots overlapping densities (ridge plot) for all features.
        """

        # Create a single figure and axis for all boxplots
        fig, axes = plt.subplots(len(self.data_analizer.data_train.columns), 1, figsize=(10, 8), sharex=True)

        # Generate a gradient of darker colors for the plots
        num_plots = len(self.data_analizer.data_train.columns)
        cmap = plt.get_cmap('Blues')
        colors = [cmap(1 - i / (num_plots + 1)) for i in range(1, num_plots + 1)]

        # Plot overlapping densities for each numerical feature
        for i, (feature, color) in enumerate(zip(self.data_analizer.data_train.columns, colors)):
            sns.kdeplot(data=self.data_analizer.data_train[feature], ax=axes[i], color=color, fill=True, linewidth=2)
            axes[i].set_ylabel(feature, rotation=0, labelpad=40)  # Rotate y-axis label
            axes[i].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position
            axes[i].spines['top'].set_visible(False)

            # Remove box structure around the plots
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)

        # Adjust plot aesthetics
        axes[-1].set_xlabel("Value")

        plt.tight_layout()
        plt.show()
