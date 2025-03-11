import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    """
    Exploratory Data Analysis (EDA) class to analyze and visualize dataset properties.

    """

    def __init__(self, df: pd.DataFrame):
        """
            Initializes the EDA class with a pandas DataFrame.

            :param df: DataFrame containing the dataset to analyze.
        """
        self.df = df


    def dataset_overview(self):
        """
            Provides a general overview of the dataset:
            - Prints dataset information (data types, non-null counts, memory usage).
            - Prints summary statistics (mean, standard deviation, min, max, etc.).
            - Displays the count of missing values per column.
        """
        print("Dataset Info:\n")
        print(self.df.info())
        print("\n Summary Statistics:\n")
        print(self.df.describe())
        print("\nMissing Values:\n")
        print(self.df.isnull().sum())

    def visualize_distributions(self):
        """
            Visualizes the distribution of numerical features using histograms.
        """
        plt.figure(figsize=(12, 6))
        self.df.hist(figsize=(12, 8), bins=30)
        plt.suptitle("Feature Distributions", fontsize=16)
        plt.show()

    def boxplot_outliers(self):
        """
            Creates boxplots for numerical features to help detect outliers.
        """
        self.df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(12, 12), notch=True)
        plt.suptitle("Boxplots for Outlier Detection", fontsize=16)
        plt.show()

    def correlation_matrix(self):
        """
            Generates a heatmap showing correlations between numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.show()

    def pairplot_relationships(self):
        """
            Displays pairplots to visualize relationships between numerical features.
            Kernel Density Estimation (KDE) is used for diagonal plots.
        """
        sns.pairplot(self.df, diag_kind='kde')
        plt.show()

    def time_based_analysis(self, date_column, target_variable):
        """
            Analyzes trends over time by extracting temporal features from a datetime column.

            :param date_column: Name of the column containing datetime values.
            :param target_variable: The target variable to visualize against time.
        """
        if date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            self.df['year'] = self.df[date_column].dt.year
            self.df['month'] = self.df[date_column].dt.month
            self.df['day_of_week'] = self.df[date_column].dt.dayofweek

            plt.figure(figsize=(12, 5))
            sns.lineplot(x=self.df[date_column], y=self.df[target_variable])
            plt.title("Trend Over Time")
            plt.show()

    def categorical_data_analysis(self):
        """
            Analyzes categorical variables by displaying their distribution as bar charts.
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(y=self.df[col], order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.show()

    def run_eda(self, date_column=None, target_variable=None):
        """
            Runs a full exploratory data analysis (EDA) process.

            :param date_column: Optional, name of the column with datetime values.
            :param target_variable: Optional, target variable to analyze trends over time.
        """
        self.dataset_overview()
        self.visualize_distributions()
        self.boxplot_outliers()
        self.correlation_matrix()
        self.pairplot_relationships()
        if date_column and target_variable:
            self.time_based_analysis(date_column, target_variable)
        self.categorical_data_analysis()
        print("EDA completed successfully.")