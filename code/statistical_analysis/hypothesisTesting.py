import itertools
from scipy.stats import f_oneway, kruskal, ttest_ind


class HypothesisTesting:
    def __init__(self, data_analizer):
        self.data_analizer = data_analizer

    def _perform_anova_test(self, feature):
        """
        Perform an analysis of variance (ANOVA) test for the given feature and multiclass target variable.

        Parameters:
            feature (str): The name of the feature for which the ANOVA test is performed.

        Returns:
            p_value (float): The p-value from the ANOVA test.
            significant (bool): True if the null hypothesis is rejected, indicating a significant relationship.
        """
        # Split data based on the target variable
        data_groups = [self.data_analizer.data_train.loc[self.data_analizer.labels_train == label, feature] for label in
                       self.data_analizer.labels_train.unique()]

        # Perform ANOVA test
        f_statistic, p_value = f_oneway(*data_groups)

        # Check significance
        significance = p_value < 0.05

        return p_value, significance

    def _perform_kruskal_test(self, feature):
        """
        Perform a Kruskal-Wallis test for the given feature and multiclass target variable.

        Parameters:
            feature (str): The name of the feature for which the Kruskal-Wallis test is performed.

        Returns:
            p_value (float): The p-value from the Kruskal-Wallis test.
            significant (bool): True if the null hypothesis is rejected, indicating a significant relationship.
        """
        # Split data based on the target variable
        data_groups = [self.data_analizer.data_train.loc[self.data_analizer.labels_train == label, feature] for label in
                       self.data_analizer.labels_train.unique()]

        # Perform Kruskal-Wallis test
        h_statistic, p_value = kruskal(*data_groups)

        # Check significance
        significance = p_value < 0.05

        return p_value, significance

    def anova_results(self):
        """
        Prints the ANOVA results
        """

        print("\n\nANOVA results:")
        for feature in self.data_analizer.data_train.columns:
            p_value, significant = self._perform_anova_test(feature)
            print(f"Feature: {feature}, p-value: {p_value}, Significant: {significant}")

    def kruskal_wallis_results(self):
        """
        Prints the kruskal wallis results in a tabular format.
        """

        print("\n\nKruskal-Wallis results:")
        for feature in self.data_analizer.data_train.columns:
            p_value, significant = self._perform_kruskal_test(feature)
            print(f"Feature: {feature}, p-value: {p_value}, Significant: {significant}")

    def t_test_results(self):
        """
        Prints the t-test results in a tabular format.
        """

        print("\n\nT-test results:")
        for feature in self.data_analizer.data_train.columns:
            print(f"Feature: {feature}")
            for class1, class2 in itertools.combinations(self.data_analizer.labels_train.unique(), 2):
                print(f"Class 1: {class1},  Class 2: {class2}")
                data_class1 = self.data_analizer.data_train.loc[self.data_analizer.labels_train == class1, feature]
                data_class2 = self.data_analizer.data_train.loc[self.data_analizer.labels_train == class2, feature]
                t_statistic, p_value = ttest_ind(data_class1, data_class2)
                print('Significant' if p_value < 0.05 else 'Not Significant')