import pandas
from sklearn import datasets
from tabulate import tabulate


iris = datasets.load_iris()  # todo: replace with other dataset
features = iris.feature_names
target_names = iris.target_names
data = pandas.DataFrame(iris.data, columns=features)
data['species'] = iris.target
data['species_names'] = data['species'].map(
    {i: name for i, name in enumerate(target_names)}
)
