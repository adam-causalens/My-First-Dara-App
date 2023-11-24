import pandas as pd
from sklearn import datasets

wine = datasets.load_wine()  # https://archive.ics.uci.edu/dataset/109/wine
features = wine.feature_names
target_names = wine.target_names
data = pd.DataFrame(wine.data, columns=features)
data['species'] = wine.target
data['species_names'] = data['species'].map(
    {i: name for i, name in enumerate(target_names)}
)

# Titanic dataset - todo: figure out how load this in Dara, not just locally
"""
# Load data
df = pd.read_csv("../data/train.csv").reset_index(drop=True).set_index("PassengerId")
data = DataVariable(df, cache=CacheType.GLOBAL)

print(df.head(2))

# Specify features
features = df.columns.to_list()
for c in ["Survived", "Name", "Ticket"]:
    print(f"Dropped column '{c}'")
    features.remove(c)

# Specify all possible values for target
target_names = df["Survived"].unique()
"""
