from dara.components import Card, Heading, Stack, Text


def home_page():
    return Card(
        Stack(
            Heading('Wine'),
            Text('This web app shows the capabilities of Dara through a classic Machine Learning Problem.'),
            Text("""These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
            I think that the initial data set had around 30 variables, but for some reason I only have the 13 dimensional version. I had a list of what the 30 or so variables were, but a.)  I lost it, and b.), I would not know which 13 variables are included in the set.
            """),
            Text("The attributes are:"),
            Text("    1) Alcohol"),
            Text("    2) Malic acid"),
            Text("    3) Ash"),
            Text("    4) Alcalinity of ash  "),
            Text("    5) Magnesium"),
            Text("    6) Total phenols"),
            Text("    7) Flavanoids"),
            Text("    8) Nonflavanoid phenols"),
            Text("    9) Proanthocyanins"),
            Text("    10)Color intensity"),
            Text("    11)Hue"),
            Text("    12)OD280/OD315 of diluted wines"),
            Text("    13)Proline "),
            Text('Data: https://archive.ics.uci.edu/dataset/109/wine'),
            direction="vertical"
        )
    )
