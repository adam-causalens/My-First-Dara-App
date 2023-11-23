from dara.components import Card, Heading, Stack, Text


def home_page():
    return Card(
        Stack(
            Heading('Titanic - Machine Learning from Disaster'),
            Text('This web shows the capabilities of Dara through a classic Machine Learning Problem.'),
            Text('Data: https://www.kaggle.com/competitions/titanic/overview'),
            direction="vertical"
        )
    )
# "https://www.kaggle.com/competitions/titanic/overview"
