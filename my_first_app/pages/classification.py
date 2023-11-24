from typing import List

from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
from bokeh.plotting import figure
from bokeh.transform import transform
from dara.core import DerivedVariable, py_component, Variable
from dara.components import Bokeh, Grid, Heading, Slider, Stack, Text
from dara.components.common.select import Select
from dara.components.plotting.palettes import SequentialDark8
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from my_first_app.data import data, features, target_names

# Constants
RANDOM_SEED = 42
METRIC_GRANULARITY = 3

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data['species'], test_size=0.33, random_state=RANDOM_SEED
)
tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)


def calculate_predictions(max_depth_map: List[int], min_samples_leaf_map, criterion_map, splitter_map):
    """
    Calculates the predictions with the new parameters.
    :param max_depth_map:
    :param min_samples_leaf_map:
    :param criterion_map:
    :param splitter_map:
    :return:
    """
    dt = DecisionTreeClassifier(max_depth=max_depth_map[0],
                                criterion=criterion_map,
                                min_samples_leaf=min_samples_leaf_map[0],
                                splitter=splitter_map,
                                random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    return dt_preds


# Calculate metrics
@py_component
def confusion_matrix_plot(preds: np.array) -> Bokeh:
    df = pd.DataFrame(
        confusion_matrix(y_test, preds), index=target_names, columns=target_names
    )
    df.index.name = 'Actual'
    df.columns.name = 'Prediction'
    df = df.stack().rename('value').reset_index()

    mapper = LinearColorMapper(
        palette=SequentialDark8, low=df.value.min(), high=df.value.max()
    )

    # define a figure
    p = figure(
        title='Confusion Matrix',
        sizing_mode='stretch_both',
        toolbar_location=None,
        x_axis_label='Predicted',
        y_axis_label='Actual',
        x_axis_location="above",
        x_range=target_names,
        y_range=target_names[::-1]
    )

    # create rectangle for heatmap
    p.rect(
        x='Actual',
        y='Prediction',
        width=1,
        height=1,
        source=df,
        line_color=None,
        fill_color=transform('value', mapper),
    )

    # add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        label_standoff=10,
        ticker=BasicTicker(desired_num_ticks=3),
    )
    p.add_layout(color_bar, 'right')

    # Add values inside the heatmap
    text_props = {
        'source': df,
        'angle': 0,
        'color': 'black',
        'text_align': 'center',
        'text_baseline': 'middle'
    }
    p.text(x='Actual', y='Prediction', text='value', text_font_size='20pt', **text_props)

    return Bokeh(p)


calculate_accuracy = lambda preds: str(round(accuracy_score(y_true=y_test, y_pred=preds), METRIC_GRANULARITY))
calculate_precision = lambda preds: str(round(precision_score(y_true=y_test, y_pred=preds, average="micro"), METRIC_GRANULARITY))
calculate_recall = lambda preds: str(round(recall_score(y_true=y_test, y_pred=preds, average="micro"), METRIC_GRANULARITY))
calculate_f1 = lambda preds: str(round(f1_score(y_true=y_test, y_pred=preds, average="micro"), METRIC_GRANULARITY))

# Define variables that can be updated
max_depth_var = Variable([5])
min_samples_leaf_var = Variable([1])
criterion_var = Variable("gini")
splitter_var = Variable("best")
predictions_var = DerivedVariable(
    calculate_predictions,
    variables=[max_depth_var, min_samples_leaf_var, criterion_var, splitter_var]
)
acc_var = DerivedVariable(calculate_accuracy, variables=[predictions_var])
prec_var = DerivedVariable(calculate_precision, variables=[predictions_var])
rec_var = DerivedVariable(calculate_recall, variables=[predictions_var])
f1_var = DerivedVariable(calculate_f1, variables=[predictions_var])


def classification_page():
    return Stack(
        Heading('Classification Results'),
        Text("Scikit-Learn Decision Tree Model"),
        Grid(
            Grid.Row(
                Grid.Column(
                    Text('Maximum Tree Depth'),
                    Slider(domain=[1, 15], value=max_depth_var, step=1, ticks=[i + 1 for i in range(0, 15)],
                           disable_input_alternative=True),
                    Text('Minimum Sample Split'),
                    Slider(domain=[1, 10], value=min_samples_leaf_var, step=1,
                           ticks=[i + 1 for i in range(0, 10, 1)],
                           disable_input_alternative=True),
                    Text("Criterion "),
                    Select(value=criterion_var, items=["gini", "entropy", "log_loss"], searchable=False),
                    Text("Splitter"),
                    Select(value=splitter_var, items=["best", "random"], searchable=False),
                    direction='vertical',
                    span=2
                ),
                Grid.Column(span=1),
                Grid.Column(
                    confusion_matrix_plot(predictions_var),
                    span=5
                ),
                Grid.Column(span=1),
                Grid.Column(
                    Text('Classification Metrics', bold=True),
                    Stack(
                        Stack(Text(f'Accuracy'), Text(acc_var), direction='horizontal'),
                        Stack(Text(f'Precision'), Text(prec_var), direction='horizontal'),
                        Stack(Text(f'Recall'), Text(rec_var), direction='horizontal'),
                        Stack(Text(f'F1'), Text(f1_var), direction='horizontal'),
                        hug=True
                    ),
                    direction='vertical',
                    span=3,
                ),
                padding='5px'
            )
        )
    )
