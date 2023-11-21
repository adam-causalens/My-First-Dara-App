from bokeh.plotting import figure
from dara.components import Bokeh, Card
from dara.components.plotting.palettes import CategoricalLight3

from my_first_app.definitions import data


def scatter_plot(x: str, y: str):
    plot_data = data.copy()
    plot_data['color'] = plot_data['species_names'].map(
        {x: CategoricalLight3[i] for i, x in enumerate(data['species_names'].unique())}
    )

    p = figure(title=f"{x} vs {y}", sizing_mode='stretch_both', toolbar_location=None)
    p.circle(
        x,
        y,
        color='color',
        source=plot_data,
        fill_alpha=0.4,
        size=10,
        legend_group='species_names'
    )
    return Bokeh(p)


def eda_page():
    return Card(
        scatter_plot('petal length (cm)', 'petal width (cm)'),
        title='Scatter Plot'
    )
