from dara.core import ConfigurationBuilder
from dara.components import Heading

from my_first_app.pages.eda import eda_page

# Create a configuration builder
config = ConfigurationBuilder()

# Register pages
# config.add_page('Hello World', Heading('Hello World!'))
config.add_page('EDA', eda_page())
