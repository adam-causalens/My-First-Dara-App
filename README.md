# My first Dara App

## Installation commands

### 1. Poetry

* [Install](https://python-poetry.org/docs/#installing-with-the-official-installer) `curl -sSL https://install.python-poetry.org | python3 -`

* Add `export PATH="$HOME/.local/bin:$PATH"` to `.zshrc` ([source](https://stackoverflow.com/a/60768677/5609328))

* Use `poetry --version`

### 2. Create new app and install dara

* Create new app
  * In PyCharm with new poetry env
  * manually using `mkdir` and `poetry init`

* Run `poetry install`

* Install `poetry add dara-core --extras all`

### 3. Hello World

* Add base code to `main.py`

* Run `poetry run dara start`

* Visit http://0.0.0.0:8000/hello-world

### 4. Add data

* Install dependencies to build a page that cna handle data: `poetry add scikit-learn`

* Create a `pages` subdir and add new pages in python there

* Add new pages to the main config using `config.add()`

* 

