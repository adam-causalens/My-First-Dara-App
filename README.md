# My first Dara App: Wine Classification Problem

![image of a glass of wine](https://archive.ics.uci.edu/static/public/109/Thumbnails/Large.jpg?37)

---

## A. Installation

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

---

## B. Creating a basic app

### 1. Hello World

* Add base code to `main.py`

* Run `poetry run dara start`

* Visit http://0.0.0.0:8000/

### 2. Add new pages

* Install dependencies to build a page that cna handle data: `poetry add scikit-learn`

* Create a `pages` subdir and add new pages in python there

* Add new pages to the main config using `config.add()`

### 3. Interactivity

* use `dara.core.Variable`
* use `dara.components.Select` for dropdowns
* add `dara.core.py_component` as a decorator

---

## 3.Future Components

todo

---

## Contact

* Email address: adam[dot]jaamour[at]causalens[dot]com
