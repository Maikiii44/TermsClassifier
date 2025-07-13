# ♻ Terms Classifier

This repo contains code to train a model that classifies terms into Nice Classes.  
It uses PEFT to finetune embedding models. The repo implements PEFT methods with  
PyTorch Lightning.

## ⚙️ Installation & Setup

To install it you just need to run the following command in an environment with Python
3.12 or higher:

```shell
poetry install
```

**📌 _NOTES:_**

* The current Poetry toml respect the nomenclature of the versions above `2.0.0`
* I would recommend to downgrade the major version, seems easier to use.

When in development mode, it is recommended to install all the extra dependencies
by running:

```shell
poetry install --all-extras
```

In case you want to update the dependencies once installed the first version, you just
need to:

```shell
poetry update
```

And the `poetry.lock` file will be updated too, if applicable.

To get help how to run the CLI you will need to run the following command:

```shell
poetry run terms --help
```

Finally, in order to run the static code analysis checks you should use the following
set of commands once installed:

```shell
poetry run ruff --fix terms
poetry run mypy termss
```

## 🔁 Package Lifecycle

First of all you'll need to install the current Python package with `poetry install --all-extras` so
that the `poetry.lock` file will be generated and a new virtual environment will be created.

Once the Python package is installed, you can do code-changes and use it via the CLI with
`poetry run terms --help` or just by running Python with `poetry run python`
so as to open a Python terminal in the created virtual environment.

In case you add/update the dependencies in `pyproject.toml` you'll need to run `poetry lock --no-update`
to update the `poetry.lock` file. After that, execute `poetry install --all-extras` to update and install
the packages that are needed.

Also, you need to remember to commit and push the changes in `poetry.lock` usually with the following
commit message "📦 Run \`poetry update\`".

## 📁 Project Structure

```text
📂 Wipo
├── 📂 docs/               - auto-generated and hand-written documentation
├── 📂 configurations/     - project configuration files (both checked-in and auto-generated)
├── 📂 data/               - raw, interim, and processed datasets
├── 📂 notebooks/          - exploratory notebooks, reports, and visual analyses
├── 📂 src/                - source code root
│   ├── 📂 app/             - FastAPI application for serving inference APIs
│   │   └── 🐍 __init__.py
│   └── 📂 terms/      - core logic for classification, preprocessing, and LLM handling
│       ├── 📂 model/               - training-related code for the classifier
│       │   ├── 🐍 data.py          - data preparation and loading logic
│       │   ├── 🐍 metrics.py       - metric definitions and evaluations
│       │   ├── 🐍 module.py        - PyTorch Lightning module definition
│       │   └── 🐍 train.py         - training entry point and logic
│       ├── 🐍 __init__.py
│       ├── 🐍 cli.py               - command-line interface entry-point (`python -m trademarks.cli`)
│       ├── 🐍 config.py            - global configuration dataclass & loaders
│       ├── 🐍 constants.py         - package-wide constant values
│       ├── 🐍 logs.py              - unified logging configuration & helpers
│       ├── 🐍 preprocess.py        - data pre-processing for training & inference
│       ├── 🐍 pipeline.py          - end-to-end inference pipeline orchestrator
│       ├── 🐍 schemas.py           - Pandera models and typing schemas
│       └── 🐍 utils.py             - assorted helper functions
└── 📂 tests/              - unit & integration tests (mirrors package structure)
```


### 🧠 Train the Model

One of the main command lines is the train command. To see how to use it
run the following:

```bash
poetry run terms --help
```

#### How to Use the Command for Generating Proposals for Testing

Here's an example of how to use the command to generate proposals for testing:

```bash
poetry run terms 
    --data-filepath "./data/data_small.parquet" 
    --pretrained-model-name "BAAI/bge-small-en-v1.5" 
    --lora-config-path "./configurations/lora_config.yaml"
```


## 📊 MLflow Tracking

Training metrics are stored locally in the `mlruns` folder at the root of the project.  
You can change this path in `train.py` under the `logger` settings.

To launch the MLflow UI locally (if you haven't changed the default paths),  
run the following command in your virtual environment:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```