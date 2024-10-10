# rayfish

![build](https://github.com/jernejfrank/rayfish/actions/workflows/merge_main.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)
[![python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Estimating parameters for discrete quantum imaging.

- **Github repository**: <https://github.com/jernejfrank/rayfish/>
<!-- - **Documentation** <https://jernejfrank.github.io/rayfish/> -->

## Usage
For now we need to manually install the package. Clone the repo and do a local installation from source.
It is recommended to do this in a virtual environment; a good tool for that is [uv](https://docs.astral.sh/uv/).

```bash
# this will download the repository into the directory your shell command is currently at
git clone https://github.com/jernejfrank/rayfish.git
cd ./rayfish
# in case you are using uv
uv pip install .
# in case you are using pip directly
pip install .
```

## Development
This repository follows [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow). Here is how to setup your work environment for contributing:

1. Fork [GitHub instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)

2. To access the contents on your computer run the following commands in your shell:
```bash
# set this to the same as your login username for github
GITHUB_USERNAME="YOUR-GITHUB-USERNAME" \
# this will download the repository into the directory your shell command is currently at
git clone https://github.com/${GITHUB_USERNAME}/rayfish.git
cd ./rayfish
# this creates an upstream connection to the original repository so that you can update your copy
git remote add upstream https://github.com/jernejfrank/rayfish.git
```

3. Installing the environment and package dependencies (requires `uv`)
```bash
make install
```
and check if the `pre-commit` hooks and tests are running
```bash
uv run pre-commit run -a
uv run pytest tests
```

### To add a feature
1. Make sure you are up to date with the main branch
```bash
git checkout main
git pull upstream main

# and might as well update your fork too!
git push origin main
```

2. Create your new branch
```bash
git checkout -b feat/some-feature
# make some changes in your code and commit them
git push origin HEAD
```

3. Open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests):

## TODO
Written in order of priority:
* Sort out existing projects and create backend module
    - coherent qfi
    - olynomial measurement basis (general way to represnt any mode sorting basis)
    - intensity estimation as benchmark
* Create Dash frontend
* Protect main branch
* Create CI/CD with Github Actions
    - automate changelog and semver
    - run tests
    - attempt to build
* Create Docker container for the app and maybe use mounting drive for saving data locally for now
* Use sphinx to create docs
* Build package and publish to pypi
* Sort out docs
