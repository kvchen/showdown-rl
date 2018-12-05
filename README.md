# Showdown RL

## Installation

Install OpenMPI as in the instructions here: https://spinningup.openai.com/en/latest/user/installation.html#installing-openmpi.

Clone and install all the other dependencies:

```
mkdir code
cd code
git clone https://github.com/openai/spinningup.git
git clone git@github.com:kvchen/gym-showdown.git
git clone git@github.com:kvchen/showdown-rl.git
cd showdown-rl
pipenv install
```

## Usage

```
pipenv shell
python3 main.py
```
