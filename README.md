# MLServer Tutorial
## Setup
Copy and paste the following block of commands into your WSL terminal:

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add pyenb to path 
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Restart your WSL shell to apply path changes

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Python 3.10 globally using pyenv
pyenv install 3.10.0

# Git clone
git clone https://github.com/matthiaskozubal/mlserver_demo.git .

# Cd to the directory
cd mlserver_demo

# Set the local Python version to Python 3.10 for this project
pyenv local 3.10.0

# Install dependencies and create a virtual environment
poetry install
```

## Run
### mushroom-xgboost
```bash
cd mushroom-xgboost

# Download data, then build, train, and save model
python save_model.py

# Start MLServer to serve the model - in a separate terminal
mlserver start .

# Serve the model using MLServer and run test inference request
python inference_request.py
```