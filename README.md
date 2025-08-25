# PhData's ML Challenge 2: House Pricing #

Project solution of the PhData's Machine Learning Challenge #2. The challenge consists on deploying a regression model for house pricing through a REST API.

The project is composed by three contenarized components (`src/components`):
* `data_ingestion`: To load the house pricing dataset.
* `model_training`: To train the regression models.
* `model_inference`: To deploy the trained models through a REST API.

## Installation ##

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/mvidela31/phdata-ml-challenge.git
cd house-pricing
pip install -r requirements.txt
```

For development purposes also install the dependencies on requirements-dev:
```bash
pip install -r requirements-dev.txt
```

## Usage ##

### Run Tests ###

Use `pre-commit` to run the lintern and formatting hook scripts:
```
pre-commit run -all
```

Use `pytest` to run the unit tests of the project sub-packages:
```
pytest . -v
```

### Deployment ###

To deploy the model endpoint using Docker, first build the model inference Docker image:
```bash
cd ./src/components/model_inference & docker build --tag house-pricing/model-inference:latest .
```

Run the builded Docker image:
```bash
docker run -p 80:80 house-pricing/model-inference
```

Now the server is deployed! You can send HTTPs requests to the server endpoints. See the `./notebooks/endpoint_invocation.ipynb` notebook for endpoint invocation examples.


### Experiments ###

See the `./notebooks/experiments.ipynb` notebook for models experimentation and evaluation.

## Contact ##
* Miguel Videla Araya (miguel.videla@ug.uchile.cl)
