# Model Inference Component

Runs a [FastAPI](https://fastapi.tiangolo.com/) application that performs inferences from a [LightGBM](https://github.com/microsoft/LightGBM) regression model for house pricing.

## Application

### Request Methods
|Name|Type|Description|
|---|---|---|
|/ping|GET|An HTTP path to health check requests to the server.|
|/invocations|POST|An HTTP path to send prediction requests to the server.|

### Inputs
The HTTP server must accept prediction requests that have the `Content-Type: application/json` HTTP header and JSON bodies with the following format:
```json
{
    "input": INSTANCES,
}
```
In these requests:
* `INSTANCES` is an array of one or more JSON values of any type. Each values represents an instance that you are providing a prediction for.

## Inputs

|Name|Type|Default|Description|
|---|---|---|---|
|host|str|"127.0.0.1"|Bind server socket to this host.|
|port|int|8000|Bind server socket to this port. If 0, an available port will be picked.|
|reload|bool|False|Enable server auto-reload.|
|proxy_headers|bool|False|Enable/Disable server X-Forwarded-Proto, X-Forwarded-For, X-Forwarded-Port to populate remote address info.|

## Usage

### Run from CLI
```bash
run-server
    --host 0.0.0.0 \
    --port 80 \
    --reload \
    --proxy-headers
```

## Build

### Upload Docker image to AWS ECR
```bash
bash build-img.sh
```

### Build and run Docker image locally
```bash
docker build --tag house-pricing/model-inference:latest .

docker run -p 80:80 house-pricing/model-inference:latest
    --host 0.0.0.0 \
    --port 80 \
    --reload \
    --proxy-headers
```
