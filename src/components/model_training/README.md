# Model Training Component

Trains a [LightGBM](https://github.com/microsoft/LightGBM) regression model.

## Inputs
|Name|Type|Default|Description|
|---|---|---|---|
|data_path|str|`os.path.join(os.environ["SM_CHANNELS"], "dataset", "dataset_df.parquet")`|Parquet file path of input dataset.|
|target_colname|str||Column name of the regression target variable.|
|model_hparams|Dict[str,Any]|{}|Regression model hyperparameters.|
|fitting_params|Dict[str,Any]|{}|Regression model fitting parameters.|
|robust_scaling|bool|False|Whether to add a robust scaling preprocessing step.|
|sampling_seed|float|0.7|Data splitting sampling seed.|
|train_size|float|0.2|Percentage size of train set.|
|val_size|float|0.2|Percentage size of validation set.|
|model_dir|str|`os.environ["SM_MODEL_DIR"]`|Directory for saving the trained model artifacts.|

## Usage

### Run as package

```python
from house_pricing.model_training import train_model

model, eval_metrics, scaler = train_model(
    df=dataset_df,
    target_colname="price",
    model_hparams={
        "boosting_type": "gbdt",
        "num_leaves": 256,
        "learning_rate": 0.1,
        "n_estimators": 5000,
        "verbose": -1,
    },
    fitting_params={
        "eval_metric": ["mse"],
    },
    robust_scaling=False,
    sampling_seed=42, 
    train_size=0.7,
    val_size=0.2,
)
```

### Run from CLI
```bash
train \
    --data-path ./dataset_df.parquet \
    --target-colname price \
    --model-hparams '{"boosting_type": "gbdt", "num_leaves": 256, "learning_rate": 0.1, "n_estimators": 5000, "verbose": -1}' \
    --fitting-params '{"eval_metric": ["mse"]}' \
    --robust-scaling False \
    --sampling-seed 42 \ 
    --train-size 0.7 \
    --val-size 0.2 \
    --model-dir ./model_dir
```

## Build

### Upload Docker image to AWS ECR
```bash
bash build-img.sh
```

### Build and run Docker image locally
```bash
docker build --tag house-pricing/model-training:latest .

docker run house-pricing/model-training:latest \
    --data-path ./dataset_df.parquet \
    --target-colname price \
    --model-hparams '{"boosting_type": "gbdt", "num_leaves": 256, "learning_rate": 0.1, "n_estimators": 5000, "verbose": -1}' \
    --fitting-params '{"eval_metric": ["mse"]}' \
    --robust-scaling False \
    --sampling-seed 42 \ 
    --train-size 0.7 \
    --val-size 0.2 \
    --model-dir ./model_dir
```
