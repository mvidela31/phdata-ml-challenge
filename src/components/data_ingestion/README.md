# Data Ingestion Component

Downloads the house pricing dataset.

## Inputs
|Name|Type|Default|Description|
|---|---|---|---|
|sales_column_selection|str||Columns of house sales dataset to select.|
|output_path|str|`os.path.join(os.environ["SM_PROCESS_DIR"], "dataset", "dataset_df.parquet")`|Loaded dataset path.|

## Usage

### Run as package

```python
from house_pricing.data_ingestion import load_house_pricing_dataset

dataset_df = load_house_pricing_dataset(
    sales_column_selection=[
        "price", 
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "zipcode",
    ]
)
```

### Run from CLI
```bash
ingest \
    --sales-column-selection '["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement", "zipcode"]' \
    --output-path ./dataset_df.parquet
```

## Build

### Upload Docker image to AWS ECR
```bash
bash build-img.sh
```

### Build and run Docker image locally
```bash
docker build --tag house-pricing/data-ingestion:latest .

docker run house-pricing/data-ingestion:latest \
    --sales-column-selection '["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement", "zipcode"]' \
    --output-path ./dataset_df.parquet
```
