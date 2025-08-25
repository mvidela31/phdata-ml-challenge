from typing import Any, Dict, List
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from house_pricing.model_inference.app import app

# TODO: Update the unit tests and mock the model file downloads.
client = TestClient(app)

@pytest.fixture
def request_instances() -> List[Dict[str, Any]]:
    instances = [
        {
            'bedrooms': 4,
            'bathrooms': 1.0,
            'sqft_living': 1680,
            'sqft_lot': 5043,
            'floors': 1.5,
            'waterfront': 0,
            'view': 0,
            'condition': 4,
            'grade': 6,
            'sqft_above': 1680,
            'sqft_basement': 0,
            'yr_built': 1911,
            'yr_renovated': 0,
            'zipcode': 98118,
            'lat': 47.5354,
            'long': -122.273,
            'sqft_living15': 1560,
            'sqft_lot15': 5765,
        },
        {
            'bedrooms': 3,
            'bathrooms': 2.5,
            'sqft_living': 2220,
            'sqft_lot': 6380,
            'floors': 1.5,
            'waterfront': 0,
            'view': 0,
            'condition': 4,
            'grade': 8,
            'sqft_above': 1660,
            'sqft_basement': 560,
            'yr_built': 1931,
            'yr_renovated': 0,
            'zipcode': 98115,
            'lat': 47.6974,
            'long': -122.313,
            'sqft_living15': 950,
            'sqft_lot15': 6380,
        }
    ]
    return instances


@pytest.fixture
def model_url() -> List[Dict[str, Any]]:
    url = "https://drive.google.com/uc?export=download&id=1U_MiVcRJwRdj3ugjYX2itBJq_xxXHcbm"
    return quote(quote(url, safe=""), safe="")


def test_ping() -> None:
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "model_url",
    [
        "https://drive.google.com/uc?export=download&id=1FWn_iOLLwHtYfItjtwh-gsEtFuB2tyrz",
        pytest.param("https://non_existant_url", marks=pytest.mark.xfail),
    ]
)
def aaaa_test_download_model(model_url: str) -> None:
    with TestClient(app) as client: 
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        assert response.json() == {"status": "Model downloaded successfully"}


@pytest.mark.parametrize(
    "n_downloads", [2, 3, pytest.param(0, marks=pytest.mark.xfail)]
)
def aaaa_test_get_versions(model_url: str, n_downloads: int) -> None:
    with TestClient(app) as client: 
        for _ in range(n_downloads):
            response = client.post(f"/download_model/{model_url}")
            assert response.status_code == 200
        response = client.get("/versions")
        assert response.status_code == 200
        assert response.json() == {"model_versions": list(range(1, n_downloads + 1))}


def aaaa_test_invocation(model_url: str, request_instances: List[Dict[str, Any]]) -> None:
    with TestClient(app) as client: 
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        response = client.post(
            "/invocations", 
            json={"instances": request_instances},
        )
        response_json = response.json()
        assert response.status_code == 200
        assert list(response_json.keys()) == ["predictions", "model_version"]
        assert isinstance(response_json["model_version"], str)
        assert isinstance(response_json["predictions"], list)
        assert len(response_json["predictions"]) == len(request_instances)
        assert list(response_json["predictions"][0].keys()) == [
            "prediction",
        ]


def aaaa_test_invocation_error(model_url: str, request_instances: List[Dict[str, Any]]) -> None:
    request_instances[1]["bedrooms"] = 3.1415926535
    with TestClient(app) as client:
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        response = client.post(
            "/invocations", 
            json={"instances": request_instances},
        )
        response_json = response.json()
        assert response.status_code == 422
        assert response_json == {
            'detail': [{
                'type': 'int_from_float',
                'loc': ['body', 'instances', 1, 'bedrooms'],
                'msg': 'Input should be a valid integer, got a number with a fractional part',
                'input': 3.1415926535,
            }]
        }


def aaaa_test_update_model(model_url: str, request_instances: List[Dict[str, Any]]) -> None:
    with TestClient(app) as client:
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        model_version = "1"
        response = client.post(
            f"/update_model/{model_version}",
        )
        response_json = response.json()
        assert response.status_code == 200
        assert response_json == {"status": "Model updated successfully"}
        response = client.post(
            "/invocations", 
            json={"instances": request_instances},
        )
        response_json = response.json()
        assert response_json["model_version"] == "1"


def aaaa_test_delete_model(model_url: str) -> None:
    with TestClient(app) as client:
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        response = client.post(f"/download_model/{model_url}")
        assert response.status_code == 200
        model_version = "2"
        response = client.post(
            f"/delete_model/{model_version}",
        )
        response = client.get("/versions")
        assert response.status_code == 200
        assert response.json() == {"model_versions": [1, 3]}


def aaaa_test_update_model_error() -> None:
    with TestClient(app) as client:
        model_version = "model_unknown.joblib"
        response = client.post(
            f"/update_model/{model_version}",
        )
        response_json = response.json()
        assert response.status_code == 500
        assert response_json == {
            "detail": f"Model {model_version} not found."
        }
