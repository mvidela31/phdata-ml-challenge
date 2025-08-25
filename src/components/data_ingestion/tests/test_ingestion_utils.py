from house_pricing.data_ingestion.utils import parse_str_dict


def test_parse_str_dict() -> None:
    src_dict = {
        "int": '1',
        "float": '0.1',
        "str": 'my_string',
        "bool": 'True',
        "null": 'null',
        "list": '[1, 0.1, "my_str", True, null, None, [1, 0.1]]',
        "dict": '{"a": 1, "b": 0.1, "c": "my_string", "d": True, "e": null, "f": [1, 0.1, True, null, [1, 0.1]]}',
    }
    parsed_dict = parse_str_dict(src_dict)
    expected_dict = {
        "int": 1,
        "float": 0.1,
        "str": "my_string",
        "bool": True,
        "null": None,
        "list": [1, 0.1, "my_str", True, None, "None", [1, 0.1]],
        "dict": {
            "a": 1,
            "b": 0.1,
            "c": "my_string",
            "d": True,
            "e": None,
            "f": [1, 0.1, True, None, [1, 0.1]],
        },
    }
    assert parsed_dict == expected_dict
