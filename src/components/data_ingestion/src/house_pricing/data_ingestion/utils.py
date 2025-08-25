from typing import Any, Dict

from yaml import Loader, load


def parse_str_dict(
    dict: Dict[str, str],
) -> Dict[str, Any]:
    """
    Parses a dictionary with string values by evaluating its \
    Python expression using the yaml loader.

    Parameters
    ----------
    dict: Dict[str, str]
        Dictionary with string values.

    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluated values.
    """
    for k, v in dict.items():
        dict[k] = load(v, Loader=Loader)
    return dict
