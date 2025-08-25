import argparse

import uvicorn


def main() -> None:
    """
    Command line invoked function that runs the application.

    Run the FastAPI application through uvicorn web server.
    """
    parser = argparse.ArgumentParser(
        description="Run the FastAPI application through uvicorn web server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind server socket to this host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind server socket to this port. If 0, an available port will be picked.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable server auto-reload.",
    )
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        help="Enable/Disable server X-Forwarded-Proto, X-Forwarded-For, X-Forwarded-Port " \
        "to populate remote address info.",
    )

    # Arguments parse
    args, _ = parser.parse_known_args()
    args = {k.replace("-", "_"): v for k, v in args.__dict__.items()}

    # Server run
    uvicorn.run(
        app="house_pricing.model_inference.app:app",
        host=args["host"],
        port=args["port"],
        reload=args["reload"],
        proxy_headers=args["proxy_headers"],
    )


if __name__ == "__main__":
    main()
