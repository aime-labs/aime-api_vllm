import argparse

from vllm import LLM
from vllm.utils import FlexibleArgumentParser


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name to download"
    )
    parser.add_argument(
        "--download-dir", type=str, required=True,
        help="Worker job type for the API Server"
    )
    return parser.parse_args()
def main():
    args = load_flags()
    _ = LLM(model=args.model, download_dir=args.download_dir, load_format='auto')


if __name__ == "__main__":
    main()
