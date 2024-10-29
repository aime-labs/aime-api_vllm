import argparse
from getpass import getpass
from pathlib import Path

AIME_HOSTED_MODELS = []


class ModelDownloader:

    def __init__(self):
        self.args = self.load_flags()
        self.prepare_download_dir()


    def load_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model', type=str, required=True,
            help='Model name to download'
        )
        parser.add_argument(
            '--download_dir', type=str, required=True,
            help='Download directory'
        )
        parser.add_argument(
            '--max_workers', type=int, required=False, default=8,
            help='Maximum number of workers for downloading'
        )
        return parser.parse_args()


    def prepare_download_dir(self):
        download_dir = Path(self.args.download_dir)
        if download_dir.is_dir():
            model_name = self.args.model.split('/')[-1]
            download_dir = download_dir / model_name
            if not download_dir.is_dir():
                download_dir.mkdir()
            self.args.download_dir = download_dir
        else:
            exit(f'Invalid path {download_dir}')



    def start_download(self):
        if self.args.model in AIME_HOSTED_MODELS:
            self.download_from_aime()
        else:
            self.download_from_hf()


    def download_from_aime(self):
        pass


    def download_from_hf(self):
        from huggingface_hub import login
        from huggingface_hub.errors import GatedRepoError
        try:
            self.start_hf_download()
        except GatedRepoError:
            hf_token = getpass(f'Access to model {self.args.model} is restricted. You need login to Huggingface. Enter your Huggingface access token (Hidden): ')
            login(hf_token)
            try:
                self.start_hf_download()
            except GatedRepoError:
                exit(f'Access to model {self.args.model} is not permitted on this account.')

    
    def start_hf_download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=self.args.model,
            max_workers=self.args.max_workers,
            local_dir=self.args.download_dir
        )


def main():
    model_downloader = ModelDownloader()
    model_downloader.start_download()


if __name__ == "__main__":
    main()
