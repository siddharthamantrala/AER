from torchvision.datasets.utils import download_and_extract_archive

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

download_and_extract_archive(
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ".",
    md5="c58f30108f718f92721af3b95e74349a",
)