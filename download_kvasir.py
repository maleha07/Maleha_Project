from pathlib import Path
import ssl
import shutil
import urllib.request
import zipfile

from config import RAW_ROOT


# Official Kvasir-SEG archive
URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def locate_dataset_folder(root):
    candidates = [
        root / "Kvasir-SEG",
        root / "kvasir-seg",
        root,
    ]

    for candidate in candidates:
        images = candidate / "images"
        masks = candidate / "masks"

        if images.exists() and masks.exists():
            return candidate

    # Search recursively if archive structure changes
    for p in root.rglob("images"):
        candidate = p.parent

        if (candidate / "masks").exists():
            return candidate

    raise RuntimeError(
        "Could not find Kvasir-SEG images/ and masks/ folders."
    )


def download_file(url, destination):
    """
    Download with normal SSL first.

    If the Colab/Python environment has a broken CA chain,
    retry using an unverified SSL context.

    This fallback should only be used for this known public dataset URL.
    """

    try:
        print("Trying normal HTTPS download...")

        urllib.request.urlretrieve(
            url,
            destination
        )

        return

    except Exception as e:
        print()
        print("Normal HTTPS download failed:")
        print(e)

        print()
        print(
            "Retrying with SSL certificate verification disabled "
            "for this dataset download..."
        )

    # Fallback for Colab / Python SSL certificate problem
    context = ssl._create_unverified_context()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        request,
        context=context
    ) as response:

        with open(
            destination,
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                response,
                output_file
            )


def download_kvasir(overwrite=False):

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_path = (
        RAW_ROOT
        / "kvasir-seg.zip"
    )

    extract_root = (
        RAW_ROOT
        / "kvasir-seg-extracted"
    )

    # --------------------------------------------------------
    # Clean if requested
    # --------------------------------------------------------

    if (
        overwrite
        and extract_root.exists()
    ):
        shutil.rmtree(
            extract_root
        )

    if (
        overwrite
        and zip_path.exists()
    ):
        zip_path.unlink()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    if not zip_path.exists():

        print()
        print(
            "Downloading Kvasir-SEG..."
        )

        print(
            URL
        )

        download_file(
            URL,
            zip_path
        )

        print()
        print(
            f"Downloaded to: {zip_path}"
        )

    else:

        print(
            f"Using existing archive: {zip_path}"
        )

    # --------------------------------------------------------
    # Validate ZIP
    # --------------------------------------------------------

    if not zipfile.is_zipfile(
        zip_path
    ):

        raise RuntimeError(
            f"The downloaded file is not a valid ZIP:\n"
            f"{zip_path}\n\n"
            "Delete it and run again."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    if not extract_root.exists():

        print(
            "Extracting Kvasir-SEG..."
        )

        extract_root.mkdir(
            parents=True,
            exist_ok=True
        )

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            z.extractall(
                extract_root
            )

    # --------------------------------------------------------
    # Locate images/masks
    # --------------------------------------------------------

    dataset_root = (
        locate_dataset_folder(
            extract_root
        )
    )

    print()
    print(
        "Dataset ready."
    )

    print(
        f"Dataset root: {dataset_root}"
    )

    return dataset_root


if __name__ == "__main__":
    download_kvasir()