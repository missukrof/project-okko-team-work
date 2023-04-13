import os
import gdown
import pandas as pd


def read_csv_from_gdrive(url: str) -> pd.DataFrame:
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    # file_id = url.split("/")[-2]
    # file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    # data = pd.read_csv(file_path)

    output_directory = f"{os.path.dirname(os.path.realpath(__file__))}/df.csv"

    data = pd.read_csv(
        gdown.download(
            url=url,
            output=output_directory,
            quiet=True,
            fuzzy=True
        )
    )

    os.remove(output_directory)

    return data
