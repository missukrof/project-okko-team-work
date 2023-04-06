import pandas as pd
import gdown


def read_csv_from_gdrive(url: str) -> pd.DataFrame:
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    # file_id = url.split("/")[-2]
    # file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    # data = pd.read_csv(file_path)

    data = pd.read_csv(
        gdown.download(
            url=url,
            output='df.csv',
            quiet=True,
            fuzzy=True
        )
    )

    return data
