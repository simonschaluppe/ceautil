# login to nextcloud
import getpass
import logging
import urllib.parse
from pathlib import Path

import requests


FHTW_NEXTCLOUD_URL = "https://cloud.technikum-wien.at/remote.php/dav/files"

LOCAL_NEXTCLOUD = [  # add your local nextcloud paths here to try and automatically load from disk, instead of cloud
    r"C:\Users\Simon Schneider\Nextcloud",
    r"C:\Users\lektor\nextcloud",
]


def get(file_path, verbose=False, force_online=False):
    """Returns a file from the FHTW nextcloud requiring command line authentication, first searching provided local paths."""
    # try finding a local nextcloud
    if not force_online:
        for path in LOCAL_NEXTCLOUD:
            print(f"trying to get {file_path=} from {LOCAL_NEXTCLOUD=}...")
            if Path(path).exists():
                local_path = Path(path, file_path)
                logging.info(f"{local_path=} found!")
                return local_path
        logging.info("...failed.")

    # only if not locally found, connect
    logging.info(f"Trying to access {FHTW_NEXTCLOUD_URL=}\nfor {file_path=}...")
    parsed_file = urllib.parse.quote(
        f"{file_path}",
    )
    user = input("user (FHTW): ")
    pw = getpass.getpass("pw (not shown, hit enter after typing): ")
    response = requests.request(
        method="GET", url=f"{FHTW_NEXTCLOUD_URL}/{user}/{parsed_file}", auth=(user, pw)
    )
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"{parsed_file=}")
        logging.debug(f"{response=}")
        logging.debug(f"{response.content=}")
    return response.content


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    file_path = "EE/6_Daten/Energie Österreich/Energiebilanzen_AT_1970-2020_Statistik_Austria.xlsx"
    file = get(file_path, verbose=True, force_online=True)
    df = pd.read_excel(file, sheet_name="Sektoraler Endverbrauch TJ")
    print(df.head())
