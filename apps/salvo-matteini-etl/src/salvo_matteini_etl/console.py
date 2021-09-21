
import logging.config
from os.path import dirname, join
logging.config.fileConfig(join(dirname(__file__), "logging.conf"))

import asyncio
# import time
# import argparse
import sys

from salvo_matteini_etl.etl import batch


def main():

    logging.info(sys.argv)

    # # parse arguments
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument("--debug", action="store_true",
    #                     help="Execute in debug mode")
    # parser.add_argument("--file", "-f", nargs='?', type=argparse.FileType('r'), default=None,
    #                     help="ETL funds in file")
    #
    # kwargs = parser.parse_args()

    asyncio.run(batch())


if __name__ == "__main__":
    main()
