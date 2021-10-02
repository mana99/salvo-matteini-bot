
import logging
import logging.config
from os.path import join, dirname

logging.config.fileConfig(join(dirname(__file__), "logging.conf"))

import argparse
import sys
import time
import logging

from salvo_matteini_bot.pipeline import execute

logger = logging.getLogger(__name__)


def main():

    start_time = time.perf_counter()
    logging.info(sys.argv)

    # todo argparse
    # todo starting point parameter


    # parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--start-from", nargs='?', type=int, default=0,
                        help="")
    # parser.add_argument("funds", nargs='*', type=int,
    #                     help="")
    parser.add_argument("--export", action='store_true',
                        help="")

    kwargs = parser.parse_args()

    execute(start=kwargs.start_from, export=kwargs.export)


if __name__ == '__main__':
    main()
