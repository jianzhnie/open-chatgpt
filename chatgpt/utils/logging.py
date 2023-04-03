import logging
import os


def paser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument(
        '--log_format',
        type=str,
        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser.add_argument('--log_date_format',
                        type=str,
                        default='%Y-%m-%d %H:%M:%S')
    return parser.parse_args()


def get_logger(
        name,
        log_file=None,
        log_level='INFO',
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_logger_from_args(name):
    args = paser_args()
    return get_logger(name,
                      log_file=args.log_file,
                      log_level=args.log_level,
                      log_format=args.log_format)
