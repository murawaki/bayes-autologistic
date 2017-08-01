import argparse
from misc import args_construction


def _parse_args():
    parser = argparse.ArgumentParser(
        description='arguments for autologistic experiment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    sub_parsers = parser.add_subparsers(
        help='Type of experiment(s)',
        dest='experiment_type'
    )
    args_construction.parser_for_autologistic(sub_parsers)

    got_args = parser.parse_args()
    return got_args


if __name__ == '__main__':
    args = _parse_args()
    args.func(args)
