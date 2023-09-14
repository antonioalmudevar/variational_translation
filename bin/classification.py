import argparse

from src.experiments import run_classification


def parse_args():

    parser = argparse.ArgumentParser(description='Train VAE')

    parser.add_argument(
        'config_data', 
        nargs='?', 
        type=str,
        help='data configuration file'
    )

    parser.add_argument(
        'config_model', 
        nargs='?', 
        type=str, 
        help='Model configuration file'
    )

    parser.add_argument(
        'config_training', 
        nargs='?', 
        type=str, 
        help='training configuration file'
    )

    parser.add_argument(
        '--freeze_encoder', 
        action='store_true',
        help=''
    )
    parser.set_defaults(freeze_encoder=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_classification(**vars(args))