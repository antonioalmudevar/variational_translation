import argparse

from src.experiments import train_model


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
        '--restart', 
        action='store_true',
        help='restart the training with this flag'
    )
    parser.set_defaults(restart=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(**vars(args))