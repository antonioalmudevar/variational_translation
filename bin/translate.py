import argparse

from src.experiments import translate_model


def parse_args():

    parser = argparse.ArgumentParser(description='Test VAE')

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    translate_model(**vars(args))