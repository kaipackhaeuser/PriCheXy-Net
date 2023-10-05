import os
import json
import argparse
from utils import utils
from agents.AgentSiameseNetwork import AgentSiameseNetwork


if __name__ == "__main__":
    print('------------------------------------')
    print('------- Patient Verification -------')
    print('------------------------------------' + '\n')

    # Define an argument parser
    parser = argparse.ArgumentParser('Patient Verification')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_retrainSNN.json')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # Read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # Parse config
    config = json.loads(config)

    # Create folder to save experiment-related files
    os.mkdir('./archive/' + config['experiment_description'])
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
    utils.make_zip(SAVINGS_PATH + config['experiment_description'] + '.zip', './', args.config)

    # Call agent and run experiment
    experiment = AgentSiameseNetwork(config)
    experiment.run()
