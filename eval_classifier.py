import json
import torch
import argparse
from torchvision import transforms
import chexnet.eval_model as E


if __name__ == "__main__":
    print('----------------------------------------')
    print('---- Evaluate Auxiliary Classifier -----')
    print('----------------------------------------' + '\n')

    # Define an argument parser
    parser = argparse.ArgumentParser('Evaluate Auxiliary Classifier')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_eval_classifier.json')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config + '\n')

    # Read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # Parse config
    config = json.loads(config)

    image_path = config['image_path']
    save_path = config['save_path']
    classifier_checkpoint = config['classifier_checkpoint']

    perturbation_type = config['perturbation_type']
    perturbation_model_file = config['perturbation_model_file']
    mu = config['mu']

    b = config['b']
    m = config['m']
    eps = config['eps']

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
    }

    checkpoint_best = torch.load(classifier_checkpoint)
    model = checkpoint_best['model']

    if perturbation_model_file is not None:
        perturbation_checkpoint = torch.load(perturbation_model_file)
    else:
        perturbation_checkpoint = None

    preds, aucs = E.make_pred_multilabel(data_transforms, model, image_path, save_path, perturbation_type, 
                                         perturbation_checkpoint, mu, b, m, eps)
