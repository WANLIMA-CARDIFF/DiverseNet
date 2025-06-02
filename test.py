from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import RoadNet
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from torch import Tensor


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/home/mwl/Desktop/ssl_roadnet/data",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='roadnet',
                        choices=['roadnet'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_multi_heads', choices=available_models, help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False, help="save segmentation results to \"./results\"")
    
    parser.add_argument("--batch_size", type=int, default=1, help='batch size (default: 16)')

    parser.add_argument("--ckpt", default="./best_deeplabv3plus_DF.pth", type=str, help="restore from checkpoint")
    #best_deeplabv3plus_multi_heads_dropout_roadnet_model1

    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")   
    parser.add_argument('--device', type=str,  default='cuda', help='computing device')

    return parser

def get_dataset(opts):

    if opts.dataset == 'roadnet':
        db_test =  RoadNet(data_root=opts.data_root, split="test", transforms = None)
    else:
        raise RuntimeError("Dataset not found")
    return db_test

def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for j, sample in tqdm(enumerate(loader)):

            images = sample['image'].to(device, dtype=torch.float32)
            labels = sample['label'].to(device, dtype=torch.long)
            outputs = torch.softmax(sum(model(images)),dim=1)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
        score = metrics.get_results()
        
    return score



def main():
    opts = get_argparser().parse_args()
    
    if opts.dataset.lower() == 'roadnet':
        opts.in_number = 3
        opts.num_classes = 2
    else:
        raise RuntimeError("Dataset not found")

    # Setup visualization
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    db_test = get_dataset(opts)
    test_loader = data.DataLoader(db_test, batch_size=opts.batch_size, shuffle=False, num_workers=12)

    print("Dataset: %s, test set: %d" %
          (opts.dataset, len(db_test)))

    model = network.modeling.__dict__[opts.model](in_number=opts.in_number, num_classes=opts.num_classes)
   
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cuda'), weights_only=False)

        model.load_state_dict(checkpoint["model_state"])

        model = nn.DataParallel(model)

        model.to(device)

        print("Model restored from %s" % opts.ckpt)

        del checkpoint  # free memory
   
    else:
        raise RuntimeError("checkpoint not found")    

    model.eval()

    val_score = validate(
        opts=opts, model=model, loader=test_loader, device=device, metrics=metrics)
    print(metrics.to_str(val_score))
               
if __name__ == '__main__':
    main()
