import os
import os.path as osp
import random
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import Tensor
from tensorboardX import SummaryWriter

import network
import utils
from utils import ext_transforms as et
from utils import losses, ramps
from utils.visualizer import Visualizer
from datasets import RoadNet
from metrics import StreamSegMetrics
import torchvision

def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root", type=str, default="/home/mwl/Desktop/ssl_roadnet/data")
    parser.add_argument("--dataset", type=str, default='roadnet', choices=['roadnet'])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument('--in_number', type=int, default=None)
    parser.add_argument('--exp', type=str, default='roadnet_deeplab/f4_one_model_ini_freeze_CE_Loss_adaptive_vote')
    
    available_models = sorted(name for name in network.modeling.__dict__
                              if name.islower() and not name.startswith("_") 
                              and callable(network.modeling.__dict__[name]))

    parser.add_argument("--model", type=str, default='deeplabv3plus_multi_heads', choices=available_models)
    parser.add_argument("--model_folder", type=str, default='checkpoints')
    parser.add_argument("--labeled_ratio", type=float, default=0.25)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--total_itrs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--val_batch_size", type=int, default=5)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'])
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    return parser

def get_dataset(opts):
    if opts.dataset == 'roadnet':
        db = RoadNet(data_root=opts.data_root, split="train", transforms=None)
        train_len = int(len(db) * 0.9)
        val_len = len(db) - train_len
        db_train, db_val = torch.utils.data.random_split(db, [train_len, val_len])

        labeled_len = int(len(db_train) * opts.labeled_ratio)
        unlabeled_len = len(db_train) - labeled_len
        db_train_l, db_train_unl = torch.utils.data.random_split(db_train, [labeled_len, unlabeled_len])
    else:
        raise RuntimeError("Dataset not found")
    return db_train_l, db_train_unl, db_val

def validate(opts, model, ce_loss, loader, device, metrics):
    metrics.reset()
    average_loss = 0.0
    with torch.no_grad():
        for sample in tqdm(loader):
            images = sample['image'].to(device)
            labels = sample['label'].to(device)
            outputs = torch.softmax(sum(model(images)), dim=1)
            loss = ce_loss(outputs, labels)
            average_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
    score = metrics.get_results()
    return score, average_loss / len(loader)

def create_model(opts, ema=False):
    model = network.modeling.__dict__[opts.model](in_number=opts.in_number, num_classes=opts.num_classes, output_stride=opts.output_stride)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model.to(opts.device)

class VoteModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1) * 5)

    def forward(self, mean, vote):
        return (self.weights * mean) + vote

def main():
    opts = get_argparser().parse_args()
    
    if opts.dataset.lower() == 'roadnet':
        opts.in_number = 3
        opts.num_classes = 2
    else:
        raise RuntimeError("Dataset not found")

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    db_train, db_train_unl, db_val = get_dataset(opts)

    train_label_loader = data.DataLoader(db_train, batch_size=opts.batch_size, shuffle=True, num_workers=12, drop_last=True)
    train_unlabel_loader = data.DataLoader(db_train_unl, batch_size=opts.batch_size, shuffle=True, num_workers=12, drop_last=True)
    val_loader = data.DataLoader(db_val, batch_size=opts.val_batch_size, shuffle=False, num_workers=8)

    print(f"Dataset: {opts.dataset}, labeled: {len(db_train)}, unlabeled: {len(db_train_unl)}, val: {len(db_val)}")

    model1 = create_model(opts)
    utils.set_bn_momentum(model1.backbone, momentum=0.01)

    optimizer1 = torch.optim.SGD([
        {'params': model1.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model1.classifier.parameters(), 'lr': opts.lr}
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    vote_max = VoteModule()
    optimizer_vote = torch.optim.Adam(vote_max.parameters(), lr=0.001)

    scheduler1 = utils.PolyLR(optimizer1, opts.total_itrs, power=0.9) if opts.lr_policy == 'poly' else None

    ce_loss = nn.CrossEntropyLoss()

    metrics = StreamSegMetrics(opts.num_classes)

    def save_ckpt1(path):
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model1.module.state_dict(),
            "optimizer_state": optimizer1.state_dict(),
            "scheduler_state": scheduler1.state_dict(),
            "best_score": best_score,
        }, path)
        print(f"Model saved: {path}")

    if opts.ckpt and os.path.isfile(opts.ckpt):
        torch.load(opts.ckpt, map_location='cpu')
    else:
        print("[!] Retrain")
        model1 = nn.DataParallel(model1).to(device)
        vote_max = nn.DataParallel(vote_max).to(device)

    snapshot_path = f"../save_model/{opts.exp}/{opts.model}"
    if osp.exists(snapshot_path):
        raise RuntimeError("Model already exists")
    os.makedirs(snapshot_path)
    shutil.copytree('.', osp.join(snapshot_path, 'code'), shutil.ignore_patterns('.git', '__pycache__'))
    os.makedirs(osp.join(snapshot_path, 'training_sample_record'))
    os.makedirs(osp.join(snapshot_path, 'checkpoints'))

    with open(osp.join(snapshot_path, 'training_sample_record', f'labeled_train_sample_{opts.dataset}.txt'), 'w') as f:
        f.writelines([f"{s['sample_name']}\n" for s in db_train])
    with open(osp.join(snapshot_path, 'training_sample_record', f'unlabeled_train_sample_{opts.dataset}.txt'), 'w') as f:
        f.writelines([f"{s['sample_name']}\n" for s in db_train_unl])

    writer_train = SummaryWriter(osp.join(snapshot_path, 'log_train'))
    writer_val = SummaryWriter(osp.join(snapshot_path, 'log_val'))

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    interval_loss = 0.0
    max_epoch = opts.total_itrs // len(train_label_loader) + 1

    while True:
        model1.train()
        cur_epochs += 1
        unlabel_iter = iter(train_unlabel_loader)

        for nimibatch_l in tqdm(train_label_loader):
            try:
                nimibatch_unl = next(unlabel_iter)
            except StopIteration:
                unlabel_iter = iter(train_unlabel_loader)
                nimibatch_unl = next(unlabel_iter)

            labeled_image = nimibatch_l['image'].to(device)
            label = nimibatch_l['label'].to(device)
            unlabeled_image = nimibatch_unl['image'].to(device)

            cur_itrs += 1

            if cur_itrs > 3500:
                for name, param in model1.named_parameters():
                    param.requires_grad = not any(name.startswith(f'module.classifier.classifier{i}') for i in np.random.randint(1, 11, 5))

            pred_sup_1 = model1(labeled_image)
            pred_unsup_1 = model1(unlabeled_image)
            pred_1 = [torch.cat([s, u], dim=0) for s, u in zip(pred_sup_1, pred_unsup_1)]

            random_index = random.randint(0, 9)
            max_preds = [p.argmax(dim=1) for p in pred_1]
            one_hot_vote = sum([F.one_hot(p, opts.num_classes) for p in max_preds])
            max_vote = one_hot_vote.argmax(dim=3)

            mean_pred = sum(pred_1).argmax(dim=1)
            one_hot_mean = F.one_hot(mean_pred, opts.num_classes)
            combined = vote_max(one_hot_mean, one_hot_vote).argmax(dim=3)

            selected_prediction = pred_1.pop(random_index)
            cps_loss = ce_loss(selected_prediction, combined) * 1.5 * 5
            loss_sup_1 = sum([ce_loss(p, label) for p in pred_sup_1])

            loss = loss_sup_1 + cps_loss

            optimizer1.zero_grad()
            optimizer_vote.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer_vote.step()

            interval_loss += loss.item()
            writer_train.add_scalar('info/cps_loss', cps_loss.item(), cur_itrs)
            writer_train.add_scalar('info/loss_sup_1', loss_sup_1.item(), cur_itrs)
            writer_train.add_scalar('info/loss', loss.item(), cur_itrs)

            if cur_itrs % opts.val_interval == 0:
                interval_loss /= opts.val_interval
                writer_train.add_scalar('info/interval_loss', interval_loss, cur_itrs)
                print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss:.4f}")
                save_ckpt1(osp.join(snapshot_path, opts.model_folder, f'latest_{opts.model}_{opts.dataset}_model1.pth'))

                model1.eval()
                val_score, val_loss = validate(opts, model1, ce_loss, val_loader, device, metrics)
                print(metrics.to_str(val_score))

                writer_val.add_scalar('info/loss', val_loss, cur_itrs)
                writer_val.add_scalar('info/mIoU', val_score['Mean IoU'], cur_itrs)
                writer_val.add_scalar('info/Overall_Acc', val_score['Overall Acc'], cur_itrs)
                writer_val.add_scalar('info/UA', val_score['UA'], cur_itrs)
                writer_val.add_scalar('info/PA', val_score['PA'], cur_itrs)
                writer_val.add_scalar('info/f1', val_score['f1'], cur_itrs)

                interval_loss = 0.0
                if val_score['Mean IoU'] > best_score:
                    best_score = val_score['Mean IoU']
                    save_ckpt1(osp.join(snapshot_path, opts.model_folder, f'best_{opts.model}_{opts.dataset}_model1.pth'))
                model1.train()

            scheduler1.step()

            if cur_itrs >= opts.total_itrs:
                writer_train.close()
                writer_val.close()
                return

if __name__ == '__main__':
    main()

