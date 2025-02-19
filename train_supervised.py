import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.multiprocessing as mp
import torch.distributed as dist


from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm

import wandb
# os.environ['WANDB_SILENT']="true"


from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811K
from utils import AverageMeter, accuracy
from utils.common import get_args, save_checkpoint, set_seed, create_model, \
    get_cosine_schedule_with_warmup
from datetime import datetime


best_f1 = 0


def prerequisite(args):
    args.n_gpu = torch.cuda.device_count()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    if not args.wandb:
        wandb_mode = 'disabled'
    else:
        wandb_mode = 'online'
    
    run_name = f"arch_{args.arch}_proportion_{args.proportion}_supervised"
    # set wandb
    wandb.init(project=args.project_name, config=args, mode=wandb_mode)
    wandb.run.name = run_name

    if args.seed is not None:
        set_seed(args)
    
    if args.out == '':
        args.out = f"results/{datetime.now().strftime('%y%m%d%H%M%S')}_" + run_name 
    os.makedirs(args.out, exist_ok=True)
    print(f'{args.out} directory created.')

    if args.dataset == 'wm811k':
        if not args.exclude_none:
            args.num_classes = 9
        else:
            args.num_classes = 8
            
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    else:
        raise ValueError('unknown dataset')


def main():
    global best_f1  # fix init params and args
    best_f1 = 0
    args = get_args()
    prerequisite(args)
    main_worker(0, args)


def main_worker(local_rank: int, args: object):
    torch.cuda.set_device(local_rank)
    args.logger = logging.getLogger(__name__)

    labeled_dataset, unlabeled_dataset, valid_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')

    # https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/20
    labeled_trainloader = DataLoader(dataset=labeled_dataset,
                      batch_size=args.batch_size,
                      sampler=ImbalancedDatasetSampler(labeled_dataset),
                      num_workers=args.num_workers,
                      drop_last=True,
                      pin_memory=True)

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(args)
    model.to(args.local_rank)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.nm_optim == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,  # grouped_parameters
                              momentum=0.9, nesterov=args.nesterov)
    elif args.nm_optim == 'adamw':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)  # grouped_parameters
    else:
        raise ValueError("unknown optim")

    total_steps = len(labeled_trainloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, total_steps)
    ema_model = None

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    args.start_epoch = 0

    args.logger.info("***** Running training *****")
    args.logger.info(f"  Task = {args.dataset}@{args.proportion}")
    args.logger.info(f"  Num Epochs = {args.epochs}")
    args.logger.info(f"  Batch size per GPU = {args.batch_size}")
    args.logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    args.logger.info(f"  Total optimization steps = {total_steps}")
    model.zero_grad()
    train(args, labeled_trainloader, valid_loader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, valid_loader, test_loader, model, optimizer, ema_model, scheduler):
    global best_f1
    end = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        p_bar = tqdm(range(len(labeled_trainloader)))

        for batch_idx, (inputs_x, targets_x) in enumerate(labeled_trainloader):
            data_time.update(time.time() - end)
            targets_x = targets_x.to(args.local_rank)
            inputs_x = inputs_x.to(args.local_rank)

            # inputs_x = F.one_hot(inputs_x.long(), num_classes=3).squeeze().float()  # make 3 channels
            inputs_x = inputs_x.permute(0, 3, 1, 2).float()  # (c, h, w)
            logits = model(inputs_x)
            loss = F.cross_entropy(logits, targets_x.long(), reduction='mean')
            loss.backward()
            losses.update(loss.item())
            
            optimizer.step()
            scheduler.step()
            
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(labeled_trainloader),
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg))
            p_bar.update()

        p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        valid_loss, valid_acc, valid_auprc, valid_f1 = test(args, valid_loader, test_model, epoch)
        test_loss, test_acc, test_auprc, test_f1 = test(args, test_loader, test_model, epoch)

        wandb.log({
            'train/1.train_loss': losses.avg,
            'valid/1.valid_acc': valid_acc,
            'valid/2.valid_loss': valid_loss,
            'valid/3.valid_auprc': valid_auprc,
            'valid/4.valid_f1': valid_f1,
            'test/1.test_acc': test_acc,
            'test/2.test_loss': test_loss,
            'test/3.test_auprc': test_auprc,
            'test/4.test_f1': test_f1
        })

        is_best = valid_f1 > best_f1
        best_f1 = max(valid_f1, best_f1)
        
        if is_best:  # save best f1
            wandb.run.summary["test_best_acc"] = test_acc
            wandb.run.summary["test_best_loss"] = test_loss
            wandb.run.summary["test_best_auprc"] = test_auprc
            wandb.run.summary["test_best_f1"] = test_f1

            model_to_save = model.module if hasattr(model, "module") else model

            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            args.logger.info('Best top-1 f1 score: {:.2f}'.format(best_f1))


def test(args, loader, model, epoch):
    fn_auprc = torchmetrics.classification.MulticlassAveragePrecision(num_classes=args.num_classes, average='macro')
    fn_f1score = torchmetrics.classification.MulticlassF1Score(num_classes=args.num_classes, average='macro')

    test_losses = AverageMeter()
    test_top1 = AverageMeter()
    test_top3 = AverageMeter()
    test_auprc = AverageMeter()
    test_f1 = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    loader = tqdm(loader)

    total_preds = []
    total_reals = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_time.update(time.time() - end)
            model.eval()

            # make 3 channels
            # inputs = F.one_hot(inputs.long(), num_classes=3).squeeze()
            inputs = inputs.permute(0, 3, 1, 2).float()  # (c, h, w)

            inputs = inputs.to(args.local_rank)
            targets = targets.to(args.local_rank)
            outputs = model(inputs)

            total_preds.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            total_reals.append(targets.cpu().detach().numpy())

            # item
            loss = F.cross_entropy(outputs, targets)
            prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
            auprc = fn_auprc.to(args.local_rank)(outputs, targets)
            f1 = fn_f1score.to(args.local_rank)(torch.argmax(outputs, dim=1), targets)

            test_losses.update(loss.item(), inputs.shape[0])
            test_top1.update(prec1.item(), inputs.shape[0])
            test_top3.update(prec3.item(), inputs.shape[0])
            test_auprc.update(auprc.item(), inputs.shape[0])
            test_f1.update(f1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            loader.set_description(
                "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {top3:.2f}. f1: {top3:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=test_losses.avg,
                    top1=test_top1.avg,
                    top3=test_top3.avg,
                    auprc=test_auprc.avg,
                    f1=test_f1.avg
                ))
        loader.close()

        total_preds = np.concatenate(total_preds)
        total_reals = np.concatenate(total_reals)

        wandb.log({f"conf_mat_{epoch}":
            wandb.plot.confusion_matrix(
                probs=None,
                y_true=total_reals,
                preds=total_preds,
                class_names=np.asarray(WM811K.idx2label)[:args.num_classes])
        }
        )
        f1 = f1_score(y_true=total_reals, y_pred=total_preds, average='macro')

        test_image = wandb.Image(inputs[0], caption="Test image")
        wandb.log({"test image": test_image})
        args.logger.info("top-1 acc: {:.2f}".format(test_top1.avg))
        args.logger.info("top-3 acc: {:.2f}".format(test_top3.avg))
        args.logger.info("auprc: {:.2f}".format(test_auprc.avg))
        args.logger.info("f1 score (torchmetrics) : {:.2f}".format(test_f1.avg))
        args.logger.info("f1: {:.2f}".format(f1))

    return test_losses.avg, test_top1.avg, test_auprc.avg, f1


if __name__ == '__main__':
    main()
