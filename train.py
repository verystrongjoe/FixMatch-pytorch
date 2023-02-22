import os
import logging
import math
import time
import numpy as np

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['WANDB_SILENT']="true"

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.multiprocessing as mp
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm

from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811K
from utils import AverageMeter, accuracy
from utils.common import get_args, de_interleave, interleave, save_checkpoint, set_seed, create_model, \
    get_cosine_schedule_with_warmup
from datetime import datetime
import yaml
from argparse import Namespace

logger = logging.getLogger(__name__)
best_valid_f1 = 0
best_test_f1 = 0

def prerequisite(args):
    args.logger = logger

    if not args.wandb:
        wandb_mode = 'disabled'
        args.logger.info('wandb disabled.')
    else:
        wandb_mode = 'online'
        args.logger.info('wandb enabled.')

    # set wandb
    wandb.init(project=args.project_name, config=args, mode=wandb_mode)
    if args.sweep:
        try:
            with open('./sweep.yaml') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)            
                wandb.config.update(config)
                args.proportion = wandb.config.proportion
                args.n_weaks_combinations = wandb.config.n_weaks_combinations
                args.tau = wandb.config.tau
                args.threshold = wandb.config.threshold
                args.lambda_u = wandb.config.lambda_u
                args.mu = wandb.config.mu
                args.nm_optim = wandb.config.nm_optim
                args.seed = wandb.config.seed
                args.keep
        except:
             args.logger.warn('there is no sweep yaml.')             

    run_name = f"keep_{args.keep}_prop_{args.proportion}_n_{args.n_weaks_combinations}_t_{args.tau}_th_{args.threshold}_mu_{args.mu}_l_{args.lambda_u}_op_{args.nm_optim}_arch_{args.arch}"
    wandb.run.name = run_name

    
    

    
    
    
    
    
    
    
    
    

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    args.out = f"results/{datetime.now().strftime('%y%m%d%H%M%S')}_" + run_name 
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == 'wm811k':
        args.num_classes = 8
        assert args.arch in ('wideresnet', 'resnext')
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    else:
        raise ValueError('unknown dataset')


def main(local_rank, args):
    global best_valid_f1, best_test_f1
    args.local_rank = local_rank
    torch.cuda.set_device(args.local_rank)

    labeled_dataset, unlabeled_dataset, valid_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')
    
    # https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/20
    labeled_trainloader = DataLoader(dataset=labeled_dataset,
                      batch_size=args.batch_size,
                      sampler=ImbalancedDatasetSampler(labeled_dataset),
                      num_workers=args.num_workers,
                      drop_last=True,
                      pin_memory=True)
    
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

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
        optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov) # grouped_parameters
    elif args.nm_optim == 'adamw':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)  # grouped_parameters   
    else:
        raise ValueError("unknown optim")

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs  * len(labeled_trainloader))
    ema_model = None

    if args.use_ema:
        logger.info('current model changes to ema model..')
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        
        best_valid_f1 = checkpoint['best_valid_f1']
        best_test_f1 = checkpoint['best_test_f1']
        
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.proportion}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.epochs  * len(labeled_trainloader)}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_valid_f1, best_test_f1
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    unlabeled_iter = iter(unlabeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        p_bar = tqdm((labeled_trainloader))
       
        for batch_idx, (inputs_x, targets_x) in enumerate(labeled_trainloader):
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                args.logger.info('train unlabeled dataset iter is reset.')
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.local_rank)
            targets_x = targets_x.to(args.local_rank)

            F.one_hot(inputs_u_s.long(), num_classes=3).squeeze().float()

            # make 3 channels
            inputs = F.one_hot(inputs.long(), num_classes=3).squeeze().float()
            inputs = inputs.permute(0, 3, 1, 2)  # (c, h, w)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits
            Lx = F.cross_entropy(logits_x, targets_x.long(), reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # threshold를 넘은 값의 logiit
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()  # cross entropy from targets_u 
            loss = Lx + args.lambda_u * Lu  # 최종 loss를 labeled와 unlabeled 합산
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(labeled_trainloader),
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_probs.avg))
            p_bar.update()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        valid_loss, valid_acc, valid_auprc, valid_f1, _, _  = test(args, valid_loader, test_model, epoch)
        test_loss, test_acc, test_auprc, test_f1, total_reals, total_preds = test(args, test_loader, test_model, epoch, valid_f1=valid_f1)

        # black/white image
        # weak_image = wandb.Image(inputs_u_w[0].detach().numpy().astype(np.uint8), caption="Weak image")       # 32 x 32 x 1
        # strong_image = wandb.Image(inputs_u_s[0].detach().numpy().astype(np.uint8), caption="Strong image")   # 32 x 32 x 1

        # rgb image
        weak_image = wandb.Image(F.one_hot(inputs_u_w[0].long(), num_classes=3).squeeze().numpy().astype(np.uint8), caption="Weak image")
        strong_image = wandb.Image(F.one_hot(inputs_u_s[0].long(), num_classes=3).squeeze().numpy().astype(np.uint8), caption="Strong image")

        wandb.log({"weak_image": weak_image})
        wandb.log({"strong_image": strong_image})

        wandb.log({
            'train/1.train_loss': losses.avg,
            'train/2.train_loss_x': losses_x.avg,
            'train/3.train_loss_u': losses_u.avg,
            'train/4.mask': mask_probs.avg,
            'valid/1.test_acc': valid_acc,
            'valid/2.test_loss': valid_loss,
            'valid/3.test_auprc': valid_auprc,
            'valid/4.test_f1': valid_f1,
            'test/1.test_acc': test_acc,
            'test/2.test_loss': test_loss,
            'test/3.test_auprc': test_auprc,
            'test/4.test_f1': test_f1
            })

        is_best = valid_f1 > best_valid_f1
        best_valid_f1 = max(valid_f1, best_valid_f1)
        best_test_f1 = max(test_f1, best_test_f1)
        
        
        if is_best:  # save best f1
            wandb.run.summary["test_best_acc"] = test_acc
            wandb.run.summary["test_best_loss"] = test_loss  
            wandb.run.summary["test_best_auprc"] = test_auprc  
            wandb.run.summary["test_best_f1_max"] = best_test_f1
            wandb.run.summary["test_best_f1"] = test_f1  
            
            # wandb.log({f"conf_mat_{epoch}" :
            #     wandb.plot.confusion_matrix(
            #         probs=None,
            #         y_true=total_reals,
            #         preds=total_preds,
            #         class_names=np.asarray(WM811K.idx2label)[:args.num_classes])
            #     }
            # )
            
            
        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_valid_f1': best_valid_f1,
            'best_test_f1': best_test_f1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)
        logger.info('Best top-1 f1 score: {:.2f}'.format(best_test_f1))


def test(args, loader, model, epoch, valid_f1=None):
    fn_auprc = torchmetrics.classification.MulticlassAveragePrecision(num_classes=args.num_classes, average='macro')
    fn_f1score = torchmetrics.classification.MulticlassF1Score(num_classes=args.num_classes, average='macro')

    test_losses = AverageMeter()
    test_top1 = AverageMeter()
    test_top3 = AverageMeter()
    test_auprc = AverageMeter()
    test_f1  = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    total_preds = []
    total_reals = []
    
    loader = tqdm(loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs3 = inputs.to(args.local_rank)
            targets = targets.to(args.local_rank)

            # make 3 channels
            inputs3 = F.one_hot(inputs3.long(), num_classes=3).squeeze().float()
            inputs3 = inputs3.permute(0, 3, 1, 2)  # (c, h, w)

            outputs = model(inputs3)

            total_preds.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            total_reals.append(targets.cpu().detach().numpy())

            # item
            loss = F.cross_entropy(outputs, targets)
            prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
            auprc = fn_auprc.to(args.local_rank)(outputs, targets)
            f1 = fn_f1score.to(args.local_rank)(torch.argmax(outputs, dim=1), targets)
        
            test_losses.update(loss.item(), inputs3.shape[0])
            test_top1.update(prec1.item(), inputs3.shape[0])
            test_top3.update(prec3.item(), inputs3.shape[0])
            test_auprc.update(auprc.item(), inputs3.shape[0])
            test_f1.update(f1.item(), inputs3.shape[0])   
            batch_time.update(time.time() - end)
            end = time.time()
            loader.set_description(f"{'Valid' if valid_f1 is None else 'Test'}" + " Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {top3:.2f}. f1: {top3:.2f}.".format(
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
        
        if valid_f1 is not None and valid_f1 > best_valid_f1:
            f1 = f1_score(y_true=total_reals, y_pred=total_preds, average='macro')
            logger.info("top-1 acc: {:.2f}".format(test_top1.avg))
            logger.info("top-3 acc: {:.2f}".format(test_top3.avg))
            logger.info("auprc: {:.2f}".format(test_auprc.avg))
            logger.info("f1 score (torchmetrics) : {:.2f}".format(test_f1.avg))
            logger.info("f1: {:.2f}".format(f1))

    return test_losses.avg, test_top1.avg, test_auprc.avg, f1, total_reals, total_preds 


if __name__ == '__main__':
    args = get_args()    
    prerequisite(args)
    args.device_id = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"GPU of {args.device_id} Device ID is training...")
    main(0, args)  # single machine, single gpu
