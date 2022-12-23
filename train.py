import logging
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.dataset import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils.common import get_args, de_interleave, interleave, save_checkpoint, set_seed, create_model, get_cosine_schedule_with_warmup
import wandb
from datasets.loaders import balanced_loader
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler


logger = logging.getLogger(__name__)
best_acc = 0


def main():

    # fix init params and args
    global best_acc
    args = get_args()
    device = torch.device('cuda', args.num_gpu)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.logger = logger

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    # set wandb
    wandb.init(project='wafermatch', config=args)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        assert args.arch in ('wideresnet', 'resnext')
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'wm811k':
        args.num_classes = 9
        assert args.arch in ('wideresnet', 'resnext')
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        assert args.arch in ('wideresnet', 'resnext')
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')

    train_sampler = RandomSampler

    
    labeled_trainloader = balanced_loader(labeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)  

    # https://github.com/ufoym/imbalanced-dataset-sampler
    # labeled_trainloader = DataLoader(labeled_dataset,  sampler=ImbalancedDatasetSampler(labeled_dataset), batch_size=args.batch_size, num_workers=1, pin_memory=False)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(args)

    NGPU = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if NGPU > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
        # torch.multiprocessing.set_start_method('spawn')
    model.to(device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.nm_optim == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                            momentum=0.9, nesterov=args.nesterov)
    elif args.nm_optim == 'adamw':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)
    else:
        raise ValueError("unknown optim")

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    ema_model = None

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
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
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.train()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next() 
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            
            weak_image = wandb.Image(inputs_u_w[0], caption="Weak image")
            strong_image = wandb.Image(inputs_u_s[0], caption="Strong image")
            wandb.log({"weak_image": weak_image})
            wandb.log({"strong_image": strong_image})

            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)

            logits = model(inputs)

            logits = de_interleave(logits, 2*args.mu+1)
            
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            del logits

            Lx = F.cross_entropy(logits_x, targets_x.long(), reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # threshold를 넘은 값의 logiit
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,  
                                  reduction='none') * mask).mean()  # cross entropy from targets_u 

            loss = Lx + args.lambda_u * Lu  # 최종 loss를 labeled와 unlabeled 합산
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model


        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        wandb.log('train/1.train_loss', losses.avg)
        wandb.log('train/2.train_loss_x', losses_x.avg)
        wandb.log('train/3.train_loss_u', losses_u.avg)
        wandb.log('train/4.mask', mask_probs.avg)
        wandb.log('test/1.test_acc', test_acc)
        wandb.log('test/2.test_loss', test_loss)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-20:])))
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
