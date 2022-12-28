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
from datetime import datetime
import torchmetrics

logger = logging.getLogger(__name__)
best_f1 = 0


def main():

    # fix init params and args
    global best_f1
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

    if not args.wandb:
        os.environ['WANDB_SILENT']="true" # it is not working..
        wandb_mode = 'disabled'
        args.logger.info('wandb disabled.......')
    else:
        wandb_mode = 'online'
        args.logger.info('wandb online.......')
    
    # set wandb
    wandb.init(project=args.project_name, config=args, mode=wandb_mode)
    wandb.run.name = f"arch_{args.arch}_proportion_{args.proportion}_n_{args.n_weaks_combinations}"

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
        args.num_classes = 8
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

    labeled_trainloader = balanced_loader(labeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)  
    # https://github.com/ufoym/imbalanced-dataset-sampler
    # labeled_trainloader = DataLoader(labeled_dataset,  sampler=ImbalancedDatasetSampler(labeled_dataset), batch_size=args.batch_size, num_workers=1, pin_memory=False)

    args.eval_step = int(len(labeled_dataset) / args.batch_size)
    args.logger.info(f'args.eval_step : {args.eval_step} reset..')

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
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

    NGPU = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if NGPU > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
        # torch.multiprocessing.set_start_method('spawn') # todo : check this.
    model.to(device)

    # no_decay = ['bias', 'bn']
    # grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
    #     {'params': [p for n, p in model.named_parameters() if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    
    
    if args.nm_optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,   # grouped_parameters
                            momentum=0.9, nesterov=args.nesterov)
    elif args.nm_optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # grouped_parameters   
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
        best_f1 = checkpoint['best_f1']
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
    train(args, labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_f1
    test_f1 = []
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
                args.logger.info('train labeled dataset iter is reset.')
                inputs_x, targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next() 
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                args.logger.info('train unlabeled dataset iter is reset.')
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
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

        # valid_losses.avg, valid_top1.avg, valid_auprc.avg, valid_f1.avg, test_losses.avg, test_top1.avg, test_auprc.avg, test_f1.avg
        # valid_loss, valid_acc, valid_auprc, valid_f1, test_loss, test_acc, test_auprc, test_f1 = test(args, valid_loader, test_loader, test_model, epoch)
        test_loss, test_acc, test_auprc, test_f1 = test(args, test_loader, test_model, epoch)

        weak_image = wandb.Image(inputs_u_w[0], caption="Weak image")
        strong_image = wandb.Image(inputs_u_s[0], caption="Strong image")
        wandb.log({"weak_image": weak_image})
        wandb.log({"strong_image": strong_image})
        wandb.log({
            'train/1.train_loss': losses.avg,
            'train/2.train_loss_x': losses_x.avg,
            'train/3.train_loss_u': losses_u.avg,
            'train/4.mask': mask_probs.avg,
            # 'valid/1.valid_acc': valid_acc,
            # 'valid/2.valid_loss': valid_loss,
            # 'valid/3.valid_aurpc': valid_auprc,
            # 'valid/4.valid_f1': valid_f1,
            'test/1.test_acc': test_acc,
            'test/2.test_loss': test_loss,
            'test/1.test_auprc': test_auprc,
            'test/2.test_f1': test_f1
            })

        is_best = test_f1 > best_f1
        best_f1 = max(test_f1, best_f1)
        if is_best:  # save best f1
            # wandb.run.summary["valid_best_acc"] = valid_acc
            # wandb.run.summary["valid_best_loss"] = valid_loss
            # wandb.run.summary["valid_best_auprc"] = valid_auprc
            # wandb.run.summary["valid_best_f1"] = valid_f1
            
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
        logger.info('Best top-1 f1 score: {:.2f}'.format(best_f1))
        args.writer.close()


def valid_and_test(args, valid_loader, test_loader, model, epoch):
    fn_auprc = torchmetrics.classification.MulticlassAveragePrecision(num_classes=args.num_classes, average='macro')
    fn_f1score = torchmetrics.classification.MulticlassF1Score(num_classes=args.num_classes, average='macro')

    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    valid_top3 = AverageMeter()
    valid_auprc = AverageMeter()
    valid_f1 = AverageMeter()

    test_losses = AverageMeter()
    test_top1 = AverageMeter()
    test_top3 = AverageMeter()
    test_auprc = AverageMeter()
    test_f1  = AverageMeter()

    for (nm, loader) in [('valid', valid_loader),('test', test_loader)]:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        if not args.no_progress:
            loader = tqdm(loader)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                data_time.update(time.time() - end)
                model.eval()

                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs)

                # item
                loss = F.cross_entropy(outputs, targets)
                prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
                auprc = fn_auprc.to(args.device)(outputs, targets)
                f1 = fn_f1score.to(args.device)(outputs, targets)
                
                if nm == 'valid':
                    valid_losses.update(loss.item(), inputs.shape[0])
                    valid_top1.update(prec1.item(), inputs.shape[0])
                    valid_top3.update(prec3.item(), inputs.shape[0])
                    valid_auprc.update(auprc.item(), inputs.shape[0])
                    valid_f1.update(f1.item(), inputs.shape[0])
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if not args.no_progress:
                        loader.set_description("Valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {top3:.2f}. f1: {top3:.2f}.".format(
                            batch=batch_idx + 1,
                            iter=len(loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            loss=valid_losses.avg,
                            top1=valid_top1.avg,
                            top3=valid_top3.avg,
                            auprc=valid_auprc.avg,
                            f1=valid_f1.avg
                        ))
                else:
                    test_losses.update(loss.item(), inputs.shape[0])
                    test_top1.update(prec1.item(), inputs.shape[0])
                    test_top3.update(prec3.item(), inputs.shape[0])
                    test_auprc.update(auprc.item(), inputs.shape[0])
                    test_f1.update(f1.item(), inputs.shape[0])   
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if not args.no_progress:
                        loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {top3:.2f}. f1: {top3:.2f}.".format(
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
            if not args.no_progress:
                loader.close()

            test_image = wandb.Image(inputs[0], caption="Test image")
            wandb.log({"test image": test_image})
        
        if nm == 'valid':
            logger.info("top-1 acc: {:.2f}".format(valid_top1.avg))
            logger.info("top-3 acc: {:.2f}".format(valid_top3.avg))
            logger.info("auprc: {:.2f}".format(valid_auprc.avg))
            logger.info("f1: {:.2f}".format(valid_f1.avg))
        else:
            logger.info("top-1 acc: {:.2f}".format(test_top1.avg))
            logger.info("top-3 acc: {:.2f}".format(test_top3.avg))
            logger.info("auprc: {:.2f}".format(test_auprc.avg))
            logger.info("f1: {:.2f}".format(test_f1.avg))

    return valid_losses.avg, valid_top1.avg, valid_auprc.avg, valid_f1.avg, test_losses.avg, test_top1.avg, test_auprc.avg, test_f1.avg



def test(args, loader, model, epoch):
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

    if not args.no_progress:
        loader = tqdm(loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)

            # item
            loss = F.cross_entropy(outputs, targets)
            prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
            auprc = fn_auprc.to(args.device)(outputs, targets)
            f1 = fn_f1score.to(args.device)(outputs, targets)
            
        
            test_losses.update(loss.item(), inputs.shape[0])
            test_top1.update(prec1.item(), inputs.shape[0])
            test_top3.update(prec3.item(), inputs.shape[0])
            test_auprc.update(auprc.item(), inputs.shape[0])
            test_f1.update(f1.item(), inputs.shape[0])   
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {top3:.2f}. f1: {top3:.2f}.".format(
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
        if not args.no_progress:
            loader.close()

        test_image = wandb.Image(inputs[0], caption="Test image")
        wandb.log({"test image": test_image})
        

        logger.info("top-1 acc: {:.2f}".format(test_top1.avg))
        logger.info("top-3 acc: {:.2f}".format(test_top3.avg))
        logger.info("auprc: {:.2f}".format(test_auprc.avg))
        logger.info("f1: {:.2f}".format(test_f1.avg))

    return test_losses.avg, test_top1.avg, test_auprc.avg, test_f1.avg



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # yymmddhhmm = datetime.now().strftime('%y%m%d%H%M')
    main()
