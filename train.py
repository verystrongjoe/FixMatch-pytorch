import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm
from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811K
from utils import AverageMeter, accuracy
from utils.common import get_args, de_interleave, interleave, save_checkpoint
from utils.common import set_seed, create_model, get_cosine_schedule_with_warmup
from datetime import datetime
import yaml
from argparse import Namespace
from PIL import Image
import collections
import pandas as pd
# from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)
best_valid_f1 = 0
best_test_f1 = 0


def prerequisite(args):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(dict(args._get_kwargs()))
    args.logger = logger

    if not args.wandb:
        wandb_mode = 'disabled'
        args.logger.info('wandb disabled.')
        assert not args.sweep 
    else:
        wandb_mode = 'online'
        args.logger.info('wandb enabled.')

        # set wandb
        # wandb.init(project=args.project_name, mode=wandb_mode)
        with open('sweep.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        wandb.init(project=args.project_name, mode=wandb_mode, config=config)


        args.logger.info(f"sweep configuraion is loaded.")
        args.logger.info(wandb.config)

        if args.sweep:
            args.logger.info('existing confiuguration will be replaced by sweep yaml.')
            try:
                args.seed = wandb.config.seed                                   # 1
                args.proportion = wandb.config.proportion                       # 2 
                args.n_weaks_combinations = wandb.config.n_weaks_combinations   # 3  
                args.tau = wandb.config.tau                                     # 4 
                args.threshold = wandb.config.threshold                         # 5 
                args.lambda_u = wandb.config.lambda_u                           # 6
                args.mu = wandb.config.mu                                       # 7
                args.nm_optim = wandb.config.nm_optim                           # 8   
                args.keep = wandb.config.keep                                   # 9 
                args.limit_unlabled = wandb.config.limit_unlabled               # 10
                args.lr = wandb.config.lr                                       # 11
                args.aug_types = wandb.config['aug_types'].split(',')           # 12

                args.logger.info(f"sweep configuraion is set.")
                args.logger.info(args)
            except Exception as e:
                args.logger.warning('there is no sweep yaml.')             
                raise e
                
    run_name = f"keep_{args.keep}_prop_{args.proportion}_n_{args.n_weaks_combinations}_t_{args.tau:.2f}_th_{args.threshold:.2f}_mu_{args.mu}_l_{args.lambda_u}_op_{args.nm_optim}_arch_{args.arch}_unlabeld_{args.limit_unlabled}"                
    wandb.run.name = run_name
    
    if args.seed is not None:
        set_seed(args)
        args.logger.info(f'seed is set to {args.seed}.')
    
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
    

def check_args(args):
    ######################## check and display args 
    print(f"we are using {len(args.aug_types)} weak augmentations such as {args.aug_types}")
    args.device_id = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"GPU of {args.device_id} Device ID is training...")


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
        logger.info(f'sgd set with learning rate {args.lr}')
    elif args.nm_optim == 'adamw':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)  # grouped_parameters   
        logger.info(f'adamw set with learning rate {args.lr}')
    else:
        raise ValueError("unknown optim")

    # TODO: warmp up 파라미터 값 변화주기
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs  * len(labeled_trainloader))
    ema_model = None

    # TODO: ema 로직 잘 작동하는걸까, ema_decay 파라미터 값 변화주기
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
    
    labeled_iter = iter(labeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        masks = AverageMeter()
        count = AverageMeter()
        p_bar = tqdm((unlabeled_trainloader))

        model.train()
        for batch_idx, (inputs_u_w, inputs_u_s, caption, saliency_map) in enumerate(unlabeled_trainloader):
            try:
                (inputs_x, targets_x) = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                args.logger.info(f'train labeled dataset iter is reset at unlabeled mini-batch step of {batch_idx}')
                (inputs_x, targets_x) = next(labeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.local_rank)
            targets_x = targets_x.to(args.local_rank)

            # inputs = F.one_hot(inputs.long(), num_classes=3).squeeze().float()  # make 3 channels
            inputs = inputs.permute(0, 3, 1, 2).float()  # (b, c, h, w)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)

            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits
            Lx = F.cross_entropy(logits_x, targets_x.long(), reduction='mean') # targets_x.cpu().numpy(), logits_u_s.detach().cpu().numpy()
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # threshold 넘은 값 logiit
            mask = max_probs.ge(args.threshold).float() # 이 값은 threshold를 넘은 값 수도 레이블 개수
          
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
            masks.update(mask.long().sum().item())
            count.update(inputs_u_w.shape[0])

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}/{mask_total:.2f}({prop:.2f}). ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(unlabeled_trainloader),
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=masks.sum,
                mask_total=count.sum,
                prop=(masks.sum/count.sum)*100
                )
            )
            p_bar.update()

        model.eval()

        valid_loss, valid_acc, valid_auprc, valid_f1, _, _  = evaluate(epoch, args, valid_loader, model)
        is_best = valid_f1 > best_valid_f1
        best_valid_f1 = max(valid_f1, best_valid_f1)

        wandb.log({
            'epoch': epoch,
            'train/1.loss': losses.avg,
            'train/2.loss_x': losses_x.avg,
            'train/3.loss_u': losses_u.avg,
            'train/4.mask': masks.sum,
            'train/4.mask_prop': (masks.sum/count.sum)*100,
            'valid/1.acc': valid_acc,
            'valid/2.loss': valid_loss,
            'valid/3.auprc': valid_auprc,
            'valid/4.f1': valid_f1,
            # 'test/1.test_acc': test_acc,
            # 'test/2.test_loss': test_loss,
            # 'test/3.test_auprc': test_auprc,
            # 'test/4.test_f1': test_f1
            })        

        if is_best:
            test_loss, test_acc, test_auprc, test_f1, total_reals, total_preds = evaluate(epoch, args, test_loader, model, valid_f1=valid_f1)
            best_test_f1 = max(test_f1, best_test_f1)
            wandb.run.summary["test_best_f1"] = best_test_f1
            wandb.run.summary["test_f1"] = test_f1
            wandb.run.summary["test_auprc"] = test_auprc
            wandb.run.summary["test_acc"] = test_acc

            if test_f1 > 0.8:
                saving_path = os.path.join(args.out, str(epoch))
                os.makedirs(saving_path, exist_ok=True)
                
                # 수도레이블 선정된 것들 인덱스 취해서..
                idxes = np.arange(batch_size*args.mu)[mask.cpu().numpy() != 0.]
                # 모든 수도레이블(레이블이 없는 상태인 -를 제외하고 0~8 값을 다 하나씩 인덱스를 뽑아서 저장)
                if len(set(targets_u[idxes])) == args.num_classes:
                    flags = [False for i in range(args.num_classes)]
                    for sample_idx, ul in zip(idxes, targets_u[idxes]):
                        if not flags[ul]:
                            flags[ul] = True
                            weak_image = (inputs_u_w[sample_idx].detach().numpy().squeeze()*127.5).astype(np.uint8)      # 96 x 96 x 1
                            strong_image = (inputs_u_s[sample_idx].detach().numpy().squeeze()*127.5).astype(np.uint8)    # 96 x 96 x 1
                            h, w = weak_image.shape[0], weak_image.shape[0]
                            three_images = Image.new('L',(3*weak_image.shape[0], weak_image.shape[0]))
                            three_images.paste(Image.fromarray(weak_image), (0,0, w, h))
                            three_images.paste(Image.fromarray(strong_image),(w, 0, w*2, h))
                            if args.keep:
                                three_images.paste(Image.fromarray(np.squeeze(np.load(saliency_map[sample_idx])*255).astype(np.uint8)),(w*2, 0, w*3, h))
                            else:
                                three_images.paste(Image.fromarray(np.squeeze(np.zeros((96,96))).astype(np.uint8)),(w*2, 0, w*3, h))
                            final_caption = caption[sample_idx].replace('./data/wm811k/unlabeled/train/-/', '').replace('.png', '')
                            # three_images = wandb.Image(three_images, caption=final_caption)
                            # wandb.log({f"pseduo label: {WM811K.idx2label[targets_u[sample_idx]]}": three_images})
                            three_images.save(os.path.join(saving_path, f'{WM811K.idx2label[targets_u[sample_idx]]}_{final_caption}.png'))

                    # wandb.log({f"conf_mat_{epoch}" :
                    #     wandb.plot.confusion_matrix(
                    #         probs=None,
                    #         y_true=total_reals,
                    #         preds=total_preds,
                    #         class_names=np.asarray(WM811K.idx2label)[:args.num_classes])
                    #     }
                    # )
                    cm = confusion_matrix(total_reals, total_preds, labels=list(range(args.num_classes)))
                    # plot confusion matrix using seaborn heatmap
                    labels = np.asarray(WM811K.idx2label)[:args.num_classes]
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

                    # set plot labels and title
                    plt.xlabel("Predicted label")
                    plt.ylabel("True label")
                    plt.title("Confusion Matrix")

                    # show plot
                    plt.savefig(os.path.join(saving_path, 'confusion.png'))
            
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


def evaluate(epoch, args, loader, model, valid_f1=None):

    fn_auprc = torchmetrics.classification.MulticlassAveragePrecision(num_classes=args.num_classes, average='macro')
    fn_f1score = torchmetrics.classification.MulticlassF1Score(num_classes=args.num_classes, average='macro')
    
    losses = AverageMeter()
    top1s = AverageMeter()
    top3s = AverageMeter()
    auprcs = AverageMeter()
    f1s  = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    total_preds = []
    total_reals = []
    total_images = []
    
    loader = tqdm(loader)
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            
            data_time.update(time.time() - end)
            inputs3 = inputs.to(args.local_rank)
            targets = targets.to(args.local_rank)
            
            # inputs3 = F.one_hot(inputs3.long(), num_classes=3).squeeze().float()    # make 3 channels
            inputs3 = inputs3.permute(0, 3, 1, 2).float()  # (b, h, w, c) -> (b, c, h, w)
            outputs = model(inputs3)
            
            total_images.append(inputs)
            total_preds.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
            total_reals.append(targets.cpu().detach().numpy())

            loss = F.cross_entropy(outputs, targets)
            prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
            auprc = fn_auprc.to(args.local_rank)(outputs, targets)
            f1 = fn_f1score.to(args.local_rank)(torch.argmax(outputs, dim=1), targets)
        
            losses.update(loss.item(), inputs3.shape[0]) 
            top1s.update(prec1.item(), inputs3.shape[0])
            top3s.update(prec3.item(), inputs3.shape[0])
            auprcs.update(auprc.item(), inputs3.shape[0])
            f1s.update(f1.item(), inputs3.shape[0])   
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            loader.set_description(f"{'Valid' if valid_f1 is None else 'Test'}" + " Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top3: {top3:.2f}. auprc: {auprc:.2f}. f1: {f1:.2f}.".format(
                batch=batch_idx + 1,
                iter=len(loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1s.avg,
                top3=top3s.avg,
                auprc=auprcs.avg,
                f1=f1s.avg
            ))
        loader.close()
        
        total_images = np.concatenate(total_images)    
        total_preds = np.concatenate(total_preds)
        total_reals = np.concatenate(total_reals)   

        final_f1 = f1_score(y_true=total_reals, y_pred=total_preds, average='macro')

        logger.info("top-1 acc: {:.2f}".format(top1s.avg))
        logger.info("top-3 acc: {:.2f}".format(top3s.avg))
        logger.info("auprc: {:.2f}".format(auprcs.avg))
        logger.info("f1: {:.2f}".format(final_f1))
        
        df_total_preds = pd.Series(total_preds).value_counts()
        df_total_preds = df_total_preds.rename(index=dict(zip(list(range(10)), WM811K.idx2label)))
        df_total_reals = pd.Series(total_reals).value_counts()
        df_total_reals = df_total_reals.rename(index=dict(zip(list(range(10)), WM811K.idx2label)))

        logger.info("======================= total_preds ======================")
        logger.info(df_total_preds)
        logger.info("======================= total_reals ======================")
        logger.info(df_total_reals)
    
    if valid_f1 is not None and valid_f1 > 0.5:  # test 데이터셋에 한해서만 
        preds = np.asarray(total_preds[total_preds!=total_reals]) 
        reals = np.asarray(total_reals[total_preds!=total_reals])
        images = np.asarray(total_images[total_preds!=total_reals])
        
        for label_idx in range(len(WM811K.idx2label)): # remove unknown
            label = WM811K.idx2label[label_idx]
            saving_path = os.path.join(args.out, str(epoch), 'wrong_predicted', label)
            os.makedirs(saving_path, exist_ok=True)
            if label != '-':
                wrong_images = images[reals==label_idx] 
                wrong_labels_idxes = preds[reals==label_idx]
                for num, (wrong_label_idx, wrong_image) in enumerate(zip(wrong_labels_idxes, wrong_images)):
                    img = Image.fromarray((wrong_image.squeeze()*127.5).astype(np.uint8))
                    img.save(os.path.join(saving_path, f"{WM811K.idx2label[wrong_label_idx]}_{str(num)}.png" )) 

    return losses.avg, top1s.avg, auprcs.avg, final_f1, total_reals, total_preds 


if __name__ == '__main__':
    args = get_args()    
    prerequisite(args)
    check_args(args)
    
    main(0, args)  # single machine, single gpu
