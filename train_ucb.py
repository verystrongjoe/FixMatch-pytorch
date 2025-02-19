import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
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
from utils.common import get_args_ucb, de_interleave, interleave, save_checkpoint
from utils.common import set_seed, create_model, get_cosine_schedule_with_warmup
from datetime import datetime
import yaml
from argparse import Namespace
from PIL import Image
import collections
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
best_valid_f1 = 0
best_test_f1 = 0


def create_pie_plot(arr, labels):
    unique_values, sizes = np.unique(arr, return_counts=True)
    explode = (0.1, 0)  # explode 1st slice for emphasis
    plt.figure(figsize=(5,5))
    plt.pie(sizes,  labels=labels,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    return plt


def prerequisite(args):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(dict(args._get_kwargs()))
    args.logger = logger

    # set wandb
    wandb.init(project=args.project_name, mode='online', config=args)

    args.logger.info(f"sweep configuraion is loaded.")
    args.logger.info(wandb.config)

    run_name = f"ucb_{args.ucb}_{args.proportion}_{args.ucb_alpha}"                
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
    args.model = model # context vector 줄이기 위해!
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

        if args.ucb:
            num_weaks = []
            num_strongs = []
            reward_w_1 = []
            reward_w_2 = []
            reward_s_1 = []
            reward_s_2 = []

        model.train()
        for batch_idx, items in enumerate(unlabeled_trainloader):
            if args.ucb:
                arm_for_weak_aug, inputs_u_w, arm_for_strong_aug, inputs_u_s, inputs_origin, caption, saliency_map = items
                num_weaks.extend(arm_for_weak_aug)
                num_strongs.extend(arm_for_strong_aug) 
            else:
                inputs_u_w, inputs_u_s, caption, saliency_map = items
            try:
                (inputs_x, targets_x) = next(labeled_iter) # labeled
            except:
                labeled_iter = iter(labeled_trainloader)
                args.logger.info(f'train labeled dataset iter is reset at unlabeled mini-batch step of {batch_idx}')
                (inputs_x, targets_x) = next(labeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            
            
            if args.ucb:
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_origin)), 3*args.mu+1).to(args.local_rank)
                targets_x = targets_x.to(args.local_rank)

                # inputs = F.one_hot(inputs.long(), num_classes=3).squeeze().float()  # make 3 channels
                inputs = inputs.permute(0, 3, 1, 2).float()  # (b, c, h, w)
                logits = model(inputs)
                logits = de_interleave(logits, 3*args.mu+1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s, logits_u_origin = logits[batch_size:].chunk(3)

                del logits
                Lx = F.cross_entropy(logits_x, targets_x.long(), reduction='mean') # targets_x.cpu().numpy(), logits_u_s.detach().cpu().numpy()
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # threshold 넘은 값 logiit
                mask = max_probs.ge(args.threshold).float() # 이 값은 threshold를 넘은 값 수도 레이블 개수

            else:
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


            if args.ucb:
                #######################################################################################################################
                # bandit 적용 부분 (리워드 및 컨텍스트 반영 및 업데이트 )
                #######################################################################################################################
                # TODO: 낮아야 좋은건지 높아야 좋을건지 고민되네. 서로 유사하긴 해야하지 않나. 둘다 실험이 필요할듯
                # -->업데이트를 하자면 cosine 유사도가 낮도록 해놓고 pseudo label이 얻어졌을때 리워드를 크게 주는것이 좋을듯
                
                rewards_weak_first = nn.CosineSimilarity(dim=1, eps=1e-6)(logits_u_w, logits_u_origin)
                rewards_weak_seocnd = -2 * (F.cross_entropy(logits_u_w, targets_u, reduction='none') * mask)

                # TODO: Strong augmentation의 경우에는 코사인 유사도가 낮아야!
                rewards_strong_first = -1 * nn.CosineSimilarity(dim=1, eps=1e-6)(logits_u_s, logits_u_origin)
                rewards_strong_second =  -2 * (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask)

                reward_w_1.append(rewards_weak_first.cpu().detach().numpy())
                reward_w_2.append(rewards_weak_seocnd.cpu().detach().numpy())

                reward_s_1.append(rewards_strong_first.cpu().detach().numpy())
                reward_s_2.append(rewards_strong_second.cpu().detach().numpy())

                # TODO: arm을 선택할때 좀 곤란해서 일단 원 이미지 입력으로..
                context_vectors = inputs_origin.flatten(start_dim=1).cpu().detach().numpy()

                weak_reward_vectors = (rewards_weak_first + rewards_weak_seocnd).cpu().detach().numpy()
                strong_reward_vectors = (rewards_strong_first + rewards_strong_second).cpu().detach().numpy()

                # update weak policy
                for arm in range(args.ucb_weak_policy.K_arms):
                    states  = context_vectors[arm_for_weak_aug == arm]
                    rewards = weak_reward_vectors[arm_for_weak_aug == arm]
                    args.ucb_weak_policy.linucb_arms[arm].add_to_buffer(states, rewards)

                # update storng policy
                for arm in range(args.ucb_strong_policy.K_arms):
                    states  = context_vectors[arm_for_strong_aug == arm]
                    rewards = strong_reward_vectors[arm_for_strong_aug == arm]
                    args.ucb_strong_policy.linucb_arms[arm].add_to_buffer(states, rewards)
                #######################################################################################################################                
                #######################################################################################################################

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


        if args.ucb:
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
                'reward_weak_first_mean': np.asarray(reward_w_1).mean(),
                'reward_weak_first_std': np.asarray(reward_w_1).std(),
                'reward_weak_second_mean': np.asarray(reward_w_2).mean(),
                'reward_weak_second_std': np.asarray(reward_w_2).std(),
                'reward_strong_first_mean': np.asarray(reward_s_1).mean(),
                'reward_strong_first_std': np.asarray(reward_s_1).std(),
                'reward_strong_second_mean': np.asarray(reward_s_2).mean(),
                'reward_strong_second_std': np.asarray(reward_s_2).std(),
                }            
                )        
            
            plt = create_pie_plot(num_weaks, args.simple_modes)
            wandb.log({"Num weaks": wandb.Image(plt, caption="Distribution of Weak Augmentations")})
            plt.close()
            plt = create_pie_plot(num_strongs, args.composite_modes)
            wandb.log({"Num strong": wandb.Image(plt, caption="Distribution of Strong Augmentations")})
            plt.close()
        else:
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
                'valid/4.f1': valid_f1
                }            
                )   
        if is_best:
            test_loss, test_acc, test_auprc, test_f1, total_reals, total_preds = evaluate(epoch, args, test_loader, model, valid_f1=valid_f1)
            best_test_f1 = max(test_f1, best_test_f1)
            wandb.run.summary["test_best_f1"] = best_test_f1
            wandb.run.summary["test_f1"] = test_f1
            wandb.run.summary["test_auprc"] = test_auprc
            wandb.run.summary["test_acc"] = test_acc



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
    args = get_args_ucb() 
    prerequisite(args)   
    main(0, args)  # single machine, single gpu
