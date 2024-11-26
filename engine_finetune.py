# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from util.stat import calculate_stats

# From BEANS
class Accuracy:
    def __init__(self):
        self.num_total = 0
        self.num_correct = 0
    
    def update(self, logits, y):
        self.num_total += logits.shape[0]
        self.num_correct += torch.sum(logits.argmax(axis=1) == y).cpu().item()

    def get_metric(self):
        return {'acc': 0. if self.num_total == 0 else self.num_correct / self.num_total}

    def get_primary_metric(self):
        return self.get_metric()['acc']


class BinaryF1Score:
    def __init__(self):
        self.num_positives = 0
        self.num_trues = 0
        self.num_tps = 0

    def update(self, logits, y):
        positives = logits.argmax(axis=1) == 1
        trues = y == 1
        tps = trues & positives
        self.num_positives += torch.sum(positives).cpu().item()
        self.num_trues += torch.sum(trues).cpu().item()
        self.num_tps += torch.sum(tps).cpu().item()

    def get_metric(self):
        prec = 0. if self.num_positives == 0 else self.num_tps / self.num_positives
        rec = 0. if self.num_trues == 0 else self.num_tps / self.num_trues
        if prec + rec > 0.:
            f1 = 2. * prec * rec / (prec + rec)
        else:
            f1 = 0.

        return {'prec': prec, 'rec': rec, 'f1': f1}

    def get_primary_metric(self):
        return self.get_metric()['f1']


class MulticlassBinaryF1Score:
    def __init__(self, num_classes):
        self.metrics = [BinaryF1Score() for _ in range(num_classes)]
        self.num_classes = num_classes

    def update(self, logits, y):
        probs = torch.sigmoid(logits)
        for i in range(self.num_classes):
            binary_logits = torch.stack((1-probs[:, i], probs[:, i]), dim=1)
            self.metrics[i].update(binary_logits, y[:, i])

    def get_metric(self):
        macro_prec = 0.
        macro_rec = 0.
        macro_f1 = 0.
        for i in range(self.num_classes):
            metrics = self.metrics[i].get_metric()
            macro_prec += metrics['prec']
            macro_rec += metrics['rec']
            macro_f1 += metrics['f1']
        return {
            'macro_prec': macro_prec / self.num_classes,
            'macro_rec': macro_rec / self.num_classes,
            'macro_f1': macro_f1 / self.num_classes
        }

    def get_primary_metric(self):
        return self.get_metric()['macro_f1']


class AveragePrecision:
    """
    Taken from https://github.com/amdegroot/tnt
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)
        self.targets = torch.tensor(torch.LongStorage(), dtype=torch.int64, requires_grad=False)
        self.weights = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)

    def update(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size
                                               + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output.detach())
        self.targets.narrow(0, offset, target.size(0)).copy_(target.detach())

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def get_metric(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


class MeanAveragePrecision:
    def __init__(self):
        self.ap = AveragePrecision()

    def reset(self):
        self.ap.reset()

    def update(self, output, target, weight=None):
        self.ap.update(output, target, weight)

    def get_metric(self):
        return {'map': self.ap.get_metric().mean().item()}

    def get_primary_metric(self):
        return self.get_metric()['map']


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        if outputs.shape == targets.shape:
            acc1, acc2 = torch.tensor(0.0), torch.tensor(0.0)
        else:
            acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # acc1 = torch.as_tensor(acc1, device=device)
        # acc2 = torch.as_tensor(acc2, device=device)
        acc1_reduce = misc.all_reduce_mean(acc1)
        acc2_reduce = misc.all_reduce_mean(acc2)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            log_writer.add_scalar('acc1', acc1_reduce, epoch_1000x)
            log_writer.add_scalar('acc2', acc2_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs=[]
    targets=[]
    file_names=[]

    multi_label = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        file_names.extend(batch[2])

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            outputs.append(output)
            targets.append(target)
        if output.shape == target.shape:
            multi_label = True
            acc1, acc2 = torch.tensor(0.0), torch.tensor(0.0)
        else:
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
        # acc1, acc2 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))

    # print("OUTPUT: ", (torch.cat(outputs)[:20]))
    # print("TARGET: ", (torch.cat(targets)[:20]))
    outputs=torch.cat(outputs).cpu().numpy()
    targets=torch.cat(targets).cpu().numpy()
    if not multi_label:
        # map from class index to one-hot encoding
        targets_one_hot = np.zeros((targets.shape[0], outputs.shape[1]))
        targets_one_hot[np.arange(targets.shape[0]), targets] = 1
        targets = targets_one_hot
    stats = calculate_stats(outputs, targets)
    AP = [stat['AP'] for stat in stats]
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    f1 = np.mean([stat['f1'] for stat in stats])
    middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
    middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
    average_precision = np.mean(middle_ps)
    average_recall = np.mean(middle_rs)
    print("mAP: {:.6f}, mAUC: {:.6f}, f1: {:.6f}, pre: {:.6f}, rec: {:.6f}".format(mAP, mAUC, f1, average_precision, average_recall))

    dct= {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    dct['mAP'] = mAP
    dct['mAUC'] = mAUC
    dct['f1'] = f1
    dct['precision'] = average_precision
    dct['recall'] = average_recall

    # dct['file'] = file_names
    # dct['pred'] = outputs
    return dct
