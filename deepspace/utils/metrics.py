"""
This file will contain the metrics of the framework
"""
import sys
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import numpy as np
from commontools.setup import config, logger
import torch


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self, device):
        self.value = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        self.avg = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        self.sum = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        self.count = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        self.device = device
        self.reset()

    def reset(self):
        self.value = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)
        self.avg = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)
        self.sum = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)
        self.count = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg.item()


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


def cls_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def get_performance_eval(P, Y):
    precision_, recall_, thresholds = precision_recall_curve(Y.astype(np.int32), P)
    FPR, TPR, _ = roc_curve(Y.astype(np.int32), P)
    AUC = auc(FPR, TPR)
    AP = average_precision_score(Y.astype(np.int32), P)

    f_measure = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)

    best_idx = np.argmax(f_measure)

    f_measure[best_idx]
    thr = thresholds[best_idx]

    FP, FN, TN, TP = calc_confusion_mat(P >= thr, Y)

    FP_, FN_, TN_, TP_ = calc_confusion_mat(P >= thresholds[np.where(recall_ >= 1)], Y)

    F_measure = (2 * TP.sum()) / float(2 * TP.sum() + FP.sum() + FN.sum())

    return TP, FP, FN, TN, TP_, FP_, FN_, TN_, F_measure, AUC, AP


def evaluate_decision(prediction, ground_truth):

    TP, FP, FN, TN, TP_, FP_, FN_, TN_, F_measure, AUC, AP = get_performance_eval(prediction, ground_truth)

    logger.info("AP: %.03f, FP/FN: %d/%d, FP@FN=0: %d" % (AP, FP.sum(), FN.sum(), FP_.sum()))

    results = {
        'TP': TP.sum(),
        'FP': FP.sum(),
        'FN': FN.sum(),
        'TN': TN.sum(),
        'FP@FN=0': FP_.sum(),
        'f-measure': F_measure,
        'AUC': AUC,
        'AP': AP,
        'prediction': prediction,
        'ground_truth': ground_truth
    }

    return results


if __name__ == "__main__":

    evaluate_decision(sys.argv[1], folds_list=[0, 1, 2])
