import torch
from ignite.metrics.metric import Metric

class MultiThresholdMeasures(Metric):
    """
    Calculates Accuracy, IoU, F1-score (Dice Coefficient) within thresholds [0.0, 0.1, ..., 1.0]
    """
    def __init__(self):
        super(MultiThresholdMeasures, self).__init__()
        self.reset()
        self._thrs = torch.FloatTensor([i/10 for i in range(11)]).to(self._device)

    def reset(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tp = torch.zeros(11).to(device)
        self._fp = torch.zeros(11).to(device)
        self._fn = torch.zeros(11).to(device)
        self._tn = torch.zeros(11).to(device)
        self._device = device

    def update(self, output):
        logit, y = output
        n = y.size(0)

        y_pred = torch.sigmoid(logit)
        y_pred = y_pred.view(n, -1, 1).repeat(1, 1, 11) > self._thrs
        y = y.byte().view(n, -1, 1).repeat(1, 1, 11)

        tp = y_pred * y == 1
        tn = y_pred + y == 0
        fp = y_pred - y == 1
        fn = y - y_pred == 1

        self._tp += torch.sum(tp, dim=[0,1]).float()
        self._tn += torch.sum(tn, dim=[0,1]).float()
        self._fp += torch.sum(fp, dim=[0,1]).float()
        self._fn += torch.sum(fn, dim=[0,1]).float()

    def compute(self):
        return

    def compute_iou(self):
        intersect = self._tp
        union = self._tp + self._fp + self._fn
        iou = intersect / union
        return [round(i.item(), 3) for i in iou]

    def compute_f1(self):
        pr = self._tp / (self._tp + self._fp)
        re = self._tp / (self._tp + self._fn)
        f1 = 2 * pr * re / (pr + re)
        return [round(f.item(), 3) for f in f1]

    def compute_accuracy(self):
        acc = (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
        return [round(a.item(), 3) for a in acc]


class Accuracy(Metric):
    def __init__(self, multi_thrs_measure):
        super(Accuracy, self).__init__()
        self.multi_thrs_measure = multi_thrs_measure

    def update(self, output):
        return

    def compute(self):
        return self.multi_thrs_measure.compute_accuracy()


class IoU(Metric):
    def __init__(self, multi_thrs_measure):
        super(IoU, self).__init__()
        self.multi_thrs_measure = multi_thrs_measure

    def update(self, output):
        return

    def compute(self):
        return self.multi_thrs_measure.compute_iou()


class F1score(Metric):
    def __init__(self, multi_thrs_measure):
        super(F1score, self).__init__()
        self.multi_thrs_measure = multi_thrs_measure

    def update(self, output):
        return

    def compute(self):
        return self.multi_thrs_measure.compute_f1()

