import torch
from ignite.metrics.metric import Metric


class Accuracy(Metric):
    """
    hard copied from https://pytorch.org/ignite/_modules/ignite/metrics/accuracy.html

    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    def __init__(self, thrs=0.5):
        super(Accuracy, self).__init__()
        self._thrs = thrs
        self.reset()

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        logit, y = output
        y_pred = torch.sigmoid(logit) >= self._thrs
        y = y.long()

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or (batch_size, ...).")

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0], ) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Maps Binary Case to Categorical Case with 2 classes
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        indices = torch.max(y_pred, dim=1)[1]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise ValueError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class IoU(Metric):
    """
    Calculates intersection over union for only foreground (hair)
    """
    def __init__(self, thrs=0.5):
        super(IoU, self).__init__()
        self._thrs = thrs
        self.reset()

    def reset(self):
        self._num_intersect = 0
        self._num_union = 0

    def update(self, output):
        logit, y = output

        y_pred = torch.sigmoid(logit) >= self._thrs
        y = y.byte()

        intersect = y_pred * y == 1
        union = y_pred + y > 0

        self._num_intersect += torch.sum(intersect).item()
        self._num_union += torch.sum(union).item()

    def compute(self):
        if self._num_union == 0:
            raise ValueError('IoU must have at least one example before it can be computed')
        return self._num_intersect / self._num_union


class F1score(Metric):
    """
    Calculates F1-score within thresholds [0.0, 0.1, ..., 1.0]
    """
    def __init__(self):
        super(F1score, self).__init__()
        self.reset()

    def reset(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tp = torch.zeros(11).to(device)
        self._fp = torch.zeros(11).to(device)
        self._fn = torch.zeros(11).to(device)
        self._device = device

    def update(self, output):
        logit, y = output
        n = y.size(0)
        thrs = torch.FloatTensor([i/10 for i in range(11)]).to(self._device)

        y_pred = torch.sigmoid(logit)
        y_pred = y_pred.view(n, -1, 1).repeat(1, 1, 11) > thrs
        y = y.byte().view(n, -1, 1).repeat(1, 1, 11)

        tp = y_pred * y == 1
        fp = y_pred - y == 1
        fn = y - y_pred == 1

        self._tp += torch.sum(tp, dim=[0,1]).float()
        self._fp += torch.sum(fp, dim=[0,1]).float()
        self._fn += torch.sum(fn, dim=[0,1]).float()


    def compute(self):
        pr = self._tp / (self._tp + self._fp)
        re = self._tp / (self._tp + self._fn)
        f1 = 2 * pr * re / (pr + re)
        return [round(f.item(), 3) for f in f1]

class DiceCoef(Metric):
    """
    Calculates intersection over union for only foreground (hair)
    """
    def __init__(self, thrs=0.5):
        super(IoU, self).__init__()
        self._thrs = thrs
        self.reset()

    def reset(self):
        self._num_intersect = 0
        self._num_union = 0

    def update(self, output):
        logit, y = output

        y_pred = torch.sigmoid(logit) >= self._thrs
        y = y.byte()

        intersect = y_pred * y == 1
        union = y_pred + y > 0

        self._num_intersect += torch.sum(intersect).item()
        self._num_union += torch.sum(union).item()

    def compute(self):
        if self._num_union == 0:
            raise ValueError('IoU must have at least one example before it can be computed')
        return 2 * self._num_intersect / (self._num_union + self._num_intersect)    
