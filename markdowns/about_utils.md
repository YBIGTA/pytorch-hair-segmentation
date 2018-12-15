## Ignite
학습 시 [pytorch-ignite](https://pytorch.org/ignite/)를 활용해보았습니다. training 시 에폭과 배치 루프를 돌면서 여러 반복작업을 수행하는데요. 예를 들어 각각의 배치마다 loss를 역전파시키고, 각각의 에폭마다 validation set에 대해서 여러 metric을 뽑아내고 일정 주기마다 모델을 저장합니다. ignite는 이와 관련된 메소드를 제공하여 보다 깔끔한 코드를 작성할 수 있도록 도와주는 라이브러리입니다. ignite 페이지에 업로드된 아래 이미지는 같은 작업에 대해서 ignite를 사용한 경우와 사용하지 않은 경우를 비교합니다.  
![](https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png)

저희는 아래와 같은 형태로 코드를 작성했습니다. 편의를 위해 중간중간 생략된 부분이 있습니다.
1. trainer / evaluater 생성
 - ignite의 큰 뼈대는 `ignite.engine.Engine`으로 이루어져있습니다. Engine은 입력 받은 연산을 반복 수행하는 역할을 합니다.
 - `ignite.engine.create_supervised_trainer` 메소드는 loss를 계산하고 이를 역전파하는 Engine을 리턴합니다.
 - `ignite.engine.create_supervised_trainer` 메소드는 모델의 output을 산출하고 입력받은 metric들을 연산하는 Engine을 리턴합니다.

2. logging 함수 구현
 - Engine.on() 메소드는 decorator를 이용하여 특정 event가 발생했을 때 작성된 함수를 실행시키도록 합니다. 이를 이용해 아래와 같은 함수를 작성했습니다. 
 - `log_training_loss`: 각각의 iteration마다 (Events.ITERATION_COMPLETED) 발생한 loss를 로깅하는 함수입니다. 
 - `log_training_results`: 각각의 iteration마다 트레이닝 셋 전체에 대해 여러 metric 값을 로깅하는 함수입니다.
 - `log_validation_results`: 각각의 iteration마다 테스트(혹은 validation) 셋 전체에 대해 여러 metric 값을 로깅하는 함수입니다.


3. train.run()
 - Engine.run()은 입력받은 max_epochs만큼 위의 작업들을 수행합니다.


```python
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss


trainer = create_supervised_trainer(model, model_optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={
                                            'pix-acc': Accuracy(),
                                            'iou': IoU(0.5),
                                            'loss': Loss(loss),
                                            'f1': F1score()
                                            },
                                        device=device)

# execution after every training iteration
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    num_iter = (trainer.state.iteration - 1) % len(train_loader) + 1
    if num_iter % 20 == 0:
        logger.info("Epoch[{}] Iter[{:03d}] Loss: {:.2f}".format(
            trainer.state.epoch, num_iter, trainer.state.output))

# execution after every training epoch
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # evaluate on training set
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    logger.info("Training Results - Epoch: {} Avg-loss: {:.3f} Pix-acc: {:.3f} IoU: {:.3f} F1: {}".format(
        trainer.state.epoch, metrics['loss'], metrics['pix-acc'], metrics['iou'], str(metrics['f1'])))

# execution after every epoch
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    # evaluate test(validation) set
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    logger.info("Validation Results - Epoch: {} Avg-loss: {:.2f} Pix-acc: {:.2f} IoU: {:.3f} F1: {}".format(
        trainer.state.epoch, metrics['loss'], metrics['pix-acc'], metrics['iou'], str(metrics['f1'])))

trainer.run(train_loader, max_epochs=epochs)
```


현재 `ignite.metrics` 에서는 Precision, Recall, Accuracy 등의 제한된 metric만을 제공하고 있습니다. 새로운 metric은 `ignite.metrics.MetricsLambda`를 활용하여 구현 가능하다고 합니다. 저희는 `ignite.metrics`의 다른 구현체들을 참고하여 클래스를 새로 작성했습니다. `ignite.metrics.Metric`을 상속받은 후 __ init __, reset, update, compute 등의 메소드를 오버라이딩했습니다. (Pixel-level) IoU를 예로 들어보겠습니다.

1. __ init __ : 다른 메소드 실행에 필요한 멤버 변수를 저장합니다. 확률값에 대한 threshold 값을 저장해두었습니다.
2. reset : 연산을 시작하기 전 멤버변수를 초기화하는 함수입니다.
3. update : metric 계산을 위해 각각의 iteration에서 연산에 필요한 값을 업데이트하는 함수입니다. IoU의 경우 모델 결과와 GT 사이의 union과 intersection 값을 더해줍니다.
4. compute :  정해진 주기가 끝나면 (데이터 셋 전체를 다 훑어본 후) metric을 계산하는 함수입니다.

```python
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
```