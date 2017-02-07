class ConvergeTrigger(object):
    def __init__(self, delta=1e-6):
        self.delta = delta
        self._prevLoss = None
        print("entered ttt")

    def __call__(self, trainer):
        observation = trainer.observation
        if trainer.updater.iteration == 0: return False
        if self._prevLoss is None:
            self._prevLoss = observation['main/loss']
            return False
        loss_diff = observation['main/loss'] - self._prevLoss
        print(loss_diff.data)
        self._prevLoss = observation['main/loss']
        return abs(loss_diff.data) < self.delta
