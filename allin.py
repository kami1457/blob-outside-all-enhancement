import strategies

class AllIn:
    def __init__(self, mode, **kwargs):

        self.strategy = self._create_strategy(mode, **kwargs)

    def _create_strategy(self, mode, **kwargs):
        from strategy_base import BaseStrategy
        strategy_class = BaseStrategy.get_strategy(mode)
        return strategy_class(**kwargs)

    def process(self, frame):
      
        return self.strategy.process(frame)