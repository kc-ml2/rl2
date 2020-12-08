from abc import ABC, abstractmethod
from rl2.utils.logger import Logger
from rl2.utils.summary import EvaluationMetrics


class AbstractAgent(ABC):
    '''
    An agent uses models, collectors, etc to orchestrate the learning
    '''
    @abstractmethod
    def train(self):
        pass


class GeneralAgent(AbstractAgent):
    name = 'General'

    def __init__(self, args, model, collector):
        self.args = args
        self.model = model
        self.collector = collector
        self.logger = Logger(self.name, args=args)

        self.info = EvaluationMetrics([
            'Time/Step',
            'Time/Item',
            'Loss/Total',
            'Loss/Value',
            'Values/Reward',
            'Values/Value',
            'Score/Train',
        ])

    def loss_func(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        self.collector.step_env(self.logger, self.info)
        for epoch in range(self.args.epoch):
            self.collector.reset_count()
            while self.collector.has_next():
                data = self.collector.step()
                loss = self.loss_func(*data, info=self.info)
                self.model.step(loss)
