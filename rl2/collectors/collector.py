from abc import ABC, abstractmethod
import time
from rl2.utils.common import safemean


class AbstractCollector(ABC):
    '''
    A collector should take env and return batches of data upon call
    '''
    def __init__(self):
        self.env = None
        self.model = None
        self.batch_size = None
        self.minibatch_size = None

    @abstractmethod
    def step(self):
        '''
        Prepare a batch of data from the stored buffer
        '''
        pass

    @abstractmethod
    def step_env(self):
        '''
        Step environment to generate data and store them appropriately
        '''
        pass

    @abstractmethod
    def reset_count(self):
        '''
        Resets the batch count.
        Needed when running multiple epochs to reset remaining batches
        '''
        pass

    @abstractmethod
    def has_next(self):
        '''
        returns true if more minibatches are left for the update period
        '''
        pass


class GeneralCollector(AbstractCollector):
    def __init__(self, args):
        self.args = args
        self.time = None

    def step(self):
        raise NotImplementedError

    def step_env(self):
        raise NotImplementedError

    def reset_count(self):
        raise NotImplementedError

    def has_next(self):
        raise NotImplementedError

    def log(self, logger, info):
        curr_time = time.time()
        if self.time is not None:
            elapsed = curr_time - self.time
            info.update('Time/Step', elapsed)
            info.update('Time/Item', elapsed / self.args.num_workers)
        self.time = curr_time
        if self.frames % self.args.log_step < self.args.num_workers:
            info.update(
                'Score/Train',
                safemean([score for score in self.epinfobuf])
            )
            logger.log(
                "Training statistics for step: {}".format(self.frames)
            )
            logger.scalar_summary(
                info.avg,
                self.frames,
                tag='train'
            )
            info.reset()
