from abc import ABC, abstractmethod


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
