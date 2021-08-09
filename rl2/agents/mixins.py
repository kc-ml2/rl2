class AdversarialImitationMixin:
    def discrimination_reward(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError