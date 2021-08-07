class AdversarialImitationMixin:
    def discrimination_reward(self):
        raise NotImplementedError

    def train_discriminator(self):
        raise NotImplementedError