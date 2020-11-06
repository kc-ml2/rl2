import copy
from models.model import AbstractModel
from utils.distributions import ScalarDist


class QvalueModel(AbstractModel):
    def __init__(self, args, encoder, q_head, optimizer,
                 optim_args={},
                 **kwargs):
        self.encoder, self.q_head = encoder, q_head
        self.nets = [self.encoder, self.q_head]
        self.set_optimizer(self.nets, optimizer, optim_args)

        self.nets_target = [copy.deepcopy(net) for net in self.nets]

    def update_param(self, source, target, alpha=0.0):
        for p, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_(alpha * p_t.data + (1 - alpha) * p.data)

    def update_target(self):
        for source, target in zip(self.nets, self.nets_target):
            self.update_param(source, target)

    def _infer_from_list(self, list_of_mods, x):
        ir = list_of_mods[0](x)
        q_dist = list_of_mods[1](ir)
        return q_dist

    def infer(self, x):
        return self._infer_from_list(self.nets, x)

    def infer_target(self, x):
        return self._infer_from_list(self.nets_target, x)


class DuelingQvalueModel(AbstractModel):
    def __init__(self, args, encoder, v_head, a_head, optimizer,
                 optim_args={},
                 **kwargs):
        self.encoder, self.v_head, self.a_head = encoder, v_head, a_head
        self.nets = [self.encoder, self.v_head, self.a_head]
        self.set_optimizer(self.nets, optim_args)

        self.nets_target = [copy.deepcopy(net) for net in self.nets]

    def _infer_from_list(self, list_of_mods, x):
        ir = list_of_mods[0](x)
        v_dist = list_of_mods[1](ir)
        a_dist = list_of_mods[2](ir)
        adv = a_dist.mean - a_dist.mean.mean(-1, keepdim=True)
        q_dist = ScalarDist(v_dist.mean + adv)
        return q_dist
