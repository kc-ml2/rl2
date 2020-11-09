def atari():
    return dict(
        num_workers=64,
        batch_size=512,
        epoch=4,
        n_step=128,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        cliprange=0.1,
    )

def bullet():
    return dict(
        num_workers=64,
        batch_size=2048,
        epoch=10,
        n_step=512,
        lr=1.0e-4,
        ent_coef=0.001,
        vf_coef=0.5,
        cliprange=0.2,
    )


def dqn():
    return dict(
        num_workers=64,
        batch_size=512,
        epoch=1,
        n_step=4,
        lr=2.5e-4,
        rb_size=int(1e5),
        init_collect=10000,
        target_update=4000,
    )

def ppo():
    return dict(
        num_workers=64,
        batch_size=512,
        epoch=1,
        n_step=4,
        lr=2.5e-4,
        rb_size=int(1e5),
        init_collect=10000,
        target_update=4000,
    )

def a2c():
    return dict(
        num_workers=64,
        batch_size=512,
        epoch=4,
        n_step=128,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        cliprange=0.1,
    )
