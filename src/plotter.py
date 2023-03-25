import numpy as np
import matplotlib.pyplot as plt
import gym
from src import ddpg as models
import os


def plot_comparision():
    r_unif = np.load("../tmp/uniform/episodes_reward.npy")
    r_unif_ma = np.convolve(r_unif, [1, 1, 1, 1, 1]) / 5
    r_prior = np.load("../tmp/priority/episodes_reward.npy")
    r_prior_ma = np.convolve(r_prior, [1, 1, 1, 1, 1]) / 5
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_runif = ax.plot(r_unif_ma, color="blue", label="Uniform Buffer")
    tx = ax.twinx()
    plot_rprior = tx.plot(r_prior_ma, color="orange", label="Priority Buffer")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")
    tx.legend(loc="upper right")
    ax.set_title("Moving Average of DDPG Rewards", fontsize=20)
    plt.savefig("../tmp/cpmparision.png")


def plot_episode(chkpt_dir=None, priority=True, name="untrained", save_path="./tmp/"):
    import matplotlib.animation as animation

    env = gym.make("Pendulum-v1")
    action_shape = env.action_space.shape
    state_shape = env.observation_space.shape
    current_state = env.reset()

    if priority:
        ddpg = models.DDPGPriority(alpha=0.1,
                                   beta=0.4,
                                   states=state_shape[-1],
                                   actions=action_shape[-1],
                                   batch_size=60,
                                   buffer_size=1e6)
    else:
        ddpg = models.DDPG(state_shape[-1], action_shape[-1])

    if chkpt_dir is not None:
        ddpg.load_models(chkpt_dir)
        ddpg.load_buffer(chkpt_dir)
    else:
        _ = ddpg(current_state[None, :])  # build current models
        _ = ddpg(current_state[None, :], target=True)  # build target models
    frames = []
    terminal = False
    while not terminal:
        action, _ = ddpg(current_state[None, :], exploration=False)
        action = action[0]
        next_state, reward, terminal, info = env.step(action)
        current_state = next_state
        frame = env.render(mode='rgb_array')
        frames.append(frame)

    env.close()
    fig, ax = plt.subplots()

    def plot_frames(i):
        ax.cla()
        ax.set_title(i)
        ax.imshow(frames[i])

    anim = animation.FuncAnimation(fig, plot_frames, frames=len(frames))
    anim.save(os.path.join(save_path, f"animation_" + name + ".gif"), fps=30)
    plt.close()
