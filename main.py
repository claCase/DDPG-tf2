import os
import numpy as np
import tensorflow as tf
import gym
import argparse
from src import ddpg as models

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--transitions", type=int, default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--chkpt", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes
    transitions = args.transitions
    gpu = args.gpu
    bn = args.bn
    load_checkpoint = args.chkpt
    train = args.train
    render = args.render
    priority = args.priority
    save = args.save
    print(f"Run Config:"
          f"\n episodes {episodes} "
          f"\n transitions {transitions} "
          f"\n gpu {gpu}"
          f"\n Train {train}"
          f"\n Render {render}"
          f"\n Priority {priority}"
          f"\n Save {save}")
    if priority:
        chkpt_dir = "./tmp/priority"
    else:
        chkpt_dir = "./tmp/uniform"

    env = gym.make("Pendulum-v1")
    action_shape = env.action_space.shape
    state_shape = env.observation_space.shape

    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        physical_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_devices)

    current_state = env.reset()
    current_state = tf.expand_dims(current_state, 0)

    if priority:
        ddpg = models.DDPGPriority(alpha=0.1,
                                 beta=0.4,
                                 states=state_shape[-1],
                                 actions=action_shape[-1],
                                 batch_size=60,
                                 buffer_size=1e6)
    else:
        ddpg = models.DDPG(state_shape[-1],
                           action_shape[-1],
                           batch_size=60,
                           buffer_size=1e6)
    if load_checkpoint:
        ddpg.load_models(chkpt_dir)
        ddpg.load_buffer(chkpt_dir)
    else:
        _ = ddpg(current_state)  # build current models
        _ = ddpg(current_state, target=True)  # build target models
    ddpg.update_targets(1)  # copy weights to target networks
    best = env.reward_range[0]
    episodes_reward = []
    rewards = []
    i = 0
    for ep in range(episodes):
        current_state = env.reset()
        terminal = False
        episode_reward = []
        while not terminal:
            action, _ = ddpg(tf.convert_to_tensor(current_state[None, :]), exploration=train)
            action = action[0]
            next_state, reward, terminal, info = env.step(action)
            if train:
                ddpg.store_transition(current_state, action, reward, next_state, terminal)
                if i > transitions:
                    ddpg.train_step()
            current_state = next_state
            episode_reward.append(reward)
            if render:
                env.render()

            if terminal:
                total_reward = tf.reduce_sum(episode_reward).numpy()
                episodes_reward.append(total_reward)
                print(f"Episode {ep} Reward {total_reward} AVG {np.mean(episodes_reward[-10:])}")
                if best < episodes_reward[-1]:
                    best = episodes_reward[-1]
                    if save:
                        print("Saving model...")
                        ddpg.save_models(chkpt_dir)
                        ddpg.save_buffer(chkpt_dir)
            i += 1

    np.save(os.path.join(chkpt_dir, "episodes_reward.npy"), episodes_reward)

    fig, ax = plt.subplots()
    ax.plot(episodes_reward)
    ax.set_title("Episodes Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    plt.savefig(os.path.join(chkpt_dir, "episodes_reward.png"))
    plt.show()
