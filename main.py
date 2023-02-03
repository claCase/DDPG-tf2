import numpy as np
import tensorflow as tf
import gym
import argparse
from src import ddpg

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
    args = parser.parse_args()
    episodes = args.episodes
    transitions = args.transitions
    gpu = args.gpu
    bn = args.bn
    load_checkpoint = args.chkpt
    train = args.train
    render = args.render
    priority = args.priority
    print(f"Run Config:"
          f"\n episodes {episodes} "
          f"\n transitions {transitions} "
          f"\n gpu {gpu} "
          f"\n Train {train} "
          f"\n Render {render}"
          f"\n Priority {priority}")
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
        ddpg = ddpg.DDPGPriority(alpha=1, beta=0.5, states=state_shape[-1], actions=action_shape[-1])
    else:
        ddpg = ddpg.DDPG(state_shape[-1], action_shape[-1])
    if load_checkpoint:
        ddpg.load_models(chkpt_dir)
        ddpg.load_buffer(chkpt_dir)
    else:
        _ = ddpg(current_state)  # build current models
        _ = ddpg(current_state, target=True)  # build target models
    ddpg.update_targets(1)  # copy weights to target networks
    best = env.reward_range[0]
    episodes_reward = []
    average_episode_reward = []
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
                ddpg.train_step()
            current_state = next_state
            episode_reward.append(reward)
            if render:
                env.render()

            if terminal:
                average_reward = tf.reduce_mean(episode_reward).numpy()
                total_reward = tf.reduce_sum(episode_reward).numpy()
                average_episode_reward.append(average_reward)
                episodes_reward.append(total_reward)
                print(f"Episode {ep} Reward {total_reward} AVG {np.mean(episodes_reward[-40:])}")
                if best < average_episode_reward[-1]:
                    best = average_episode_reward[-1]
                    if train:
                        print("Saving model...")
                        ddpg.save_models(chkpt_dir)
                        ddpg.save_buffer(chkpt_dir)

    fig, ax = plt.subplots()
    sec = ax.twinx()
    ax.plot(episodes_reward, color="blue")
    sec.plot(average_episode_reward, color="green")
    plt.show()
