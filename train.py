import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from a1_mujoco_env import A1MujocoEnv
from a1_real_env import A1RealEnv
from tqdm import tqdm

MODEL_DIR = "models"
LOG_DIR = "logs"


def train(args):
    vec_env = make_vec_env(
        A1MujocoEnv,
        env_kwargs={"ctrl_type": args.ctrl_type},
        n_envs=args.num_parallel_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        run_name = f"{train_time}"
    else:
        run_name = f"{train_time}-{args.run_name}"

    model_path = f"{MODEL_DIR}/{run_name}"
    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

    # Evaluate the model every eval_frequency for 5 episodes and save
    # it if it's improved over the previous best model.
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=model_path,
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        # Default PPO model hyper-parameters give good results
        # TODO: Use dynamic learning rate
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )
    # Save final model
    model.save(f"{model_path}/final_model")


def test(args):
    model_path = Path(args.model_path)

    if not args.record_test_episodes:
        # Render the episodes live
        env = A1MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="human",
        )
        inter_frame_sleep = 0.016
    else:
        # Record the episodes
        env = A1MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="rgb_array",
            camera_name="tracking",
            width=1920,
            height=1080,
        )
        env = gym.wrappers.RecordVideo(
            env, video_folder="recordings/", name_prefix=model_path.parent.name
        )
        inter_frame_sleep = 0.0

    model = PPO.load(path=model_path, env=env, verbose=1)

    num_episodes = args.num_test_episodes
    total_reward = 0
    total_length = 0
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Slow down the rendering
            time.sleep(inter_frame_sleep)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward

    print(
        f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}"
    )

def test_real(args):
    # 创建实际环境
    env = A1RealEnv()

    # 加载模型
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f"Model path {model_path} does not exist!")

    model = PPO.load(path=model_path, env=env, verbose=1)

    # 让机器人进入站立状态
    env.robot.stand_up(standup_time=1.5, reset_time=5)
    # 如果需要初始化观测值，调用 reset()
    obs, _ = env.reset()

    max_episode_length = 20.0 / 0.002  # 20s
    max_steps = 10 * int(max_episode_length)  # 根据模拟环境中的逻辑计算

    try:
        for i in range(max_steps):
            start_time = time.time()
            # 根据策略生成动作
            action, _ = model.predict(obs, deterministic=True)

            # 将动作应用到环境，并获取新状态
            obs = env.step(action)

            print("policy step time (ms): ", (time.time() - start_time) * 1000, "\tHz: ", 1.0 / (time.time() - start_time))

    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        print("Test finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test", "real"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=1,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=5,
        help="Number of episodes to test the model",
    )
    parser.add_argument(
        "--record_test_episodes",
        action="store_true",
        help="Whether to record the test episodes or not. If false, the episodes are rendered in the window.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="The frequency of evaluating the models while training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="position",
        help="Whether the model should control the robot using torque or position control.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args)
    elif args.run == "real":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test_real(args)
