import argparse
import os.path
import pdb

import gymnasium
import wandb
import time

from rl_games.optimization import ContinuousPPO
from eaa_rl_algorithms.models.algorithms.algorithms import PPO
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="The name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="DegenerateGridWorld-v0",
        help="The name of the gym environment",
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-3, help="Learning of the actor network"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=1e-3, help="Learning of the critic network"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Lambda coefficient for generalized advantage function.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )

    parser.add_argument(
        "--action-dim", type=int, default=25, help="Action dimensionality"
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed to use",
    )
    parser.add_argument(
        "--num-mini-batches",
        type=int,
        default=32,
        help="Number of minibatches per rollout",
    )
    parser.add_argument(
        "--num-update-epochs",
        type=int,
        default=10,
        help="Number of gradient steps per rollout",
    )
    parser.add_argument(
        "--alpha-entropy", type=float, default=0, help="Weight of entropy loss"
    )
    parser.add_argument(
        "--clip-coeff", type=float, default=0.2, help="Clipping value for ppo objective"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm to apply grad clipping",
    )
    parser.add_argument(
        "--total-time-steps",
        type=int,
        default=50_000,
        help="Total number of environment time steps",
    )
    parser.add_argument(
        "--use-gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Whether to use generalized advantage functions",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Whether to anneal actor learning rates",
    )
    parser.add_argument(
        "--normalize-advantage",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Whether to normalize advantages each epoch",
    )
    parser.add_argument(
        "--clip-critic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Whether to clip critic objective",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Whether to track experiment on wandb",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="bgRL",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity name",
    )
    args, _ = parser.parse_known_args()
    return args


def make_env(gym_id):
    def thunk():
        env = gymnasium.make(gym_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def objective(trial):
    model = ContinuousPPO(
        actor_lr=trial.suggest_float("actor_lr", 1e-5, 1e-1),
        critic_lr=trial.suggest_float("critic_lr", 1e-5, 1e-1),
        num_mini_batches=trial.suggest_int("num_mini_batches", 4, 64),
        num_update_epochs=trial.suggest_int("num_update_epochs", 1, 20),
        capture_videos=False,
        gym_id="Pendulum-v1",
        exp_name="tuning",
        seed=1,
        num_envs=1,
        num_steps=1024,
    )

    model.learning()
    final_reward = model.evaluate()
    return final_reward


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.gym_id,
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
            config=vars(args),
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "Hyperparameters",
        f"|param|value|\n|-|-\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    model = PPO(
        summary_writer=writer,
        action_space="continuous",
        capture_videos=True,
        **vars(args),
    )
    model.learning()
