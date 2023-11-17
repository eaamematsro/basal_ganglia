import pdb
import random
import time
import torch
import gymnasium
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from model_factory.networks import MLP
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class PPO(nn.Module):
    def __init__(
        self, summary_writer: Optional[SummaryWriter] = None, **hyperparameters
    ):
        """"""
        super().__init__()
        self._init_hyperparameters(**hyperparameters)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        env = gymnasium.vector.SyncVectorEnv(
            [self.make_env(self.gym_id) for _ in range(self.num_envs)]
        )
        if summary_writer is None:
            run_name = (
                f"{hyperparameters['gym_id']}__{hyperparameters['exp_name']}"
                f"__{hyperparameters['seed']}__{int(time.time())}"
            )
            summary_writer = SummaryWriter(f"runs/{run_name}")
            summary_writer.add_text(
                "Hyperparameters",
                f"|param|value|\n|-|-\n%s"
                % (
                    "\n".join(
                        [
                            f"|{key}|{value}|"
                            for key, value in vars(hyperparameters).items()
                        ]
                    )
                ),
            )
        self.writer = summary_writer
        pdb.set_trace()
        self.env = env
        self.obs_dim = np.prod(env.single_observation_space.shape)
        self.act_dim = np.prod(env.single_action_space.n)

        self.critic = MLP(
            input_size=self.obs_dim,
            layer_sizes=(64, 64),
            output_size=1,
            non_linearity=nn.Tanh(),
            std=1,
            return_nnl=False,
        ).to(self.device)

        self.actor = MLP(
            input_size=self.obs_dim,
            layer_sizes=(64, 64),
            output_size=np.array(self.env.single_action_space.n).prod(),
            non_linearity=nn.Tanh(),
            std=2,
            return_nnl=False,
        ).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), eps=1e-5, lr=self.lr)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), eps=1e-5, lr=self.lr_crit
        )

    def _init_hyperparameters(
        self,
        actor_lr: float = 2.5e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        seed: int = 1,
        num_envs: int = 4,
        num_steps: int = 128,
        num_mini_batches: int = 4,
        num_update_epochs: int = 4,
        alpha_entropy: float = 1e-2,
        clip_coeff: float = 0.2,
        max_grad_norm: float = 1.0,
        total_time_steps: int = 25000,
        use_gae: bool = True,
        anneal_lr: bool = True,
        normalize_advantage: bool = True,
        clip_critic: bool = True,
        gym_id: str = "CartPole-v1",
        **kwargs,
    ):
        """Initializes network hyperparameters

        Args:
            seed:
            actor_lr: Actor learning rate, by default 2.5e-3.
            critic_lr: Critic learning rate, by default 1e-3
            gamma: Discount factor, by default 0.99
            gae_lambda: Generalized advantage lambda coefficient, by default 0.95
            seed: Seed used for randomization
            num_envs: Number of parallel environments to run, by default 4
            num_steps: Number of steps per environment, by default 128
            num_mini_batches: Number of minibatches, by default 4
            num_update_epochs: Number of gradient steps per rollout, by default 4
            alpha_entropy: Weight of entropy loss, by default 0.01
            clip_coeff: PPO clipping coefficient, by default 0.2
            max_grad_norm: Maximum grad norm used to clip gradients, by default 1
            total_time_steps: Number of environment steps used for training, by default 25,000
            use_gae: A boolean flag that specifies whether to use a traditional advantage or the generalized advantage
                by default True
            anneal_lr: Whether to anneal the actor learning rate, by default True
            normalize_advantage: Whether to normalize the advantages each roll out, by default True
            clip_critic: Whether to clip the critic objective function, by default True
            gym_id: Name of gym environment

        Returns:

        """

        self.lr = actor_lr
        self.lr_crit = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.seed = seed
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.minibatches = num_mini_batches
        self.update_epochs = num_update_epochs
        self.alpha_entropy = alpha_entropy
        self.max_norm = max_grad_norm
        self.total_time_steps = total_time_steps
        self.clip = clip_coeff
        self.anneal_lr = anneal_lr
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage
        self.clip_critic = clip_critic
        self.gym_id = gym_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(self.num_steps * self.num_envs)

    @staticmethod
    def make_env(gym_id):
        def thunk():
            env = gymnasium.make(gym_id)
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    def get_value(self, obs):
        """"""
        value = self.critic(obs)
        return value

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

    def rollout(self):
        """"""

        obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim)).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        n_updates = self.total_timesteps // self.batch_size
        global_step = 0
        start_time = time.time()

        next_obs = torch.Tensor(self.env.reset()[0]).to(self.device)
        next_done = torch.Tensor(self.num_envs).to(self.device)

        for update in range(1, n_updates + 1):
            if self.anneal_lr:
                frac = 1 - (update - 1) / n_updates
                new_lr = self.lr * frac
                self.optimizer_actor.param_groups[0]["lr"] = new_lr

            for step in range(self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Step environment

                next_obs, reward, done, _, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                for key, value in info.items():
                    if key == "final_info":
                        for item in value:
                            if isinstance(item, dict) and "episode" in item.keys():
                                print(
                                    f"global_step: {global_step} Episode return: {item['episode']['r']}"
                                )

            with torch.no_grad():
                next_value = self.get_value(next_obs).reshape(1, -1)
                if self.use_gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + (self.gamma * self.gae_lambda) ** t
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            minibatch_size = self.batch_size // self.minibatches
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, new_values = self.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )

                    log_ratio = newlogprob - b_logprobs[mb_inds]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl_approx = ((ratio - 1) - log_ratio).mean()
                    mb_advantages = b_advantages[mb_inds]

                    if self.normalize_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + torch.finfo().eps
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip, 1 + self.clip
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    new_value = new_values.view(-1)
                    if self.clip_critic:
                        v_loss_u = (new_value - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            new_value - b_values[mb_inds], -self.clip, self.clip
                        )

                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_u, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.alpha_entropy * entropy_loss

                    self.optimizer_actor.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm)
                    self.optimizer_actor.step()
                    self.optimizer_critic.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm)
                    self.optimizer_critic.step()
                    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = (
                        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    )
