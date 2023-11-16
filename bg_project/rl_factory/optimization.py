import pdb
import time
import torch
import gymnasium
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from model_factory.networks import MLP


class PPO(nn.Module):
    def __init__(self, env):
        """"""
        super().__init__()
        self._init_hyperparameters()
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

    def _init_hyperparameters(self):
        """"""

        self.lr = 2.5e-3
        self.lr_crit = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.num_envs = 4
        self.num_steps = 128
        self.batch_size = int(self.num_steps * self.num_envs)
        self.minibatches: int = 4
        self.update_epochs: int = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anneal_lr: bool = True
        self.use_gae: bool = True
        self.normalize_advantage: bool = True
        self.clip = 0.2
        self.clip_critic: bool = True
        self.alpha_critic: float = 1e-9
        self.alpha_entropy: float = 0.01
        self.max_norm = 0.5

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

    def rollout(self, total_timesteps: int = 25000):
        """"""

        obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim)).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        n_updates = total_timesteps // self.batch_size
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
