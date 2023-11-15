import torch
import numpy as np
import model_factory.networks as networks
from torch.distributions.multivariate_normal import MultivariateNormal
from model_factory.architectures import BaseArchitecture
from model_factory.networks import Module
from typing import Union, Optional, List
from gymnasium import Env


class PPO:
    """"""
    def __init__(self, env: Env, actor: Optional[Union[BaseArchitecture, Module]] = None,
                 critic: Optional[Union[BaseArchitecture, Module]] = None):
        """"""
        self._init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.cov_mat = torch.eye(self.act_dim)

        if actor is None:
            self.actor = networks.MLP(input_size=self.obs_dim,
                                      output_size=self.act_dim)
        else:
            self.actor = actor

        self.actor_optim = torch.optim.Adam(self.actor.parameters())

        if critic is None:
            self.critic = networks.MLP(input_size=self.obs_dim,
                                       output_size=1)
        else:
            self.critic = critic

        self.critic_optim = torch.optim.Adam(self.critic.parameters())


    def _init_hyperparameters(self):
        """"""

        self.max_episode_time = 500
        self.max_batch_time = 4800
        self.updates_per_iteration = 1
        self.lr = 5e-3
        self.gamma = 0.95
        self.clip = .2

    def get_action(self, observation):
        """"""
        with torch.no_grad():
            mean = self.actor(observation)

            distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.numpy(), log_prob

    def compute_ratings(self, batch_rews: List):
        """"""

        batch_rtgs = [] # batch ratings

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for reward in reversed(ep_rews):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs: torch.Tensor, batch_acts: torch.Tensor):

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        V = self.critic(batch_obs).squeeze()

        return V, log_probs

    def learn(self, total_timesteps: int = 1000, normalize: bool = True):
        """"""

        time = 0
        while time < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout()
            time += batch_acts.shape[0]
            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            if normalize:
                A_k = (A_k - A_k.mean()) / (A_k.std() + torch.finfo().eps)

            for _ in range(self.updates_per_iteration):

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp( curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clip(ratios, (1 - self.clip), (1 + self.clip)) * A_k

                actor_loss = - surr2.mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss.backward()
                self.critic_optim.step()


    def rollout(self):
        """"""

        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # batch action log probs
        batch_rews = [] # batch rewards

        t = 0

        while t < self.max_batch_time:
            ep_rews = [] # episode rewards
            obs = self.env.reset()

            for ep_t in range(self.max_episode_time):
                t += 1

                batch_obs.append(obs)
                action, log_probs = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_probs)

                if done:
                    break
            batch_rews.append(ep_rews)

            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_obs, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_obs, dtype=torch.float)

            batch_rtgs = self.compute_ratings(batch_rews) # batch ratings

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs
