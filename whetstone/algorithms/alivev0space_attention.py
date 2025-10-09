import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.tensorboard import SummaryWriter

from whetstone.modules.actor import HabitActor, ThinkingActor
from whetstone.modules.attention import (
    AttentiveDecoder,
    AttentiveEncoder,
    SpaceAttention,
)
from whetstone.modules.critic import CriticQ, CriticV
from whetstone.modules.decoder import VisionDecoder
from whetstone.modules.encoder import VisionEncoder
from whetstone.modules.model import RSSM, ContinueModel, RewardModel
from whetstone.utils.buffer import ReplayBuffer
from whetstone.utils.distribution import calculate_information_gain_proxy
from whetstone.utils.utils import (
    DynamicInfos,
    compute_lambda_values,
    create_normal_dist,
)


def attrdict_monkeypatch_fix():
    import collections
    import collections.abc

    for type_name in collections.abc.__all__:
        setattr(collections, type_name, getattr(collections.abc, type_name))


from attrdict import AttrDict


class AliveV0SpaceAttention:
    """
    A new reinforcement learning algorithm.
    Now we testing its theory and implementation.
    """

    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer: SummaryWriter | None,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        self.config = config.parameters.dreamer
        self.stochastic_size = self.config.stochastic_size
        self.deterministic_size = self.config.deterministic_size
        self.embedding_size = self.config.embedded_state_size
        # World Model
        self.rssm = RSSM(action_size, config).to(self.device)
        self.encoder = AttentiveEncoder(
            observation_shape, self.deterministic_size, 16, 5, self.embedding_size
        ).to(device)
        self.decoder = AttentiveDecoder(
            self.stochastic_size + self.deterministic_size, observation_shape, 16, 5
        ).to(device)
        self.reward_model = RewardModel(config).to(self.device)
        self.model_params = (
            list(self.encoder.patch_encoder1.parameters())
            + list(self.encoder.linear.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_model.parameters())
        )
        if self.config.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(device=self.device)
            self.model_params += list(self.continue_predictor.parameters())
        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.continue_criterion = nn.BCELoss()
        # Task
        self.desired_reward_dist = torch.distributions.Normal(
            config.environment.max_step_reward, config.environment.max_step_reward_std
        )
        self.actor = HabitActor(discrete_action_bool, action_size, config).to(self.device)
        self.critic_v = CriticV(config).to(self.device)
        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)
        self.policy_params = itertools.chain(self.actor.parameters(), self.critic_v.parameters())
        self.policy_optimizer = torch.optim.Adam(
            self.policy_params, lr=self.config.actor_learning_rate
        )

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)
        self.writer = writer
        self.num_total_episode = 0

    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        with Progress(
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>4.0f}%",
            BarColumn(bar_width=None),
            "[",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            "]",
        ) as progress:
            train_task = progress.add_task("train", total=self.config.train_iterations)
            for _ in range(self.config.train_iterations):
                for _ in range(self.config.collect_interval):
                    data = self.buffer.sample(self.config.batch_size, self.config.batch_length)
                    infos = self.dynamic_learning(data)
                    self.behavior_learning(data, infos)
                    progress.update(train_task, advance=1 / self.config.collect_interval)
                self.environment_interaction(env, self.config.num_interaction_episodes)
                self.evaluate(env)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(prior, data.action[:, t - 1], deterministic)
            prior_dist, prior = self.rssm.transition_model(deterministic)
            o_embed, ground_truth_patches = self.encoder(
                data.observation[:, t], deterministic.detach()
            )
            posterior_dist, posterior = self.rssm.representation_model(o_embed, deterministic)
            prior_dist: torch.distributions.Normal
            posterior_dist: torch.distributions.Normal
            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
                o_embed=o_embed,
                # ground_truth_patches=ground_truth_patches,
            )
            prior = posterior
        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos

    def _model_update(self, data: dict, infos: AttrDict):
        B, OriginL, C, W, H = data.observation.shape
        L = OriginL - 1
        reconstruction, patches = self.decoder(
            infos.posteriors.flatten(end_dim=1), infos.deterministics.flatten(end_dim=1)
        )
        reconstructed_loss = (reconstruction.view(B, L, C, W, H) - data.observation[:, 1:]) ** 2
        reconstructed_loss = reconstructed_loss.sum(dim=[-1, -2, -3])
        reward_dist: torch.distributions.Independent = self.reward_model(
            infos.posteriors, infos.deterministics
        )
        reward_loss = -reward_dist.log_prob(data.reward[:, 1:])
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(infos.posteriors, infos.deterministics)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.done[:, 1:])
        prior_dist = create_normal_dist(
            infos.prior_dist_means, infos.prior_dist_stds, event_shape=1
        )
        posterior_dist = create_normal_dist(
            infos.posterior_dist_means, infos.posterior_dist_stds, event_shape=1
        )
        prior_dist_detach = create_normal_dist(
            infos.prior_dist_means.detach(), infos.prior_dist_stds.detach(), event_shape=1
        )
        posterior_dist_detach = create_normal_dist(
            infos.posterior_dist_means.detach(), infos.posterior_dist_stds.detach(), event_shape=1
        )
        kl_divergence1 = torch.distributions.kl_divergence(
            posterior_dist, prior_dist_detach
        ).unsqueeze(-1)
        kl_divergence2 = torch.distributions.kl_divergence(
            posterior_dist_detach, prior_dist
        ).unsqueeze(-1)
        kl_divergence = 0.8 * kl_divergence1 + 0.2 * kl_divergence2
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats, device=self.device), kl_divergence
        ).mean()
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            + reconstructed_loss.mean()
            + reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params, self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.model_optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar(
                "loss/kl_div_loss", kl_divergence_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/kl_div", kl_divergence.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "loss/reconstruction_loss",
                reconstructed_loss.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "loss/reward_loss", reward_loss.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/prior_mean",
                infos.prior_dist_means.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/prior_std",
                infos.prior_dist_stds.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/posterior_mean",
                infos.posterior_dist_means.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/posterior_std",
                infos.posterior_dist_stds.mean().item(),
                global_step=self.num_total_episode,
            )

    def behavior_learning(self, data, infos):
        """
        TODO: last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        prior = infos.posteriors.reshape(-1, self.config.stochastic_size).detach()
        deterministic = infos.deterministics.reshape(-1, self.config.deterministic_size).detach()
        # continue_predictor reinit
        for _ in range(self.config.horizon_length):
            ori_action, ori_dist = self.actor(prior, deterministic, True, False)
            action_log_prob = self.actor.log_prob(
                ori_dist.mean, ori_dist.scale, ori_action, squashing=True
            )
            action = ori_action.tanh()
            deterministic = self.rssm.recurrent_model(prior, action, deterministic)
            priors_dist, prior = self.rssm.transition_model(deterministic)
            with torch.no_grad():
                image, _ = self.decoder(prior, deterministic)
                o_embed, _ = self.encoder(image, deterministic)
            posterior_dist, post = self.rssm.representation_model(o_embed, deterministic)
            self.behavior_learning_infos.append(
                priors=prior,
                mu_p=priors_dist.mean,
                std_p=priors_dist.scale,
                posts=post,
                mu_q=posterior_dist.mean,
                std_q=posterior_dist.scale,
                actions=action,
                action_log_probs=action_log_prob.sum(dim=-1, keepdim=True),
                deterministics=deterministic,
                o_embed=o_embed,
            )
        imagine_infos = self.behavior_learning_infos.get_stacked()
        self._update_agent(imagine_infos)

    def _update_agent(self, infos: AttrDict):
        # Actor Loss
        values = self.critic_v(infos.priors, infos.deterministics).mean
        pragmatic_value = self.desired_reward_dist.log_prob(
            self.reward_model(infos.priors, infos.deterministics).mean
        )
        p_dist = create_normal_dist(infos.mu_p, infos.std_p, event_shape=1)
        q_dist = create_normal_dist(infos.mu_q, infos.std_q, event_shape=1)
        epistemic_value = torch.distributions.kl_divergence(q_dist, p_dist).unsqueeze(-1)
        neg_efe = pragmatic_value + epistemic_value
        if self.config.use_continue_flag:
            continues = self.continue_predictor(infos.priors, infos.deterministics).mean
        else:
            continues = self.config.discount * torch.ones_like(values)
        lambda_values = compute_lambda_values(
            neg_efe,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )
        actor_loss = -torch.mean(lambda_values)
        # advantage: torch.Tensor = lambda_values - values[:, :-1].detach()
        # actor_loss = -torch.mean(infos.action_log_probs[:, :-1] * advantage)
        # Value Loss
        value_dist = self.critic_v(
            infos.priors.detach()[:, :-1], infos.deterministics.detach()[:, :-1]
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))
        loss = value_loss + actor_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_params, self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.policy_optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar(
                "loss/actor", actor_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "loss/critic", value_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/pragmatic", pragmatic_value.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/epistemic", epistemic_value.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/image_mu_p",
                infos.mu_p.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/image_std_p",
                infos.std_p.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/image_mu_q",
                infos.mu_q.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/image_std_q",
                infos.std_q.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/neg_expected_free_energy",
                neg_efe.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/value", values.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/lambda_value",
                lambda_values.mean().item(),
                global_step=self.num_total_episode,
            )
            if self.config.use_continue_flag:
                self.writer.add_scalar(
                    "value/continue", continues.mean().item(), global_step=self.num_total_episode
                )

    @torch.inference_mode()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for _ in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation, _ = self.encoder(
                torch.from_numpy(observation).float().unsqueeze(0).to(self.device), deterministic
            )
            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(embedded_observation, deterministic)
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, truncated, info = env.step(env_action)
                if train:
                    self.buffer.add(observation, buffer_action, reward, next_observation, done)
                score += reward
                embedded_observation, _ = self.encoder(
                    torch.from_numpy(next_observation).float().unsqueeze(0).to(self.device),
                    deterministic,
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        if self.writer is not None:
                            self.writer.add_scalar("training score", score, self.num_total_episode)
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            if self.writer is not None:
                self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)

    def select_action(self, posterior, deterministic, explore=True):
        """
        A helper function to encapsulate the full runtime action selection logic.
        This replaces your old `search_and_act` but is used only at runtime.
        """
        ...

    def search_state(
        self,
        state: torch.Tensor,
        deterministic: torch.Tensor,
        step: int = 10,
        lr=0.001,
        eps: float = 0.01,
    ):
        """
        TODO:
            1. Add reachability constraint to the state space
            2. Use diffusion process to search for valid state
        """
        ...
