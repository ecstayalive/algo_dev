import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.tensorboard import SummaryWriter

from thunder.nn import clone_net
from thunder.rl import soft_update
from whetstone.modules.actor import Gating, HabitActor, ThinkingActor
from whetstone.modules.critic import CriticQ, CriticV
from whetstone.modules.decoder import VisionDecoder
from whetstone.modules.encoder import VisionEncoder
from whetstone.modules.model import (
    RSSM,
    ContinueModel,
    InformationGainModel,
    RewardModel,
)
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


class AliveV0Origin:
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
        self.discount = self.config.discount
        self.stochastic_size = self.config.stochastic_size
        self.deterministic_size = self.config.deterministic_size
        self.embedding_size = self.config.embedded_state_size
        # World Model
        self.rssm = RSSM(action_size, config).to(self.device)
        self.encoder = VisionEncoder(observation_shape, config).to(self.device)
        self.decoder = VisionDecoder(observation_shape, config).to(self.device)
        self.reward_model = RewardModel(config).to(self.device)
        self.information_gain_model = InformationGainModel(
            self.deterministic_size + self.stochastic_size
        ).to(self.device)
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_model.parameters())
            + list(self.information_gain_model.parameters())
        )

        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(device=self.device)
            self.model_params += list(self.continue_predictor.parameters())

        self.desired_reward_dist = torch.distributions.Normal(
            config.environment.max_step_reward, config.environment.max_step_reward_std
        )
        # Policy
        self.actor = HabitActor(discrete_action_bool, action_size, config).to(self.device)
        self.critic_v = CriticV(config).to(self.device)
        self.target_v = clone_net(self.critic_v, requires_grad=False)
        self.critic_q1 = CriticQ(action_size, config).to(device=self.device)
        self.critic_q2 = CriticQ(action_size, config).to(device=self.device)
        self.q_params = list(self.critic_q1.parameters()) + list(self.critic_q2.parameters())
        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)
        self.config = config.parameters.dreamer
        # Optimizer
        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.v_optimizer = torch.optim.Adam(
            self.critic_v.parameters(), lr=self.config.critic_learning_rate
        )
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=self.config.critic_learning_rate)
        self.continue_criterion = nn.BCELoss()
        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)
        self.writer = writer
        self.num_total_episode = 0
        self.eps = 1e-6

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
                    # self.critic_learning(data, infos)
                    self.actor_learning(data, infos)
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
            o_embed = self.encoder(data.observation[:, t])
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
            )
            prior = posterior
        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos

    def _model_update(self, data: AttrDict, infos: AttrDict):
        reconstruction_dist = self.decoder(infos.posteriors, infos.deterministics)
        reconstructed_loss = reconstruction_dist.log_prob(data.observation[:, 1:])
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(infos.posteriors, infos.deterministics)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.done[:, 1:])
        prior_dist = create_normal_dist(
            infos.prior_dist_means, infos.prior_dist_stds, event_shape=1
        )
        posterior_dist = create_normal_dist(
            infos.posterior_dist_means, infos.posterior_dist_stds, event_shape=1
        )
        kl_divergence = torch.distributions.kl_divergence(posterior_dist, prior_dist).unsqueeze(-1)
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence
        ).mean()
        log_gain_dist = self.information_gain_model(
            torch.cat([infos.priors[:, :-1], infos.deterministics[:, :-1]], dim=-1)
        )
        gain_model_loss = log_gain_dist.log_prob(
            torch.log(kl_divergence[:, 1:].detach() + self.eps)
        )
        reward_dist = self.reward_model(infos.posteriors, infos.deterministics)
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])
        model_loss: torch.Tensor = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstructed_loss.mean()
            - reward_loss.mean()
            - gain_model_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar(
                "loss/kl_div_loss", kl_divergence_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "loss/reconstruction_loss",
                -reconstructed_loss.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "loss/reward_loss", -reward_loss.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "loss/gain_model_loss",
                -gain_model_loss.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/kl_div", kl_divergence.mean().item(), global_step=self.num_total_episode
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
            self.writer.add_scalar(
                "value/information_gain_mu",
                log_gain_dist.mean.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "value/information_gain_std",
                log_gain_dist.scale.mean().item(),
                global_step=self.num_total_episode,
            )

    def actor_learning(self, data, infos):
        """
        last posterior truncation(last can be last step)
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
            approximate_epistemic = torch.exp(
                self.information_gain_model(torch.cat([prior, deterministic], dim=-1)).mean
            )
            self.behavior_learning_infos.append(
                priors=prior,
                mu_p=priors_dist.mean,
                std_p=priors_dist.scale,
                # posts=post,
                # mu_q=posterior_dist.mean,
                # std_q=posterior_dist.scale,
                actions=action,
                action_log_probs=action_log_prob.sum(dim=-1, keepdim=True),
                deterministics=deterministic,
                epistemic=approximate_epistemic,
                # o_embed=o_embed,
            )
        imagine_infos = self.behavior_learning_infos.get_stacked()
        self._update_agent(imagine_infos)

    def _update_agent(self, infos: AttrDict):
        # q_values1 = self.critic_q1(infos.priors, infos.deterministics, infos.actions).mean
        # q_values2 = self.critic_q2(infos.priors, infos.deterministics, infos.actions).mean
        # q_value = torch.minimum(q_values1, q_values2)
        # actor_loss: torch.Tensor = -torch.mean(q_value)
        # actor_loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.actor.parameters(), self.config.clip_grad, norm_type=self.config.grad_norm_type
        # )
        # self.actor_optimizer.step()
        values = self.critic_v(infos.priors, infos.deterministics).mean
        pragmatic_value = self.desired_reward_dist.log_prob(
            self.reward_model(infos.priors, infos.deterministics).mean
        )
        epistemic_value = infos.epistemic
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
        value_dist = self.critic_v(
            infos.priors.detach()[:, :-1], infos.deterministics.detach()[:, :-1]
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))
        loss = value_loss + actor_loss
        self.actor_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_v.parameters()) + list(self.actor.parameters()),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()
        self.v_optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar(
                "loss/actor", actor_loss.item(), global_step=self.num_total_episode
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
            # self.writer.add_scalar(
            #     "value/image_mu_q",
            #     infos.mu_q.mean().item(),
            #     global_step=self.num_total_episode,
            # )
            # self.writer.add_scalar(
            #     "value/image_std_q",
            #     infos.std_q.mean().item(),
            #     global_step=self.num_total_episode,
            # )
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

    def critic_learning(self, data: AttrDict, infos: AttrDict):
        """"""
        for attr in infos:
            infos[attr] = infos[attr].detach()
        values = self.critic_v(infos.posteriors, infos.deterministics).mean
        with torch.no_grad():
            actions = self.actor(infos.posteriors, infos.deterministics)
            q_values1 = self.critic_q1(infos.posteriors, infos.deterministics, actions).mean
            q_values2 = self.critic_q2(infos.posteriors, infos.deterministics, actions).mean
            q_values = torch.minimum(q_values1, q_values2)
        values_loss: torch.Tensor = torch.mean(0.5 * (values - q_values) ** 2)
        self.v_optimizer.zero_grad()
        values_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_v.parameters(), self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.v_optimizer.step()
        q_values1 = self.critic_q1(
            infos.posteriors[:, :-1], infos.deterministics[:, :-1], data.action[:, 1:-1]
        ).mean
        q_values2 = self.critic_q2(
            infos.posteriors[:, :-1], infos.deterministics[:, :-1], data.action[:, 1:-1]
        ).mean
        with torch.no_grad():
            pragmatic_value = self.desired_reward_dist.log_prob(data.reward[:, 2:])
            p_dist = create_normal_dist(
                infos.prior_dist_means[:, 1:], infos.prior_dist_stds[:, 1:], event_shape=1
            )
            q_dist = create_normal_dist(
                infos.posterior_dist_means[:, 1:], infos.posterior_dist_stds[:, 1:], event_shape=1
            )
            epistemic_value = torch.distributions.kl.kl_divergence(q_dist, p_dist).unsqueeze(-1)
            neg_efe = pragmatic_value + epistemic_value
            value_t1 = self.target_v(infos.posteriors[:, 1:], infos.deterministics[:, 1:]).mean
            q_target = neg_efe + self.discount * (1 - data.done[:, 2:]) * value_t1
        q_loss1 = 0.5 * (q_values1 - q_target) ** 2
        q_loss2 = 0.5 * (q_values2 - q_target) ** 2
        q_loss: torch.Tensor = torch.mean(0.5 * (q_loss1 + q_loss2))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_params, self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.q_optimizer.step()
        if self.writer is not None:
            # self.writer.add_scalar(
            #     "loss/actor", actor_loss.item(), global_step=self.num_total_episode
            # )
            self.writer.add_scalar(
                "loss/v_loss", values_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar("loss/q_loss", q_loss.item(), global_step=self.num_total_episode)
            self.writer.add_scalar(
                "value/pragmatic", pragmatic_value.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/epistemic", epistemic_value.mean().item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "value/neg_expected_free_energy",
                neg_efe.mean().item(),
                global_step=self.num_total_episode,
            )
        soft_update(self.target_v, self.critic_v, 0.005)

    @torch.inference_mode()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for _ in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation: torch.Tensor = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
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
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
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
