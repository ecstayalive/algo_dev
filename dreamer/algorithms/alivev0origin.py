import itertools

import numpy as np
import torch
import torch.nn as nn
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.tensorboard import SummaryWriter

from dreamer.modules.actor import Gating, HabitActor, ThinkingActor
from dreamer.modules.critic import CriticQ, CriticV
from dreamer.modules.decoder import Decoder
from dreamer.modules.encoder import Encoder
from dreamer.modules.model import RSSM, ContinueModel, RewardModel
from dreamer.utils.buffer import ReplayBuffer
from dreamer.utils.distribution import calculate_information_gain_proxy
from dreamer.utils.utils import DynamicInfos, compute_lambda_values, create_normal_dist


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

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = HabitActor(discrete_action_bool, action_size, config).to(self.device)
        self.critic_v = CriticV(config).to(self.device)
        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)
        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.rssm.parameters(),
            self.reward_predictor.parameters(),
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_v_optimizer = torch.optim.Adam(
            self.critic_v.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

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
        data.embedded_observation = self.encoder(data.observation)
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(prior, data.action[:, t - 1], deterministic)
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )
            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )
            prior = posterior
        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos

    def _model_update(self, data: dict, infos):
        reconstructed_observation_dist = self.decoder(infos.posteriors, infos.deterministics)
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(infos.posteriors, infos.deterministics)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.done[:, 1:])
        prior_dist = create_normal_dist(
            infos.prior_dist_means, infos.prior_dist_stds, event_shape=1
        )
        posterior_dist = create_normal_dist(
            infos.posterior_dist_means, infos.posterior_dist_stds, event_shape=1
        )
        kl_divergence = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).unsqueeze(
            -1
        )
        reward_dist = self.reward_predictor(infos.posteriors, infos.deterministics)
        reward_loss = reward_dist.log_prob(data.reward[:, 1:] + 0.05 * kl_divergence)
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence
        ).mean()
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
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
                -reconstruction_observation_loss.mean().item(),
                global_step=self.num_total_episode,
            )
            self.writer.add_scalar(
                "loss/reward_loss", -reward_loss.mean().item(), global_step=self.num_total_episode
            )

    def behavior_learning(self, data, infos):
        """
        TODO: last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = infos.posteriors.reshape(-1, self.config.stochastic_size).detach()
        deterministic = infos.deterministics.reshape(-1, self.config.deterministic_size).detach()
        # continue_predictor reinit
        for _ in range(self.config.horizon_length):
            ori_action, ori_dist = self.actor(state, deterministic, True, False)
            action_log_prob = self.actor.log_prob(
                ori_dist.mean, ori_dist.scale, ori_action, squashing=True
            )
            action = ori_action.tanh()
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state,
                actions=action,
                action_log_probs=action_log_prob.sum(dim=-1, keepdim=True),
                deterministics=deterministic,
            )
        imagine_infos = self.behavior_learning_infos.get_stacked()
        self._update_agent(imagine_infos)

    def _update_agent(self, infos: AttrDict):
        values = self.critic_v(infos.priors, infos.deterministics).mean
        neg_efp = self.reward_predictor(infos.priors, infos.deterministics).mean
        if self.config.use_continue_flag:
            continues = self.continue_predictor(infos.priors, infos.deterministics).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            neg_efp,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )
        # advantage: torch.Tensor = lambda_values - values[:, :-1].detach()
        actor_loss = -torch.mean(lambda_values)
        # actor_loss = -torch.mean(behavior_learning_infos.action_log_probs[:, :-1] * advantage)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic_v(
            infos.priors.detach()[:, :-1],
            infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))
        self.critic_v_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_v.parameters(),
            self.config.clip_grad,
            self.config.grad_norm_type,
        )
        self.critic_v_optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar(
                "loss/actor", actor_loss.item(), global_step=self.num_total_episode
            )
            self.writer.add_scalar(
                "loss/critic", value_loss.item(), global_step=self.num_total_episode
            )
            # self.writer.add_scalar(
            #     "value/pragmatic", pragmatic_value.mean().item(), global_step=self.num_total_episode
            # )
            # self.writer.add_scalar(
            #     "value/epistemic", epistemic_value.mean().item(), global_step=self.num_total_episode
            # )
            self.writer.add_scalar(
                "value/neg_expected_free_energy",
                neg_efp.mean().item(),
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

    def soft_update(self, target: CriticQ, source: CriticQ, tau: float = 0.005):
        """
        target <- (1 - tau) * target + tau * source
        """
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for _ in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
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
