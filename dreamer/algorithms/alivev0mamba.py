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


class AliveV0Mamba:
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
                    world_model_infos = self.dynamic_learning(data)
                    self.behavior_learning(data, world_model_infos)
                    progress.update(train_task, advance=1 / self.config.collect_interval)
                self.environment_interaction(env, self.config.num_interaction_episodes)
                self.evaluate(env)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))
        prior = self.world_model.state0(len(data.action))
        data.embedded_observation = self.encoder(data.observation)
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(prior, data.action[:, t - 1], deterministic)
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )
            # ensemble_prior_dist = self.rssm.transition_ensemble_model(deterministic)
            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                # ensemble_prior_dist_means=ensemble_prior_dist.mean,
                # ensemble_prior_dist_stds=ensemble_prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )
            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        # infos.ensemble_prior_dist_means = infos.ensemble_prior_dist_means.transpose(1, 2)
        # infos.ensemble_prior_dist_stds = infos.ensemble_prior_dist_stds.transpose(1, 2)
        self._model_update(data, infos)
        return infos

    def _model_update(self, data: dict, infos: dict):
        reconstructed_observation_dist = self.decoder(infos.z_post, infos.H1)
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(infos.z_post, infos.H1)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.done[:, 1:])

        prior_dist = create_normal_dist(infos.mu_p, infos.std_p, event_shape=1)
        # ensemble_prior_dist = create_normal_dist(
        # infos.ensemble_prior_dist_means,
        # infos.ensemble_prior_dist_stds,
        # event_shape=1,
        # )
        posterior_dist = create_normal_dist(infos.mu_q, infos.std_q, event_shape=1)
        # kl_divs = torch.distributions.kl.kl_divergence(posterior_dist, ensemble_prior_dist)
        # with torch.no_grad():
        #     # Mask shape will be (K, B, T), matching kl_divs
        #     mask = torch.distributions.Bernoulli(0.5).sample(kl_divs.shape).to(self.device)
        #     # Ensure every data point is used by at least one head
        #     mask[0, :, :] = 1.0
        kl_divs = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        reward_dist = self.reward_predictor(infos.z_post, infos.H1)
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])
        # reward_loss = reward_dist.log_prob(data.reward[:, 1:] + 0.05 * kl_divs.unsqueeze(-1))
        # kl_divergence_loss = ((kl_divs * mask).clamp_min(self.config.free_nats)).sum() / mask.sum().clamp_min(1.0)
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divs
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

    def behavior_learning(self, data, world_model_infos: WorldModelInfos):
        """
        TODO: last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = world_model_infos.z_post.reshape(-1, self.config.stochastic_size).detach()
        deterministic = world_model_infos.H1.reshape(-1, self.config.deterministic_size).detach()
        # continue_predictor reinit
        for _ in range(self.config.horizon_length):
            # action = self.select_action(state, deterministic)
            ori_action, ori_dist = self.actor(state, deterministic, True, False)
            action_log_prob = self.actor.log_prob(
                ori_dist.mean, ori_dist.scale, ori_action, squashing=True
            )
            action = ori_action.tanh()
            state_mu, state_std, deterministic = self.world_model.imagine(action, state, None)
            state = (state_mu + torch.randn_like(state_mu) * state_std).squeeze(1)
            deterministic = deterministic.squeeze(1)
            # ensemble_prior_dist = self.rssm.transition_ensemble_model(deterministic)
            # state = ensemble_prior_dist.rsample()[0]
            self.behavior_learning_infos.append(
                priors=state,
                actions=action,
                action_log_probs=action_log_prob.sum(dim=-1, keepdim=True),
                deterministics=deterministic,
                prior_dist_means=state_mu.squeeze(1),
                prior_dist_stds=state_std.squeeze(1),
                # ensemble_prior_dist_means=ensemble_prior_dist.mean,
                # ensemble_prior_dist_stds=ensemble_prior_dist.scale,
            )
        imagine_infos = self.behavior_learning_infos.get_stacked()
        # imagine_infos.ensemble_prior_dist_means = imagine_infos.ensemble_prior_dist_means.transpose(
        #     1, 2
        # )
        # imagine_infos.ensemble_prior_dist_stds = imagine_infos.ensemble_prior_dist_stds.transpose(
        #     1, 2
        # )
        self._update_agent(imagine_infos)

    def _update_agent(self, behavior_learning_infos):
        # pragmatic_value = self.reward_predictor(
        #     behavior_learning_infos.priors, behavior_learning_infos.deterministics
        # ).mean
        # epistemic_value = calculate_information_gain_proxy(
        #     behavior_learning_infos.ensemble_prior_dist_means,
        #     behavior_learning_infos.ensemble_prior_dist_stds.pow(2).log(),
        # ).unsqueeze(
        #     -1
        # )  # (B*T, H, 1)
        # neg_expected_free_energy = pragmatic_value + epistemic_value
        values = self.critic_v(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        neg_expected_free_energy = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            neg_expected_free_energy,
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
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
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
                neg_expected_free_energy.mean().item(),
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
            # posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            # posterior, deterministic = self.world_model.latent0(1)
            posterior = self.world_model.state0(1)
            action = torch.zeros(1, self.action_size).to(self.device)
            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )
            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                world_model_returns: WorldModelInfos = self.world_model(
                    action.unsqueeze(1), posterior, embedded_observation.unsqueeze(1)
                )
                posterior = world_model_returns.z_post.squeeze(1)
                embedded_observation = embedded_observation.reshape(1, 1, -1)
                action = (
                    self.actor(world_model_returns.z_post, world_model_returns.H1)
                    .squeeze(1)
                    .detach()
                )
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
