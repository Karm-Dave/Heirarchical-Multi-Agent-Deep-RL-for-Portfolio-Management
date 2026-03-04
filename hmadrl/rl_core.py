"""PyTorch RL utilities for discrete and continuous policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev, hidden))
        layers.append(activation())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _mlp_layernorm(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.LayerNorm(hidden))
        layers.append(activation())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _to_device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(list(self._items), batch_size)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.model = _mlp(state_dim, hidden_dims, action_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.model(state_batch)


class DQNPolicy:
    """Discrete-action DQN, kept for backward compatibility tests."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.02,
        batch_size: int = 32,
        replay_size: int = 5000,
        epsilon_start: float = 0.2,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.999,
        temperature: float = 1.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = max(1e-4, temperature)
        self.rng = random.Random(seed)

        self.device = _to_device(device)
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_size, seed=seed)

    def select_action(self, state: np.ndarray, stochastic: bool = True) -> int:
        if stochastic and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).squeeze(0)

        if not stochastic:
            return int(torch.argmax(q_values).item())

        probs = torch.softmax(q_values / self.temperature, dim=0)
        action = torch.multinomial(probs, num_samples=1)
        return int(action.item())

    def store(
        self,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        transition = Transition(
            state=np.asarray(list(state), dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(list(next_state), dtype=np.float32),
            done=bool(done),
        )
        self.replay.push(transition)

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        done_mask = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + (1.0 - done_mask) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.copy_(self.tau * param + (1.0 - self.tau) * target_param)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())


class ContinuousActor(nn.Module):
    def __init__(self, state_dim: int, simplex_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        trunk_out = hidden_dims[-1] if hidden_dims else 128
        self.backbone = _mlp(state_dim, hidden_dims, trunk_out)
        self.alpha_head = nn.Linear(trunk_out, simplex_dim)
        self.beta_a_head = nn.Linear(trunk_out, 1)
        self.beta_b_head = nn.Linear(trunk_out, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(states)
        alpha = torch.nn.functional.softplus(self.alpha_head(x)) + 1.05
        beta_a = torch.nn.functional.softplus(self.beta_a_head(x)) + 1.05
        beta_b = torch.nn.functional.softplus(self.beta_b_head(x)) + 1.05
        return alpha, beta_a, beta_b


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        state_dim: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = max(1, int(state_dim))
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.state_dim, d_model))
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, int(nhead)),
            dim_feedforward=max(2 * d_model, 128),
            dropout=float(max(0.0, min(0.5, dropout))),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = states.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)


class TransformerContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        simplex_dim: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(
            state_dim=state_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.alpha_head = nn.Linear(d_model, simplex_dim)
        self.beta_a_head = nn.Linear(d_model, 1)
        self.beta_b_head = nn.Linear(d_model, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(states)
        alpha = torch.nn.functional.softplus(self.alpha_head(x)) + 1.05
        beta_a = torch.nn.functional.softplus(self.beta_a_head(x)) + 1.05
        beta_b = torch.nn.functional.softplus(self.beta_b_head(x)) + 1.05
        return alpha, beta_a, beta_b


class TransformerCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(
            state_dim=state_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.backbone(states)
        return self.head(x)


def _build_action_distribution(
    alpha: torch.Tensor,
    beta_a: torch.Tensor,
    beta_b: torch.Tensor,
) -> tuple[torch.distributions.Dirichlet, torch.distributions.Beta]:
    dirichlet = torch.distributions.Dirichlet(alpha)
    beta = torch.distributions.Beta(beta_a.squeeze(-1), beta_b.squeeze(-1))
    return dirichlet, beta


def hold_from_fraction(
    hold_fraction: float,
    min_hold_steps: int,
    max_hold_steps: int,
    previous_hold: int,
    inertia: float,
) -> int:
    hold_fraction = max(0.0, min(1.0, hold_fraction))
    raw = min_hold_steps + int(round(hold_fraction * (max_hold_steps - min_hold_steps)))
    raw = max(min_hold_steps, min(max_hold_steps, raw))
    inertia = max(0.0, min(0.99, inertia))
    smoothed = int(round(inertia * previous_hold + (1.0 - inertia) * raw))
    return max(min_hold_steps, min(max_hold_steps, smoothed))


@dataclass
class PPORecord:
    state: np.ndarray
    action: np.ndarray
    logprob: float
    value: float
    reward: float
    done: bool


class PPOContinuousPolicy:
    """Continuous-action PPO over simplex weights and hold fraction."""

    def __init__(
        self,
        state_dim: int,
        simplex_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs: int = 4,
        network_type: str = "mlp",
        transformer_d_model: int = 96,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        min_hold_steps: int = 2,
        max_hold_steps: int = 10,
        hold_inertia: float = 0.75,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.device = _to_device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.simplex_dim = simplex_dim
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.min_hold_steps = min_hold_steps
        self.max_hold_steps = max_hold_steps
        self.hold_inertia = hold_inertia
        self._last_hold = max(min_hold_steps, int(round((min_hold_steps + max_hold_steps) / 2)))
        network_type = str(network_type).lower()
        if network_type == "transformer":
            self.actor = TransformerContinuousActor(
                state_dim=state_dim,
                simplex_dim=simplex_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
            ).to(self.device)
            self.critic = TransformerCritic(
                state_dim=state_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
            ).to(self.device)
        else:
            self.actor = ContinuousActor(state_dim, simplex_dim, hidden_dims).to(self.device)
            self.critic = _mlp(state_dim, hidden_dims, 1).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.records: list[PPORecord] = []

    def _distribution(self, states: torch.Tensor):
        alpha, beta_a, beta_b = self.actor(states)
        return _build_action_distribution(alpha, beta_a, beta_b)

    def select_action(
        self,
        state: np.ndarray,
        stochastic: bool = True,
    ) -> tuple[np.ndarray, int, np.ndarray, float, float, float]:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            dirichlet, beta = self._distribution(state_t)
            if stochastic:
                weights = dirichlet.sample()
                hold_frac = beta.sample()
            else:
                weights = dirichlet.mean
                hold_frac = beta.mean
            logprob = dirichlet.log_prob(weights) + beta.log_prob(hold_frac)
            entropy = dirichlet.entropy() + beta.entropy()
            value = self.critic(state_t).squeeze(0).squeeze(0)

        weights_np = weights.squeeze(0).detach().cpu().numpy().astype(np.float32)
        hold_fraction = float(hold_frac.squeeze(0).detach().cpu().item())
        hold_steps = hold_from_fraction(
            hold_fraction=hold_fraction,
            min_hold_steps=self.min_hold_steps,
            max_hold_steps=self.max_hold_steps,
            previous_hold=self._last_hold,
            inertia=self.hold_inertia,
        )
        self._last_hold = hold_steps
        action_vec = np.concatenate([weights_np, np.array([hold_fraction], dtype=np.float32)]).astype(np.float32)
        return (
            weights_np,
            hold_steps,
            action_vec,
            float(logprob.item()),
            float(value.item()),
            float(entropy.item()),
        )

    def remember(
        self,
        state: Iterable[float],
        action_vec: Iterable[float],
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.records.append(
            PPORecord(
                state=np.asarray(list(state), dtype=np.float32),
                action=np.asarray(list(action_vec), dtype=np.float32),
                logprob=float(logprob),
                value=float(value),
                reward=float(reward),
                done=bool(done),
            )
        )

    def learn(self) -> float | None:
        if not self.records:
            return None

        states = np.stack([r.state for r in self.records], axis=0)
        actions = np.stack([r.action for r in self.records], axis=0)
        old_logprobs = np.array([r.logprob for r in self.records], dtype=np.float32)
        rewards = np.array([r.reward for r in self.records], dtype=np.float32)
        values = np.array([r.value for r in self.records], dtype=np.float32)
        dones = np.array([float(r.done) for r in self.records], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_t = torch.as_tensor(old_logprobs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        losses: list[float] = []
        for _ in range(self.epochs):
            dirichlet, beta = self._distribution(states_t)
            weights = actions_t[:, : self.simplex_dim]
            hold_frac = actions_t[:, self.simplex_dim]
            new_log = dirichlet.log_prob(weights) + beta.log_prob(hold_frac)
            ratio = torch.exp(new_log - old_log_t)
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            actor_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
            entropy = (dirichlet.entropy() + beta.entropy()).mean()
            actor_obj = actor_loss - self.entropy_coef * entropy

            values_pred = self.critic(states_t).squeeze(-1)
            critic_loss = torch.mean((values_pred - ret_t) ** 2)

            self.actor_opt.zero_grad()
            actor_obj.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
            self.critic_opt.step()

            losses.append(float((actor_loss + critic_loss).item()))

        self.records = []
        return float(np.mean(losses)) if losses else None


@dataclass(frozen=True)
class ContinuousTransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    n_steps: int = 1


class ContinuousReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._items: deque[ContinuousTransition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, transition: ContinuousTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[ContinuousTransition]:
        return self._rng.sample(list(self._items), batch_size)


class QContinuous(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = _mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.model(x).squeeze(-1)


class DuelingQContinuous(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.state_net = _mlp_layernorm(state_dim, hidden_dims, hidden_dims[-1] if hidden_dims else 128)
        trunk_out = hidden_dims[-1] if hidden_dims else 128
        self.adv_net = _mlp_layernorm(trunk_out + action_dim, hidden_dims, 1)
        self.val_net = _mlp_layernorm(trunk_out, hidden_dims, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.state_net(states)
        adv = self.adv_net(torch.cat([s, actions], dim=-1)).squeeze(-1)
        val = self.val_net(s).squeeze(-1)
        return val + adv


class SACContinuousPolicy:
    """SAC-style continuous policy over simplex weights and hold fraction."""

    def __init__(
        self,
        state_dim: int,
        simplex_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.02,
        alpha: float = 0.15,
        auto_alpha: bool = False,
        target_entropy: float | None = None,
        use_dueling_critic: bool = True,
        batch_size: int = 64,
        replay_size: int = 20000,
        min_hold_steps: int = 2,
        max_hold_steps: int = 8,
        hold_inertia: float = 0.75,
        n_step: int = 3,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.device = _to_device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.simplex_dim = simplex_dim
        self.action_dim = simplex_dim + 1
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = bool(auto_alpha)
        self.batch_size = batch_size
        self.n_step = max(1, int(n_step))
        self.min_hold_steps = min_hold_steps
        self.max_hold_steps = max_hold_steps
        self.hold_inertia = hold_inertia
        self._last_hold = max(min_hold_steps, int(round((min_hold_steps + max_hold_steps) / 2)))

        self.actor = ContinuousActor(state_dim, simplex_dim, hidden_dims).to(self.device)
        critic_cls: type[nn.Module] = DuelingQContinuous if use_dueling_critic else QContinuous
        self.q1 = critic_cls(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2 = critic_cls(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q1_target = critic_cls(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2_target = critic_cls(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=learning_rate)
        if self.auto_alpha:
            init_log_alpha = float(np.log(max(1e-4, alpha)))
            self.log_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.target_entropy = float(
                target_entropy if target_entropy is not None else -float(self.action_dim)
            )
            self.alpha = float(np.exp(init_log_alpha))
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = float(target_entropy if target_entropy is not None else -float(self.action_dim))
        self.replay = ContinuousReplayBuffer(replay_size, seed=seed)
        self._nstep_buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(
            maxlen=max(2, self.n_step + 1)
        )

    def _distribution(self, states: torch.Tensor):
        alpha, beta_a, beta_b = self.actor(states)
        return _build_action_distribution(alpha, beta_a, beta_b)

    def _sample_action_batch(
        self,
        states: torch.Tensor,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dirichlet, beta = self._distribution(states)
        if stochastic:
            weights = dirichlet.rsample()
            hold_frac = beta.rsample()
        else:
            weights = dirichlet.mean
            hold_frac = beta.mean
        logprob = dirichlet.log_prob(weights) + beta.log_prob(hold_frac)
        entropy = dirichlet.entropy() + beta.entropy()
        action = torch.cat([weights, hold_frac.unsqueeze(-1)], dim=-1)
        return action, logprob, entropy

    def select_action(
        self,
        state: np.ndarray,
        stochastic: bool = True,
    ) -> tuple[np.ndarray, int, np.ndarray, float]:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t, _, entropy_t = self._sample_action_batch(state_t, stochastic=stochastic)
        action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        weights = action_np[: self.simplex_dim]
        hold_frac = float(action_np[self.simplex_dim])
        hold_steps = hold_from_fraction(
            hold_fraction=hold_frac,
            min_hold_steps=self.min_hold_steps,
            max_hold_steps=self.max_hold_steps,
            previous_hold=self._last_hold,
            inertia=self.hold_inertia,
        )
        self._last_hold = hold_steps
        uncertainty = float(max(0.0, entropy_t.squeeze(0).item()))
        return weights, hold_steps, action_np, uncertainty

    def store(
        self,
        state: Iterable[float],
        action_vec: Iterable[float],
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        state_arr = np.asarray(list(state), dtype=np.float32)
        action_arr = np.asarray(list(action_vec), dtype=np.float32)
        next_state_arr = np.asarray(list(next_state), dtype=np.float32)
        self._nstep_buffer.append((state_arr, action_arr, float(reward), next_state_arr, bool(done)))

        def _push_from_buffer() -> None:
            if not self._nstep_buffer:
                return
            total_reward = 0.0
            next_s = self._nstep_buffer[0][3]
            done_flag = False
            used = 0
            for used, (_, _, r, n_s, d) in enumerate(list(self._nstep_buffer)[: self.n_step], start=1):
                total_reward += (self.gamma ** (used - 1)) * float(r)
                next_s = n_s
                done_flag = bool(d)
                if d:
                    break
            s0, a0, _, _, _ = self._nstep_buffer[0]
            self.replay.push(
                ContinuousTransition(
                    state=s0,
                    action=a0,
                    reward=float(total_reward),
                    next_state=next_s,
                    done=done_flag,
                    n_steps=int(used),
                )
            )

        if len(self._nstep_buffer) >= self.n_step:
            _push_from_buffer()
            self._nstep_buffer.popleft()

        if done:
            while self._nstep_buffer:
                _push_from_buffer()
                self._nstep_buffer.popleft()

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)

        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(b.done) for b in batch], dtype=torch.float32, device=self.device)
        n_steps = torch.as_tensor([float(b.n_steps) for b in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_logprob, _ = self._sample_action_batch(next_states, stochastic=True)
            q_next = torch.min(self.q1_target(next_states, next_actions), self.q2_target(next_states, next_actions))
            discount = torch.pow(torch.full_like(n_steps, self.gamma), n_steps)
            target_q = rewards + (1.0 - dones) * discount * (q_next - self.alpha * next_logprob)

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = torch.mean((q1_pred - target_q) ** 2)
        q2_loss = torch.mean((q2_pred - target_q) ** 2)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=5.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=5.0)
        self.q2_opt.step()

        actions_new, logprob_new, _ = self._sample_action_batch(states, stochastic=True)
        q_actor = torch.min(self.q1(states, actions_new), self.q2(states, actions_new))
        actor_loss = torch.mean(self.alpha * logprob_new - q_actor)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if self.auto_alpha and self.log_alpha is not None and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logprob_new + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.alpha = float(self.log_alpha.exp().item())
            alpha_loss_value = float(alpha_loss.item())

        with torch.no_grad():
            for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.copy_(self.tau * source + (1.0 - self.tau) * target)
            for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.copy_(self.tau * source + (1.0 - self.tau) * target)

        return float((q1_loss + q2_loss + actor_loss).item() + alpha_loss_value)
