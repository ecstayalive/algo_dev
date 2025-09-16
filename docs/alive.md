# Alive

## Generative model or World model


## Infer actions
According to active inference, creatures infer actions based on the minimisation of expected free energy. The below is the original free energy formula.
$$
G(\pi, s_0) = \sum_{t=0}^{N} \left[ \mathbb{E}_{q(o_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \ln p(r_{t+1}) \right]
$$

We note that when this problem is extended to an infinite time domain, the objective becomes:
$$
G(\pi, s_0) = \sum_{t=0}^{\infty} \left[ \gamma^t g(t) \right] \\
g(t) = \mathbb{E}_{q(s_t|s_{t-1}, a_{t-1})} \left[ \mathbb{E}_{q(o_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \ln p(r_{t+1}) \right]
$$

In this context, $G(\pi, s_0)$ comprises two components: one is epistemic value, which drives the agent to search for actions yielding the greatest internal state information gain. The other component is extrinsic value (reward), incentivising the agent to perform specific tasks to obtain greater rewards. $p$ denotes the expected reward probability distribution.

It is imperative that an alternative approach to optimisation is adopted. In active inference, the reinforcement learning optimisation problem is transformed into a inference problem. In other words, the preceding optimisation problem – namely, the action to be taken in order to maximise the expected cumulative reward – is thus rendered: knowing the existence of a successful state that maximises the expected cumulative reward, what action is most likely to have been taken? This view has also been used in [MPO](https://arxiv.org/abs/1806.06920)

However, relying solely on $G(\pi, s_0)$ does not appear to achieve learning as efficient as that of biological organisms. I suspect one reason is that, although we select actions based on free energy, we do not actively seek higher-value states. One example is that when undertaking tasks, we contemplate the future target state and then derive a reasonable course of action based on the current state. For state $s_t$ and our generative model, we believe it can reach any state after a finite number of time steps. One objective of the actor is to find a suitable path along this trajectory. Particularly, if we possess a structured world model, it seems unreasonable not to actively seek out states within this structured space(We use RSSM currently).One question is how we should incorporate this cost into our objective function? And how should we design this architecture?
