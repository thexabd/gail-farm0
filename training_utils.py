import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

target_update = 0
adv_update = 0

# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts):
    # Check if the function is dealing with scalar tensors and reshape them if necessary
    return {k: torch.cat([d[k].reshape(1) if d[k].dim() == 0 else d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}
  

# Indicate absorbing states
def indicate_absorbing(states, actions, terminals, next_states=None):
    absorbing_idxs = terminals.to(dtype=torch.bool)
    abs_states = torch.cat([states, torch.zeros(states.size(0), 1)], axis=1)
    abs_states[absorbing_idxs] = 0
    abs_states[absorbing_idxs, -1] = 1
    abs_actions = actions.clone()
    abs_actions[absorbing_idxs] = 0
    if next_states is not None:
        abs_next_states = torch.cat([next_states, torch.zeros(next_states.size(0), 1)], axis=1)
        return abs_states, abs_actions, abs_next_states
    else:
        return abs_states, abs_actions


# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
    def __init__(self, transitions):
        super().__init__()
        #actions = []
        transitions['states'] = torch.FloatTensor(transitions["states"])
        print(type(transitions["actions"][0]))
        numpy_array = np.stack(transitions["actions"]).astype(np.int64)
        transitions['actions'] = torch.from_numpy(numpy_array)
        print(type(transitions['actions']))
        #transitions['actions'] = torch.LongTensor(actions)
        transitions['rewards'] = torch.FloatTensor(transitions["rewards"])
        transitions['terminals'] = torch.Tensor(transitions["terminals"])
        self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']  # Detach actions

    # Allows string-based access for entire data of one type, or int-based access for single transition
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
            elif idx == 'terminals':
                return self.terminals
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

    def __len__(self):
        return self.terminals.size(0) - 1  # Need to return state and next state


# Computes and stores generalised advantage estimates ψ in the set of trajectories
def compute_advantages(trajectories, next_value, discount, trace_decay):
    with torch.no_grad():  # Do not differentiate through advantage calculation
        reward_to_go, advantage = torch.tensor([0.]), torch.tensor([0.])
        trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(trajectories['rewards']), torch.empty_like(trajectories['rewards'])
        for t in reversed(range(trajectories['states'].size(0))):
            reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (discount * reward_to_go)  # Reward-to-go/value R
            trajectories['rewards_to_go'][t] = reward_to_go
            td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * discount * next_value - trajectories['values'][t]  # TD-error δ
            advantage = td_error + (1 - trajectories['terminals'][t]) * discount * trace_decay * advantage  # Generalised advantage estimate ψ
            trajectories['advantages'][t] = advantage
            next_value = trajectories['values'][t]


# Performs one PPO update (assumes trajectories for first epoch are attached to agent)
def ppo_update(agent, trajectories, agent_optimiser, ppo_clip, epoch, value_loss_coeff=1, entropy_reg_coeff=1):
    # Recalculate outputs for subsequent iterations
    if epoch > 0:
      policy, trajectories['values'] = agent(trajectories['states'])
      trajectories['log_prob_actions'], trajectories['entropies'] = policy.log_prob(trajectories['actions'].detach()), policy.entropy()

    policy_ratio = (trajectories['log_prob_actions'] - trajectories['old_log_prob_actions']).exp()
    policy_loss = -torch.min(policy_ratio * trajectories['advantages'], torch.clamp(policy_ratio, min=1 - ppo_clip, max=1 + ppo_clip) * trajectories['advantages']).mean()  # Update the policy by maximising the clipped PPO objective
    value_loss = F.mse_loss(trajectories['values'].squeeze(), trajectories['rewards_to_go'])  # Fit value function by regression on mean squared error
    entropy_reg = -trajectories['entropies'].mean()  # Add entropy regularisation

    agent_optimiser.zero_grad()
    (policy_loss + value_loss_coeff * value_loss + entropy_reg_coeff * entropy_reg).backward()
    clip_grad_norm_(agent.parameters(), 1)  # Clamp norm of gradients
    agent_optimiser.step()


# Performs a target estimation update
def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size, absorbing):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']
        if absorbing: expert_state, expert_action = indicate_absorbing(expert_state, expert_action, expert_transition['terminals'])

        discriminator_optimiser.zero_grad(set_to_none=True)
        prediction, target = discriminator(expert_state, expert_action)
        regression_loss = F.mse_loss(prediction, target)
        #writer.add_scalar("Regression Loss", regression_loss, target_update)
        #target_update += 1
        regression_loss.backward()
        discriminator_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(discriminator, expert_trajectories, policy_trajectories, discriminator_optimiser, batch_size, absorbing=False, r1_reg_coeff=1, pos_class_prior=1, nonnegative_margin=0):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    policy_dataloader = DataLoader(policy_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    # Iterate over expert and policy data
    for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
        expert_state, expert_action, expert_next_state, expert_terminal = expert_transition['states'], expert_transition['actions'], expert_transition['next_states'], expert_transition['terminals']
        policy_state, policy_action, policy_next_state, policy_terminal = policy_transition['states'], policy_transition['actions'], policy_transition['next_states'], policy_transition['terminals']

        if absorbing:
            expert_state, expert_action, policy_state, policy_action = *indicate_absorbing(expert_state, expert_action, expert_terminal), *indicate_absorbing(policy_state, policy_action, policy_terminal)

        d_expert = discriminator(expert_state, expert_action)
        d_policy = discriminator(policy_state, policy_action)

        # Binary logistic regression
        discriminator_optimiser.zero_grad()
        expert_loss = F.binary_cross_entropy(d_expert, torch.ones_like(d_expert))  # Loss on "real" (expert) data
        #writer.add_scalar("Adversarial Expert Loss", expert_loss, step)
        autograd.backward(expert_loss, create_graph=True)
        r1_reg = 0
        for param in discriminator.parameters():
            r1_reg += param.grad.norm()  # R1 gradient penalty
        policy_loss = F.binary_cross_entropy(d_policy, torch.zeros_like(d_policy))  # Loss on "fake" (policy) data
        #writer.add_scalar("Adversarial Policy Loss", policy_loss, step)
        #adv_update += 1
        (policy_loss + r1_reg_coeff * r1_reg).backward()
        discriminator_optimiser.step()
