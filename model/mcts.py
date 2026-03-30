"""
mcts.py — Monte Carlo Tree Search for Chess Obscur inference.

v11: MCTS for stronger play at evaluation/inference time.
Uses the policy network for move priors and the value network for leaf evaluation.
Handles Chess Obscur's stochastic defense phase via chance nodes.

Usage:
    action = mcts_search(env, network, env_idx=0, cfg=cfg)

This is NOT used during training (too expensive for 2048 parallel envs).
It's for ai_server.py evaluation and evaluate_model.py.
"""
import math
import torch
from typing import Dict, Optional

from env.chess_obscur_env import (
    ChessObscurEnv, PHASE_MOVE, PHASE_DEFENSE, PHASE_PARRY, PHASE_FINISHED,
    ACTION_ATTEMPT_BLOCK, ACTION_ATTEMPT_PARRY, ACTION_ACCEPT_LOSS,
)
from model.network import ChessObscurNetwork


class MCTSNode:
    __slots__ = ("children", "visit_count", "value_sum", "prior", "is_terminal",
                 "is_chance", "action")

    def __init__(self, prior: float = 0.0, action: int = -1):
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.is_terminal: bool = False
        self.is_chance: bool = False
        self.action: int = action

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCB score."""
        best = None
        best_score = -float("inf")
        for child in self.children.values():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best = child
        return best

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


def _get_obs_and_mask(env: ChessObscurEnv, idx: int) -> tuple:
    """Get observation and legal mask for a single env index."""
    obs = env._build_obs()[idx:idx+1]
    mask = env.get_legal_mask()[idx:idx+1]
    return obs, mask


def _expand_node(node: MCTSNode, env: ChessObscurEnv, network: ChessObscurNetwork,
                 idx: int, device: torch.device) -> float:
    """Expand a leaf node: get policy priors and value estimate."""
    if env.phase[idx] == PHASE_FINISHED:
        node.is_terminal = True
        # Terminal value: +1 for win, -1 for loss, 0 for draw (from active player's perspective)
        result = env.result[idx].item()
        is_white = env.turn_is_white[idx].item()
        if result == 1:  # white wins
            return 1.0 if is_white else -1.0
        elif result == 2:  # black wins
            return -1.0 if is_white else 1.0
        else:
            return 0.0

    obs, legal_mask = _get_obs_and_mask(env, idx)
    obs = obs.to(device)
    legal_mask = legal_mask.to(device)

    with torch.no_grad():
        policy_logits, value = network(obs, legal_mask)

    value_estimate = value.item()

    # Create children for legal actions
    legal_actions = legal_mask[0].nonzero(as_tuple=True)[0]
    if legal_actions.shape[0] == 0:
        node.is_terminal = True
        return value_estimate

    probs = torch.softmax(policy_logits[0], dim=0)

    for action in legal_actions:
        a = action.item()
        node.children[a] = MCTSNode(prior=probs[a].item(), action=a)

    # Defense phase is stochastic — mark as chance node
    if env.phase[idx] == PHASE_DEFENSE:
        node.is_chance = True

    return value_estimate


def _simulate(root: MCTSNode, env: ChessObscurEnv, network: ChessObscurNetwork,
              idx: int, c_puct: float, device: torch.device):
    """Run one MCTS simulation: select → expand → evaluate → backpropagate."""
    path = [root]
    node = root
    snapshots = []
    saved_hist_idx = env._hist_idx  # save circular buffer index (global, not per-env)

    # Selection: traverse tree until we find an unexpanded node
    while node.is_expanded and not node.is_terminal:
        if node.is_chance:
            # Chance node (defense phase): pick randomly among children
            # weighted by prior (approximation of the stochastic outcome)
            import random
            children_list = list(node.children.values())
            node = random.choice(children_list)
        else:
            node = node.best_child(c_puct)

        path.append(node)

        # Apply the action in the environment
        snapshots.append(env.snapshot(idx))
        action_tensor = torch.full((env.N,), 0, dtype=torch.int64, device=env.device)
        action_tensor[idx] = node.action

        # Temporarily disable stats collection during MCTS simulation
        env.step(action_tensor)

    # Expansion + evaluation
    if not node.is_terminal:
        value = _expand_node(node, env, network, idx, device)
    else:
        # Re-evaluate terminal
        result = env.result[idx].item()
        is_white = env.turn_is_white[idx].item()
        if result == 1:
            value = 1.0 if is_white else -1.0
        elif result == 2:
            value = -1.0 if is_white else 1.0
        else:
            value = 0.0

    # Backpropagation
    # Negate value at each level (zero-sum game, alternating perspective)
    for i, n in enumerate(reversed(path)):
        # Value alternates sign as we go up the tree
        sign = 1.0 if (i % 2 == 0) else -1.0
        n.visit_count += 1
        n.value_sum += sign * value

    # Restore env state (reverse order)
    for snap in reversed(snapshots):
        env.restore(idx, snap)
    env._hist_idx = saved_hist_idx  # restore circular buffer index


def mcts_search(env: ChessObscurEnv, network: ChessObscurNetwork,
                env_idx: int = 0, num_simulations: int = 50,
                c_puct: float = 1.5, temperature: float = 0.0,
                device: Optional[torch.device] = None) -> int:
    """
    Run MCTS search and return the best action for env_idx.

    Args:
        env: The vectorized environment (we only use slot env_idx).
        network: The policy/value network (in eval mode).
        env_idx: Which environment slot to search from.
        num_simulations: Number of MCTS simulations to run.
        c_puct: Exploration constant for UCB.
        temperature: 0 = pick most visited, >0 = sample proportional to visits^(1/T).
        device: Device for network inference.

    Returns:
        Best action index.
    """
    if device is None:
        device = next(network.parameters()).device

    root = MCTSNode()
    _expand_node(root, env, network, env_idx, device)

    if root.is_terminal or not root.children:
        # No legal moves or game over — return accept_loss
        return ACTION_ACCEPT_LOSS

    for _ in range(num_simulations):
        _simulate(root, env, network, env_idx, c_puct, device)

    # Select action based on visit counts
    if temperature <= 0:
        # Greedy: most visited child
        best_action = max(root.children.keys(),
                          key=lambda a: root.children[a].visit_count)
    else:
        # Sample proportional to visit_count^(1/temperature)
        actions = list(root.children.keys())
        visits = torch.tensor([root.children[a].visit_count for a in actions],
                              dtype=torch.float32)
        if temperature != 1.0:
            visits = visits.pow(1.0 / temperature)
        probs = visits / visits.sum()
        idx = torch.multinomial(probs, 1).item()
        best_action = actions[idx]

    return best_action
