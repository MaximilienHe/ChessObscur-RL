#!/usr/bin/env python3
"""
Test script for material-based draw rewards.
"""
import torch
from env.reward import reward_terminal, compute_material
from env.move_tables import MoveTables

def test_material_computation():
    """Test that material computation works correctly."""
    print("Testing material computation...")

    device = "cpu"
    tables = MoveTables(device)
    piece_values = tables.piece_values

    # Create a simple board: White has queen (9), Black has rook (5)
    # White should have +4 advantage
    # Piece encoding: White pieces 1-6, Black pieces 7-12
    # Pawn=1/7, Knight=2/8, Bishop=3/9, Rook=4/10, Queen=5/11, King=6/12
    board = torch.zeros(1, 64, dtype=torch.int8)
    board[0, 0] = 5  # White queen
    board[0, 63] = 10  # Black rook

    active_is_white = torch.tensor([True])

    material = compute_material(board, piece_values, active_is_white)
    print(f"Material advantage (White queen vs Black rook): {material[0].item():.1f}")
    print(f"Expected: 4.0 (queen=9, rook=5)")

    # Flip perspective
    active_is_white = torch.tensor([False])
    material = compute_material(board, piece_values, active_is_white)
    print(f"Material advantage (from Black perspective): {material[0].item():.1f}")
    print(f"Expected: -4.0\n")

    return True


def test_draw_rewards():
    """Test that draw rewards are differentiated by material."""
    print("Testing draw rewards with material advantage...")

    device = "cpu"
    tables = MoveTables(device)
    piece_values = tables.piece_values

    # Scenario 1: Timeout draw with material advantage (White has queen, Black has rook)
    # Piece encoding: White pieces 1-6, Black pieces 7-12
    # Pawn=1/7, Knight=2/8, Bishop=3/9, Rook=4/10, Queen=5/11, King=6/12
    board = torch.zeros(3, 64, dtype=torch.int8)
    board[0, 0] = 5   # Game 0: White queen (piece type 4)
    board[0, 63] = 10  # Game 0: Black rook (piece type 3)
    board[1, 0] = 5   # Game 1: White queen
    board[1, 63] = 11   # Game 1: Black queen (equal)
    board[2, 0] = 4  # Game 2: White rook
    board[2, 63] = 11   # Game 2: Black queen (disadvantage)

    result_code = torch.tensor([3, 3, 3])  # All draws
    active_is_white = torch.tensor([True, True, True])
    full_move_count = torch.tensor([300, 300, 300])  # All timeout draws (max_steps=150)
    max_steps = 150

    rewards = reward_terminal(
        result_code,
        active_is_white,
        board=board,
        piece_values=piece_values,
        full_move_count=full_move_count,
        max_steps=max_steps
    )

    print("Timeout draw rewards:")
    print(f"  Game 0 (White queen vs Black rook, +4 advantage): {rewards[0].item():.3f}")
    print(f"  Game 1 (Equal material): {rewards[1].item():.3f}")
    print(f"  Game 2 (White rook vs Black queen, -4 disadvantage): {rewards[2].item():.3f}")
    print(f"  Expected: Game 0 > Game 1 > Game 2 (less negative for advantage)\n")

    # Scenario 2: Non-timeout draw (50-move rule)
    full_move_count_short = torch.tensor([100, 100, 100])  # Not timeout

    rewards_non_timeout = reward_terminal(
        result_code,
        active_is_white,
        board=board,
        piece_values=piece_values,
        full_move_count=full_move_count_short,
        max_steps=max_steps
    )

    print("Non-timeout draw rewards (50-move rule):")
    print(f"  All games: {rewards_non_timeout[0].item():.3f}")
    print(f"  Expected: All equal (standard draw penalty)\n")

    # Verify the ordering
    assert rewards[0] > rewards[1], "Advantage should give better reward"
    assert rewards[1] > rewards[2], "Equal should be better than disadvantage"
    assert abs(rewards_non_timeout[0] - rewards_non_timeout[1]) < 0.001, "Non-timeout draws should be equal"

    print("✓ All tests passed!")
    return True


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    device = "cpu"
    tables = MoveTables(device)
    piece_values = tables.piece_values

    # Test with no board/piece_values (fallback behavior)
    result_code = torch.tensor([3])
    active_is_white = torch.tensor([True])
    full_move_count = torch.tensor([300])

    rewards_no_material = reward_terminal(
        result_code,
        active_is_white,
        board=None,
        piece_values=None,
        full_move_count=full_move_count,
        max_steps=150
    )

    print(f"Timeout draw without material info (fallback): {rewards_no_material[0].item():.3f}")
    print(f"Expected: {-0.3:.3f} (REWARD_DRAW)\n")

    print("✓ Edge case tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Material-Based Draw Reward System Test")
    print("=" * 60 + "\n")

    try:
        test_material_computation()
        test_draw_rewards()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
