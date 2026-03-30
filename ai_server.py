"""
ai_server.py — HTTP API pour servir le modèle Chess Obscur entraîné.

Le serveur Node.js appelle ce service pour obtenir les coups de l'IA.

Usage:
    pip install fastapi uvicorn torch
    python ai_server.py --checkpoint checkpoints/step_36962304.pt --port 8100

Endpoints:
    POST /move       — demande un coup (phase move, parry_move ou defense)
    GET  /health     — health check
"""
import argparse
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn

# ── Imports du projet RL ──
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.network import ChessObscurNetwork
from config import Config
from env.chess_obscur_env import (
    ChessObscurEnv,
    PHASE_MOVE,
    PHASE_DEFENSE,
    PHASE_PARRY,
    ACTION_ATTEMPT_BLOCK,
    ACTION_ATTEMPT_PARRY,
    ACTION_ACCEPT_LOSS,
)
from model.mcts import mcts_search
from utils.checkpoint import prepare_model_state_dict

TOTAL_ACTIONS = ACTION_ACCEPT_LOSS + 1

# ─────────────────────────────────────────────
#  Pydantic models pour l'API
# ─────────────────────────────────────────────

class QteZones(BaseModel):
    """Zones QTE envoyées par server.js."""
    blockStartMs: int = 0
    blockEndMs: int = 0
    parryStartMs: int = 0
    parryEndMs: int = 0


class MoveRequest(BaseModel):
    """État du jeu envoyé par server.js pour demander un coup."""
    board: List[Optional[str]]          # 64 cases, null ou "P","k", etc.
    turn: str                           # "w" ou "b"
    castling: Dict[str, bool]           # {wK, wQ, bK, bQ}
    enPassant: Optional[int] = None     # index ou null
    phase: str                          # "move", "defense", "parry_move"
    checkAttempts: Dict[str, int] = {"w": 0, "b": 0}
    halfMoves: int = 0

    # Pour parry_move : quelle case contrôler
    parrySquare: Optional[int] = None
    parryController: Optional[str] = None

    # Coups légaux envoyés par le serveur (format: {fromIdx: [{from, to, capture, ...}]})
    legal: Optional[Dict[str, list]] = None

    # Infos de défense (pour phase defense)
    pendingAttackerPiece: Optional[str] = None
    pendingDefenderPiece: Optional[str] = None
    pendingAttackerSq: Optional[int] = None
    pendingTargetSq: Optional[int] = None
    pendingAttackerColor: Optional[str] = None
    fullMoveCount: Optional[int] = None

    # ── FIX: zones QTE transmises par ai_client.js ──
    qteZones: Optional[QteZones] = None
    qteDurationMs: Optional[int] = 2000


class MoveResponse(BaseModel):
    """Réponse de l'IA."""
    action: str                # "move", "defense"
    fromSq: Optional[str] = None     # ex: "e2"
    toSq: Optional[str] = None       # ex: "e4"
    promotion: Optional[str] = None  # "q","r","b","n" ou null
    defenseAction: Optional[str] = None  # "stop" ou "accept_loss"
    stopMs: Optional[int] = None     # position du slider si stop


# ─────────────────────────────────────────────
#  Conversion état JS → observation tensor
# ─────────────────────────────────────────────

PIECE_TO_INT = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}


def _in_bounds(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def _board_piece(board: torch.Tensor, file_idx: int, rank_idx: int) -> int:
    if not _in_bounds(file_idx, rank_idx):
        return 0
    return int(board[file_idx + rank_idx * 8].item())


def _find_king(board: torch.Tensor, king_is_white: bool) -> Optional[int]:
    target = 6 if king_is_white else 12
    for idx in range(64):
        if int(board[idx].item()) == target:
            return idx
    return None


def _is_square_attacked(board: torch.Tensor, sq_idx: int, by_white: bool) -> bool:
    file_idx = sq_idx % 8
    rank_idx = sq_idx // 8

    # Pawn attacks (reverse lookup from target square).
    pawn_piece = 1 if by_white else 7
    pawn_rank = rank_idx - 1 if by_white else rank_idx + 1
    for df in (-1, 1):
        pf = file_idx + df
        if _in_bounds(pf, pawn_rank) and _board_piece(board, pf, pawn_rank) == pawn_piece:
            return True

    # Knight attacks.
    knight_piece = 2 if by_white else 8
    knight_offsets = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    for df, dr in knight_offsets:
        nf, nr = file_idx + df, rank_idx + dr
        if _in_bounds(nf, nr) and _board_piece(board, nf, nr) == knight_piece:
            return True

    # King attacks.
    king_piece = 6 if by_white else 12
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            kf, kr = file_idx + df, rank_idx + dr
            if _in_bounds(kf, kr) and _board_piece(board, kf, kr) == king_piece:
                return True

    # Sliding attacks.
    bishop_piece = 3 if by_white else 9
    rook_piece = 4 if by_white else 10
    queen_piece = 5 if by_white else 11

    def _ray(df: int, dr: int, bishop_ok: bool, rook_ok: bool) -> bool:
        rf, rr = file_idx + df, rank_idx + dr
        while _in_bounds(rf, rr):
            piece = _board_piece(board, rf, rr)
            if piece != 0:
                if piece == queen_piece:
                    return True
                if bishop_ok and piece == bishop_piece:
                    return True
                if rook_ok and piece == rook_piece:
                    return True
                return False
            rf += df
            rr += dr
        return False

    # Diagonals
    if _ray(1, 1, True, False) or _ray(1, -1, True, False) or _ray(-1, 1, True, False) or _ray(-1, -1, True, False):
        return True
    # Straights
    if _ray(1, 0, False, True) or _ray(-1, 0, False, True) or _ray(0, 1, False, True) or _ray(0, -1, False, True):
        return True

    return False


def _is_in_check(board: torch.Tensor, side_is_white: bool) -> bool:
    king_sq = _find_king(board, side_is_white)
    if king_sq is None:
        # Mirror training env semantics: missing king = in check / losing state.
        return True
    return _is_square_attacked(board, king_sq, by_white=not side_is_white)


def board_js_to_tensor(board_js: list) -> torch.Tensor:
    """Convertit le board JS (64 éléments, null ou string) en tensor int8."""
    t = torch.zeros(64, dtype=torch.int8)
    for i, p in enumerate(board_js):
        if p is not None and p in PIECE_TO_INT:
            t[i] = PIECE_TO_INT[p]
    return t


def build_obs_from_request(req: MoveRequest, frame_stack: int = 4) -> torch.Tensor:
    """Construit l'observation (1, obs_planes, 8, 8) à partir de la requête.

    v11: produces frame-stacked obs with attack planes and move number.
    Since the server has no frame history, all history slots are filled with the
    current board. The network still benefits from the attack/move-number planes.
    """
    board = board_js_to_tensor(req.board)
    is_white = (req.turn == "w")

    P = 12  # piece planes per frame
    M = 7   # metadata planes
    E = 3   # extra planes (attacks + move number)
    total_planes = P * frame_stack + M + E
    obs = torch.zeros(total_planes, 8, 8)

    # Build piece planes for current board (used for all history frames)
    piece_planes = torch.zeros(12, 8, 8)
    for pt in range(6):
        w_code = pt + 1
        b_code = pt + 7
        w_mask = (board == w_code).float().view(8, 8)
        b_mask = (board == b_code).float().view(8, 8)
        if is_white:
            piece_planes[pt] = w_mask
            piece_planes[6 + pt] = b_mask
        else:
            piece_planes[pt] = b_mask
            piece_planes[6 + pt] = w_mask

    # Fill all frame_stack slots with the same piece planes (no history in server)
    for t in range(frame_stack):
        obs[t * P:(t + 1) * P] = piece_planes

    # ── Metadata planes ──
    off = P * frame_stack

    # En passant
    ep = req.enPassant
    if ep is not None and 0 <= ep < 64:
        r, f = ep // 8, ep % 8
        obs[off, r, f] = 1.0

    # Castling
    c = req.castling
    wK = float(c.get("wK", False))
    wQ = float(c.get("wQ", False))
    bK = float(c.get("bK", False))
    bQ = float(c.get("bQ", False))
    if is_white:
        obs[off+1, 0, :] = wK; obs[off+1, 1, :] = wQ
        obs[off+1, 2, :] = bK; obs[off+1, 3, :] = bQ
    else:
        obs[off+1, 0, :] = bK; obs[off+1, 1, :] = bQ
        obs[off+1, 2, :] = wK; obs[off+1, 3, :] = wQ

    # Turn
    obs[off+2] = 1.0 if is_white else 0.0

    # In-check
    obs[off+3] = 1.0 if _is_in_check(board, is_white) else 0.0

    # Phase
    phase_map = {"move": 0, "defense": 1, "parry_move": 2}
    obs[off+4] = phase_map.get(req.phase, 0) / 3.0

    # Check attempts
    ca = req.checkAttempts.get(req.turn, 0)
    obs[off+5] = ca / 3.0

    # Half-move clock
    obs[off+6] = min(req.halfMoves / 100.0, 1.0)

    # ── Extra planes (v11) ──
    eoff = off + M

    # Approximate attack planes (x-ray, no blocking for sliding pieces)
    # Simplified scalar version for single-board inference
    w_attacks = torch.zeros(64)
    b_attacks = torch.zeros(64)
    for sq in range(64):
        piece = int(board[sq].item())
        if piece == 0:
            continue
        is_w_piece = 1 <= piece <= 6
        pt_val = piece - 1 if is_w_piece else piece - 7
        target = w_attacks if is_w_piece else b_attacks

        if pt_val == 0:  # pawn
            f_idx, r_idx = sq % 8, sq // 8
            if is_w_piece:
                for df in (-1, 1):
                    nf = f_idx + df
                    if 0 <= nf < 8 and r_idx + 1 < 8:
                        target[nf + (r_idx + 1) * 8] = 1.0
            else:
                for df in (-1, 1):
                    nf = f_idx + df
                    if 0 <= nf < 8 and r_idx - 1 >= 0:
                        target[nf + (r_idx - 1) * 8] = 1.0
        elif pt_val == 1:  # knight
            f_idx, r_idx = sq % 8, sq // 8
            for df, dr in [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]:
                nf, nr = f_idx + df, r_idx + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    target[nf + nr * 8] = 1.0
        elif pt_val == 5:  # king
            f_idx, r_idx = sq % 8, sq // 8
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    nf, nr = f_idx + df, r_idx + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        target[nf + nr * 8] = 1.0
        else:  # sliding pieces (bishop=2, rook=3, queen=4)
            f_idx, r_idx = sq % 8, sq // 8
            dirs = []
            if pt_val in (2, 4):  # bishop or queen: diagonals
                dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]
            if pt_val in (3, 4):  # rook or queen: straights
                dirs += [(1,0),(-1,0),(0,1),(0,-1)]
            for df, dr in dirs:
                nf, nr = f_idx + df, r_idx + dr
                while 0 <= nf < 8 and 0 <= nr < 8:
                    target[nf + nr * 8] = 1.0
                    nf += df
                    nr += dr

    if is_white:
        obs[eoff] = w_attacks.view(8, 8)
        obs[eoff+1] = b_attacks.view(8, 8)
    else:
        obs[eoff] = b_attacks.view(8, 8)
        obs[eoff+1] = w_attacks.view(8, 8)

    # Move number (normalized — use halfMoves as proxy since we don't have full_move_count)
    obs[eoff+2] = min(req.halfMoves / 200.0, 1.0)

    return obs.unsqueeze(0)  # (1, total_planes, 8, 8)


def build_legal_mask_from_request(req: MoveRequest) -> torch.Tensor:
    """
    Construit le masque d'actions légales (1, 4099) à partir des coups
    légaux envoyés par le serveur Node.js.
    """
    mask = torch.zeros(1, TOTAL_ACTIONS, dtype=torch.bool)

    if req.phase == "defense":
        # En défense : 3 actions possibles
        mask[0, ACTION_ATTEMPT_BLOCK] = True
        mask[0, ACTION_ATTEMPT_PARRY] = True
        mask[0, ACTION_ACCEPT_LOSS] = True
        return mask

    # Phase move ou parry_move : utiliser les coups légaux
    legal = req.legal or {}
    for from_idx_str, moves in legal.items():
        from_idx = int(from_idx_str)
        for mv in moves:
            to_idx = mv.get("to", -1)
            if isinstance(to_idx, str):
                f = ord(to_idx[0]) - ord('a')
                r = int(to_idx[1]) - 1
                to_idx = f + r * 8
            if 0 <= from_idx < 64 and 0 <= to_idx < 64:
                action_idx = from_idx * 64 + to_idx
                mask[0, action_idx] = True

    # S'assurer qu'au moins une action est légale
    if not mask.any():
        mask[0, ACTION_ACCEPT_LOSS] = True

    return mask


def build_env_from_request(req: MoveRequest) -> Optional[ChessObscurEnv]:
    """Reconstruct a 1-env state for stronger search-based inference."""
    if server_cfg is None or device is None:
        return None

    phase_map = {"move": PHASE_MOVE, "defense": PHASE_DEFENSE, "parry_move": PHASE_PARRY}
    phase = phase_map.get(req.phase, PHASE_MOVE)

    if phase == PHASE_DEFENSE:
        if req.pendingAttackerSq is None or req.pendingTargetSq is None or req.pendingAttackerColor is None:
            return None
    if phase == PHASE_PARRY:
        if req.parrySquare is None or req.parryController is None:
            return None

    env = ChessObscurEnv(
        1,
        device=str(device),
        max_steps=server_cfg.curriculum_max_steps_cap,
        frame_stack=server_cfg.frame_stack,
    )

    board = board_js_to_tensor(req.board).to(device)
    env.board[0] = board
    env.turn_is_white[0] = (req.turn == "w")
    env.phase[0] = phase
    env.result[0] = 0
    env.castling[0] = torch.tensor([
        bool(req.castling.get("wK", False)),
        bool(req.castling.get("wQ", False)),
        bool(req.castling.get("bK", False)),
        bool(req.castling.get("bQ", False)),
    ], dtype=torch.bool, device=device)
    env.en_passant[0] = -1 if req.enPassant is None else int(req.enPassant)
    env.check_attempts[0, 0] = int(req.checkAttempts.get("w", 0))
    env.check_attempts[0, 1] = int(req.checkAttempts.get("b", 0))
    env.half_moves[0] = int(req.halfMoves)
    inferred_full_move = req.fullMoveCount
    if inferred_full_move is None:
        inferred_full_move = min(int(req.halfMoves), env.max_steps * 2)
    env.full_move_count[0] = int(inferred_full_move)

    env.pending_attacker_sq[0] = -1
    env.pending_target_sq[0] = -1
    env.pending_attacker_piece[0] = PIECE_TO_INT.get(req.pendingAttackerPiece, 0)
    env.pending_defender_piece[0] = PIECE_TO_INT.get(req.pendingDefenderPiece, 0)
    env.pending_attacker_color_white[0] = False
    env.parry_square[0] = -1
    env.parry_controller_is_white[0] = False
    env.agent_is_white[0] = env.turn_is_white[0]

    if phase == PHASE_DEFENSE:
        env.pending_attacker_sq[0] = int(req.pendingAttackerSq)
        env.pending_target_sq[0] = int(req.pendingTargetSq)
        env.pending_attacker_color_white[0] = (req.pendingAttackerColor == "w")

    if phase == PHASE_PARRY:
        env.parry_square[0] = int(req.parrySquare)
        env.parry_controller_is_white[0] = (req.parryController == "w")

    env.board_history[0] = board.unsqueeze(0).expand(env.frame_stack, -1)
    env._hist_idx = 0
    env.zobrist_history[0] = 0
    env.zobrist_len[0] = 0
    env._cached_move_legal_valid = False
    env._update_zobrist(torch.tensor([0], device=device))
    return env


def idx_to_sq(idx: int) -> str:
    f = idx % 8
    r = idx // 8
    return chr(ord('a') + f) + str(r + 1)


# ─────────────────────────────────────────────
#  FIX: Calcul intelligent du stopMs à partir des zones QTE
# ─────────────────────────────────────────────

def compute_stop_ms_for_zone(action: int, zones: Optional[QteZones], duration_ms: int = 2000) -> Optional[int]:
    """
    Place le stopMs au MILIEU de la zone QTE correcte.
    
    En entraînement, le modèle choisit BLOCK ou PARRY comme action discrète.
    En production, on doit convertir ça en un stopMs qui tombe dans la bonne zone.
    
    Si les zones ne sont pas fournies, utilise un fallback raisonnable.
    """
    if action == ACTION_ACCEPT_LOSS:
        # ACCEPT_LOSS — pas de stopMs
        return None

    if zones is not None:
        if action == ACTION_ATTEMPT_BLOCK:
            # ATTEMPT_BLOCK → milieu de la zone de blocage
            mid = (zones.blockStartMs + zones.blockEndMs) // 2
            return max(0, min(duration_ms, mid))
        elif action == ACTION_ATTEMPT_PARRY:
            # ATTEMPT_PARRY → milieu de la zone de parade
            mid = (zones.parryStartMs + zones.parryEndMs) // 2
            return max(0, min(duration_ms, mid))

    # Fallback: pas de zones fournies, placer à 80% / 95% de la durée
    # C'est moins fiable mais mieux que des valeurs fixes
    if action == ACTION_ATTEMPT_BLOCK:
        return int(duration_ms * 0.80)
    elif action == ACTION_ATTEMPT_PARRY:
        return int(duration_ms * 0.95)

    return None


# ─────────────────────────────────────────────
#  Chargement du modèle
# ─────────────────────────────────────────────

app = FastAPI(title="Chess Obscur AI")
model: ChessObscurNetwork = None
device: torch.device = None
temperature: float = 0.0
server_cfg: Config = None  # v11: store config for frame_stack inference
mcts_enabled: bool = False
mcts_num_simulations: int = 0
mcts_c_puct: float = 1.5
mcts_temperature: float = 0.0


def _infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor], cfg: Config) -> None:
    """
    Infer model architecture from checkpoint weights.
    This keeps ai_server compatible with older checkpoints.
    """
    if "input_conv.0.weight" in state_dict:
        w = state_dict["input_conv.0.weight"]
        cfg.num_filters = int(w.shape[0])
        cfg.obs_planes = int(w.shape[1])

    if "policy_conv.0.weight" in state_dict:
        cfg.policy_head_filters = int(state_dict["policy_conv.0.weight"].shape[0])

    if "policy_fc.weight" in state_dict:
        cfg.total_actions = int(state_dict["policy_fc.weight"].shape[0])

    if "value_fc.0.weight" in state_dict:
        cfg.value_head_hidden = int(state_dict["value_fc.0.weight"].shape[0])

    # v11: infer value head channels
    if "value_conv.0.weight" in state_dict:
        cfg.value_head_channels = int(state_dict["value_conv.0.weight"].shape[0])

    # v11: detect attention layer
    cfg.use_attention = "attention.attn.in_proj_weight" in state_dict

    res_block_indices = []
    for key in state_dict.keys():
        if key.startswith("res_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                res_block_indices.append(int(parts[1]))
    if res_block_indices:
        cfg.num_res_blocks = max(res_block_indices) + 1

    # v11: infer frame_stack from obs_planes
    # obs_planes = 12 * frame_stack + 7 + 3 => frame_stack = (obs_planes - 10) / 12
    inferred_fs = (cfg.obs_planes - cfg.meta_planes - cfg.extra_planes) // cfg.piece_planes
    if inferred_fs >= 1:
        cfg.frame_stack = inferred_fs


def load_model(checkpoint_path: str, dev: str = "cpu",
               value_head_hidden: Optional[int] = None,
               num_res_blocks: Optional[int] = None,
               num_filters: Optional[int] = None,
               policy_head_filters: Optional[int] = None):
    global model, device, server_cfg
    device = torch.device(dev)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    new_state_dict, _, migration = prepare_model_state_dict(ckpt)

    cfg = Config()
    _infer_arch_from_state_dict(new_state_dict, cfg)

    # Optional explicit overrides for older/newer checkpoints.
    if value_head_hidden is not None:
        cfg.value_head_hidden = value_head_hidden
    if num_res_blocks is not None:
        cfg.num_res_blocks = num_res_blocks
    if num_filters is not None:
        cfg.num_filters = num_filters
    if policy_head_filters is not None:
        cfg.policy_head_filters = policy_head_filters

    model = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
        value_head_channels=cfg.value_head_channels,
        use_attention=cfg.use_attention,
        attention_heads=cfg.attention_heads,
    ).to(device)

    strict = not migration.get("attention", False)
    model.load_state_dict(new_state_dict, strict=strict)
    model.eval()
    server_cfg = cfg
    print(f"[ai] Modèle chargé: {checkpoint_path} sur {device}")
    if migration["action_head"]:
        print("[ai] Checkpoint legacy adapte automatiquement de 4163 a 4099 actions")
    if migration["value_head"]:
        print("[ai] Checkpoint legacy adapte automatiquement le value head de 1 a 4 canaux")
    print(
        f"[ai] Arch: num_filters={cfg.num_filters}, num_res_blocks={cfg.num_res_blocks}, "
        f"policy_head_filters={cfg.policy_head_filters}, value_head_hidden={cfg.value_head_hidden}, "
        f"obs_planes={cfg.obs_planes}, attention={'ON' if cfg.use_attention else 'OFF'}"
    )


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    """Retourne le meilleur coup selon le modèle."""
    fs = server_cfg.frame_stack if server_cfg is not None else 4
    obs = build_obs_from_request(req, frame_stack=fs).to(device)
    legal_mask = build_legal_mask_from_request(req).to(device)
    action = None

    with torch.inference_mode():
        if mcts_enabled and req.phase != "defense":
            mcts_env = build_env_from_request(req)
            if mcts_env is not None:
                mcts_action = mcts_search(
                    mcts_env,
                    model,
                    env_idx=0,
                    num_simulations=mcts_num_simulations,
                    c_puct=mcts_c_puct,
                    temperature=mcts_temperature,
                    device=device,
                )
                if 0 <= mcts_action < TOTAL_ACTIONS and legal_mask[0, mcts_action]:
                    action = int(mcts_action)

        if action is None:
            policy_logits, value = model(obs, legal_mask)

            # True greedy mode for temperature <= 0
            if temperature <= 0:
                greedy_logits = policy_logits.masked_fill(~legal_mask, -1e8)
                action = torch.argmax(greedy_logits, dim=-1).item()
            else:
                # Appliquer température
                policy_logits = policy_logits / temperature
                # Masquer les actions illégales
                policy_logits = policy_logits.masked_fill(~legal_mask, -1e8)
                probs = F.softmax(policy_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

    # Décoder l'action
    if req.phase == "defense":
        return _decode_defense_action(action, req.qteZones, req.qteDurationMs or 2000)
    else:
        return _decode_board_action(action)


def _decode_defense_action(action: int, zones: Optional[QteZones] = None,
                           duration_ms: int = 2000) -> MoveResponse:
    """
    Décode une action de défense.
    
    FIX: Utilise les zones QTE transmises par le serveur pour placer
    le stopMs au bon endroit au lieu de valeurs fixes.
    """
    stop_ms = compute_stop_ms_for_zone(action, zones, duration_ms)

    if action == ACTION_ATTEMPT_BLOCK:
        # ATTEMPT_BLOCK
        return MoveResponse(
            action="defense",
            defenseAction="stop",
            stopMs=stop_ms,
        )
    elif action == ACTION_ATTEMPT_PARRY:
        # ATTEMPT_PARRY
        return MoveResponse(
            action="defense",
            defenseAction="stop",
            stopMs=stop_ms,
        )
    else:
        # ACCEPT_LOSS
        return MoveResponse(
            action="defense",
            defenseAction="accept_loss",
        )


def _decode_board_action(action: int) -> MoveResponse:
    """Décode une action de mouvement sur le plateau."""
    if action >= 4096:
        return MoveResponse(action="move", fromSq="a1", toSq="a1")

    from_sq = action // 64
    to_sq = action % 64

    to_rank = to_sq // 8
    promotion = None
    if to_rank == 7 or to_rank == 0:
        promotion = "q"

    return MoveResponse(
        action="move",
        fromSq=idx_to_sq(from_sq),
        toSq=idx_to_sq(to_sq),
        promotion=promotion,
    )


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Chemin vers le checkpoint .pt")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Température de sampling (0=greedy)")
    parser.add_argument("--mcts-simulations", type=int, default=0,
                        help="Nombre de simulations MCTS (0=desactive)")
    parser.add_argument("--mcts-c-puct", type=float, default=None,
                        help="Constante d'exploration MCTS")
    parser.add_argument("--mcts-temperature", type=float, default=0.0,
                        help="Temperature de selection finale MCTS")
    parser.add_argument("--value-head-hidden", type=int, default=None,
                        help="Override de l'architecture checkpoint si besoin")
    parser.add_argument("--num-res-blocks", type=int, default=None,
                        help="Override de l'architecture checkpoint si besoin")
    parser.add_argument("--num-filters", type=int, default=None,
                        help="Override de l'architecture checkpoint si besoin")
    parser.add_argument("--policy-head-filters", type=int, default=None,
                        help="Override de l'architecture checkpoint si besoin")
    args = parser.parse_args()

    temperature = args.temperature
    load_model(
        args.checkpoint,
        args.device,
        value_head_hidden=args.value_head_hidden,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        policy_head_filters=args.policy_head_filters,
    )
    mcts_enabled = args.mcts_simulations > 0
    mcts_num_simulations = args.mcts_simulations
    mcts_c_puct = args.mcts_c_puct if args.mcts_c_puct is not None else server_cfg.mcts_c_puct
    mcts_temperature = args.mcts_temperature
    uvicorn.run(app, host="0.0.0.0", port=args.port)
