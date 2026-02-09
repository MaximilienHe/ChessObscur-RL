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

def board_js_to_tensor(board_js: list) -> torch.Tensor:
    """Convertit le board JS (64 éléments, null ou string) en tensor int8."""
    t = torch.zeros(64, dtype=torch.int8)
    for i, p in enumerate(board_js):
        if p is not None and p in PIECE_TO_INT:
            t[i] = PIECE_TO_INT[p]
    return t


def build_obs_from_request(req: MoveRequest) -> torch.Tensor:
    """Construit l'observation (1, 19, 8, 8) à partir de la requête."""
    board = board_js_to_tensor(req.board)
    is_white = (req.turn == "w")

    obs = torch.zeros(19, 8, 8)

    for pt in range(6):
        w_code = pt + 1
        b_code = pt + 7
        w_mask = (board == w_code).float().view(8, 8)
        b_mask = (board == b_code).float().view(8, 8)
        if is_white:
            obs[pt] = w_mask
            obs[6 + pt] = b_mask
        else:
            obs[pt] = b_mask
            obs[6 + pt] = w_mask

    # En passant
    ep = req.enPassant
    if ep is not None and 0 <= ep < 64:
        r, f = ep // 8, ep % 8
        obs[12, r, f] = 1.0

    # Castling
    c = req.castling
    wK = float(c.get("wK", False))
    wQ = float(c.get("wQ", False))
    bK = float(c.get("bK", False))
    bQ = float(c.get("bQ", False))
    if is_white:
        obs[13, 0, :] = wK; obs[13, 1, :] = wQ
        obs[13, 2, :] = bK; obs[13, 3, :] = bQ
    else:
        obs[13, 0, :] = bK; obs[13, 1, :] = bQ
        obs[13, 2, :] = wK; obs[13, 3, :] = wQ

    # Turn
    obs[14] = 1.0 if is_white else 0.0

    # Phase
    phase_map = {"move": 0, "defense": 1, "parry_move": 2}
    obs[16] = phase_map.get(req.phase, 0) / 3.0

    # Check attempts
    ca = req.checkAttempts.get(req.turn, 0)
    obs[17] = ca / 3.0

    # Half-move clock
    obs[18] = min(req.halfMoves / 100.0, 1.0)

    return obs.unsqueeze(0)  # (1, 19, 8, 8)


def build_legal_mask_from_request(req: MoveRequest) -> torch.Tensor:
    """
    Construit le masque d'actions légales (1, 4163) à partir des coups
    légaux envoyés par le serveur Node.js.
    """
    mask = torch.zeros(1, 4163, dtype=torch.bool)

    if req.phase == "defense":
        # En défense : 3 actions possibles
        mask[0, 4160] = True  # ATTEMPT_BLOCK
        mask[0, 4161] = True  # ATTEMPT_PARRY
        mask[0, 4162] = True  # ACCEPT_LOSS
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
        mask[0, 4162] = True  # fallback

    return mask


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
    if action == 4162:
        # ACCEPT_LOSS — pas de stopMs
        return None

    if zones is not None:
        if action == 4160:
            # ATTEMPT_BLOCK → milieu de la zone de blocage
            mid = (zones.blockStartMs + zones.blockEndMs) // 2
            return max(0, min(duration_ms, mid))
        elif action == 4161:
            # ATTEMPT_PARRY → milieu de la zone de parade
            mid = (zones.parryStartMs + zones.parryEndMs) // 2
            return max(0, min(duration_ms, mid))

    # Fallback: pas de zones fournies, placer à 80% / 95% de la durée
    # C'est moins fiable mais mieux que des valeurs fixes
    if action == 4160:
        return int(duration_ms * 0.80)
    elif action == 4161:
        return int(duration_ms * 0.95)

    return None


# ─────────────────────────────────────────────
#  Chargement du modèle
# ─────────────────────────────────────────────

app = FastAPI(title="Chess Obscur AI")
model: ChessObscurNetwork = None
device: torch.device = None
temperature: float = 0.5


def load_model(checkpoint_path: str, dev: str = "cpu"):
    global model, device
    device = torch.device(dev)
    cfg = Config()
    model = ChessObscurNetwork(
        obs_planes=cfg.obs_planes,
        num_filters=cfg.num_filters,
        num_res_blocks=cfg.num_res_blocks,
        policy_head_filters=cfg.policy_head_filters,
        value_head_hidden=cfg.value_head_hidden,
        total_actions=cfg.total_actions,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Remove _orig_mod. prefix if present (from torch.compile)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"[ai] Modèle chargé: {checkpoint_path} sur {device}")


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    """Retourne le meilleur coup selon le modèle."""
    obs = build_obs_from_request(req).to(device)
    legal_mask = build_legal_mask_from_request(req).to(device)

    with torch.no_grad():
        policy_logits, value = model(obs, legal_mask)

        # Appliquer température
        if temperature > 0:
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

    if action == 4160:
        # ATTEMPT_BLOCK
        return MoveResponse(
            action="defense",
            defenseAction="stop",
            stopMs=stop_ms,
        )
    elif action == 4161:
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
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Température de sampling (0=greedy)")
    args = parser.parse_args()

    temperature = args.temperature
    load_model(args.checkpoint, args.device)
    uvicorn.run(app, host="0.0.0.0", port=args.port)