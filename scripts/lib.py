"""Fonctions partagées par les scripts du projet AI-Image.

Tout est en stdlib + PyYAML pour rester léger et portable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# Racine du projet (le dossier qui contient `styles/`, `scripts/`, etc.)
ROOT = Path(__file__).resolve().parent.parent
STYLES_DIR = ROOT / "styles"
OUTPUT_DIR = ROOT / "output"

# Extensions d'images reconnues (identiques à celles que mflux sait lire)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def die(message: str) -> "NoReturn":  # type: ignore[name-defined]
    """Affiche une erreur lisible et quitte avec un code non nul."""
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def slugify(text: str) -> str:
    """Transforme un texte libre en identifiant sûr (dossiers, trigger words)."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def style_dir(style: str) -> Path:
    """Chemin du dossier d'un style donné."""
    return STYLES_DIR / style


def list_styles() -> list[str]:
    """Liste les styles existants (dossiers contenant un style.yaml)."""
    if not STYLES_DIR.exists():
        return []
    styles = []
    for path in sorted(STYLES_DIR.iterdir()):
        if path.is_dir() and (path / "style.yaml").exists() and path.name != "_TEMPLATE":
            styles.append(path.name)
    return styles


def load_style(style: str) -> dict:
    """Charge et valide le style.yaml d'un style."""
    cfg_path = style_dir(style) / "style.yaml"
    if not cfg_path.exists():
        die(
            f"Style '{style}' introuvable ({cfg_path} manquant).\n"
            f"   Styles disponibles : {', '.join(list_styles()) or '(aucun)'}\n"
            f"   Crée-le avec : python scripts/new_style.py {style}"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("name", style)
    if not data.get("trigger"):
        data["trigger"] = f"{slugify(style)}_style"
    data.setdefault("description", "")
    data.setdefault("base_prompt", "album cover art")
    data.setdefault("base_model", "dev")  # FLUX.1 [dev] par défaut
    return data


def find_images(folder: Path) -> list[Path]:
    """Liste les images d'un dossier (non récursif), triées."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def latest_checkpoint(style: str) -> Path | None:
    """Retourne le checkpoint .zip le plus récent (le plus entraîné) d'un style.

    Les checkpoints mflux sont nommés `NNNNNNN_checkpoint.zip` où NNNNNNN est le
    nombre d'itérations. On prend donc le plus grand numéro disponible.
    """
    training_root = style_dir(style) / "training"
    if not training_root.exists():
        return None
    checkpoints = sorted(training_root.rglob("*_checkpoint.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: _checkpoint_iterations(p))


def _checkpoint_iterations(path: Path) -> int:
    m = re.match(r"(\d+)_checkpoint\.zip", path.name)
    return int(m.group(1)) if m else 0
