#!/usr/bin/env python3
"""Génère une cover à partir d'un style entraîné + un titre.

Charge automatiquement le dernier checkpoint LoRA du style, construit le prompt
à partir de style.yaml (base_prompt + trigger) et du titre, puis appelle
`mflux-generate`.

Utilisation :
    python scripts/generate_cover.py phonk --title "Midnight Drive"
    python scripts/generate_cover.py phonk --title "Midnight Drive" \
        --extra "neon city, rain, cinematic" --seed 7 --steps 28
    python scripts/generate_cover.py phonk --title "X" --lora-scale 0.8 --no-title-text

⚠️  À lancer sur ton Mac Apple Silicon (mflux requis).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from lib import OUTPUT_DIR, die, latest_checkpoint, load_style, slugify, style_dir


def ensure_mflux() -> str:
    exe = shutil.which("mflux-generate")
    if not exe:
        die(
            "Commande 'mflux-generate' introuvable.\n"
            "   Installe les dépendances sur ton Mac : pip install -r requirements.txt"
        )
    return exe


def extract_adapter(checkpoint_zip: Path) -> Path:
    """Extrait le LoRA (*_adapter.safetensors) d'un checkpoint .zip mflux.

    Le fichier extrait est mis en cache à côté du zip pour éviter de réextraire.
    """
    with zipfile.ZipFile(checkpoint_zip) as zf:
        members = [n for n in zf.namelist() if n.endswith("_adapter.safetensors")]
        if not members:
            die(f"Aucun adaptateur LoRA trouvé dans {checkpoint_zip.name}.")
        member = members[0]
        target = checkpoint_zip.with_name(checkpoint_zip.stem + "_adapter.safetensors")
        if not target.exists():
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return target


def build_prompt(cfg: dict, title: str, extra: str, title_as_text: bool) -> str:
    parts = [cfg["base_prompt"], cfg["trigger"]]
    if title:
        if title_as_text:
            parts.append(f'album cover with the title text "{title}"')
        else:
            parts.append(f'album cover inspired by "{title}"')
    else:
        parts.append("album cover")
    if extra:
        parts.append(extra)
    return ", ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer une cover pour un style.")
    parser.add_argument("style", help="Nom du style entraîné.")
    parser.add_argument("--title", default="", help="Titre du morceau.")
    parser.add_argument("--extra", default="", help="Détails additionnels au prompt.")
    parser.add_argument(
        "--no-title-text",
        action="store_true",
        help="Ne pas demander d'écrire le titre dans l'image (titre = thème seulement).",
    )
    parser.add_argument("--steps", type=int, default=25, help="Pas d'inférence (défaut 25).")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (défaut 3.5).")
    parser.add_argument("--seed", type=int, default=None, help="Graine (défaut : aléatoire).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--quantize", type=int, default=8, choices=[3, 4, 6, 8])
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Force du LoRA (0-1.5).")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .zip précis à utiliser (défaut : le plus récent).",
    )
    parser.add_argument("--output", default=None, help="Chemin de sortie de l'image.")
    args = parser.parse_args()

    exe = ensure_mflux()
    cfg = load_style(args.style)

    # 1. Trouver et extraire le LoRA
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            die(f"Checkpoint introuvable : {ckpt}")
    else:
        ckpt = latest_checkpoint(args.style)
        if ckpt is None:
            die(
                f"Aucun checkpoint pour '{args.style}'.\n"
                f"   Entraîne d'abord : python scripts/train_style.py {args.style}"
            )
    adapter = extract_adapter(ckpt)

    # 2. Construire le prompt
    prompt = build_prompt(cfg, args.title, args.extra, not args.no_title_text)

    # 3. Chemin de sortie
    if args.output:
        out = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        slug = slugify(args.title) or "cover"
        out = OUTPUT_DIR / f"{args.style}_{slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # 4. Lancer mflux-generate
    cmd = [
        exe,
        "--model", cfg["base_model"],
        "--prompt", prompt,
        "--lora-paths", str(adapter),
        "--lora-scales", str(args.lora_scale),
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--width", str(args.width),
        "--height", str(args.height),
        "--output", str(out),
        "--metadata",
    ]
    if not args.no_quantize:
        cmd += ["-q", str(args.quantize)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    print(f"🎨 Style : {args.style}  | LoRA : {ckpt.name}")
    print(f"📝 Prompt : {prompt}")
    print(f"💾 Sortie : {out}")
    print(f"▶️  {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✅ Cover générée : {out}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
