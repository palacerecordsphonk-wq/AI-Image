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

from lib import (GRAIN_PROMPT, OUTPUT_DIR, die, latest_checkpoint, load_style,
                 random_extra, slugify, style_dir)


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


def resolve_title_mode(cfg: dict, force_text: bool, force_no_text: bool) -> bool:
    """Décide si le titre doit être écrit dans l'image.

    Priorité aux flags CLI, sinon on suit la convention `title_text` du style :
    no -> jamais écrit, yes/mixed -> écrit par défaut.
    """
    if force_no_text:
        return False
    if force_text:
        return True
    return cfg.get("title_text", "mixed") != "no"


def build_prompt(
    cfg: dict, title: str, extra: str, title_as_text: bool, grain: str = "auto"
) -> str:
    # Le trigger en tête déclenche tout l'univers appris (couleurs, ambiance, typo).
    parts = [cfg["trigger"], cfg["class_word"]]
    if title:
        if title_as_text:
            parts.append(f'with the title text "{title}"')
        else:
            parts.append(f'inspired by "{title}"')
    if extra:
        parts.append(extra)
    # Dosage du grain : "auto" laisse le modèle faire comme appris ; sinon on guide.
    grain_note = GRAIN_PROMPT.get(grain, "")
    if grain_note:
        parts.append(grain_note)
    return ", ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer une cover pour un style.")
    parser.add_argument("style", help="Nom du style entraîné.")
    parser.add_argument("--title", default="", help="Titre du morceau.")
    parser.add_argument("--extra", default="", help="Détails additionnels au prompt.")
    parser.add_argument(
        "--no-title-text",
        action="store_true",
        help="Forcer : ne pas écrire le titre dans l'image (titre = thème seulement).",
    )
    parser.add_argument(
        "--title-text",
        dest="force_title_text",
        action="store_true",
        help="Forcer : écrire le titre dans l'image (outrepasse title_text=no du style).",
    )
    parser.add_argument(
        "--grain",
        choices=["auto", "none", "light", "medium", "heavy"],
        default="auto",
        help="Dosage du grain : auto (comme appris), none (clean forcé), light/medium/heavy.",
    )
    parser.add_argument(
        "--random",
        dest="random_mode",
        action="store_true",
        help="Sans inspiration : compose un prompt aléatoire depuis le vocabulaire appris du style.",
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

    # 2. Construire le prompt (le mode titre suit la convention du style sauf override)
    title_as_text = resolve_title_mode(cfg, args.force_title_text, args.no_title_text)
    extra = args.extra
    # Mode aléatoire : pas d'inspiration -> on pioche dans le vocabulaire appris du
    # style. On respecte un --extra explicite s'il est fourni (l'aléatoire le complète).
    if args.random_mode and not extra.strip():
        extra = random_extra(args.style)
        if extra:
            print(f"🎲 Prompt aléatoire : {extra}")
        else:
            print("🎲 Aléatoire : pas de vocabulaire descriptif appris, "
                  "génération sur le trigger + titre + seed aléatoire.")
    prompt = build_prompt(cfg, args.title, extra, title_as_text, args.grain)

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
