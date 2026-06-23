#!/usr/bin/env python3
"""Assistant tout-en-un pour fine-tuner un style.

Il te demande (ou prend en options) :
  • le NOM du style (ex: rock, funk, ambiance)
  • le CHEMIN du dossier contenant tes images de ce style

…puis il enchaîne automatiquement : import des images -> préparation +
captioning IA (détection auto du style, des couleurs, de l'atmosphère ET du
titre) -> entraînement du LoRA.

Usage interactif (le plus simple) :
    python scripts/finetune.py

Usage direct :
    python scripts/finetune.py --style rock --images "/Users/moi/Desktop/rock" \
        --title-text mixed --max-resolution 768

Astuce macOS : pour saisir le chemin, tu peux GLISSER-DÉPOSER le dossier
directement dans le Terminal.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from lib import create_style, die, import_images, list_styles, slugify, style_dir

SCRIPTS = Path(__file__).resolve().parent


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"➤ {prompt}{suffix}\n  ").strip()
    except EOFError:
        val = ""
    return val or (default or "")


def clean_path(raw: str) -> Path:
    """Nettoie un chemin saisi/collé/glissé (guillemets, espaces échappés macOS)."""
    raw = raw.strip().strip('"').strip("'").strip()
    raw = raw.replace("\\ ", " ")  # glisser-déposer Terminal échappe les espaces
    return Path(raw).expanduser()


def has_mlx_vlm() -> bool:
    return importlib.util.find_spec("mlx_vlm") is not None


def run(cmd: list[str]) -> int:
    print(f"\n▶️  {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant de fine-tune d'un style.")
    parser.add_argument("--style", default=None, help="Nom du style.")
    parser.add_argument("--images", default=None, help="Dossier des images du style.")
    parser.add_argument(
        "--title-text",
        choices=["yes", "no", "mixed"],
        default=None,
        help="Convention de titre (défaut demandé : mixed = détection auto).",
    )
    parser.add_argument("--class-word", default=None, help="Ce que sont les covers.")
    parser.add_argument(
        "--simple-caption",
        action="store_true",
        help="Légendes simples au lieu de l'auto-captioning IA.",
    )
    parser.add_argument("--max-resolution", type=int, default=1024, help="Résolution d'entraînement (768 = + rapide).")
    parser.add_argument("--total-steps", type=int, default=1200)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--no-train", action="store_true", help="Tout préparer sans lancer l'entraînement.")
    args = parser.parse_args()

    print("=" * 60)
    print("🎛️  ASSISTANT DE FINE-TUNE — un style à la fois")
    print("=" * 60)

    # 1. Nom du style
    style = slugify(args.style or ask("Quel STYLE veux-tu entraîner ? (ex: rock, funk, ambiance)"))
    if not style:
        die("Nom de style requis.")

    # 2. Dossier d'images
    images_path = clean_path(
        args.images or ask("CHEMIN du dossier contenant les images de ce style (glisse-le ici)")
    )

    # 3. Créer le style (ou l'enrichir s'il existe)
    existing = style in list_styles()
    if existing:
        print(f"\nℹ️  Le style '{style}' existe déjà : on lui ajoute ces nouvelles images.")
    else:
        class_word = args.class_word or ask(
            "Décris en quelques mots ce que sont ces covers", f"{style} album cover"
        )
        title_text = args.title_text or ask(
            "Y a-t-il un titre écrit sur les covers ? yes / no / mixed", "mixed"
        )
        if title_text not in {"yes", "no", "mixed"}:
            title_text = "mixed"
        create_style(style, class_word=class_word, title_text=title_text)
        print(f"\n✅ Style '{style}' créé (class_word='{class_word}', title_text='{title_text}').")

    # 4. Importer les images
    n = import_images(images_path, style_dir(style) / "raw")
    print(f"✅ {n} image(s) importée(s) dans styles/{style}/raw/")

    # 5. Choix du captioning
    auto = not args.simple_caption
    if auto and not has_mlx_vlm():
        print("\n⚠️  L'auto-captioning IA nécessite 'mlx-vlm' (non installé).")
        choice = ask("Que faire ? [i]nstaller mlx-vlm / [s]imple captioning", "i").lower()
        if choice.startswith("i"):
            if run([sys.executable, "-m", "pip", "install", "mlx-vlm"]) != 0:
                die("Échec d'installation de mlx-vlm. Relance avec --simple-caption.")
        else:
            auto = False

    # 6. Préparer + légender
    prep = [sys.executable, str(SCRIPTS / "prepare_dataset.py"), style]
    if auto:
        prep.append("--auto-caption")
    if run(prep) != 0:
        die("La préparation des images a échoué.")

    # 7. Entraîner
    if args.no_train:
        print(f"\n✅ Prêt. Lance l'entraînement quand tu veux :")
        print(f"   python scripts/train_style.py {style} --max-resolution {args.max_resolution}")
        return

    train = [
        sys.executable, str(SCRIPTS / "train_style.py"), style,
        "--max-resolution", str(args.max_resolution),
        "--total-steps", str(args.total_steps),
        "--rank", str(args.rank),
    ]
    code = run(train)
    if code == 0:
        print("\n" + "=" * 60)
        print(f"🎉 Style '{style}' entraîné ! Génère une cover :")
        print(f'   python scripts/generate_cover.py {style} --title "Mon titre"')
        print("=" * 60)
    sys.exit(code)


if __name__ == "__main__":
    main()
