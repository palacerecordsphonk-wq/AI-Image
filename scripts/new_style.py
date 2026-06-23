#!/usr/bin/env python3
"""Crée un nouveau style : dossier + style.yaml pré-rempli.

Exemple :
    python scripts/new_style.py phonk \
        --description "covers phonk sombres, grain, esthétique cassette" \
        --base-prompt "dark phonk album cover, vintage grain, high contrast"

Ensuite : dépose tes images dans styles/phonk/raw/ et lance prepare_dataset.py.
"""

from __future__ import annotations

import argparse

from lib import IMAGE_EXTS, STYLES_DIR, create_style, slugify


def main() -> None:
    parser = argparse.ArgumentParser(description="Créer un nouveau style de cover.")
    parser.add_argument("style", help="Nom du style (ex: phonk, drill, lofi).")
    parser.add_argument(
        "--description", default="", help="Description libre du style (pour mémoire)."
    )
    parser.add_argument(
        "--base-prompt",
        default="album cover art, high quality",
        help="Phrase de base injectée dans chaque prompt de ce style.",
    )
    parser.add_argument(
        "--trigger",
        default=None,
        help="Mot-déclencheur unique. Par défaut : <style>_style.",
    )
    parser.add_argument(
        "--base-model",
        default="dev",
        help="Modèle de base mflux (dev = FLUX.1 [dev], défaut).",
    )
    parser.add_argument(
        "--class-word",
        default="album cover",
        help="Ce que SONT les images (ancrage sémantique). Ex: 'phonk album cover'.",
    )
    parser.add_argument(
        "--title-text",
        default="mixed",
        choices=["yes", "no", "mixed"],
        help="Convention de titre du style : yes=titre écrit, no=jamais, mixed=les deux.",
    )
    args = parser.parse_args()

    style = slugify(args.style)
    sdir = create_style(
        style,
        trigger=args.trigger,
        description=args.description,
        base_prompt=args.base_prompt,
        base_model=args.base_model,
        class_word=args.class_word,
        title_text=args.title_text,
    )
    trigger = args.trigger or f"{style}_style"

    exts = ", ".join(sorted(IMAGE_EXTS))
    print(f"✅ Style '{style}' créé dans {sdir.relative_to(STYLES_DIR.parent)}")
    print(f"   • trigger word : {trigger}")
    print(f"   • base_prompt  : {args.base_prompt}")
    print()
    print("Étapes suivantes :")
    print(f"   1. Dépose tes images ({exts}) dans : styles/{style}/raw/")
    print(f"   2. python scripts/prepare_dataset.py {style}")
    print(f"   3. python scripts/train_style.py {style}")
    print(f'   4. python scripts/generate_cover.py {style} --title "Mon titre"')


if __name__ == "__main__":
    main()
