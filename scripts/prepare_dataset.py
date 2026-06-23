#!/usr/bin/env python3
"""Prépare les images brutes d'un style pour l'entraînement.

raw/  ->  dataset/

- convertit en RGB (gère PNG transparents, CMJN, etc.)
- redimensionne pour que le plus grand côté ne dépasse pas --size (défaut 1024)
- ré-encode proprement en PNG, renommé img_0001.png, img_0002.png, ...
- génère automatiquement les légendes (.txt) via caption.py

Utilisation :
    python scripts/prepare_dataset.py phonk
    python scripts/prepare_dataset.py phonk --size 768        # plus rapide à entraîner
    python scripts/prepare_dataset.py phonk --no-caption      # ne pas (re)générer les .txt
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageOps

from caption import generate_captions
from lib import (IMAGE_EXTS, die, find_images, safe_filename, style_dir,
                 title_from_filename)

# Quantité minimale d'images conseillée pour qu'un style s'apprenne correctement.
MIN_RECOMMENDED = 15


def resize_keep_ratio(img: Image.Image, max_side: int) -> Image.Image:
    """Redimensionne en gardant le ratio, plus grand côté = max_side maximum."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Préparer le dataset d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Taille max du plus grand côté, px (768 rapide, 1024 défaut, "
             "jusqu'à 3000 ultra). Ne fait que réduire : pas d'upscale artificiel.",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Ne pas générer automatiquement les légendes après préparation.",
    )
    parser.add_argument(
        "--auto-caption",
        action="store_true",
        help="Légender avec un VLM (mlx-vlm) au lieu du mode simple.",
    )
    args = parser.parse_args()

    sdir = style_dir(args.style)
    if not (sdir / "style.yaml").exists():
        die(
            f"Style '{args.style}' introuvable.\n"
            f"   Crée-le d'abord : python scripts/new_style.py {args.style}"
        )

    raw_dir = sdir / "raw"
    dataset_dir = sdir / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    sources = find_images(raw_dir)
    if not sources:
        exts = ", ".join(sorted(IMAGE_EXTS))
        die(f"Aucune image ({exts}) dans {raw_dir}. Dépose tes images d'abord.")

    # On repart d'un dataset propre pour éviter les doublons/orphelins.
    for old in dataset_dir.iterdir():
        if old.is_file():
            old.unlink()

    count = 0
    used_names: set[str] = set()
    for src in sources:
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)  # respecte l'orientation EXIF
                im = im.convert("RGB")
                im = resize_keep_ratio(im, args.size)
                # Nom du dataset = TITRE seul (artiste retiré). Ce titre servira
                # de "vérité" au captioning : l'IA saura quel texte chercher sur la cover.
                title = safe_filename(title_from_filename(src.name))
                stem = title
                n = 1
                while stem.lower() in used_names:
                    stem = f"{title} ({n})"
                    n += 1
                used_names.add(stem.lower())
                count += 1
                im.save(dataset_dir / f"{stem}.png", "PNG")
        except Exception as exc:  # image corrompue / format exotique
            print(f"⚠️  Ignorée ({src.name}) : {exc}")

    if count == 0:
        die("Aucune image valide n'a pu être préparée.")

    print(f"✅ {count} image(s) préparée(s) dans {dataset_dir} (max {args.size}px)")
    if count < MIN_RECOMMENDED:
        print(
            f"⚠️  Seulement {count} images : c'est peu pour un style. "
            f"Vise au moins {MIN_RECOMMENDED} (idéalement 30-150) pour un bon résultat."
        )

    if not args.no_caption:
        generate_captions(args.style, overwrite=True, auto=args.auto_caption, quiet=False)

    print()
    print(f"Étape suivante : python scripts/train_style.py {args.style}")


if __name__ == "__main__":
    main()
