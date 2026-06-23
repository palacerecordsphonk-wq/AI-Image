#!/usr/bin/env python3
"""Génère les fichiers de légende (.txt) attendus par mflux pour l'entraînement.

mflux fait de l'auto-découverte : chaque image `img_0001.png` doit avoir un
fichier `img_0001.txt` contenant son prompt. Pour entraîner un *style* (et non
un objet précis), la bonne pratique est d'utiliser des légendes courtes et
cohérentes qui contiennent toujours le mot-déclencheur du style.

Légende par défaut générée :  "<base_prompt>, <trigger>"

Tu peux ensuite éditer librement chaque .txt à la main pour affiner.

Utilisation :
    python scripts/caption.py phonk                # légendes par défaut
    python scripts/caption.py phonk --overwrite    # écrase les .txt existants
    python scripts/caption.py phonk --caption "moody phonk cover, {trigger}"
"""

from __future__ import annotations

import argparse

from lib import die, find_images, load_style, style_dir


def generate_captions(
    style: str,
    *,
    caption: str | None = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> int:
    """Crée un .txt par image du dataset. Retourne le nombre de fichiers écrits."""
    cfg = load_style(style)
    dataset = style_dir(style) / "dataset"
    images = find_images(dataset)
    if not images:
        die(
            f"Aucune image dans {dataset}.\n"
            f"   Lance d'abord : python scripts/prepare_dataset.py {style}"
        )

    template = caption or f"{cfg['base_prompt']}, {cfg['trigger']}"
    text = template.format(trigger=cfg["trigger"], base_prompt=cfg["base_prompt"])

    written = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists() and not overwrite:
            continue
        txt.write_text(text + "\n", encoding="utf-8")
        written += 1

    # mflux attend au moins un preview*.txt dans le dossier de données pour les
    # aperçus pendant l'entraînement.
    preview = dataset / "preview.txt"
    if overwrite or not preview.exists():
        preview.write_text(
            f"{cfg['base_prompt']}, {cfg['trigger']}, album cover, masterpiece\n",
            encoding="utf-8",
        )

    if not quiet:
        skipped = len(images) - written
        print(f"✅ {written} légende(s) écrite(s) dans {dataset} (texte : « {text} »)")
        if skipped:
            print(f"   ({skipped} déjà présentes, garde --overwrite pour les remplacer)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer les légendes d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument(
        "--caption",
        default=None,
        help="Modèle de légende. Variables : {trigger}, {base_prompt}.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Écraser les .txt existants."
    )
    args = parser.parse_args()
    generate_captions(args.style, caption=args.caption, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
