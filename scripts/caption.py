#!/usr/bin/env python3
"""Génère les fichiers de légende (.txt) attendus par mflux pour l'entraînement.

PHILOSOPHIE DU CAPTIONING DE STYLE
----------------------------------
Pour qu'un LoRA apprenne TOUT l'univers d'un style (couleurs, atmosphère, grain,
composition, typographie), il ne faut PAS sur-décrire les images :

  • Ce que tu NE décris PAS est absorbé par le mot-déclencheur -> devient LE style.
  • Ce que tu DÉCRIS devient une variable modifiable à la génération.

D'où deux modes :

1) Mode SIMPLE (défaut) — légende minimale et cohérente sur toutes les images :
       "<trigger>, <class_word>[, with stylized title text]"
   Tout l'univers visuel se fond dans le trigger. Idéal pour un style très cohérent.

2) Mode AUTO (--auto) — un VLM (Qwen2.5-VL via mlx-vlm) décrit chaque image. La
   légende garde TOUJOURS le trigger en tête, ajoute une description -> capte les
   variations internes au style. (Mac Apple Silicon uniquement.)

LE TITRE VIENT DU NOM DE FICHIER
--------------------------------
Le nom du fichier préparé EST le titre du morceau (l'artiste a été retiré par
prepare_dataset). On ne fait donc PAS deviner le texte au modèle : on lui DONNE
le titre connu, et le VLM dit seulement s'il est présent sur la cover et COMMENT
il est écrit (style, placement, partiel, caché, répété…). Le LoRA apprend ainsi
la typographie réelle du style, sans halluciner les lettres.

La présence de titre est pilotée par `title_text` dans style.yaml :
   yes  -> le titre est toujours écrit (légende = titre exact + style d'écriture)
   no   -> jamais de titre (le LoRA apprend "pas de texte")
   mixed-> les deux (utilise --auto : le VLM décide image par image)

Utilisation :
    python scripts/caption.py phonk                 # mode simple
    python scripts/caption.py phonk --overwrite
    python scripts/caption.py phonk --auto          # auto-captioning par IA
    python scripts/caption.py phonk --caption "moody phonk cover, {trigger}"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

from lib import die, find_images, load_style, style_dir

DEFAULT_VLM = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# Note de typographie ajoutée selon la convention de titre du style.
TITLE_NOTE = {
    "yes": ", with stylized title text",
    "no": "",
    "mixed": "",  # géré image par image en mode --auto
}

# On DONNE le titre connu (= nom du fichier) au VLM, au lieu de lui faire deviner
# les lettres. Il juge la PRÉSENCE et décrit le STYLE d'écriture du titre.
VLM_PROMPT_TMPL = (
    'This album cover is for a track titled "{title}". '
    "Reply with ONLY a compact JSON object, no extra text, with keys: "
    '"caption" (short phrase: colors, mood, subject, composition; max 20 words; '
    "do NOT transcribe any text), "
    '"has_title" (true/false: is that title text visibly written anywhere on the '
    "cover, even partially, stylized, hidden or repeated?), "
    '"title_style" (if has_title: a few words on HOW it is rendered — font style, '
    "placement, size, color, partial, repeated, hidden, glitched; else empty)."
)


def _show_title(cfg: dict, vlm_has_title: bool) -> bool:
    """Décide si le titre doit figurer dans la légende, selon la convention du style."""
    tt = cfg["title_text"]
    if tt == "yes":
        return True
    if tt == "no":
        return False
    return vlm_has_title  # mixed -> ce que voit le VLM


def _build_simple_caption(cfg: dict, title: str, caption: str | None) -> str:
    """Légende minimale (mode simple), avec le vrai titre si le style en a."""
    if caption:
        return caption.format(trigger=cfg["trigger"], base_prompt=cfg["base_prompt"],
                              class_word=cfg["class_word"], title=title)
    parts = [cfg["trigger"], cfg["class_word"]]
    # En mode simple on ne "voit" pas l'image : on ajoute le titre seulement si le
    # style l'écrit toujours (yes). En "mixed", préférer --auto.
    if cfg["title_text"] == "yes" and title:
        parts.append(f'with the title text "{title}"')
    return ", ".join(parts)


def _extract_json(text: str) -> dict | None:
    """Récupère le dernier objet JSON présent dans la sortie du VLM."""
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for raw in reversed(matches):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _vlm_caption(image_path, cfg: dict, title: str, model: str) -> str:
    """Décrit une image avec un VLM, en lui FOURNISSANT le titre connu.

    Le VLM ne devine pas le texte : on lui donne le titre (nom du fichier) et il
    indique s'il est présent et COMMENT il est écrit. Le trigger reste en tête.
    """
    cmd = [
        sys.executable, "-m", "mlx_vlm.generate",
        "--model", model,
        "--max-tokens", "180",
        "--temperature", "0.2",
        "--prompt", VLM_PROMPT_TMPL.format(title=title),
        "--image", str(image_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "échec mlx_vlm")

    data = _extract_json(proc.stdout) or {}
    desc = str(data.get("caption", "")).strip().strip(".")
    parts = [cfg["trigger"], cfg["class_word"]]
    if desc:
        parts.append(desc)

    if _show_title(cfg, bool(data.get("has_title"))) and title:
        note = f'with the title text "{title}"'
        style = str(data.get("title_style", "")).strip().strip(".")
        if style:
            note += f", {style}"
        parts.append(note)
    return ", ".join(p for p in parts if p)


def generate_captions(
    style: str,
    *,
    caption: str | None = None,
    overwrite: bool = False,
    auto: bool = False,
    vlm_model: str = DEFAULT_VLM,
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

    if auto and not shutil.which("mlx_vlm") and not _module_available("mlx_vlm"):
        die(
            "Mode --auto demandé mais mlx-vlm est introuvable.\n"
            "   Installe-le sur ton Mac : pip install mlx-vlm\n"
            "   (ou retire --auto pour le mode simple)."
        )

    written = 0
    for i, img in enumerate(images, 1):
        txt = img.with_suffix(".txt")
        if txt.exists() and not overwrite:
            continue
        # Le nom de fichier (sans extension) EST le titre du morceau.
        title = img.stem
        if auto:
            try:
                text = _vlm_caption(img, cfg, title, vlm_model)
            except Exception as exc:  # repli sur la légende simple en cas d'échec
                if not quiet:
                    print(f"⚠️  VLM KO sur {img.name} ({exc}); légende simple utilisée.")
                text = _build_simple_caption(cfg, title, caption)
            if not quiet:
                print(f"   [{i}/{len(images)}] {img.name} -> {text}")
        else:
            text = _build_simple_caption(cfg, title, caption)
        txt.write_text(text + "\n", encoding="utf-8")
        written += 1

    # mflux attend au moins un preview*.txt dans le dossier de données.
    preview = dataset / "preview.txt"
    if overwrite or not preview.exists():
        note = TITLE_NOTE.get(cfg["title_text"], "")
        preview.write_text(
            f"{cfg['trigger']}, {cfg['class_word']}{note}, masterpiece, highly detailed\n",
            encoding="utf-8",
        )

    if not quiet:
        skipped = len(images) - written
        mode = "AUTO (VLM)" if auto else "simple"
        print(f"✅ {written} légende(s) écrite(s) dans {dataset} [mode {mode}]")
        if not auto:
            example = _build_simple_caption(cfg, "<titre>", caption)
            print(f"   Modèle : « {example} »")
        if skipped:
            print(f"   ({skipped} déjà présentes ; --overwrite pour les remplacer)")
    return written


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer les légendes d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument(
        "--caption",
        default=None,
        help="Modèle de légende manuel. Variables : {trigger}, {class_word}, {base_prompt}.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-captioning par IA (VLM mlx-vlm). Capte les variations + détecte le titre.",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM,
        help=f"Modèle VLM mlx-vlm pour --auto (défaut : {DEFAULT_VLM}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Écraser les .txt existants."
    )
    args = parser.parse_args()
    generate_captions(
        args.style,
        caption=args.caption,
        overwrite=args.overwrite,
        auto=args.auto,
        vlm_model=args.vlm_model,
    )


if __name__ == "__main__":
    main()
