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

2) Mode AUTO — un VLM décrit chaque image. La légende garde TOUJOURS le trigger
   en tête, ajoute une description -> capte les variations internes au style.
   Deux backends, au choix, exactement le MÊME prompt et la MÊME logique :
       • --auto    : modèle LOCAL (Qwen2.5-VL via mlx-vlm). Hors-ligne, gratuit,
                     Mac Apple Silicon uniquement, ~5 Go au 1er usage.
       • --gemini  : API Gemini (gemini-2.5-flash par défaut). Meilleur OCR du
                     texte stylisé/caché et descriptions plus fines ; nécessite
                     une clé (GEMINI_API_KEY ou --gemini-api-key) + internet.

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
    python scripts/caption.py phonk --auto          # VLM local (mlx-vlm)
    python scripts/caption.py phonk --gemini        # VLM via API Gemini
    python scripts/caption.py phonk --caption "moody phonk cover, {trigger}"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from lib import GRAIN_CAPTION, detect_grain, die, find_images, load_style, style_dir

DEFAULT_VLM = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DEFAULT_GEMINI = "gemini-2.5-flash"

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Note de typographie ajoutée selon la convention de titre du style.
TITLE_NOTE = {
    "yes": ", with stylized title text",
    "no": "",
    "mixed": "",  # géré image par image en mode --auto
}

# On DONNE le titre connu (= nom du fichier) au VLM, au lieu de lui faire deviner
# les lettres. Il juge la PRÉSENCE et décrit le STYLE d'écriture du titre.
VLM_PROMPT_TMPL = (
    'You are an expert art director labeling an album cover to train an image '
    'style model. The track is titled "{title}". '
    "Reply with ONLY one compact JSON object, no extra text, with these keys:\n"
    '"caption": ONE rich descriptive phrase, ~25-45 words, comma-separated fragments '
    "(no full sentences, no repeated words, NO filler like beautiful/stunning/amazing/"
    "masterpiece). Describe ONLY what is actually visible, in this order when present: "
    "(1) main subject and its pose/action/expression, (2) secondary elements, "
    "(3) setting/background, (4) color palette and dominant tones, (5) lighting "
    "(direction, glow, contrast), (6) composition and framing (close-up, centered, "
    "rule-of-thirds, depth), (7) distinctive visual effects and rendering "
    "(neon glow, bloom, halftone dots, chrome, splatter, painterly, 3D, comic, "
    "double-exposure, vignette). Be specific and concrete so the model can recreate "
    "AND recombine this look. Do NOT transcribe any title or letters. Do NOT mention "
    "film grain or noise (handled separately).\n"
    '"has_title": true ONLY if the title (or a clear part of it) is visibly written on '
    "the cover as graphic text — even if stylized, distorted, partial, integrated into "
    "the artwork or repeated. false only if there is no readable title text at all.\n"
    '"title_style": if has_title, 4-12 words on HOW the title looks (font, weight, color, '
    "placement, effects: bubble/inflated/glossy/chrome/graffiti/neon-glow/outline/"
    "stacked/repeated/3D); else \"\"."
)


def _show_title(cfg: dict, vlm_has_title: bool) -> bool:
    """Décide si le titre doit figurer dans la légende, selon la convention du style."""
    tt = cfg["title_text"]
    if tt == "yes":
        return True
    if tt == "no":
        return False
    return vlm_has_title  # mixed -> ce que voit le VLM


def _grain_note(grain: str | None) -> str:
    """Mention de grain à insérer dans la légende (vide si non détecté/désactivé)."""
    return GRAIN_CAPTION.get(grain or "", "")


def _build_simple_caption(
    cfg: dict, title: str, caption: str | None, grain: str | None = None
) -> str:
    """Légende minimale (mode simple), avec le vrai titre si le style en a."""
    if caption:
        return caption.format(trigger=cfg["trigger"], base_prompt=cfg["base_prompt"],
                              class_word=cfg["class_word"], title=title)
    parts = [cfg["trigger"], cfg["class_word"]]
    # En mode simple on ne "voit" pas l'image : on ajoute le titre seulement si le
    # style l'écrit toujours (yes). En "mixed", préférer --auto.
    if cfg["title_text"] == "yes" and title:
        parts.append(f'with the title text "{title}"')
    note = _grain_note(grain)
    if note:
        parts.append(note)
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


def _compose_caption(cfg: dict, title: str, data: dict, grain: str | None = None) -> str:
    """Assemble la légende finale à partir du JSON renvoyé par un VLM.

    Identique quel que soit le backend (local ou Gemini) : trigger en tête, puis
    la description, puis le titre réel s'il est présent + son style d'écriture, et
    enfin la mention de grain détectée (le grain est mesuré, pas deviné par le VLM).
    """
    parts = [cfg["trigger"], cfg["class_word"]]
    desc = str(data.get("caption", "")).strip().strip(".")
    if desc:
        parts.append(desc)
    if _show_title(cfg, bool(data.get("has_title"))) and title:
        note = f'with the title text "{title}"'
        style = str(data.get("title_style", "")).strip().strip(".")
        if style:
            note += f", {style}"
        parts.append(note)
    gnote = _grain_note(grain)
    if gnote:
        parts.append(gnote)
    return ", ".join(p for p in parts if p)


def _vlm_caption(image_path, cfg: dict, title: str, model: str, grain: str | None = None) -> str:
    """Décrit une image avec un VLM LOCAL (mlx-vlm), en lui FOURNISSANT le titre.

    Le VLM ne devine pas le texte : on lui donne le titre (nom du fichier) et il
    indique s'il est présent et COMMENT il est écrit. Le trigger reste en tête.
    """
    cmd = [
        sys.executable, "-m", "mlx_vlm.generate",
        "--model", model,
        "--max-tokens", "320",
        "--temperature", "0.2",
        "--prompt", VLM_PROMPT_TMPL.format(title=title),
        "--image", str(image_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "échec mlx_vlm")

    data = _extract_json(proc.stdout) or {}
    return _compose_caption(cfg, title, data, grain)


def _gemini_caption(
    image_path, cfg: dict, title: str, model: str, api_key: str, grain: str | None = None
) -> str:
    """Décrit une image via l'API Gemini, avec EXACTEMENT le même prompt/logique.

    Meilleur OCR du texte stylisé/caché et descriptions plus fines qu'un petit
    modèle local 4-bit. On envoie l'image en base64 + le prompt, et on force une
    réponse JSON (mêmes clés : caption / has_title / title_style).
    """
    mime = _MIME_BY_EXT.get(image_path.suffix.lower(), "image/png")
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": VLM_PROMPT_TMPL.format(title=title)},
                    {"inline_data": {"mime_type": mime, "data": img_b64}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 768,
            "responseMimeType": "application/json",
            # Pas de "réflexion" : plus rapide et moins cher pour ce labeling.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = GEMINI_ENDPOINT.format(model=model) + "?key=" + urllib.parse.quote(api_key)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Gemini HTTP {e.code} : {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini injoignable : {e.reason}")

    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Réponse Gemini inattendue : {str(resp)[:200]}")

    data = _extract_json(text) or {}
    return _compose_caption(cfg, title, data, grain)


def generate_captions(
    style: str,
    *,
    caption: str | None = None,
    overwrite: bool = False,
    backend: str = "simple",
    vlm_model: str = DEFAULT_VLM,
    gemini_model: str = DEFAULT_GEMINI,
    gemini_api_key: str | None = None,
    detect_grain_flag: bool = True,
    quiet: bool = False,
) -> int:
    """Crée un .txt par image du dataset. Retourne le nombre de fichiers écrits.

    backend : "simple" (légende minimale, sans IA), "local" (VLM mlx-vlm) ou
    "gemini" (API Gemini). Les deux backends VLM partagent prompt et logique.

    detect_grain_flag : si True (défaut), mesure le grain de chaque image et l'inscrit
    dans la légende -> le grain devient un attribut appris, dosable à la génération.
    """
    if backend not in {"simple", "local", "gemini"}:
        die(f"Backend de captioning inconnu : {backend!r} (simple|local|gemini).")

    cfg = load_style(style)
    dataset = style_dir(style) / "dataset"
    images = find_images(dataset)
    if not images:
        die(
            f"Aucune image dans {dataset}.\n"
            f"   Lance d'abord : python scripts/prepare_dataset.py {style}"
        )

    if backend == "local" and not shutil.which("mlx_vlm") and not _module_available("mlx_vlm"):
        die(
            "Captioning local demandé mais mlx-vlm est introuvable.\n"
            "   Installe-le sur ton Mac : pip install mlx-vlm\n"
            "   (ou utilise --gemini, ou retire l'option pour le mode simple)."
        )

    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if backend == "gemini" and not api_key.strip():
        die(
            "Captioning Gemini demandé mais aucune clé API.\n"
            "   Fournis --gemini-api-key ou définis la variable GEMINI_API_KEY.\n"
            "   (Dans l'interface web, colle ta clé dans la zone dédiée.)"
        )
    api_key = api_key.strip()

    written = 0
    for i, img in enumerate(images, 1):
        txt = img.with_suffix(".txt")
        if txt.exists() and not overwrite:
            continue
        # Le nom de fichier (sans extension) EST le titre du morceau.
        title = img.stem
        # Grain MESURÉ sur l'image (déterministe, indépendant du backend de légende).
        grain = detect_grain(img) if detect_grain_flag else None
        if backend == "simple":
            text = _build_simple_caption(cfg, title, caption, grain)
        else:
            try:
                if backend == "gemini":
                    text = _gemini_caption(img, cfg, title, gemini_model, api_key, grain)
                else:
                    text = _vlm_caption(img, cfg, title, vlm_model, grain)
            except Exception as exc:  # repli sur la légende simple en cas d'échec
                if not quiet:
                    print(f"⚠️  VLM KO sur {img.name} ({exc}); légende simple utilisée.")
                text = _build_simple_caption(cfg, title, caption, grain)
            if not quiet:
                print(f"   [{i}/{len(images)}] {img.name} -> {text}")
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
        mode = {"local": "AUTO local (mlx-vlm)", "gemini": "AUTO Gemini"}.get(
            backend, "simple"
        )
        print(f"✅ {written} légende(s) écrite(s) dans {dataset} [mode {mode}]")
        if backend == "simple":
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-captioning par VLM LOCAL (mlx-vlm). Variations + détection du titre.",
    )
    group.add_argument(
        "--gemini",
        action="store_true",
        help="Auto-captioning via l'API Gemini (meilleur OCR du texte). Nécessite une clé.",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM,
        help=f"Modèle VLM mlx-vlm pour --auto (défaut : {DEFAULT_VLM}).",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI,
        help=f"Modèle Gemini pour --gemini (défaut : {DEFAULT_GEMINI}).",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Clé API Gemini (sinon variable d'environnement GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Écraser les .txt existants."
    )
    parser.add_argument(
        "--no-grain",
        dest="detect_grain",
        action="store_false",
        help="Ne pas détecter/annoter le grain dans les légendes.",
    )
    args = parser.parse_args()
    backend = "gemini" if args.gemini else "local" if args.auto else "simple"
    generate_captions(
        args.style,
        caption=args.caption,
        overwrite=args.overwrite,
        backend=backend,
        vlm_model=args.vlm_model,
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        detect_grain_flag=args.detect_grain,
    )


if __name__ == "__main__":
    main()
