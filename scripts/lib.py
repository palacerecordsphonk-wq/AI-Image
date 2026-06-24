"""Fonctions partagées par les scripts du projet AI-Image.

Tout est en stdlib + PyYAML pour rester léger et portable.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# NB : PyYAML n'est importé QUE dans les fonctions qui en ont besoin (load_style /
# create_style), pour que les helpers purs (slugify, strip_version_tags, …) restent
# importables sans dépendance — utile côté spotify_covers.py et pour les tests.

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


# --------------------------------------------------------------------------- #
# Nettoyage des "versions" de morceaux (slowed / sped up / reverb…)
# --------------------------------------------------------------------------- #
# Un "token de version" = un mot-clé de remix de tempo/effet. On en retire toute
# occurrence en fin de titre (après tiret, entre parenthèses/crochets, ou même
# collée sans séparateur) afin de retrouver le TITRE de la version normale.
_VER_TOKEN = (
    r"(?:(?:super|ultra|mega|extra|hyper)\s*)?slowed(?:\s*down)?(?:\s*(?:to\s*perfection|tf))?"
    r"|sp(?:ed|eed)\s*-?\s*up"
    r"|reverb(?:ed|ered)?"
    r"|nightcore|daycore"
    r"|chopped\s*(?:and|&|n)\s*screwed|screwed"
)
# Liaisons possibles entre deux tokens : symbole, mot, ou simple espace.
_VER_CONNECT = r"(?:\s*[+&/,xX×·]\s*|\s+(?:and|n|with|et|x)\s+|\s+)"
# Un "bloc de version" = un ou plusieurs tokens enchaînés par des liaisons.
_VER_BLOB = rf"(?:{_VER_TOKEN})(?:{_VER_CONNECT}(?:{_VER_TOKEN}))*"

_VER_PATTERNS = [
    # (Slowed + Reverb) / [Sped Up] / {Slowed} — n'importe où.
    re.compile(rf"\s*[\(\[\{{]\s*{_VER_BLOB}\s*[\)\]\}}]\s*", re.IGNORECASE),
    # - Slowed + Reverb  (tiret puis bloc, jusqu'à la fin).
    re.compile(rf"\s*[-–—|]\s*{_VER_BLOB}\s*$", re.IGNORECASE),
    # Slowed  (bloc collé en fin de titre, sans séparateur).
    re.compile(rf"\s+{_VER_BLOB}\s*$", re.IGNORECASE),
]


def strip_version_tags(title: str) -> str:
    """Retire les mentions de version (slowed, sped up, reverb…) d'un titre.

    Gère toutes les formes : '- Slowed', '(Slowed)', 'Slowed + Reverb',
    'Slowed and Reverb', '(Sped Up)', 'super slowed', etc. — avec ou sans tiret,
    avec ou sans parenthèses. Renvoie le titre de la version NORMALE. Si tout
    disparaîtrait (titre = juste un mot-clé), on garde le titre d'origine.
    """
    out = title
    for pat in _VER_PATTERNS:
        out = pat.sub(" ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" -–—|·")
    return out or title.strip()


def title_from_filename(filename: str) -> str:
    """Extrait le TITRE (version normale) depuis un nom de fichier 'Artiste - Titre'.

    On retire D'ABORD les mentions de version (slowed / sped up / reverb…) sur le
    nom entier — car elles sont en fin de nom et utilisent souvent un ' - ' qui
    fausserait le découpage. On coupe ENSUITE au PREMIER ' - ' : avant = artiste(s),
    après = titre. S'il ne reste pas de ' - ', on renvoie le nom tel quel.

    Ex : 'Mc Gw - BAILA LENTO - Slowed'  -> 'BAILA LENTO'
         'BAILA LENTO - Slowed + Reverb' -> 'BAILA LENTO'
         'Artist - Normal Title'         -> 'Normal Title'
    """
    cleaned = strip_version_tags(Path(filename).stem)
    if " - " in cleaned:
        return cleaned.split(" - ", 1)[1].strip()
    return cleaned.strip()


def safe_filename(name: str, max_len: int = 120) -> str:
    """Nom de fichier lisible et sûr : garde accents/casse/espaces, retire l'illégal."""
    name = re.sub(r'[\\/:*?"<>|]', "", name).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "cover"


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
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("name", style)
    if not data.get("trigger"):
        data["trigger"] = f"{slugify(style)}_style"
    data.setdefault("description", "")
    data.setdefault("base_prompt", "album cover art")
    data.setdefault("base_model", "dev")  # FLUX.1 [dev] par défaut
    # Ancrage sémantique : ce que SONT les images (aide le modèle à situer le style).
    data.setdefault("class_word", "album cover")
    # Convention de titre du style : "yes" (titre écrit), "no" (jamais de titre),
    # "mixed" (les deux). Pilote le captioning ET le comportement par défaut à la génération.
    title_text = str(data.get("title_text", "mixed")).lower()
    if title_text not in {"yes", "no", "mixed"}:
        title_text = "mixed"
    data["title_text"] = title_text
    return data


def create_style(
    style: str,
    *,
    trigger: str | None = None,
    description: str = "",
    base_prompt: str = "album cover art, high quality",
    base_model: str = "dev",
    class_word: str = "album cover",
    title_text: str = "mixed",
) -> Path:
    """Crée l'arborescence d'un style + son style.yaml. Retourne le dossier.

    Lève SystemExit si le style existe déjà ou si le nom est invalide.
    """
    style = slugify(style)
    if not style:
        die("Nom de style invalide.")
    sdir = style_dir(style)
    if sdir.exists():
        die(f"Le style '{style}' existe déjà ({sdir}).")

    (sdir / "raw").mkdir(parents=True)
    (sdir / "dataset").mkdir()

    data = {
        "name": style,
        "trigger": trigger or f"{style}_style",
        "description": description,
        "base_prompt": base_prompt,
        "class_word": class_word,
        "title_text": title_text,
        "base_model": base_model,
    }
    import yaml
    with open(sdir / "style.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return sdir


def import_images(source: Path, dest_raw: Path) -> int:
    """Copie les images d'un dossier source vers le dossier raw/ d'un style.

    Recherche récursive (gère les sous-dossiers). Retourne le nombre copié.
    """
    if not source.exists() or not source.is_dir():
        die(f"Dossier source introuvable : {source}")
    found = sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not found:
        exts = ", ".join(sorted(IMAGE_EXTS))
        die(f"Aucune image ({exts}) trouvée dans {source}.")
    dest_raw.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in found:
        target = dest_raw / src.name
        # Évite d'écraser un fichier homonyme venant d'un autre sous-dossier.
        n = 1
        while target.exists():
            target = dest_raw / f"{src.stem}_{n}{src.suffix}"
            n += 1
        shutil.copy2(src, target)
        copied += 1
    return copied


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
