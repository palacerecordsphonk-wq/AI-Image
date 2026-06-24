#!/usr/bin/env python3
"""Entraîne un LoRA de style avec mflux.

Génère (ou régénère) la config d'entraînement puis lance `mflux-train`.
Un LoRA = un style. Les checkpoints sont écrits dans styles/<style>/training/.

Utilisation :
    python scripts/train_style.py phonk
    python scripts/train_style.py phonk --total-steps 1500 --max-resolution 768
    python scripts/train_style.py phonk --resume styles/phonk/training/0000500_checkpoint.zip

⚠️  À lancer sur ton Mac Apple Silicon (mflux requis). Compte plusieurs heures
    selon la résolution et le nombre d'images. 768px = nettement plus rapide.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from build_train_config import build_config, write_config
from lib import die, style_dir


def ensure_mflux() -> str:
    exe = shutil.which("mflux-train")
    if not exe:
        die(
            "Commande 'mflux-train' introuvable.\n"
            "   Installe les dépendances sur ton Mac : pip install -r requirements.txt\n"
            "   (mflux ne s'installe que sur macOS Apple Silicon.)"
        )
    return exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîner le LoRA d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument(
        "--model",
        dest="model_key",
        default=None,
        help="Modèle à entraîner (flux1-dev, flux2-klein-9b, flux2-klein-4b). "
             "Défaut : le base_model du style. Les checkpoints sont rangés par modèle.",
    )
    parser.add_argument("--total-steps", type=int, default=1200, help="Itérations cibles (~).")
    parser.add_argument("--rank", type=int, default=16, help="Rang du LoRA (8-32).")
    parser.add_argument("--quantize", type=int, default=8, choices=[3, 4, 6, 8])
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--max-resolution", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--resume",
        default=None,
        help="Reprendre depuis un checkpoint .zip (ignore les autres options de config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Générer et valider la config sans lancer l'entraînement.",
    )
    args = parser.parse_args()

    exe = ensure_mflux()

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            die(f"Checkpoint introuvable : {resume_path}")
        cmd = [exe, "--resume", str(resume_path)]
        print(f"▶️  Reprise de l'entraînement depuis {resume_path}")
    else:
        config = build_config(
            args.style,
            model_key=args.model_key,
            total_steps=args.total_steps,
            rank=args.rank,
            quantize=None if args.no_quantize else args.quantize,
            max_resolution=args.max_resolution,
            learning_rate=args.learning_rate,
        )
        config_path = write_config(args.style, config, args.model_key)
        loop = config["training_loop"]
        print(f"✅ Config : {config_path}")
        print(
            f"   modèle={config['model']} | epochs={loop['num_epochs']} | "
            f"quantize={config['quantize']} | rang={args.rank} | max_res={config['max_resolution']}"
        )
        cmd = [exe, "--config", str(config_path)]
        if args.dry_run:
            cmd.append("--dry-run")

    # On exécute depuis le dossier du style pour que les chemins relatifs
    # ("dataset/", "training") du config se résolvent correctement.
    cwd = style_dir(args.style)
    print(f"▶️  {' '.join(cmd)}  (cwd={cwd})")
    try:
        result = subprocess.run(cmd, cwd=cwd)
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu (les checkpoints déjà sauvegardés sont conservés).")
        sys.exit(130)

    if result.returncode == 0 and not args.dry_run:
        print()
        print(f"🎉 Entraînement terminé. Checkpoints dans : {cwd / 'training'}")
        print(f"   Génère une cover : python scripts/generate_cover.py {args.style} --title \"Mon titre\"")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
