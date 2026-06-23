#!/usr/bin/env python3
"""Génère le fichier de configuration d'entraînement mflux pour un style.

Le JSON produit suit exactement le schéma attendu par `mflux-train` (voir
TrainingSpec dans mflux). Les `module_path` ciblent les couches d'attention du
transformer FLUX.1 (19 blocs joints + 38 blocs simples), ce qui donne un LoRA de
style léger et efficace.

Le nombre d'epochs est calculé automatiquement à partir du nombre d'images pour
viser ~`--total-steps` itérations (1200 par défaut), un bon point de départ pour
un LoRA de style.

Utilisation :
    python scripts/build_train_config.py phonk
    python scripts/build_train_config.py phonk --total-steps 1500 --rank 16 --quantize 8
"""

from __future__ import annotations

import argparse
import json
import math

from lib import die, find_images, load_style, style_dir

# Familles de modèles FLUX gérées et leur nombre de blocs transformer.
FLUX_DEV_FAMILY = {"dev", "schnell", "krea-dev", "dev-krea"}
FLUX_NUM_JOINT_BLOCKS = 19
FLUX_NUM_SINGLE_BLOCKS = 38


def build_lora_targets(model: str, rank: int) -> list[dict]:
    """Construit les cibles LoRA (couches d'attention) selon le modèle."""
    if model not in FLUX_DEV_FAMILY:
        die(
            f"Modèle '{model}' non géré par ce générateur de config.\n"
            f"   Ce projet est optimisé pour FLUX.1 (base_model: dev). "
            f"   Pour Z-Image/Qwen, adapte les module_path manuellement."
        )

    joint = {"start": 0, "end": FLUX_NUM_JOINT_BLOCKS}
    single = {"start": 0, "end": FLUX_NUM_SINGLE_BLOCKS}

    targets: list[dict] = []
    for proj in ("to_q", "to_k", "to_v", "to_out.0"):
        targets.append(
            {"module_path": f"transformer_blocks.{{block}}.attn.{proj}", "blocks": joint, "rank": rank}
        )
    for proj in ("to_q", "to_k", "to_v"):
        targets.append(
            {"module_path": f"single_transformer_blocks.{{block}}.attn.{proj}", "blocks": single, "rank": rank}
        )
    return targets


def build_config(
    style: str,
    *,
    total_steps: int = 1200,
    rank: int = 16,
    quantize: int | None = 8,
    max_resolution: int = 1024,
    learning_rate: float = 1e-4,
    steps: int = 20,
    guidance: float = 1.0,
    seed: int = 42,
) -> dict:
    """Construit le dictionnaire de configuration d'entraînement pour un style."""
    cfg = load_style(style)
    model = cfg["base_model"]

    dataset = style_dir(style) / "dataset"
    images = find_images(dataset)
    n = len(images)
    if n == 0:
        die(
            f"Aucune image préparée dans {dataset}.\n"
            f"   Lance d'abord : python scripts/prepare_dataset.py {style}"
        )

    # Vise ~total_steps itérations : itérations = epochs * n_images (batch_size=1).
    num_epochs = max(1, math.ceil(total_steps / n))
    effective_steps = num_epochs * n
    # ~5 checkpoints sur la durée, et au moins tous les 50 pas.
    save_frequency = max(50, effective_steps // 5)

    return {
        "model": model,
        "data": "dataset/",  # relatif à l'emplacement du fichier de config
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "quantize": quantize,
        "max_resolution": max_resolution,
        "low_ram": False,
        "training_loop": {
            "num_epochs": num_epochs,
            "batch_size": 1,
            "timestep_low": 0,
            "timestep_high": steps,
        },
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate},
        "checkpoint": {"save_frequency": save_frequency, "output_path": "training"},
        "monitoring": {
            "preview_width": 1024,
            "preview_height": 1024,
            "plot_frequency": min(25, save_frequency),
            "generate_image_frequency": save_frequency,
        },
        "lora_layers": {"targets": build_lora_targets(model, rank)},
    }


def write_config(style: str, config: dict) -> "Path":  # type: ignore[name-defined]
    """Écrit le JSON dans styles/<style>/train_config.json et retourne le chemin."""
    path = style_dir(style) / "train_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer la config d'entraînement d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument("--total-steps", type=int, default=1200, help="Itérations cibles (~).")
    parser.add_argument("--rank", type=int, default=16, help="Rang du LoRA (8-32).")
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[3, 4, 6, 8],
        help="Quantification (8 = bon compromis qualité/mémoire sur Mac).",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Désactiver la quantification (qualité max, + de RAM).")
    parser.add_argument("--max-resolution", type=int, default=1024, help="Résolution max d'entraînement.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    config = build_config(
        args.style,
        total_steps=args.total_steps,
        rank=args.rank,
        quantize=None if args.no_quantize else args.quantize,
        max_resolution=args.max_resolution,
        learning_rate=args.learning_rate,
    )
    path = write_config(args.style, config)
    loop = config["training_loop"]
    print(f"✅ Config écrite : {path}")
    print(
        f"   modèle={config['model']} | epochs={loop['num_epochs']} | "
        f"quantize={config['quantize']} | rang={args.rank} | "
        f"max_res={config['max_resolution']}"
    )


if __name__ == "__main__":
    main()
