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

from lib import (die, find_images, load_style, model_slug, model_spec,
                 resolve_model_key, style_dir)

# FLUX.1 : 19 blocs joints + 38 blocs simples, LoRA sur les couches d'attention.
FLUX1_NUM_JOINT_BLOCKS = 19
FLUX1_NUM_SINGLE_BLOCKS = 38

# FLUX.2 Klein : cibles reproduites VERBATIM de la config d'exemple officielle de
# mflux (src/mflux/models/flux2/README.md). Architecture différente de FLUX.1 :
# blocs joints 0-5 (attn + ff + ff_context), blocs simples 0-20 (to_qkv_mlp_proj,
# to_out). Ne pas modifier ces chemins sans vérifier contre ta version de mflux.
FLUX2_JOINT_BLOCKS = {"start": 0, "end": 5}
FLUX2_SINGLE_BLOCKS = {"start": 0, "end": 20}
FLUX2_JOINT_PROJ = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.linear_in", "ff.linear_out",
    "ff_context.linear_in", "ff_context.linear_out",
)
FLUX2_SINGLE_PROJ = ("attn.to_qkv_mlp_proj", "attn.to_out")


def build_lora_targets(family: str, rank: int) -> list[dict]:
    """Construit les cibles LoRA selon la FAMILLE du modèle (flux1 ou flux2)."""
    targets: list[dict] = []
    if family == "flux1":
        joint = {"start": 0, "end": FLUX1_NUM_JOINT_BLOCKS}
        single = {"start": 0, "end": FLUX1_NUM_SINGLE_BLOCKS}
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            targets.append(
                {"module_path": f"transformer_blocks.{{block}}.attn.{proj}", "blocks": joint, "rank": rank}
            )
        for proj in ("to_q", "to_k", "to_v"):
            targets.append(
                {"module_path": f"single_transformer_blocks.{{block}}.attn.{proj}", "blocks": single, "rank": rank}
            )
    elif family == "flux2":
        for proj in FLUX2_JOINT_PROJ:
            targets.append(
                {"module_path": f"transformer_blocks.{{block}}.{proj}", "blocks": dict(FLUX2_JOINT_BLOCKS), "rank": rank}
            )
        for proj in FLUX2_SINGLE_PROJ:
            targets.append(
                {"module_path": f"single_transformer_blocks.{{block}}.{proj}", "blocks": dict(FLUX2_SINGLE_BLOCKS), "rank": rank}
            )
    else:
        die(f"Famille de modèle non gérée : {family!r} (flux1|flux2).")
    return targets


def build_config(
    style: str,
    *,
    model_key: str | None = None,
    total_steps: int = 1200,
    rank: int = 16,
    quantize: int | None = 8,
    max_resolution: int = 1024,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Construit le dictionnaire de configuration d'entraînement pour un style.

    Le `model_key` (clé du registre : flux1-dev, flux2-klein-9b, …) décide du modèle
    entraîné, des couches LoRA ciblées, de la fenêtre de timesteps et du dossier de
    sortie. Par défaut on prend le modèle préféré du style (style.yaml: base_model).
    """
    cfg = load_style(style)
    key = resolve_model_key(model_key or cfg["base_model"])
    spec = model_spec(key)

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
    train_steps = spec["train_steps"]

    return {
        "model": spec["train_model"],
        "data": "dataset/",  # relatif à l'emplacement du fichier de config
        "seed": seed,
        "steps": train_steps,
        "guidance": spec["train_guidance"],
        "quantize": quantize,
        "max_resolution": max_resolution,
        "low_ram": False,
        "training_loop": {
            "num_epochs": num_epochs,
            "batch_size": 1,
            "timestep_low": spec["timestep_low"],
            "timestep_high": spec["timestep_high"],
        },
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate},
        # Checkpoints rangés PAR MODÈLE -> un style peut avoir v1 et v2 côte à côte.
        "checkpoint": {"save_frequency": save_frequency, "output_path": f"training/{model_slug(key)}"},
        "monitoring": {
            "preview_width": 1024,
            "preview_height": 1024,
            "plot_frequency": min(25, save_frequency),
            "generate_image_frequency": save_frequency,
        },
        "lora_layers": {"targets": build_lora_targets(spec["family"], rank)},
    }


def write_config(style: str, config: dict, model_key: str | None = None) -> "Path":  # type: ignore[name-defined]
    """Écrit le JSON de config et retourne le chemin (un fichier par modèle)."""
    if model_key:
        name = f"train_config.{model_slug(model_key)}.json"
    else:
        name = "train_config.json"
    path = style_dir(style) / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer la config d'entraînement d'un style.")
    parser.add_argument("style", help="Nom du style.")
    parser.add_argument(
        "--model",
        dest="model_key",
        default=None,
        help="Modèle à entraîner (flux1-dev, flux2-klein-9b, flux2-klein-4b). "
             "Défaut : le base_model du style.",
    )
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
        model_key=args.model_key,
        total_steps=args.total_steps,
        rank=args.rank,
        quantize=None if args.no_quantize else args.quantize,
        max_resolution=args.max_resolution,
        learning_rate=args.learning_rate,
    )
    path = write_config(args.style, config, args.model_key)
    loop = config["training_loop"]
    print(f"✅ Config écrite : {path}")
    print(
        f"   modèle={config['model']} | epochs={loop['num_epochs']} | "
        f"quantize={config['quantize']} | rang={args.rank} | "
        f"max_res={config['max_resolution']}"
    )


if __name__ == "__main__":
    main()
