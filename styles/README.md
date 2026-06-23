# Dossier `styles/`

Un sous-dossier = un **style musical** = un **LoRA**.

## Structure d'un style

```
styles/
└── phonk/
    ├── style.yaml          # métadonnées (trigger word, base_prompt, modèle)
    ├── raw/                # ⬅️ TU déposes ici tes images d'origine (jpg/png/webp)
    ├── dataset/            # images préparées + légendes .txt (généré)
    ├── train_config.json   # config d'entraînement mflux (généré)
    └── training/           # checkpoints LoRA (généré, volumineux, gitignoré)
```

## Cycle de vie

1. `python scripts/new_style.py phonk` → crée le dossier + `style.yaml`
2. Tu déposes tes images dans `styles/phonk/raw/`
3. `python scripts/prepare_dataset.py phonk` → remplit `dataset/` + légendes
4. `python scripts/train_style.py phonk` → entraîne le LoRA dans `training/`
5. `python scripts/generate_cover.py phonk --title "..."` → génère une cover

## Bon à savoir

- **15 images minimum** par style, **30 à 150 idéalement**. Un dossier vide ne peut rien apprendre.
- Les dossiers `raw/`, `dataset/`, `training/` et les `.safetensors` sont **gitignorés**
  (trop lourds). Seuls `style.yaml` et la structure sont versionnés.
- Le dossier `_TEMPLATE/` est un exemple, il est ignoré par les scripts.
