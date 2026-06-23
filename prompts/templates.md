# Gabarits de prompts pour covers

Le prompt final est construit automatiquement par `generate_cover.py` :

```
<base_prompt du style>, <trigger du style>, album cover ..., <extra>
```

Tu n'as donc qu'à donner le **titre** et éventuellement des **détails** (`--extra`).

## Exemples

### Phonk
```bash
python scripts/generate_cover.py phonk \
  --title "Midnight Drive" \
  --extra "neon-lit tokyo street at night, rain reflections, motion blur, cinematic"
```

### Drill
```bash
python scripts/generate_cover.py drill \
  --title "Cold Block" \
  --extra "foggy london estate, grayscale, gritty, balaclava silhouette"
```

### Lofi
```bash
python scripts/generate_cover.py lofi \
  --title "Rainy Sunday" \
  --extra "cozy bedroom, warm lamp light, anime style, soft grain" \
  --no-title-text
```

## Conseils

- **Titre dans l'image** : par défaut le script demande au modèle d'écrire le titre.
  FLUX gère bien le texte court. Pour un titre long ou pour ne pas l'afficher,
  ajoute `--no-title-text` (le titre ne sert alors que de thème).
- **Force du style** : `--lora-scale 1.0` par défaut. Baisse à `0.7-0.8` si le style
  écrase trop la composition, monte à `1.1-1.2` s'il n'est pas assez marqué.
- **Variations** : change `--seed` (ou laisse vide pour de l'aléatoire) et relance.
- **Qualité vs vitesse** : `--steps 25` est un bon défaut ; monte à `30-35` pour
  plus de détail, descends à `20` pour aller plus vite.
- **Format** : carré `1024x1024` par défaut (pochette). Pour une bannière :
  `--width 1280 --height 720`.
```
