# 🎵 AI-Image — Générateur de covers par style musical

Pipeline **100 % local sur Mac Apple Silicon** pour entraîner une IA d'image sur
**tes propres styles** de pochettes et générer des covers à partir d'un titre.

- **Modèle** : [FLUX.1 \[dev\]](https://blackforestlabs.ai/) — la meilleure qualité open-weights, niveau MidJourney.
- **Moteur** : [mflux](https://github.com/filipstrand/mflux) — FLUX natif Metal (MLX), génération **et** entraînement LoRA sur Mac.
- **Principe** : **1 dossier de style = 1 LoRA**. Tu entraînes chaque style séparément,
  puis tu génères en choisissant le style + le titre.

> 💡 Pourquoi un LoRA par style ? Tes styles sont visuellement différents et tu as
> des quantités d'images inégales. Un LoRA par style évite que les styles se
> mélangent, te laisse en ré-entraîner un sans toucher aux autres, et tu peux même
> combiner deux styles à la génération (`--lora-scale`).

> 📖 **Pour bien faire apprendre tout l'univers d'un style** (couleurs, ambiance,
> compo, typographie) et obtenir des covers cohérentes : lis le
> [**guide de méthode**](docs/GUIDE.md). C'est le document le plus important du repo.

---

## 🖥️ Prérequis

- **Mac Apple Silicon** (M1/M2/M3/M4). Testé pour M4 Pro 48 Go.
- **Python 3.10+**.
- mflux ne s'installe **que sur macOS Apple Silicon** (il dépend de `mlx`).

```bash
git clone <ce-repo>
cd AI-Image
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Au premier lancement, mflux télécharge FLUX.1 [dev] depuis Hugging Face
(~24 Go). Accepte la licence du modèle sur sa page HF si demandé.

---

## 🚀 Le plus simple : l'assistant

Une seule commande, qui te demande le nom du style et le dossier d'images, puis
fait TOUT (import → captioning IA → entraînement) :

```bash
python scripts/finetune.py
```

Il te pose 2-3 questions :
- **Quel style ?** (ex. `rock`, `funk`, `ambiance`)
- **Chemin du dossier d'images ?** (sur Mac : glisse-dépose le dossier dans le Terminal)
- **Titre écrit sur les covers ?** → réponds `mixed` : l'IA détecte elle-même, image
  par image, s'il y a un titre, et apprend la typographie quand il y en a un.

Refais-le pour chaque style (`rock`, puis `funk`, puis `ambiance`…). Ensuite :
```bash
python scripts/generate_cover.py rock --title "Thunder Road"
```

> Tu peux aussi tout passer en options :
> `python scripts/finetune.py --style rock --images "/Users/moi/Desktop/rock" --max-resolution 768`

---

## 🚀 Étape par étape (si tu préfères contrôler)

```bash
# 1) Créer un style (class-word = ancrage ; title-text = convention de titre)
python scripts/new_style.py phonk \
  --description "phonk sombre, grain cassette, contraste élevé" \
  --base-prompt "dark phonk album cover, vintage grain, high contrast" \
  --class-word "dark phonk album cover" \
  --title-text mixed          # yes = titre toujours écrit, no = jamais, mixed = les deux

# 2) Déposer tes images dans styles/phonk/raw/  (jpg / png / webp)

# 3) Préparer le dataset (redimensionne + crée les légendes)
python scripts/prepare_dataset.py phonk
#    Variante : --auto-caption  -> légende chaque image avec un VLM (capte les
#    variations internes + détecte le titre). Nécessite : pip install mlx-vlm

# 4) Entraîner le LoRA du style  (plusieurs heures ; 768px = + rapide)
python scripts/train_style.py phonk

# 5) Générer une cover
python scripts/generate_cover.py phonk --title "Midnight Drive" \
  --extra "neon tokyo street, rain, cinematic"
```

La cover sort dans `output/phonk_midnight_drive.png`.

> Raccourcis `make` : `make new STYLE=phonk`, `make prepare STYLE=phonk`,
> `make train STYLE=phonk`, `make cover STYLE=phonk TITLE="Midnight Drive"`.

---

## 📂 Structure

```
AI-Image/
├── scripts/
│   ├── finetune.py           # ⭐ assistant tout-en-un (nom + dossier -> LoRA)
│   ├── new_style.py          # crée un style (dossier + style.yaml)
│   ├── prepare_dataset.py    # raw/ -> dataset/ (resize + légendes auto)
│   ├── caption.py            # (re)génère les légendes .txt
│   ├── build_train_config.py # génère la config mflux d'un style
│   ├── train_style.py        # lance l'entraînement LoRA (mflux-train)
│   ├── generate_cover.py     # titre + style -> cover (mflux-generate)
│   └── lib.py                # helpers partagés
├── styles/                   # un sous-dossier par style (voir styles/README.md)
│   └── _TEMPLATE/            # gabarit d'exemple
├── prompts/templates.md      # exemples de prompts par style
├── output/                   # covers générées (gitignoré)
├── requirements.txt
└── Makefile
```

---

## 🎛️ Réglages utiles

### Entraînement (`train_style.py`)
| Option | Défaut | Quand le changer |
|---|---|---|
| `--total-steps` | 1200 | Monte (1500-2000) si le style n'est pas assez appris. |
| `--max-resolution` | 1024 | Mets **768** pour entraîner ~2× plus vite. |
| `--rank` | 16 | 8 = LoRA + léger ; 32 = capte des styles + complexes. |
| `--quantize` | 8 | `--no-quantize` = qualité max mais + de RAM. |
| `--resume <ckpt.zip>` | — | Reprendre un entraînement interrompu. |

Le nombre d'epochs est calculé **automatiquement** selon le nombre d'images
pour viser `--total-steps` itérations. Donc 30 images ou 150 images : pas besoin
de régler quoi que ce soit, le script s'adapte.

### Génération (`generate_cover.py`)
| Option | Défaut | Effet |
|---|---|---|
| `--title` | — | Titre du morceau (écrit dans l'image par défaut). |
| `--no-title-text` | off | Le titre devient un simple thème, non écrit. |
| `--extra` | — | Détails additionnels (lieu, ambiance, couleurs…). |
| `--lora-scale` | 1.0 | Force du style (0.7 = discret, 1.2 = très marqué). |
| `--steps` | 25 | Plus = + de détail (et + lent). |
| `--seed` | aléatoire | Fixe-la pour reproduire/varier une image. |
| `--width/--height` | 1024 | `1024x1024` = pochette ; `1280x720` = bannière. |
| `--checkpoint` | dernier | Utiliser un checkpoint précis. |

Voir [`prompts/templates.md`](prompts/templates.md) pour des exemples.

---

## ❓ FAQ

**Combien d'images par style ?** Minimum ~15, idéalement **30 à 150**. Inutile d'aller
au-delà de quelques centaines pour un style. Un dossier vide ne peut rien apprendre.

**Faut-il légender chaque image ?** Non, c'est automatique. En mode simple, une
légende minimale et cohérente (`trigger + class_word`) est générée pour chaque
image — c'est ce qui fait « fondre » tout l'univers du style dans le trigger. Pour
les styles variés ou avec titres, le mode `--auto-caption` (VLM) décrit chaque
image et détecte le texte. Détails et stratégie : [docs/GUIDE.md](docs/GUIDE.md).

**Comment lui faire apprendre la façon d'écrire les titres (ou l'absence de titre) ?**
Via `title_text` dans `style.yaml` (`yes` / `no` / `mixed`). Le LoRA apprend alors
la typographie du style (ou le look sans texte), et la génération s'y conforme par
défaut. Voir la section 4 du guide.

**Combien de temps / quel coût ?** 100 % local et gratuit. Compte plusieurs heures
d'entraînement par style sur M4 Pro (moins à 768px). La génération d'une cover
prend ~30-60 s.

**Et FLUX.2 / Z-Image ?** FLUX.2 (32B) est meilleur en absolu mais lourd et son
fine-tuning local est encore immature. FLUX.1 [dev] offre le meilleur rapport
qualité / facilité de fine-tuning. Pour changer de base, modifie `base_model`
dans `style.yaml` (note : `build_train_config.py` ne génère les `module_path`
LoRA que pour la famille FLUX.1).

**Alternative sans ligne de commande ?** L'app **Draw Things** (gratuite, native Mac)
fait aussi génération + entraînement LoRA via une interface graphique. Ce repo
vise l'approche scriptable/automatisable.
