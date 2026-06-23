# Raccourcis pratiques. Usage : make <cible> STYLE=phonk TITLE="Mon titre"
# (nécessite mflux installé sur Mac Apple Silicon pour train/cover)

STYLE ?= phonk
TITLE ?= Untitled

.PHONY: help install new prepare train cover styles

help:
	@echo "Cibles disponibles :"
	@echo "  make install                      # installe les dépendances (sur Mac)"
	@echo "  make new STYLE=phonk              # crée un nouveau style"
	@echo "  make prepare STYLE=phonk          # prépare images + légendes"
	@echo "  make train STYLE=phonk            # entraîne le LoRA"
	@echo "  make cover STYLE=phonk TITLE=...  # génère une cover"
	@echo "  make styles                       # liste les styles existants"

install:
	pip install -r requirements.txt

new:
	python scripts/new_style.py $(STYLE)

prepare:
	python scripts/prepare_dataset.py $(STYLE)

train:
	python scripts/train_style.py $(STYLE)

cover:
	python scripts/generate_cover.py $(STYLE) --title "$(TITLE)"

styles:
	@ls -1 styles | grep -v '^_TEMPLATE$$' || echo "(aucun style)"
