# Mini-projet — Recherche heuristique, A* et Processus Décisionnels de Markov
## ENSET Mohammedia — Master SDIA 2025-2026

- **Étudiant :** Felouach Oussama
- **Encadrant :** Prof. Mohammed Mestari
- **Filière :** Master SDIA — ENSET Mohammedia
- **Module :** Bases de l'Intelligence Artificielle
- **Année :** 2025–2026

---

## Structure du projet

```
project/
├── main.py                    # Point d'entrée principal
├── README.md
├── search/
│   ├── astar.py               # A* et Weighted A*
│   ├── ucs.py                 # Uniform-Cost Search
│   ├── greedy.py              # Greedy Best-First Search
│   └── utils.py               # Logger, reconstruction de chemin, type Graph
├── markov/
│   ├── absorbing_chain.py     # Analyse analytique (N, B, t)
│   └── simulation.py          # Monte Carlo + comparaison
├── experiments/
│   ├── graphs.py              # Graphes de référence, heuristiques, générateur
│   └── benchmarks.py          # Expériences comparatives + graphiques
└── data/                      # Graphiques générés automatiquement
```

---

## Installation

Python ≥ 3.9 requis.

```bash
pip install numpy matplotlib
```

---

## Exécution

### Toutes les expériences

```bash
cd project
python main.py all
```

### Seulement les algorithmes de recherche (Parties I–IV)

```bash
python main.py search
```

### Seulement la chaîne de Markov (Partie V)

```bash
python main.py markov
```

### Extensions (E1, E3, E4, E5)

```bash
python main.py extensions
```

---

## Description des modules

### `search/astar.py`
Implémente A* avec :
- File de priorité (heapq)
- Mise à jour des coûts g
- Réouverture des nœuds CLOSED (pour heuristiques non cohérentes)
- Logger d'expansion pas à pas

### `search/ucs.py`
Uniform-Cost Search — cas dégénéré de A* avec h ≡ 0.

### `search/greedy.py`
Greedy Best-First — trie la frontière par h(n) uniquement.

### `markov/absorbing_chain.py`
- Construction de la matrice de transition P
- Forme canonique (états absorbants en premier)
- Matrice fondamentale N = (I − Q)⁻¹
- Probabilités d'absorption B = NR
- Espérance du temps d'absorption t = N·1

### `markov/simulation.py`
- Simulation Monte Carlo (200 000 essais par défaut)
- Comparaison résultats analytiques vs empiriques
- Analyse de sensibilité : E[T|i] et P(ruine|i) pour tous les états

### `experiments/benchmarks.py`
- Traces pas à pas (Partie II)
- Comparaison des heuristiques (admissible/cohérente, admissible/non-cohérente, non-admissible)
- Étude Weighted A* (Extension E3)
- Étude de scalabilité sur graphes aléatoires (Partie IV)
- Étude de dominance heuristique (Extension E1)

---

## Sorties

Les graphiques sont sauvegardés dans `data/` :
- `weighted_astar.png` — coût et expansions en fonction de w
- `scalability.png` — nœuds développés vs taille du graphe
- `dominance.png` — dominance heuristique sur 20 graphes aléatoires