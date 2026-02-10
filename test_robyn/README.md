LE PROCESS GLOBAL (pipeline complet)

On part de données brutes multimodales et on arrive à une prédiction d’action.

🧱 Étape 1 — Les données brutes

Tu as, pour chaque sujet :

plusieurs séquences

chaque séquence contient des capteurs IMU

et un fichier Events qui dit :

entre t0 et t1, l’action = X

Donc tes données sont continues dans le temps, mais les labels sont segmentés.

👉 Problème fondamental :
Un réseau de neurones ne sait pas traiter du temps continu brut.
Il lui faut des segments de taille fixe.

🧱 Étape 2 — Segmentation par Events (clé du projet)

Le fichier Events fournit :

un label (ex : Walking)

un intervalle temporel [t0, t1]

👉 Pour chaque ligne du fichier Events, tu crées UN échantillon :

(segment IMU entre t0 et t1) → classe = Walking


➡️ C’est ce qu’on appelle :

une reconnaissance d’actions basée sur des segments temporels

C’est standard en HAR (Human Activity Recognition).

🧱 Étape 3 — Lecture + découpage temporel

Pour chaque segment :

tu lis le CSV IMU

tu gardes uniquement les lignes où :

t0 ≤ timestamp ≤ t1


Tu obtiens :

[Tseg, 96]


Tseg varie selon l’action (marcher ≠ sauter)

96 = nombre fixe de canaux IMU

🧱 Étape 4 — Problème des longueurs variables

Les réseaux convolutifs n’acceptent pas :

[120, 96], [430, 96], [50, 96]


👉 Il faut une longueur temporelle fixe.

🧱 Étape 5 — Resampling (normalisation temporelle)

Tu choisis :

L = 256


Et tu transformes chaque segment :

[Tseg, 96] → [256, 96]


Par interpolation linéaire.

👉 Sens physique :

on ne change pas l’action

on normalise juste sa durée

on conserve la dynamique globale

➡️ C’est une pratique très courante en reconnaissance d’actions.

🧱 Étape 6 — Format final pour le réseau

Avant d’entrer dans le modèle :

[256, 96] → [96, 256]


Car en PyTorch :

Conv1D attend [channels, time]


Donc chaque échantillon est :

IMU ∈ ℝ^(96 × 256)
Label ∈ {0,…,30}

🧱 Étape 7 — Split par sujets (très important)

Tu sépares :

sujets d’entraînement

sujets de validation

👉 Le modèle n’a jamais vu les personnes du test.

C’est crucial, car :

sinon il apprend la personne

pas l’action

➡️ C’est un vrai test de généralisation.

2️⃣ L’ARCHITECTURE DU MODÈLE IMU_CNN

Maintenant, voyons comment le réseau comprend une action.

🎯 Entrée du réseau
x ∈ ℝ^(96 × 256)


96 capteurs (accéléromètres, gyroscopes)

256 instants temporels

👉 On veut détecter :

des motifs temporels

dans plusieurs capteurs à la fois

🧠 Pourquoi une CNN 1D ?

Parce que :

le temps est 1D

les actions sont des motifs dynamiques

une CNN peut apprendre :

oscillations

impacts

périodicité (marche, course)

👉 CNN 1D = standard en HAR IMU

🔹 Bloc 1 — Conv1D (96 → 64)
Conv1D(kernel=7, stride=2)

Ce que ça fait :

regarde des fenêtres de 7 instants

combine les 96 capteurs

détecte des motifs simples :

début de mouvement

changements d’accélération

➡️ Sortie :

[64, ~128]

🔹 Bloc 2 — Conv1D (64 → 128)
Conv1D(kernel=5, stride=2)

Ce que ça fait :

combine les motifs précédents

détecte des patterns plus complexes :

pas de marche

flexion / extension

balancement

➡️ Sortie :

[128, ~64]

🔹 Bloc 3 — Conv1D (128 → 256)
Conv1D(kernel=3, stride=2)

Ce que ça fait :

capte des structures de mouvement complètes

représentation abstraite de l’action

➡️ Sortie :

[256, ~32]

🧠 Global Average Pooling (clé)
x = x.mean(dim=-1)


👉 On moyenne sur le temps.

Pourquoi c’est intelligent :

le modèle devient invariant à la position temporelle

seul compte :

“est-ce que ce motif apparaît ?”

➡️ Sortie :

[256]


C’est un résumé global de l’action.

🎯 Classification finale
Linear(256 → 31)


Chaque neurone correspond à une action :

marcher

sauter

s’accroupir

etc.

La softmax donne une probabilité par classe.

🧠 Résumé en une phrase (très utile à l’oral)

“Les signaux IMU sont segmentés à partir des annotations temporelles, normalisés en durée par resampling, puis analysés par un réseau convolutionnel 1D qui apprend des motifs dynamiques caractéristiques des actions humaines.”

✅ Ce que TON modèle fait bien

✔️ respecte la structure temporelle
✔️ généralise entre sujets
✔️ simple mais robuste
✔️ justifiable scientifiquement