#!/usr/bin/env python3
import numpy as np
import getfem as gf

# Paramètres pour le caoutchouc
E = 69e6  # Module de Young pour le caoutchouc en Pascal par mètre carré (Pa/m^2)
v= -0.2 # Coefficient de Poisson pour le caoutchouc

# Calcul des coefficients lambda et mu pour l'élasticité linéaire isotrope
clambda = E * v / ((1 + v) * (1 - 2 * v))
cmu = E / (2 * (1 + v))

# Gravité
gravity = 9.82  # Accélération gravitationnelle en m/s^2

# Création d'un maillage cartésien simple 2x4
mesh = gf.Mesh('import', 'structured', 'GT="GT_PK(2,1)";SIZES=[1.,2.];NOISED=0;NSUBDIV=[2, 4];')
# Création d'un maillage cartésien plus fin 15*30
#mesh = gf.Mesh('import', 'structured', 'GT="GT_PK(2,1)";SIZES=[1.,2.];NOISED=0;NSUBDIV=[15, 30];')
mesh.display()

# Création d'un MeshFem pour les champs u de dimension 2 (c'est-à-dire, un champ vectoriel)
mf = gf.MeshFem(mesh, 2)
mf.set_fem(gf.Fem('FEM_PK(2,1)'))

# Méthode d'intégration utilisée
mim = gf.MeshIm(mesh, gf.Integ('IM_TRIANGLE(4)'))

# Sélection des frontières extérieures
faces_exterieures = mesh.outer_faces()

# Calcul des normales des faces
normales_faces = mesh.normal_of_faces(faces_exterieures)

# Identification des faces de la frontière gauche, supérieure et inférieure
gauche = abs(normales_faces[0, :]) < 1e-14
haut = abs(normales_faces[1, :] - 2) < 1e-14
bas = abs(normales_faces[1, :]) < 1e-14

# Compression des faces pour obtenir les faces de chaque frontière spécifiée
faces_gauche = np.compress(gauche, faces_exterieures, axis=1)
faces_haut = np.compress(haut, faces_exterieures, axis=1)
faces_bas = np.compress(bas, faces_exterieures, axis=1)


# Marquage des frontières
NUM_FRONTIERE_DIRICHLET = 1
NUM_FRONTIERE_NEUMANN = 2
mesh.set_region(NUM_FRONTIERE_DIRICHLET, faces_gauche)  # Γ0: Frontière gauche
mesh.set_region(NUM_FRONTIERE_NEUMANN, np.concatenate((faces_haut, faces_bas), axis=1))  # Γ1: Frontières supérieure et inférieure


# Modèle
md = gf.Model('real')

# Variable principale inconnue
md.add_fem_variable('u', mf)

# Déformation élastique de la membrane
md.add_initialized_data('cmu', [cmu])
md.add_initialized_data('clambda', [clambda])

# Élasticité linéaire isotrope
md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')

# Initialisation des données de Dirichlet (déplacement nul dans les deux directions)
cond_dirichlet = np.array([0, 0])

# Ajout de la condition de Dirichlet sur la frontière gauche avec un déplacement nul
md.add_initialized_data('DirichletData', cond_dirichlet)
md.add_Dirichlet_condition_with_multipliers(mim, 'u', mf, NUM_FRONTIERE_DIRICHLET, 'DirichletData')

# Force de gravité
Force_gravite = np.array([0, -9.82])
md.add_initialized_data('VolumicData', Force_gravite)
md.add_source_term_brick(mim, 'u', 'VolumicData')

# Résolution du problème
md.solve()

# Exportation de la solution
U = md.variable('u')
mfvm = gf.MeshFem(mesh, 1)
mfvm.set_classical_discontinuous_fem(1)
VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'clambda', 'cmu', mfvm)

# Exportation des contraintes de Von Mises pour la visualisation
mfvm.export_to_pos('Sol_Exercice_1_maillage1_v02.pos', mfvm, VM, 'Von Mises Stresses', mf, U, 'Displacements')

