#!/usr/bin/env python3
# Importation des modules de base
import numpy as np
import getfem as gf

###########################################
# Données du problème
########################################### 

# Module de Young pour le caoutchouc en pascals par mètre carré (Pa/m^2)
module_young = 690  
# Coefficient de Poisson pour le caoutchouc
coef_poisson = 0.3 

# Premier coefficient de Lame
lambda_lame = module_young * coef_poisson / ((1 + coef_poisson) * (1 - 2 * coef_poisson)) 
# Second coefficient de Lame
mu_lame = module_young / (2 * (1 + coef_poisson))               
# Coefficient de Lame pour les contraintes planes
lambda_star = 2 * lambda_lame * mu_lame / (lambda_lame + 2 * mu_lame) 

# Nombre d'éléments en x et y pour le maillage
nombre_elements_x = 30
nombre_elements_y = 15

# Utilisation de multiplicateurs pour les conditions de Dirichlet
utiliser_multiplicateurs_dirichlet = True  
# Coefficient de pénalisation
coefficient_penalisation = 1e10       

# Degré des éléments finis
degre_elements_fini = 1  

###########################################
# Création d'un maillage cartésien simple 4x2
###########################################
maillage = gf.Mesh('import', 'structured', 'GT="GT_PK(2,1)";SIZES=[2.,1.];NOISED=0;NSUBDIV=[4,2];')
###########################################
# Création d'un maillage cartésien plus simple 30x15
###########################################
#maillage = gf.Mesh('import', 'structured', 'GT="GT_PK(2,1)";SIZES=[2.,1.];NOISED=0;NSUBDIV=[30,15];')




maillage.display()

# Création d'un MeshFem pour les champs u de dimension 2 (champ vectoriel)
mf = gf.MeshFem(maillage, 2)
# Attribution du fem P1 à tous les éléments du MeshFem
mf.set_fem(gf.Fem('FEM_PK(2,1)'))  # Passage au fem P1
# Méthode d'intégration utilisée
mim = gf.MeshIm(maillage, gf.Integ('IM_TRIANGLE(4)'))  # Éléments triangulaires

# Sélection des frontières
faces_exterieures = maillage.outer_faces()
normales_faces = maillage.normal_of_faces(faces_exterieures)
gauche = abs(normales_faces[0, :]) < 1e-14
haut = abs(normales_faces[1, :] - 1) < 1e-14  # Ajustement pour les nouvelles dimensions du domaine
bas = abs(normales_faces[1, :]) < 1e-14
faces_gauche = np.compress(gauche, faces_exterieures, axis=1)
faces_haut = np.compress(haut, faces_exterieures, axis=1)
faces_bas = np.compress(bas, faces_exterieures, axis=1)

# Marquage des frontières
NUM_FRONTIERE_DIRICHLET = 1
NUM_FRONTIERE_NEUMANN = 2  
maillage.set_region(NUM_FRONTIERE_DIRICHLET, faces_gauche)  # Γ0: Frontière gauche
maillage.set_region(NUM_FRONTIERE_NEUMANN, np.concatenate((faces_haut, faces_bas), axis=1))  # Γ1: Frontières supérieure et inférieure

###########################################
# Modèle
###########################################
md = gf.Model('real')



# Variable principale inconnue
md.add_fem_variable('u', mf)

# Déformation élastique de la membrane
md.add_initialized_data('cmu', [mu_lame])
md.add_initialized_data('clambda', [lambda_lame])

# Assemblage de la matrice de rigidité
# (les coefficients c11, c12, etc. sont définis ici)
c11 = lambda_lame+2*mu_lame    ; c22 = lambda_lame+2*mu_lame   ;
c12 = lambda_lame;
c44 = mu_lame ; c55 = mu_lame ; c66 = mu_lame;
#
K11=[[c11,0. ],[0.,c66]];
K21=[[0. ,c12],[c66,0.]];
K22=[[c66,0. ],[0.,c22]];
K12=np.transpose(K21)
md.add_initialized_data('K11', K11,[2,2])
md.add_initialized_data('K21', K21,[2,2])
md.add_initialized_data('K12', K12,[2,2])
md.add_initialized_data('K22', K22,[2,2])
#
md.add_linear_generic_assembly_brick(mim,"(K11*(Grad_u.[1;0])).(Grad_Test_u.[1;0])")
md.add_linear_generic_assembly_brick(mim,"(K12*(Grad_u.[1;0])).(Grad_Test_u.[0;1])")
md.add_linear_generic_assembly_brick(mim,"(K21*(Grad_u.[0;1])).(Grad_Test_u.[1;0])")
md.add_linear_generic_assembly_brick(mim,"(K22*(Grad_u.[0;1])).(Grad_Test_u.[0;1])")


# Imposition des conditions de Dirichlet "u=0" sur la frontière inférieure
md.add_Dirichlet_condition_with_multipliers(mim, 'u', degre_elements_fini - 1, NUM_FRONTIERE_DIRICHLET)

# Application des forces extérieures
md.add_initialized_data('Fdata', [0, 0])

# Terme source volumique
densite = 1.0  # Densité en kg/m^3 (ajuster selon les besoins)
gravite = 9.82  # Accélération gravitationnelle en m/s^2


# Terme source volumique pour la gravité
force_gravite = densite * np.array([0, -gravite])  # Force de la gravité appliquée dans la direction négative de y
md.add_initialized_data('DonneesVolumiques', force_gravite)
md.add_source_term_brick(mim, 'u', 'DonneesVolumiques')

# Ajout de la brique d'élasticité linéaire isotrope
md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')

###########################################
# Assemblage du système linéaire et résolution
###########################################
md.solve()

###########################################
# Exportation de la solution
###########################################
U = md.variable('u')
mfvm = gf.MeshFem(maillage, 1)
mfvm.set_classical_discontinuous_fem(1)
VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'clambda', 'cmu', mfvm)
mfvm.export_to_pos('ex2.pos', mfvm, VM, 'Contraintes Von Mises', mf, U, 'Deplacements')

# Visualisation des résultats par exportation au format Gmsh
mf.export_to_pos('ex2sol_maillage1.pos', U, 'Solution calculée')

