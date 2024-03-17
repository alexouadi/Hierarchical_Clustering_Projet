import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



#Code de plusieurs distances possibles
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def Minkowski_distance(x,y,p):
    return (np.sum(np.abs(x-y)**p))**(1/p)

def Manhattan_distance(x,y):
    return np.sum(np.abs(x-y))

def Tchebychev_distance(x,y):
    return max(np.abs(x-y))

def Pearson_distance(x,y):
    return 1 - (np.sum((x-np.mean(x))(y-np.mean(y))))/((np.sqrt(np.sum((x - np.mean(x)) ** 2)))*(np.sqrt(np.sum((y - np.mean(y)) ** 2))))

def Canberra_distance(x,y):
    return np.sum(np.abs(x-y)/(x+y))

def cosinus_distance(x,y):
    return 1 - (np.sum(x*y))/((np.sqrt(np.sum(x**2)))*(np.sqrt(np.sum(y**2))))

#Code critère de distance
def distance_minimal(cluster1, cluster2):
    n1 = len(cluster1)
    n2 = len(cluster2)
    dist = []
    for i in range(n1):
        for j in range(n2):
            dist.append(euclidean_distance(cluster1[i], cluster2[j]))
    return min_list(dist)

def distance_maximal(cluster1, cluster2):
    n1 = len(cluster1)
    n2 = len(cluster2)
    dist = []
    for i in range(n1):
        for j in range(n2):
            dist.append(euclidean_distance(cluster1[i], cluster2[j]))
    return max_list(dist)

def min_list(L):
    n = len(L)
    a = L[0]
    for i in range(1, n):
        if L[i] < a:
            a = L[i]
    return a

def max_list(L):
    n = len(L)
    a = L[0]
    for i in range(1, n):
        if L[i] > a:
            a = L[i]
    return a

# Calcule la distance de Ward entre deux clusters
def ward_distance(cluster1, cluster2):
    n1 = len(cluster1)
    n2 = len(cluster2)
    mean1 = np.mean(cluster1, axis=0)
    mean2 = np.mean(cluster2, axis=0)
    return n1 * n2 / (n1 + n2) * euclidean_distance(mean1, mean2)

def hierarchical_clustering(data):
    # Récupère le nombre de points dans les données
    n = len(data)
    # Initialise chaque point comme un cluster
    clusters = [[i] for i in range(n)]
    # Initialise une liste de clusters pour stocker la hiérarchie de clustering
    tree = [[i] for i in range(n)]
    # Initialise une matrice de linkage
    linkage_matrix = np.zeros((n-1, 4))
    # Effectue le clustering jusqu'à ce qu'il ne reste qu'un cluster
    for k in range(n-1):
        # Initialise les variables pour stocker les clusters avec la distance minimale
        min_distance = float('inf')
        min_i = 0
        min_j = 0
        # Parcourt toutes les paires de clusters pour trouver la distance minimale
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                # Calcule la distance de Ward entre les deux clusters
                distance = ward_distance(data[cluster1], data[cluster2])
                # Met à jour les variables avec le cluster avec la distance minimale
                if distance < min_distance:
                    min_distance = distance
                    min_i = i
                    min_j = j
        # Ajoute le nouveau cluster à la liste de la hiérarchie de clustering
        tree.append(clusters[min_i] + clusters[min_j])
        # Stocke les informations de linkage dans la matrice de linkage
        linkage_matrix[k] = [tree.index(clusters[min_i]), tree.index(clusters[min_j]), min_distance, len(clusters[min_i]) + len(clusters[min_j])]
        # Fusionne les deux clusters avec la distance minimale
        clusters[min_i].extend(clusters[min_j])
        del clusters[min_j]
    # Retourne la matrice de linkage et la liste de la hiérarchie de clustering
    return linkage_matrix, tree


def dendrogramme(L,tree):
    # Création d'une nouvelle figure et ajout d'un titre et un nom d'axe y
    plt.figure()
    plt.title('Dendrogramme')
    plt.ylabel('Distance')
    # Nombre de feuilles dans l'arbre
    N=len(L)
    # Initialisation de la matrice M pour stocker les coordonnées de chaque noeud
    M=np.zeros((2*N+1,2))
    # Récupération des indices des feuilles dans la dernière couche de l'arbre
    for i in range(N+1):
        M[i][0]=tree[-1].index(i)
    # Boucle pour parcourir les étapes de l'agglomération en partant des feuilles pour les regrouper et former l'arbre
    for i in range(N):
        # Récupération des deux clusters à fusionner ainsi que leur distance
        i_1=int(L[i][0])
        i_2=int(L[i][1])
        C_1=M[i_1]
        C_2=M[i_2]
        dist=L[i][2]
        # Calcul des coordonnées du nouveau cluster qui regroupe les deux précédents
        Cbis_1=[C_1[0],dist]
        Cbis_2=[C_2[0],dist]
        # Trace les lignes pour relier les clusters
        color = plt.cm.nipy_spectral(float(i) / N)
        plt.plot([C_1[0], Cbis_1[0]], [C_1[1], Cbis_1[1]], color=color)
        plt.plot([C_2[0], Cbis_2[0]], [C_2[1], Cbis_2[1]], color=color)
        plt.plot([Cbis_1[0], Cbis_2[0]], [Cbis_1[1], Cbis_2[1]], color=color)
        # Calcul des coordonnées du nouveau cluster et stockage dans la matrice M
        M[N+i+1]=[(C_1[0]+C_2[0])/2,dist]
    # Modification de l'axe x pour afficher les labels de chaque cluster
    plt.xticks(np.arange(N+1),tree[-1],fontsize = 'xx-small')
    # Affichage de la figure
    plt.show()
    
def coude(linkage_matrix,tree, X, max_clusters):
    # Initialisation des valeurs WCSS vides
    wcss_values = []
    
    # Définition de la plage de nombre de clusters à considérer
    n_clusters_range = range(1, min(linkage_matrix.shape[0] + 1, max_clusters + 1))
    
    # Pour chaque nombre de clusters, calcul du WCSS et stockage de la valeur
    for n_clusters in n_clusters_range:
        # Récupération des labels de clusters à partir de la matrice de linkage
        labels = np.array(fcluster(linkage_matrix, tree, n_clusters))
        
        # Calcul des centres de clusters
        cluster_centers = np.array([np.mean(X[labels == i], axis=0) for i in range(1, n_clusters+1)])
        
        # Calcul du WCSS
        wcss = np.sum((X - cluster_centers[labels-1])**2)
        
        # Ajout de la valeur WCSS à la liste des valeurs
        wcss_values.append(wcss)
    
    # Tracé de la courbe "Méthode du coude"
    plt.figure()
    plt.plot(n_clusters_range, wcss_values)
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    
# Définition de la fonction de récupération des clusters
def clusters(L, tree, n):
    clusters = []
    # Si n est égal à 1, renvoyer le dernier élément de l'arbre
    if n == 1:
        return [tree[-1]]
    N = len(L)
    index = []
    # Ajouter les deux dernières feuilles à la liste des index
    index.append(int(L[-1][-4]))
    index.append(int(L[-1][-3]))
    # Ajouter les clusters correspondant à ces index
    clusters.append(tree[index[0]])
    clusters.append(tree[index[1]])
    # Tant que le nombre de clusters est inférieur à n
    while len(clusters) < n:
        # Retirer le cluster de la liste à partir de l'index maximal
        clusters.remove(tree[max(index)])
        i = int(L[max(index) - (N + 1)][-4])
        j = int(L[max(index) - (N + 1)][-3])
        # Ajouter les nouveaux index
        index.remove(max(index))
        index.append(i)
        index.append(j)
        # Ajouter les nouveaux clusters
        clusters.append(tree[i])
        clusters.append(tree[j])
    return clusters


# Définition de la fonction de clustering plat
def fcluster(L, tree, n):
    # Récupérer les clusters
    clust = clusters(L, tree, n)
    n = len(clust)
    # Initialisation d'un vecteur f de même taille que le nombre de points
    f = (len(L) + 1) * [0]
    # Pour chaque cluster, assigner l'indice du cluster à chaque point correspondant
    for i in range(n):
        for j in clust[i]:
            f[j] = i + 1
    return f



def plot_clusters(data, n_clusters):
    # Effectue un clustering hiérarchique sur les données
    L_data,T_data = hierarchical_clustering(data)
    # Récupère les clusters avec la méthode fcluster
    labels = fcluster(L_data,T_data, n_clusters)
    # Effectue une réduction de dimension avec PCA pour afficher les données en 2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    # Affiche les points colorés selon leur cluster
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
    # Titre du plot
    plt.title(f'Répartition des données en {n_clusters} clusters')
    # Affiche le plot
    plt.show()
