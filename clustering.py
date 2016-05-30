import numpy as np
import matplotlib.pyplot as plt


def random_distribution(n_item, n_cluster): 

	random_matrix = np.random.random_sample((n_item, n_cluster))
	#print(random_matrix, "\n")
	normalise = random_matrix / random_matrix.sum(axis=1).reshape(n_item,1)
	#print(normalise, "\n")
	return(normalise)


def info_based_clustering(s_matrix, n_item, n_cluster, t, e):
	# initialisation 
	## p(m)(C|i)
	p_clusters_for_items = random_distribution(n_item, n_cluster)
	#print("p(m)(C|i)\n", p_clusters_for_items, "\n")

	## p(i)
	p_items = 1/n_item

	while True :
		## p(C)
		p_clusters = p_clusters_for_items.sum(axis=0) * p_items
		#print("p(C)\n", p_clusters, "\n")
		## p(i|C)
		p_items_in_clusters = p_clusters_for_items * p_items / p_clusters
		#print("p(i|C)\n", p_items_in_clusters, "\n")

		## equation 6 : s(C;i)
		items_similarity_to_clusters = np.dot(s_matrix, p_items_in_clusters)
		#print("s(C;i)\n", items_similarity_to_clusters, "\n")

		## equation 1 : s(c) = p(i|C) * s(C;i)
		product = np.multiply(p_items_in_clusters, items_similarity_to_clusters)
		similarity_in_clusters = np.sum(product, axis=0)
		#print("s(C)\n", similarity_in_clusters, "\n")

		## equation 2 : <s> = p(c) * s(c)
		average_similarity = np.multiply(p_clusters, similarity_in_clusters).sum()

		## p(m+1)(C|i)
		## 1er point
		p_next_clusters_for_items = np.multiply(p_clusters, np.exp((1/t) * ((2*items_similarity_to_clusters) - similarity_in_clusters)))
		#print("avant normaliser p(m+1)(C|i)\n", p_next_clusters_for_items, "\n")
		## 2eme point
		p_next_clusters_for_items = p_next_clusters_for_items / p_next_clusters_for_items.sum(axis=1).reshape(n_item, 1)
		#print("p(m+1)(C|i)\n", p_next_clusters_for_items, "\n")

		## condition break
		diff = np.absolute(p_next_clusters_for_items - p_clusters_for_items)
		#print("|p(m+1)(C|i) - p(m)(C|i)|\n", diff, "\n")
		comparison = np.less_equal(diff, e)
		#print("condition", comparison, "\n")

		if comparison.all() == True :
			break
		else : 
			## p(m) <- p(m+1) pour la suite
			p_clusters_for_items = p_next_clusters_for_items

	return(p_next_clusters_for_items, average_similarity)


def q1_verification(estimation):

	f = np.loadtxt("scatter_ds1.d")
	coordinates_x = f[:, 0]
	coordinates_y = f[:, 1]
	real_partition = f[:, 2]
	real_partition = real_partition.astype(int)

	## real partition
	plt.title('visualisation de scatter_ds1')
	plt.scatter(coordinates_x, coordinates_y, c=real_partition)
	plt.savefig('real_scatter.png')
	plt.show()

	## estimated partition 
	plt.title('visualisation résultat obtenu')
	plt.scatter(coordinates_x, coordinates_y, c=estimation)
	plt.savefig('estimated.png')
	plt.show()


def question1(f, t, e) :
	## similarity matrix : s(i1,i2)
	similarity_matrix = np.loadtxt(f)
	items = len(similarity_matrix)
	clusters = 3

	clustering_estimation, av = info_based_clustering(similarity_matrix, items, clusters, t, e)

	## liste de partitionnement des elements selon les estimations obtenues
	estimated_partition = np.argmax(clustering_estimation, axis=1)
	## comparer avec la solution
	q1_verification(estimated_partition)
	

def q2_visualisation(s_m, i, c, t, e) : 

	clustering_estimation, av = info_based_clustering(s_m, i, c, t, e)
	plt.pcolormesh(clustering_estimation, cmap='Blues')
	plt.colorbar()
	plt.show()


def question2(f, t, e) : 
	## pour chercher les bons nombres de cluster, il faut regarder average similarity de clusters.
	## s'il baisse l'augmentation, c'est le bon nombre.
	## => ici, le nombre de cluster est probablement 4.

	## similarity matrix : s(i1,i2)
	similarity_matrix = np.loadtxt(f)
	items = len(similarity_matrix)
	average_similarities = list()

	#iteration en changeant le nombre de cluster
	for nbr_clusters in range(1, 11) :
		clustering_estimation, average_similarity = info_based_clustering(similarity_matrix, items, nbr_clusters, t, e)
		#print("average similarity\n", average_similarity, "\n")
		average_similarities.append(average_similarity)

	iteration = range(1, len(average_similarities)+1)

	plt.plot(iteration, average_similarities)
	plt.xlabel('nombre de cluster')
	plt.ylabel('average similarity')
	plt.title('transition de average similarity')
	plt.grid(True)
	plt.show()

	## si on veut visualiser
	## avec le nombre de cluster = 4
	q2_visualisation(similarity_matrix, items, 4, t, e)


def verification(items, clusters, estimated_partition, f1, f2, f3) :

	## liste des noms d'entreprise / films
	names = []
	with open(f1, "r") as fichier :
		for line in fichier :
			names.append(line.rstrip())
	#print(names)

	## liste des entreprises/films selon les clusters obtenus
	clusters_items = [[] for i in range(clusters)]
	for i in range(items) :
		n_cluster = estimated_partition[i]
		clusters_items[n_cluster].append(i)
	#print(clusters_items)

	## croiser les entreprises/films et ses secteurs/genres
	mat_croise = np.genfromtxt(f2, dtype=int)
	croise = np.transpose(np.nonzero(mat_croise))
	#print(croise)
	types_names = [[] for i in range(items)]
	for row in croise :
		n_name = row[0]
		n_type = row[1]
		types_names[n_name].append(n_type)
	#print(types_names)

	## liste des noms de secteurs / genres
	types = []
	with open(f3, "r") as fichier :
		for line in fichier :
			types.append(line.rstrip())
	#print(types)

	## affiche les resultats de croise
	for c in clusters_items : 
		print("Cluster :")
		for n_n in c: ## numeros d'entreprise / de films
			print("- ", names[n_n]) ## affiche nom d'entreprise / de films
			n_t = types_names[n_n] ## numeros de secteurs / genres
			for t in n_t:
				print("    ", types[t]) ## affiche nom de secteurs / genres
		print("") ## nouvelle ligne


def question3(f, t, e) :
	## similarity matrix : s(i1,i2)
	similarity_matrix = np.loadtxt(f)
	items = len(similarity_matrix)
	clusters = 20

	clustering_estimation, av = info_based_clustering(similarity_matrix, items, clusters, t, e)
	#print("p(c|i)\n", clustering_estimation)
	#print("<s>\n", av)

	## liste de partitionnement des elements selon les estimations obtenues
	estimated_partition = np.argmax(clustering_estimation, axis=1)
	#print("estimated clustering :\n", estimated_partition)

	verification(items, clusters, estimated_partition, "./sp500_data_id/sp500_names.d", "./sp500_data_id/sp500_matType.d", "./sp500_data_id/sp500_TypeNames.d")


def question4(f, t, e) : 
	## similarity matrix : s(i1,i2)
	similarity_matrix = np.loadtxt(f)
	items = len(similarity_matrix)
	clusters = 20

	clustering_estimation, av = info_based_clustering(similarity_matrix, items, clusters, t, e)
	#print("p(c|i)\n", clustering_estimation)
	#print("<s>\n", av)

	## liste de partitionnement des elements selon les estimations obtenues
	estimated_partition = np.argmax(clustering_estimation, axis=1)
	#print("estimated clustering :\n", estimated_partition)

	verification(items, clusters, estimated_partition, "./movie_data_id/movie_name", "./movie_data_id/movie_labels.d", "./movie_data_id/movie_typename")


##### MAIN #####

### input ###
tradeoff = 1/25
epsilon = 0.00001

## Question 1
question1("simil_ds1.d", tradeoff, epsilon)

## Question 2
#question2("simil_ds2.d", tradeoff, epsilon)

## Quesiton 3
tradeoff_q3 = 1/35
#question3("mi_sp500.d", tradeoff_q3, epsilon)

## Quesiton 4
tradeoff_q4 = 1/40
epsilon_q4 = 0.1
#question4("mi_movie.d", tradeoff_q4, epsilon_q4)
