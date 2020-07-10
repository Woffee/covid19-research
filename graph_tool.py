"""

Some graph indicators using Networkx.

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import asyn_fluid
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms import bipartite


class MyGraph():
    def __init__(self, arr):
        # arr = np.array([
        #     [0, 0, 0, 0.5],
        #     [0, 0, 0, 0.6],
        #     [0, 0.2, 0, 0.9],
        #     [0, 0, 0, 0],
        # ])

        weighted_edges = []
        m, n = arr.shape
        for i in range(m):
            for j in range(n):
                if arr[i, j] > 0:
                    weighted_edges.append((i, j, arr[i, j]))

        G = nx.DiGraph()
        G.add_weighted_edges_from(weighted_edges)
        self.G = G

    def to_array(self):
        return nx.to_numpy_matrix(self.G, nodelist=list(range(len(self.G.nodes()))))

    # Global cluster coefficient
    def global_cluster_coefficient(self):
        return nx.average_clustering(self.G)

    # Local cluster coefficients
    # Returns: A dictionary, keyed by degree, with local cluster coefficient values
    def local_cluster_coefficient(self):
        return nx.clustering(self.G)


    def rich_club_coefficient(self):
        """
        Rich-club coefficient
        Returns: A dictionary, keyed by degree, with rich-club coefficient values
        """
        undirected_g = self.G.to_undirected()
        return nx.rich_club_coefficient(undirected_g, normalized=False)


    def spectral_centrality(self):
        pass


    def spectral_bipartivity(self):
        """
        Returns the spectral bipartivity.
        https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.spectral.spectral_bipartivity.html
        """
        return bipartite.spectral_bipartivity(self.G)


    def page_rank(self, alpha=0.9):
        """
        PageRank
        Returns: Dictionary of nodes with PageRank as value
        """
        return nx.pagerank(self.G, alpha)

    def leader_rank(self):
        pass


    def modularity_based_community(self):
        """
        Find communities in graph using Clauset-Newman-Moore greedy modularity maximization. This method currently supports the Graph class and does not consider edge weights.

        Greedy modularity maximization begins with each node in its own community and joins the pair of communities that most increases modularity until no such pair exists.

        https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
        """
        undirected_g = self.G.to_undirected()
        return list(greedy_modularity_communities(undirected_g))


    def fluid_community(self, k=2):
        """
        Returns communities in G as detected by Fluid Communities algorithm.
        """
        undirected_g = self.G.to_undirected()
        return list(asyn_fluid.asyn_fluidc(undirected_g, k))

    def minimum_spanning_arborescence(self):
        b = nx.minimum_spanning_arborescence(self.G)
        return nx.to_numpy_matrix(b, nodelist=list(range(len(self.G.nodes()))))


if __name__ == '__main__':
    arr = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0.6],
        [0, 0.2, 0, 0.9],
        [0.5, 0, 0, 0],
    ])

    G = MyGraph(arr)
    print("global_cluster_coefficient:", G.global_cluster_coefficient())
    print("local_cluster_coefficient:", G.local_cluster_coefficient())
    print("rich_club_coefficient:", G.rich_club_coefficient())
    print("page_rank:", G.page_rank())
    print("modularity_based_community:", G.modularity_based_community())
    print("fluid_community:", G.fluid_community())

    b = G.minimum_spanning_arborescence()
    print("minimum_spanning_arborescence", b)
