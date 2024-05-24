import numpy as np
from scipy.sparse import csr_array, load_npz
import pandas as pd


class LocationDistribution:
    """Class for storing and sampling a spatial distribution
    with uber H3 tiles
    """
    def __init__(self, df_file):
        """
        constructor method

        Parameters
        ----------
        df_file : str
            path to a csv file with columns: probability, h3
        """

        df = pd.read_csv(df_file)
        self.p = df.probability.values
        self.h3 = df.h3.values
        self.h3_sorted = np.sort(self.h3.copy())
        self.idx_map = {x: i for i, x in enumerate(self.h3_sorted)}

    @property
    def locations(self):
        """Get list of Uber H3 files contained in the distribution

        Returns
        -------
        list(string)
            list of uber h3 tiles contained in the distribution
        """
        return self.h3_sorted

    def __iter__(self):
        while True:
            yield self.sample()

    def sample(self):
        """Choose a random h3 tile from the distribution using the
        probability columns as weights

        Returns
        -------
        string
            chosen h3 tile
        """
        return np.random.choice(self.h3, p=self.p)

    def get_index(self, x):
        """Get the index of the uber h3 tile in the data frame

        Parameters
        ----------
        x : string
            uber h3 tile

        Returns
        -------
        int
            index of the uber h3 tile in the dataframe
        """
        return self.idx_map[x]


class PairedLocationSampler:
    """Sample paired locations using two distributions and a joint
    probability distribution.
    """
    def __init__(
            self,
            primary_location_dist,
            secondary_location_dist,
            pairwise_probabilities):
        """
        Constructor method
        
        Parameters
        ----------
        primary_location_dist : str
            Path to csv with primary sampling distribution
        secondary_location_dist : str
            Path to csv with secondary sampling distribution
        pairwise_probabilities : str
            Path to *.npz file containing a sparse array of 
            pairwise probabilities between the primary and secondary distributions
        """
        self.primary_distribution = LocationDistribution(primary_location_dist)
        self.secondary_distribution = LocationDistribution(
            secondary_location_dist)
        self.pairwise_probabilities = load_npz(pairwise_probabilities)

    def sample(self):
        """Sample a pair of locations using the primary and joint
        proability distributions

        Returns
        -------
        tuple(str, str)
            - prime_loc : uber h3 tile of primary location
            - secondary_loc : uber h3 tile of secondary location
        """
        prime_loc = self.primary_distribution.sample()
        prime_loc_idx = self.primary_distribution.get_index(prime_loc)
        p = self.pairwise_probabilities[prime_loc_idx].toarray()[0]
        p = p/p.sum()
        secondary_idx = np.random.choice(
            len(self.secondary_distribution.locations),
            p=p)
        # print('WARNING NOT USING PAIRWISE DIST')
        # secondary_idx = np.random.choice(
        #     len(self.secondary_distribution.locations))
        secondary_loc = self.secondary_distribution.locations[secondary_idx]
        return prime_loc, secondary_loc

    def __iter__(self):
        while True:
            yield self.sample()


class SumoPairedLocationSampler(PairedLocationSampler):
    """Sample paired locations in a SUMO road network using two
    distributions and a joint probability distribution.
    """
    def __init__(self, primary_location_dist, secondary_location_dist,
                 pairwise_probabilities, edge_map):
        """
        Constructor method
        
        Parameters
        ----------
        primary_location_dist : str
            Path to csv with primary sampling distribution
        secondary_location_dist : str
            Path to csv with secondary sampling distribution
        pairwise_probabilities : str
            Path to *.npz file containing a sparse array of 
            pairwise probabilities between the primary and secondary distributions
        edge_map : str
            path to a csv file with columns h3, id_road, length_along_edge
        """
        super(
            SumoPairedLocationSampler,
            self).__init__(
            primary_location_dist,
            secondary_location_dist,
            pairwise_probabilities)

        self.edge_df = pd.read_csv(edge_map)
        self.edge_df = self.edge_df.sort_values(['h3'])
        self.h3_index = pd.Index(self.edge_df.h3)
        self.valid_h3 = self.edge_df.h3.unique()

    def sample(self):
        """Sample a pair of locations in the SUMO road network
        using a primary distributiona and joint probability
        distribution.

        Returns
        -------
        tuple(string, float)
            - prime_loc : primary location as SUMO edge, and length along edge
            - second_loc : secondary location as SUMO edge, and length along edge
        """
        while True:
            prim_h3, second_h3 = super().sample()
            if (prim_h3 in self.valid_h3) and (second_h3 in self.valid_h3):
                break
        prime_loc = self.sample_h3(prim_h3)
        second_loc = self.sample_h3(second_h3)
        return prime_loc, second_loc
    
    def sample_h3(self, h3_loc):
        """Sample a random location within an Uber h3 tile.

        Parameters
        ----------
        h3_loc : string
            h3 tile to sample from

        Returns
        -------
        tuple(string, float)
            - road_id : Location in the h3 tile as encoded as an edge
            - length : length along the edge.
        """
        sub = self.edge_df.loc[self.h3_index.isin([h3_loc])]
        idx = np.random.choice(len(sub))
        road_id = sub.id_road.values[idx]
        length = sub.length_along_edge.values[idx]
        return (road_id, length)