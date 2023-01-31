from .base_representer import BaseEmbeddingRepresenter
from src.rta.utils import mean_FM

class FMRepresenter(BaseEmbeddingRepresenter):
        
    def forward(self, x):
        albs = self.song_album[x]
        arts = self.song_artist[x]
        pops = self.data_manager.get_pop_bucket(self.song_pop[x]).int()
        durs = self.data_manager.get_duration_bucket(self.song_dur[x]).int()

        X = self.embedding(x)
        X_albs = self.album_embedding(albs)
        X_arts = self.artist_embedding(arts)
        X_pops = self.pop_embedding(pops)
        X_durs = self.dur_embedding(durs)

        return mean_FM((X, X_albs, X_arts, X_pops, X_durs))