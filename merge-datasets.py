import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper, cross_val_score
import numpy as np

chartdf = pd.read_csv('data/engineered-data.csv')
trackdf = pd.read_csv('data/engineered-track-data.csv')

namedf = chartdf[['Track Name','Artist','URL']].copy()

trackdf = trackdf.drop(labels=['artist_name','track_name','popularity'], axis=1)
trackdf.columns = ['URL', 'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
       'valence', 'Opera', 'A Capella', 'Alternative', 'Blues', 'Dance', 'Pop',
       'Electronic', 'R&B', 'Children’s Music', 'Folk', 'Anime', 'Rap',
       'Classical', 'Reggae', 'Hip-Hop', 'Comedy', 'Country', 'Reggaeton',
       'Ska', 'Indie', 'Rock', 'Soul', 'Soundtrack', 'Jazz', 'World', 'Movie',
       'time_sig_x', 'time_sig_y', 'key_x', 'key_y']
chartdf = chartdf.drop(labels=['Track Name','Artist','Streams','Date'], axis=1)

print(chartdf.columns)
print(trackdf.columns)
print(namedf.columns)


trackdf['URL'] = 'https://open.spotify.com/track/' + \
                      trackdf['URL'].astype(str)
print(trackdf['URL'].iloc[9999])

finaldf = chartdf.merge(trackdf.drop_duplicates(), how='left')
print(finaldf.columns)

finaldf.loc[(finaldf['Position'] <= 10), 'Position'] = 10
finaldf.loc[(finaldf['Position'] <= 50)  & (finaldf['Position'] > 10), 'Position'] = 50
finaldf.loc[(finaldf['Position'] <= 100) & (finaldf['Position'] > 50), 'Position'] = 100
finaldf.loc[(finaldf['Position'] <= 150) & (finaldf['Position'] > 100), 'Position'] = 150
finaldf.loc[(finaldf['Position'] >  150), 'Position'] = 200

finaldf.to_csv(path_or_buf='data/finaldata.csv',index=False)

# finaldf.set_index('URL')

mapper = DataFrameMapper([
       ('Position', None),
       ('URL', None),
       ('Region', None),
], input_df=True, default=preprocessing.MinMaxScaler())
normalizeddf = pd.DataFrame(mapper.fit_transform(finaldf))


normalizeddf.to_csv(path_or_buf='data/normalizeddata.csv',index=False)