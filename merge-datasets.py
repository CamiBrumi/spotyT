import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

COUNTRIES = False

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

emptydf = finaldf[finaldf.isnull().any(axis=1)].copy()
emptydf.drop(labels=['acousticness','danceability','duration_ms','energy',
                'instrumentalness','liveness','loudness','speechiness',
                'tempo','valence','Opera','A Capella','Alternative',
                'Blues','Dance','Pop','Electronic','R&B','Children’s Music',
                'Folk','Anime','Rap','Classical','Reggae','Hip-Hop',
                'Comedy','Country','Reggaeton','Ska','Indie','Rock','Soul',
                'Soundtrack','Jazz','World','Movie','time_sig_x','time_sig_y','Position','Region','key_x','key_y'], inplace=True, axis=1)
emptydf.replace('https://open.spotify.com/track/', '', regex=True, inplace=True)
print(emptydf.dtypes)
emptydf.to_csv(path_or_buf='data/empty.csv')
finaldf.dropna(inplace=True)

country = ''
if COUNTRIES:
       finaldf = pd.get_dummies(finaldf, columns=['Region'])
       country = '_countries'

columns = finaldf.columns.values

finaldf_train, finaldf_test = train_test_split(finaldf, test_size=2000)

finaldf_train.to_csv(path_or_buf='data/finaldata_train'+country+'.csv',index=False)
finaldf_test.to_csv(path_or_buf='data/finaldata_test'+country+'.csv',index=False)

# finaldf.set_index('URL')
if COUNTRIES:
       mapper = DataFrameMapper([
              (['Position','URL'], None),
              (['acousticness','danceability','duration_ms','energy',
                'instrumentalness','liveness','loudness','speechiness',
                'tempo','valence','Opera','A Capella','Alternative',
                'Blues','Dance','Pop','Electronic','R&B','Children’s Music',
                'Folk','Anime','Rap','Classical','Reggae','Hip-Hop',
                'Comedy','Country','Reggaeton','Ska','Indie','Rock','Soul',
                'Soundtrack','Jazz','World','Movie','time_sig_x','time_sig_y',
                'key_x','key_y'], preprocessing.MinMaxScaler()),
              (['Region_be','Region_ca',
                'Region_ch','Region_cl','Region_ec','Region_ee',
                'Region_es','Region_fi','Region_gb','Region_gr',
                'Region_hk','Region_hn','Region_ie','Region_is',
                'Region_it','Region_jp','Region_nz','Region_pa',
                'Region_pe','Region_pl','Region_py','Region_se',
                'Region_sg','Region_sk','Region_sv','Region_tw',
                'Region_uy'], None)
       ], input_df=True)
else:
       mapper = DataFrameMapper([
              ('Position', None),
              ('URL', None),
              ('Region', None),
       ], input_df=True, default=preprocessing.MinMaxScaler())
normalizeddf_train = pd.DataFrame(mapper.fit_transform(finaldf_train))
normalizeddf_test = pd.DataFrame(mapper.fit_transform(finaldf_test))
normalizeddf_train.columns = columns
normalizeddf_test.columns = columns

normalizeddf_train.to_csv(path_or_buf='data/normalizeddata_train'+country+'.csv',index=False)
normalizeddf_test.to_csv(path_or_buf='data/normalizeddata_test'+country+'.csv',index=False)