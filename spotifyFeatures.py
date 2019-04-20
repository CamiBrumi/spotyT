import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# cid ="4f1f4d0e715e4832bd412a58caa0d842"
# secret = "facf04c849754f58959d2ca3d525d60c"
#
# client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#
# df_tracks = pd.read_csv('data/empty.csv')
# df_tracks.dropna(inplace=True)

# # The following code is reused from https://github.com/tgel0/spotify-data/blob/master/notebooks/SpotifyDataRetrieval.ipynb
#
# rows = []
# batchsize = 100
# None_counter = 0
#
# for i in range(0, len(df_tracks['URL']), batchsize):
#     batch = df_tracks['URL'][i:i + batchsize]
#     # print(batch)
#     feature_results = sp.audio_features(batch)
#     for i, t in enumerate(feature_results):
#         if t == None:
#             None_counter = None_counter + 1
#         else:
#             rows.append(t)
#
# df = pd.DataFrame.from_records(rows)
# df.to_csv('data/not-empty.csv', index=False)
#
# print('Number of tracks where no audio features were available:', None_counter)

def translateKey(key):
    switch = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }
    return switch.get(key, '')

df = pd.read_csv('data/not-empty.csv')

columns_to_drop = ['analysis_url','track_href','type','uri']
df.drop(columns_to_drop, axis=1,inplace=True)

df.rename(columns={'id': 'track_id'}, inplace=True)

df['mode'] = df['mode'].apply(lambda x: 'Major' if x == 1 else 'Minor')
df['key'] = df['key'].apply(lambda x: translateKey(x))

df.to_csv('data/new-songs.csv', index=False)
