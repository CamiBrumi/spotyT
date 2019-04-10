import pandas as pd
filename = 'SpotifyFeatures.csv'
names = ['genre','artist_name','track_name','track_id','popularity','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']
df = pd.read_csv(filename)
df.replace(["Minor", "Major"], [-1, 1], inplace = True)
print(df)