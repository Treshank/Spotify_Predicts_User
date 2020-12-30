import pandas as pd


treshank = pd.read_csv("Datasets/Treshank.csv")
tushar = pd.read_csv("Datasets/Tushar.csv")
vaibhav = pd.read_csv("Datasets/Vaibhav.csv")

df = pd.DataFrame(columns=['owner', 'id', 'name', 'artists', 'danceability', 'energy', 'key', 'loudness', 'mode',
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                           'duration_ms', 'time_signature'])

for index, row in treshank.iterrows():
    df.loc[df.shape[0]] = ['Treshank', row['id'], row['name'], row['artists'], row['danceability'], row['energy'], row['key'], row['loudness'], row['mode'], row['speechiness'], row['acousticness'], row['instrumentalness'], row['liveness'], row['valence'], row['tempo'], row['duration_ms'], row['time_signature']]

for index, row in tushar.iterrows():
    df.loc[df.shape[0]] = ['Tushar', row['id'], row['name'], row['artists'], row['danceability'], row['energy'], row['key'], row['loudness'], row['mode'], row['speechiness'], row['acousticness'], row['instrumentalness'], row['liveness'], row['valence'], row['tempo'], row['duration_ms'], row['time_signature']]

for index, row in vaibhav.iterrows():
    df.loc[df.shape[0]] = ['Vaibhav', row['id'], row['name'], row['artists'], row['danceability'], row['energy'], row['key'], row['loudness'], row['mode'], row['speechiness'], row['acousticness'], row['instrumentalness'], row['liveness'], row['valence'], row['tempo'], row['duration_ms'], row['time_signature']]


df.to_csv("Datasets/merged.csv", index=False)
print("Success!")