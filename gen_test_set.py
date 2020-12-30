import sys
import spotipy
import spotipy.util as util
import re
import pandas as pd

scope = 'user-library-read'

sp = spotipy.Spotify(auth_manager=spotipy.SpotifyOAuth(scope=scope, client_id='f43cb43d7d5d48489253d3eb5a30a965',
                                                       client_secret='12e5999517774bc1b4cf8ab7a6dd6f99',
                                                       redirect_uri='http://127.0.0.1:9090'))

df = pd.DataFrame(
    columns=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
             'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])

uris = ['spotify:track:0jdso14vaFnpRazMLEZovF',
        'spotify:track:6zDs6zI94L761vd0cVScTT',
        'spotify:track:2f0pn9DkEJwAzXApnFh5cr',
        'spotify:track:6L4F4btXioJPhEwz6T7YYt',
        'spotify:track:6UelLqGlWMcVH1E5c4H7lY',
        'spotify:track:3Uo7WG0vmLQ07WB4BDwy7D',
        'spotify:track:2nZq5WQOW4FEPxCVTdNGfB',
        'spotify:track:4VrWlk8IQxevMvERoX08iC',
        'spotify:track:6or1bKJiZ06IlK0vFvY75k']


for uri in uris:
    response = str(sp.audio_features(uri)[0])
    response = response[1:-1]
    response = response.split(",")
    response.pop(11)
    response.pop(11)
    response.pop(11)
    response.pop(11)
    response.pop(11)
    features = []
    for x in response:
        m = re.search("\d", x)
        if m:
            if "," in x:
                x = x[:-1]
            if x[m.start() - 1] == "-":
                feature = x[m.start() - 1:]
            else:
                feature = x[m.start():]
            try:
                feature = float(feature)
            except ValueError:
                feature = feature
            features.append(feature)
    df.loc[df.shape[0]] = [features[0], features[1], features[2], features[3], features[4], features[5],
                           features[6], features[7], features[8], features[9], features[10], features[11],
                           features[12]]

df.to_csv('Datasets/testdf.csv', index=False)
print('Success!')