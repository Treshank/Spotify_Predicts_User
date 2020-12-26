import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import re

scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id='', client_secret='', redirect_uri='http://127.0.0.1:9090'))


def track_info(tracks):
    idlist = []
    for i, item in enumerate(tracks['items']):
        tracks = item['track']
        idlist.append([tracks['id'], tracks['name'], tracks['artists'][0]['name']])
        # print(" %d %32.32s %s" % (i, track['artists'][0]['name'],track['name']))
    return idlist


df = pd.DataFrame(columns=['id', 'name', 'artists', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])


def music(uri, name, artists):
    global df
    response = str(sp.audio_features(uri)[0])
    response = response[1:-1]
    response = response.split(",")
    response.pop(11)
    response.pop(11)
    response.pop(11)
    response.pop(11)
    response.pop(11)
    features = [uri, name, artists]
    for x in response:
        m = re.search("\d",x)
        if m:
            if "," in x:
                x = x[:-1]
            if x[m.start()-1] == "-":
                feature = x[m.start()-1:]
            else:
                feature = x[m.start():]
            try:
                feature = float(feature)
            except ValueError:
                feature = feature
            features.append(feature)
    df.loc[df.shape[0]] = [features[0][:-1], features[1], features[2], features[3], features[4], features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13],features[14],features[15]]


r1 = sp.current_user_saved_tracks(limit=50)
r2 = sp.current_user_saved_tracks(limit=50, offset=50)
for idx, item in enumerate(r1['items']):
    idlist = []
    track = item['track']
    idlist.append([track['id'], track['name'], track['artists'][0]['name']])
    for x in idlist:
        music(x[0], x[1], x[2])

for idx, item in enumerate(r2['items']):
    idlist = []
    track = item['track']
    idlist.append([track['id'], track['name'], track['artists'][0]['name']])
    for x in idlist:
        music(x[0], x[1], x[2])

df.to_csv('Tresh.csv', index=False)
