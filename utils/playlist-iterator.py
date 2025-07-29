from pytubefix import Playlist
import re
import pandas as pd
from dotenv import load_dotenv
import os
import json

load_dotenv()
playlist_urls = json.loads(os.environ.get('PLAYLISTS'))

# Iterate through genres
for pl in playlist_urls:
    genre = f"{pl}".lower()
    file_path = f"playlists/{genre}.json"

    # Make pandas datafame
    if os.path.exists(file_path):
        df = pd.read_json(file_path, orient="records")
    else:
        df = pd.DataFrame(columns=["title", "author", "url"])
        
    # Iterate through playlists
    for link in playlist_urls[pl]:
        playlist = Playlist(link)
        
        # Iterate through videos in playlist
        for i in range(len(playlist.videos)):
            try:
                parts = re.split(r'-|–| \"|\|| \(| \[', playlist.videos[i].title)
            except Exception as e:
                print(e)
                continue
            else:
                url = playlist.videos[i].watch_url
                if url not in df["url"].values:
                    author = parts[0].strip()
                    title = parts[1].strip() if len(parts) > 1 else playlist.videos[i].title
                    df.loc[len(df)] = {"title": title, "author": author, "url": playlist.videos[i].watch_url}
    df.to_json(file_path, orient="records", indent=4)
