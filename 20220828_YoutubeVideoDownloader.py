import requests
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import urllib.request
from contextlib import suppress
import json

#Description:
#This programm downloads the audio from youtube videos from a provided URL and
#then allows to cut it according to the provided timestamps if wanted

#Get the Youtube Url

def getID(url):
          
    if url.startswith(('youtu', 'www')):
        url = 'http://' + url

    query = urlparse(url)

    if 'youtube' in query.hostname:
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        elif query.path.startswith(('/embed/', '/v/')):
            return query.path.split('/')[2]
    elif 'youtu.be' in query.hostname:
        return query.path[1:]
    else:
        raise ValueError

def downloadVideo(ID): 
    url = "https://youtube-mp3-download1.p.rapidapi.com/dl"

    querystring = {"id":ID}

    headers = {
        "X-RapidAPI-Key": "455df0560amsh06a4a1fe47ef87fp1e4476jsne08671195039",
        "X-RapidAPI-Host": "youtube-mp3-download1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    
    x = json.loads(response.text)

    return x


print("Lets go")
url = input("Please enter the URL here: ")
ID = getID(url)
VideoDetails = downloadVideo(ID)
DownloadLink= VideoDetails["link"]
print(DownloadLink)

# open a connection to a URL using urllib
webUrl = urllib.request.urlopen(DownloadLink)

#get the result code and print it
print("result code: " + str(webUrl.getcode()))

# read the data from the URL and print it
#data = webUrl.read()
#print (data)

#next steps: 1) it only sends you a link to a website wihich when openeed downoaed the video
#for that retrieve the website, go to it and download it, make sure to get the progress shown in the file 
#show where it was saved
