from __future__ import print_function
from __future__ import unicode_literals
from apiclient import discovery
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from oauth2client.tools import argparser
import datetime
import time
import youtube_dl

DEVELOPER_KEY = "AIzaSyA8pfZr7qtJ3_eWgvAev58HrKMLFHgBKaM"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
#######################################################################################
# Authored by WESLEY J. CHUN,
# Retrieved from https://stackoverflow.com/questions/41935427/cant-download-video-captions-using-youtube-api-v3-in-
# python?noredirect=1&lq=1 ,
# September 2017
SCOPES = 'https://www.googleapis.com/auth/youtube.force-ssl'
store = file.Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
    creds = tools.run_flow(flow, store)
YOUTUBE = discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, http=creds.authorize(Http()))
#######################################################################################

videos = []
successfulVideos=[]

def youtube_search_first(searchTerm):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=searchTerm,
        part="id,snippet",
        maxResults=50,
        relevanceLanguage="en",
        type="video",
        videoCaption="closedCaption",
        videoDuration="medium"
    ).execute()

    nextPageToken= search_response.get("nextPageToken")

    for search_result in search_response.get("items", []):
        videos.append("%s" % (search_result["id"]["videoId"]))
    return nextPageToken, videos

def youtube_search(searchTerm,nextPageToken):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=searchTerm,
        part="id,snippet",
        maxResults=50,
        pageToken=nextPageToken,
        relevanceLanguage="en",
        type="video",
        videoCaption="closedCaption",
        videoDuration="medium"
    ).execute()

    nextPageToken= search_response.get("nextPageToken")

    for search_result in search_response.get("items", []):
        videos.append("%s" % (search_result["id"]["videoId"]))

    return nextPageToken, videos

def downlaod_caption(vid):
    #######################################################################################
    # Authored by WESLEY J. CHUN,
    # Retrieved from https://stackoverflow.com/questions/41935427/cant-download-video-captions-using-youtube-api-v3-in-
    # python?noredirect=1&lq=1,
    # September 2017
    caption_info = YOUTUBE.captions().list(
        part='id', videoId=vid).execute().get('items', [])
    caption_str = YOUTUBE.captions().download(
        id=caption_info[0]['id'], tfmt='srt',tlang='en').execute()
    caption_data = caption_str.decode().split('\n\n')
    #######################################################################################
    starttimes=[]
    captions=[]
    for line in caption_data:
        if line.count('\n') > 1:
            idx, cap_time, caption = line.split('\n', 2)
            start_time=cap_time.split('-->')[0]
            start_time_strip = time.strptime(start_time.split(',')[0],'%H:%M:%S')
            start_time_seconds=datetime.timedelta(hours=start_time_strip.tm_hour,
                                                  minutes=start_time_strip.tm_min,
                                                  seconds=start_time_strip.tm_sec).total_seconds()
            start_time_milliseconds=int(float(start_time.split(',')[1]))
            start_time_seconds=start_time_seconds+start_time_milliseconds/1000
            start_time_seconds=round(start_time_seconds,3)
            starttimes.append(str(start_time_seconds))

            captions.append(caption)

            if(int(float(idx))==len(caption_data)-1):
                end_time=cap_time.split('-->')[1]
                end_time=end_time.strip()
                end_time_strip = time.strptime(end_time.split(',')[0],'%H:%M:%S')
                end_time_seconds=datetime.timedelta(hours=end_time_strip.tm_hour,
                                                    minutes=end_time_strip.tm_min,
                                                    seconds=end_time_strip.tm_sec).total_seconds()
                end_time_milliseconds=int(float(start_time.split(',')[1]))
                end_time_seconds=end_time_seconds+end_time_milliseconds/1000

                last_duration=float(end_time_seconds-start_time_seconds)
                last_duration=str(round(last_duration,3))

    durations=[None] * len(starttimes)
    for i in range(len(durations)-1):
        duration=float(starttimes[i+1])-float(starttimes[i])
        duration=round(duration,3)
        durations[i]=str(duration)
    durations[len(durations)-1]=last_duration

    with open('captions/'+vid+'.txt', 'w') as txtfile:
        for i in range(len(captions)):
            txtfile.write(str(i)+'--'+starttimes[i]+'--'+durations[i]+'--'+captions[i]+'\n')
    print('captions for '+vid+' downloaded')

def downlaod_audio(vid):
    #######################################################################################
    # Authored by Philipp Hagemeister,
    # Retrieved from https://stackoverflow.com/questions/27473526/download-only-audio-from-youtube-video-using-youtube-
    # dl-in-python-script,
    # September 2017
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': '/audio/%(id)s.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['http://www.youtube.com/watch?v='+vid])
    #######################################################################################
if __name__ == '__main__':
    argparser.add_argument("--searchterm",
                           help="Required; searchterm for video searching", default="Englsih")
    args = argparser.parse_args()
    searchterm=args.searchterm

    #Searching Videos
    nextPageToken, videos=youtube_search_first(searchterm)
    for i in range(10):
        try:
            nextPageToken, videos=youtube_search(searchterm, nextPageToken)
        except Exception as e:
            print (e)

    print(videos)

    #Donwloading captions
    for videoID in videos:
        try:
            downlaod_caption(videoID)
            successfulVideos.append(videoID)
        except Exception as e:
            print (e)

    #Donwloading audio
    for videoID in successfulVideos:
        try:
            downlaod_audio(videoID)
        except Exception as e:
            print (e)


