from datetime import datetime
from time import sleep
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import subprocess
import webbrowser
import ddgs
import python_weather
import asyncio
import os

load_dotenv()




#------INITIALIZATION-------#
BROWSER_PATHS = {
    "firefox incognito" : "C:\\Program Files\\Mozilla Firefox\\private_browsing.exe",
    "firefox regular" : "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
    "chrome" : "C:\\Program Files\\Google\Chrome\\Application\\chrome.exe"
}
for name, path in BROWSER_PATHS.items():
    webbrowser.register(name, None, webbrowser.BackgroundBrowser(path))



def getTime():
    """
    returns the current time and date
    """
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")


def web_search(query: str):
    """
    performs a web search and returns the result
    :param query: the web search query
    :return: the search result
    """
    try:
        results = ddgs.DDGS().text(query=query, max_results=3)
        results_array = [{'title': r['title'], 'description': r['body'], 'url': r['href']} for r in results]
        if results_array:
            return "\n".join([f"Title: {r['title']}\nDescription: {r['description']}" for r in results_array])
    except Exception as e:
        print(f'error in web search: {query}, error: {e}')
    return f'No results found for {query}'


async def weather_grabber(location):
    """
    returns the current weather and date
    :param location: the location of the weather
    :return: returns the weather for the given location
    """
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        weather = await client.get(location)

    return f"The temperature is {weather.temperature} and is {weather.kind} with a description of {weather.description} and a windspeed of {weather.wind_speed} mph and coordinates of {weather.coordinates}"


def get_weather(location):
    return asyncio.run(weather_grabber(location))

def start_timer(duration):
    """
    do not use this it fucking sucks but it's not cool enough to get fixed
    :param duration: its a timer take a guess
    :return: ur timer is done go do whatever u need to now.
    """
    print(f"Jarvis is starting a timer for {duration} seconds...")
    asyncio.run(asyncio.sleep(duration-15))
    print("Timer has 15 seconds left")
    asyncio.run(asyncio.sleep(15))
    return f"Timer for {duration} seconds has finished, Sir."


def search_files(query, result_amount: int = 5, otherFunction: bool = False):
    command = ["es.exe", "-s", "-n", str(result_amount), query]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split("\n")
        if otherFunction:
            return result
        return "\n".join(f"- {path}" for path in result)
    except Exception as e:
        print(f'error in search_files: {query}, error: {e}')
        return f'error in search_files: {query}, error: {e}'


def run_program(query):
    if query.endswith(".exe"):
        result = search_files(query, otherFunction=True)
    else:
        result = search_files(f"{query}.exe", otherFunction=True)
    try:
        for path in result:
            if path.find('.exe') != -1:
                subprocess.run([path])
                return 'process ran successfully'
        return 'process could not be ran, or possibly not found.'
    except Exception as e:
        print(f'error in program: {query}, error: {e}')
        return 'program could not be ran'

def open_tabs(url: str, browser: str = 'firefox regular'):
    try:
        webbrowser.get(browser).open_new_tab(url)
    except webbrowser.Error as e:
        return f"error could not find or open the requested browser: {browser}, with error message: {e}"
    except Exception as e:
        return f"error {e}"


#--------SPOTIFY CONTROL-----#


def _authenticate_spotify():
    """
    Authenticate with Spotify and return a Spotify client.
    """
    scope = (
        "user-read-playback-state user-modify-playback-state "
        "user-read-currently-playing user-read-playback-position "
        "user-library-read user-library-modify user-read-email user-read-private"
    )
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri="http://127.0.0.1:8888/callback",
        scope=scope,
        username=os.getenv("SPOTIFY_USERNAME"),
    )
    return spotipy.Spotify(auth_manager=auth_manager)


_spotify_client = _authenticate_spotify()


def _active_devices_id_spotify(_spotify_client):
    devices = _spotify_client.devices()
    active_device = None
    for device in devices['devices']:
        if device['is_active']:
            active_device = device['id']
            break
    if not active_device and devices['devices']:
        active_device = devices['devices'][0]['id']
        print(f"no active devices found. resorting to fallback device {devices['devices'][0]['name']}")
    return active_device


def main_controller_spotify(action: str, title: str = None, form: str = 'track', artist: str = None, amount: int = 1):
    """
    control spotify playback functions. if no device found, use device issuing commands.
    :param action: what is going to be performed.
    :param title: name of the spotify playback title. Should not be filled out if simply unpausing
    :param form: what playback medium is being listened to
    :return: an action.
    """
    try:
        active_device = _active_devices_id_spotify(_spotify_client)
        # noinspection PyInconsistentReturns
        if action == 'play':
            if title:
                _play_music_spotify(title, form, artist, active_device)
                return f"playing title {title} of type: {form}"
            else:
                _spotify_client.start_playback(active_device)
                return "Music Unpaused"
        elif action == 'get_track_info':
            return _get_track_info()
        elif action == 'pause':
            _spotify_client.pause_playback(active_device)
            return "Music Paused"
        elif action == 'skip_to_next_track':
            for i in range(amount):
                _spotify_client.next_track(active_device)
                sleep(1)
            return f"{amount} Tracks Skipped"
        elif action == 'skip_to_previous_track':
            _spotify_client.previous_track(active_device)
            return "Backed up to last track"
        else:
            return f"Unknown spotify action attempted {action}"
    except spotipy.SpotifyException as e:
        return f"Spotipy Error: {str(e)}"

def _get_track_info():
    global _spotify_client
    try:
        current_track = _spotify_client.current_user_playing_track()['item']
        track_title = current_track['name']
        artist_name = current_track['artists'][0]['name']
        album_name = current_track['album']['name']
        next_song = _spotify_client.queue()['queue'][0]['name']
        return f"Currently playing: {track_title} by {artist_name} from the album {album_name}. The next song is {next_song}"

    except spotipy.SpotifyException as e:
        return f"Error getting current track: {str(e)}"


def _play_music_spotify(title: str, form: str, artist: str, active_device_id: str):
    """
    play a song from spotify
    :param query: the song to search for
    :return: plays the song that is requested
    """
    if form == 'album':
        if artist:
            search = _spotify_client.search(f"album:{title} artist:{artist}", type='album')['albums']['items'][0]['uri']
            _spotify_client.start_playback(active_device_id, search)
            return None
        search = _spotify_client.search(f"album:{title}", type='album')['albums']['items'][0]['uri']
        _spotify_client.start_playback(active_device_id, search)
    elif form == 'track':
        if artist:
            search = [_spotify_client.search(f"track:{title} artist:{artist}")['tracks']['items'][0]['uri']]
            _spotify_client.start_playback(active_device_id, search)
            return None
        search = [_spotify_client.search(f"track:{title}")['tracks']['items'][0]['uri']]
        _spotify_client.start_playback(active_device_id, uris=search)
    elif form == 'playlist':
        search = _spotify_client.search(f"{title}", type='playlist')['playlists']['items'][0]['uri']
        _spotify_client.start_playback(active_device_id, search)
    return None



