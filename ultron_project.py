import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser as wb
import os
import ctypes
import time
import random
import pyautogui
import pyjokes
import platform
import requests  # For APIs like weather and news
import smtplib  # For sending emails
import subprocess
import pytz
import math
import psutil
import cmath
import numpy as np
import sympy as sp
import pywhatkit as kit
import pywhatkit
import datetime
import calendar
import logging
import schedule
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
from google.auth.transport.requests import Request
import re
import glob
from bs4 import BeautifulSoup
import wikipediaapi
import sounddevice as sd
import wave
# API Keys (Replace with your credentials)
GOOGLE_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_custom_search_engine_id"
import json
import threading
import cv2
# import video_control
import vlc
import yt_dlp as youtube_dl
import sys
import webbrowser
import os
import threading
import time
import logging
import pygame
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import time

from PIL import Image

# Open an image
image = Image.open("jarvis_image.jpg")  # Replace with your image path





logo="""                                          
                                        
          ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██║██╔═══╝ ██║   ██║██║╚════██║
 ╚█████╔╝██║  ██║██║     ╚██████╔╝██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝╚══════╝

   """






# Initialize global variables
engine = pyttsx3.init()
engine.setProperty('rate', 150)


# Get available voices
voices = engine.getProperty('voices')

# Set Male Voice (Usually voices[0] is male, but check with print)
engine.setProperty('voice', voices[1].id)  # Change index if needed



instance = vlc.Instance('--no-xlib')
player = instance.media_player_new()
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300

is_playing = False
current_volume = 50
playlist = []
current_track_index = 0

# Logging setup
logging.basicConfig(filename='jarvis.log', level=logging.INFO, format='%(asctime)s - %(message)s')




# Google Calendar API setup
SCOPES = ['https://www.googleapis.com/auth/calendar.events']




# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




# Initialize pygame mixer for music playback
pygame.mixer.init()




def speak(text):
    """Convert text to speech"""
    logging.info(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()




def play_youtube_audio(song_name):
    """Search and play audio from YouTube"""
    global is_playing, playlist, current_track_index
    try:
        speak(f"Searching for {song_name} on YouTube")
        ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{song_name}", download=False)
            
            if not result or 'entries' not in result or len(result['entries']) == 0:
                speak("No results found")
                return False
            
            video = result['entries'][0]
            video_url = video['url']
            speak(f"Playing {video['title']}")
            
            media = instance.media_new(video_url)
            player.set_media(media)
            player.audio_set_volume(current_volume)
            if player.play() == -1:
                speak("Error playing the audio")
                return False
            
            is_playing = True
            playlist.append(video_url)
            current_track_index = len(playlist) - 1
            return True
            
    except Exception as e:
        logging.error(f"Error playing YouTube audio: {e}")
        speak("Failed to play the song")
        return False





def play_spotify_song(song_name):
    """Search and play a song on Spotify"""
    speak(f"Searching for {song_name} on Spotify")
    query = song_name.replace(" ", "%20")
    spotify_url = f"https://open.spotify.com/search/{query}"
    webbrowser.open(spotify_url)
    speak("Playing on Spotify")






def play_local_media(file_path):
    """Play local media file"""
    global is_playing, playlist, current_track_index
    if os.path.exists(file_path):
        speak(f"Playing {os.path.basename(file_path)}")
        media = instance.media_new(file_path)
        player.set_media(media)
        player.audio_set_volume(current_volume)
        player.play()
        is_playing = True
        playlist.append(file_path)
        current_track_index = len(playlist) - 1
    else:
        speak("File not found")






def pause_media():
    """Pause media playback"""
    global is_playing
    if player.is_playing():
        player.pause()
        is_playing = False
        speak("Media paused")
    else:
        speak("No media is playing")





def resume_media():
    """Resume media playback"""
    global is_playing
    if not player.is_playing() and is_playing:
        player.play()
        speak("Resuming media")
    else:
        speak("Media is already playing")





def stop_media():
    """Stop media playback"""
    global is_playing, playlist, current_track_index
    player.stop()
    is_playing = False
    playlist = []
    current_track_index = 0
    speak("Media stopped")





def set_volume(volume_level):

    """Set volume level (0-100)"""
    global current_volume
    if 0 <= volume_level <= 100:
        current_volume = volume_level
        player.audio_set_volume(volume_level)
        speak(f"Volume set to {volume_level}%")
    else:
        speak("Volume level must be between 0 and 100")







def increase_volume():
    """Increase volume by 10%"""
    set_volume(min(current_volume + 10, 100))







def decrease_volume():
    """Decrease volume by 10%"""
    set_volume(max(current_volume - 10, 0))










def next_track():
    """Play the next track in the playlist"""
    global current_track_index

    if current_track_index < len(playlist) - 1:
        current_track_index += 1
        media = instance.media_new(playlist[current_track_index])
        player.set_media(media)
        player.play()
        speak("Playing next track")
    else:
        speak("No more tracks in the playlist")





def search_files_by_content(directory, search_term, file_extension='.py'):
    """
    Search for files containing specific text content.
    
    :param directory: Directory to search in
    :param search_term: Text to search for in files
    :param file_extension: File extension to filter by
    :return: List of files containing the search term
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if search_term.lower() in f.read().lower():
                            matching_files.append(file_path)
                except UnicodeDecodeError:
                    continue
    return matching_files

def search_files_by_size(directory, min_size=0, max_size=float('inf'), file_extension='*'):
    """
    Search for files within a specific size range.
    
    :param directory: Directory to search in
    :param min_size: Minimum file size in bytes
    :param max_size: Maximum file size in bytes
    :param file_extension: File extension to filter by
    :return: List of files matching the size criteria
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file_extension == '*' or file.endswith(file_extension):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                if min_size <= file_size <= max_size:
                    matching_files.append((file_path, file_size))
    return matching_files


def search_files_by_date(directory, days_old=7, file_extension='*'):
    """
    Search for files modified within the last N days.
    
    :param directory: Directory to search in
    :param days_old: Maximum age of files in days
    :param file_extension: File extension to filter by
    :return: List of recently modified files
    """
    matching_files = []
    cutoff_time = time.time() - (days_old * 86400)
    for root, _, files in os.walk(directory):
        for file in files:
            if file_extension == '*' or file.endswith(file_extension):
                file_path = os.path.join(root, file)
                mod_time = os.path.getmtime(file_path)
                if mod_time >= cutoff_time:
                    matching_files.append((file_path, 
                                         datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')))
    return matching_files



def search_functions_with_docstrings(directory):
    """
    Search for functions along with their docstrings.
    
    :param directory: Directory to search in
    :return: Dictionary with file paths and function info (name, docstring)
    """
    functions_info = {}
    python_files = search_files(directory, '.py')
    
    for file in python_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find function definitions and their docstrings
        pattern = re.compile(
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\):\s*\n\s*"""(.*?)"""',
            re.DOTALL
        )
        matches = pattern.finditer(content)
        
        functions = []
        for match in matches:
            func_name = match.group(1)
            docstring = match.group(2).strip()
            functions.append({
                'name': func_name,
                'docstring': docstring
            })
        
        if functions:
            functions_info[file] = functions
    
    return functions_info

def search_functions_by_complexity(directory, threshold=10):
    """
    Find functions with high complexity (based on rough line count).
    
    :param directory: Directory to search in
    :param threshold: Minimum number of lines to consider complex
    :return: Dictionary of complex functions with line counts
    """
    complex_functions = {}
    python_files = search_files(directory, '.py')
    
    for file in python_files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_func = None
        func_start = 0
        functions = []
        
        for i, line in enumerate(lines):
            # Check for function definition
            if line.strip().startswith('def '):
                if current_func:
                    # Save previous function
                    func_lines = i - func_start
                    if func_lines >= threshold:
                        functions.append({
                            'name': current_func,
                            'lines': func_lines
                        })
                # Start new function
                current_func = line.split('def ')[1].split('(')[0].strip()
                func_start = i
        
        if current_func:
            # Save last function in file
            func_lines = len(lines) - func_start
            if func_lines >= threshold:
                functions.append({
                    'name': current_func,
                    'lines': func_lines
                })
        
        if functions:
            complex_functions[file] = functions
    
    return complex_functions








def handle_file_search_command(query):
    """Process file search commands from voice input."""
    if "search functions" in query:
        speak("Please specify the directory to search in.")
        directory = takecommand()
        
        if directory:
            if "with docstrings" in query:
                results = search_functions_with_docstrings(directory)
                if results:
                    response = "Found functions with docstrings:\n"
                    for file, funcs in results.items():
                        response += f"In {os.path.basename(file)}:\n"
                        for func in funcs:
                            response += f"- {func['name']}: {func['docstring']}\n"
                    speak(response)
                else:
                    speak("No functions with docstrings found.")
            
            elif "complex" in query:
                threshold = 15  # Default threshold
                if "threshold" in query:
                    try:
                        threshold = int(query.split("threshold")[1].strip().split()[0])
                    except:
                        pass
                results = search_functions_by_complexity(directory, threshold)
                if results:
                    response = f"Found complex functions (>{threshold} lines):\n"
                    for file, funcs in results.items():
                        response += f"In {os.path.basename(file)}:\n"
                        for func in funcs:
                            response += f"- {func['name']}: {func['lines']} lines\n"
                    speak(response)
                else:
                    speak(f"No complex functions found with threshold {threshold}.")
            
            else:
                functions = search_functions_in_directory(directory)
                if functions:
                    response = "Found the following functions:\n"
                    for file, funcs in functions.items():
                        response += f"In {os.path.basename(file)}, functions: {', '.join(funcs)}\n"
                    speak(response)
                else:
                    speak("No functions found in the Python files.")
        else:
            speak("No directory specified.")

    elif "search files by content" in query:
        speak("Please specify the directory and search term.")
        directory = takecommand()
        speak("What content should I search for?")
        search_term = takecommand()
        if directory and search_term:
            results = search_files_by_content(directory, search_term)
            if results:
                speak(f"Found {len(results)} files containing '{search_term}':")
                for file in results[:5]:  # Limit to first 5 results
                    speak(os.path.basename(file))
            else:
                speak(f"No files found containing '{search_term}'.")

    elif "search recent files" in query:
        days = 7  # Default
        if "last" in query:
            try:
                days = int(query.split("last")[1].strip().split()[0])
            except:
                pass
        speak(f"Searching for files modified in the last {days} days.")
        directory = takecommand()
        if directory:
            results = search_files_by_date(directory, days)
            if results:
                speak(f"Found {len(results)} recently modified files:")
                for file, mod_date in results[:5]:  # Limit to first 5 results
                    speak(f"{os.path.basename(file)} modified on {mod_date}")
            else:
                speak(f"No files modified in the last {days} days found.")





def get_directory_size(directory):
    """Calculate total size of a directory in MB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)  # MB

def count_code_lines(directory, file_extension='.py'):
    """Count total lines of code in a directory."""
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                with open(os.path.join(root, file), 'r') as f:
                    total_lines += len(f.readlines())
    return total_lines






def previous_track():   
    """Play the previous track in the playlist"""
    global current_track_index
    if current_track_index > 0:
        current_track_index -= 1
        media = instance.media_new(playlist[current_track_index])
        player.set_media(media)
        player.play()
        speak("Playing previous track")


    else:
        speak("This is the first track")




def handles_play_command(query):  # Accept the 'query' parameter
    """Process play requests"""
    if "on spotify" in query:
        song_name = query.replace("play", "").replace("on spotify", "").strip()
        play_spotify_song(song_name)


    elif "from local" in query:
        file_path = query.replace("play", "").replace("from local", "").strip()
        play_local_media(file_path)


    else:
        song_name = query.replace("play", "").strip()
        if not play_youtube_audio(song_name):
            speak("Failed to play that song")




def handle_command_mode():
    """Handle interactive command session"""
    speak("How can I assist you?")
    while True:
        query = takecommand()
        if not query:
            continue
        
        if "play" in query:
            handles_play_command(query)
        elif "pause" in query:
            pause_media()
        elif "resume" in query:
            resume_media()
        elif "stop" in query:
            stop_media()
        elif "next" in query:
            next_track()
        elif "previous" in query:
            previous_track()
        elif "volume up" in query:
            increase_volume()
        elif "volume down" in query:
            decrease_volume()
        elif "set volume to" in query:
            try:
                volume_level = int(query.replace("set volume to", "").strip())
                set_volume(volume_level)
            except ValueError:
                speak("Please specify a valid volume level between 0 and 100")
        elif "exit" in query or "quit" in  query:
            speak("Goodbye!")
            exist=False
        else:
            speak("Command not recognized")




def run_assistant():
    """Main execution loop without wake word"""
    speak("Jarvis Assistant initialized")
    exist = True  # Keep assistant running

    while exist:
        try:
            handle_command_mode()  # Directly enter command mode
        except KeyboardInterrupt:
            speak("Shutting down")
            player.stop()
            sys.exit()





def listen_for_wake_word():
    """Detect 'Hey Jarvis' wake phrase"""
    with sr.Microphone() as source:
        print("Waiting for wake word...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)
            phrase = recognizer.recognize_google(audio).lower()
            print(f"Wake word detected: {phrase}")
            return phrase
        except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError):
            return ""


def run_assistants():
    """Main execution loop"""
    speak("Jarvis Assistant initialized")
    exist=True;
    while exist:
        try:
            wake_word = listen_for_wake_word()
            if "hey " in wake_word:
                handle_command_mode()
        except KeyboardInterrupt:
            speak("Shutting down")
            player.stop()
            sys.exit()




def extract_filename(command):
    """Function to extract filename from the command"""
    # Allow multiple file extensions and simpler "open [filename]" commands
    pattern = r"open (\S+\.?\S*.py)"
    match = re.search(pattern, command)
    if match:
        return match.group(1)
    else:
        print("No valid command found.")
        return None






def open_file_in_vscode(file_name):
    """Function to open the file in VS Code"""
    try:
        subprocess.run(["code", file_name], check=True)
        print(f"Opening {file_name} in VS Code...")
    except subprocess.CalledProcessError:
        print(f"Failed to open {file_name} in VS Code.")







def search_functions_in_file(file_path):
    """
    Search for function definitions in a Python file.
    
    :param file_path: Path to the Python file.
    :return: List of function names found in the file.
    """
    functions = []
    with open(file_path, 'r') as file:
        # Search for lines starting with 'def' (Python function definitions)
        for line in file:
            match = re.match(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            if match:
                functions.append(match.group(1))
    return functions





def search_files(directory, file_extension='*'):
    """
    Search for files with a specific extension in a given directory.
    
    :param directory: Directory to search in.
    :param file_extension: The file extension to search for (e.g., '.py').
    :return: List of file paths that match the search criteria.
    """
    # Create the search pattern based on the extension
    search_pattern = os.path.join(directory, f'**/*{file_extension}')
    
    # Use glob with recursive pattern matching
    return glob.glob(search_pattern, recursive=True)










def search_functions_in_directory(directory):
    """
    Search for all Python files in a directory and list their functions.
    
    :param directory: Directory to search in.
    :return: Dictionary where the keys are file paths and the values are lists of function names.
    """
    python_files = search_files(directory, '.py')  # Search for Python files
    functions_in_files = {}

    for file in python_files:
        functions = search_functions_in_file(file)  # Search functions in the file
        if functions:
            functions_in_files[file] = functions

    return functions_in_files









def close_browser_website(browser_name=None):
    """
    Close the browser or a specific tab in the browser.
    
    :param browser_name: Optionally specify the browser window to close (e.g., 'Chrome', 'Firefox').
    """
    try:
        # Platform-specific key combinations
        current_platform = platform.system()
        if current_platform == 'Windows':
            close_tab_key = 'ctrl+w'
            close_window_key = 'alt+f4'
        elif current_platform == 'Darwin':  # macOS
            close_tab_key = 'command+w'
            close_window_key = 'command+q'
        else:  # Linux (most common)
            close_tab_key = 'ctrl+w'
            close_window_key = 'alt+f4'

        # Log the action
        logging.info("Attempting to close the browser or tab...")

        # If a specific browser is given, attempt to focus on it
        if browser_name:
            logging.info(f"Trying to close browser: {browser_name}")
            # Additional logic here could be added if you want to focus a specific browser using pygetwindow or other libraries.

        # Activate the browser window (switch to it)
        pyautogui.hotkey('alt', 'tab')  # Switch to the active browser window
        time.sleep(1)

        # Close the current tab
        pyautogui.hotkey(*close_tab_key.split('+'))  # Close the tab (Ctrl+W or Command+W)
        time.sleep(1)

        # Close the entire browser window if needed (Alt+F4 or Command+Q)
        pyautogui.hotkey(*close_window_key.split('+'))  # Close the entire window
        time.sleep(1)

        # Log success
        logging.info("Browser or tab closed successfully.")

    except Exception as e:
        logging.error(f"An error occurred while closing the browser: {e}")
        raise









# add battery status function 
def get_battery_status():
    """
    Function to check the battery status and return details.
    """
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery status not available on this system."
    
    percent = battery.percent
    charging = battery.power_plugged  # True if charging, False otherwise
    time_left = battery.secsleft if not charging else None

    # Prepare the status message
    status = f"Battery is at {percent}%."
    if charging:
        status += " The laptop is currently charging."
    else:
        status += f" Estimated time left: {time_left // 60} minutes." if time_left != psutil.POWER_TIME_UNLIMITED else " Time left is unlimited."

    return status





# Load settings from a JSON file
def load_settings():
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            return json.load(f)
    else:
        return {
            "output_dir": "recordings",
            "default_duration": 10,
            "language": "en-US"
        }
    






# Save settings to a JSON file
def save_settings(settings):
    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)



settings = load_settings()




# Screen recording function
def screen_recording(duration=10, output_filename="screen_recording"):
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(settings["output_dir"], "screen_recordings")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{output_filename}_{timestamp}.mp4")

    out = cv2.VideoWriter(filename, fourcc, 20.0, screen_size)

    print("Screen recording started...")
    speak("Screen recording started. You can minimize this window.")
    start_time = datetime.datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    speak(f"Screen recording saved successfully as {filename}")
    print(f"Screen recording saved as {filename}")






# Audio recording function
def audio_recording(duration=10, output_filename="audio_recording"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(settings["output_dir"], "audio_recordings")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{output_filename}_{timestamp}.wav")

    print("Audio recording started...")
    speak("Audio recording started. Speak now.")
    
    # Record audio
    fs = 44100  # Sample rate
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype="int16")
    sd.wait()  # Wait until recording is finished

    # Save audio to file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

    speak(f"Audio recording saved successfully as {filename}")
    print(f"Audio recording saved as {filename}")











# Transcription of audio recordings
def transcribe_audio(audio_filename):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_filename) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio, language=settings
["language"])
        print(f"Transcription: {transcription}")
        speak("Here is the transcription of the audio.")
        speak(transcription)
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        speak("Sorry, I couldn't transcribe the audio.")
        return None






# Background recording function
def background_recording(duration, output_filename):
    threading.Thread(target=screen_recording, args=(duration, output_filename)).sta()
    threading.Thread(target=audio_recording, args=(duration, output_filename)).start()







class Jarvis:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)  # Adjust speaking speed
        self.creator_info = {
            "name": "Mohammod Jamil Uddin",
            "skills": [
                "Embedded Systems",
                "Circuit Design",
                "IoT Development",
                "Firmware Development",
                "Signal Processing",
                "Automation",
                "Machine Learning in Computer Science",
                "Open Source Hardware"
            ],
            "contact": "jamiluddin3282003@email.com"
        }



    def speak(self, text):
        """Speak the given text."""
        self.engine.say(text)
        self.engine.runAndWait()
        
    


    def respond(self, query):
        """Respond to the user query."""
        if query and "creator" in query.lower():
            return self.get_creator_info()
        else:
            return "I'm sorry, I can't help with that."


    def get_creator_info(self):
        """Get information about the creator."""
        info = self.creator_info
        response = (
            f"My creator is {info['name']}.\n"
            f"Here are some of their skills:\n"
            + "\n".join(f"- {skill}" for skill in info['skills']) +
            f"\nYou can contact them at: {info['contact']}"
        )
        return response




def search_google(query):
    """
    Perform a Google search.
    """
    speak(f"Searching Google for {query}")
    print(f"Searching Google for: {query}")
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    wb.open(url)







def search_bing(query):
    """
    Perform a Bing search.
    """
    speak(f"Searching Bing for {query}")
    print(f"Searching Bing for: {query}")
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    wb.open(url)





def search_duckduckgo(query):
    """
    Perform a DuckDuckGo search.
    """
    speak(f"Searching DuckDuckGo for {query}")
    print(f"Searching DuckDuckGo for: {query}")
    url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
    wb.open(url)






def search_youtube(query):
    """
    Search YouTube for videos.
    """
    speak(f"Searching YouTube for {query}")
    print(f"Searching YouTube for: {query}")
    url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    wb.open(url)






def search_wikipedia(query):
    """
    Search Wikipedia for an article and read a summary.
    """
    wiki = wikipediaapi.Wikipedia("en")
    page = wiki.page(query)
    
    if page.exists():
        speak(f"Found Wikipedia article for {query}")
        print(f"Title: {page.title}")
        summary = page.summary[:500]  # Limit to 500 characters
        speak(f"Summary: {summary}")
        print(f"Summary: {summary}")
    else:
        speak(f"Sorry, I couldn't find a Wikipedia article for {query}.")
        print("Wikipedia page not found.")








def search_amazon(query):
    """
    Search Amazon for products.
    """
    speak(f"Searching Amazon for {query}")
    print(f"Searching Amazon for: {query}")
    url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
    wb.open(url)





def search_ebay(query):
    """
    Search eBay for products.
    """
    speak(f"Searching eBay for {query}")
    print(f"Searching eBay for: {query}")
    url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
    wb.open(url)





def search_google_api(query):
    """
    Use Google Custom Search API for precise results.
    """
    speak(f"Using Google API to search for {query}")
    print(f"Using Google API to search for: {query}")
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID).execute()
    results = res.get("items", [])
    
    for i, item in enumerate(results[:3]):  # Limit to top 3 results
        print(f"Result {i + 1}: {item['title']}")
        speak(f"Result {i + 1}: {item['title']}")
        print(f"Snippet: {item['snippet']}")
        print(f"Link: {item['link']}")
    



def search():
    speak("Welcome to the voice-enabled search assistant!")
    speak("You can ask me to search Google, Bing, DuckDuckGo, YouTube, Wikipedia, Amazon, or eBay. Say 'exit' to quit.")

    while True:
        speak("Please say your search platform or command.")
        command = takecommand()

        # Skip this iteration if no command was heard
        if not command:
            continue




        if "exit" in command or "quit" in command:
            speak("Okay, stopping the search assistant. Goodbye!")
            break




        elif "google" in command:
            speak("What would you like to search on Google?")
            query = takecommand()
            if query:
                search_google(query)




        elif "bing" in command:
            speak("What would you like to search on Bing?")
            query = takecommand()
            if query:
                search_bing(query)




        elif "duckduckgo" in command:
            speak("What would you like to search on DuckDuckGo?")
            query = takecommand()
            if query:
                search_duckduckgo(query)



        elif "youtube" in command:
            speak("What would you like to search on YouTube?")
            query = takecommand()
            if query:
                search_youtube(query)




        elif "wikipedia" in command:
            speak("What topic would you like to search on Wikipedia?")
            query = takecommand()
            if query:
                search_wikipedia(query)




        elif "amazon" in command:
            speak("What product would you like to search on Amazon?")
            query = takecommand()
            if query:
                search_amazon(query)



        elif "ebay" in command:
            speak("What product would you like to search on eBay?")
            query = takecommand()
            if query:
                search_ebay(query)




        elif "google api" in command:
            speak("What would you like to search using Google API?")
            query = takecommand()
            if query:
                search_google_api(query)
        else:
            speak("Sorry, I didn't understand. Please try again.")







def play_youtube(search_query):
    """Search and play content on YouTube with added randomness to the query."""
    try:
        if search_query.strip():
            # Add a random tag to the search query
            random_tags = ["HD", "live", "trending", "official"]
            query_with_randomness = f"{search_query} {random.choice(random_tags)}"
            
            # Play on YouTube
            kit.playonyt(query_with_randomness)
            
            # Randomize the response message
            responses = [
                f"Playing {query_with_randomness} on YouTube.",
                f"Sure, let's watch {query_with_randomness}.",
                f"Here's what I found: {query_with_randomness}. Enjoy!",
                f"Starting {query_with_randomness}. Sit back and relax.",
            ]
            return random.choice(responses)
        else:
            return "No query provided to search on YouTube."
    except Exception as e:
        return f"An error occurred while trying to play on YouTube: {e}"








def close_youtube():
    """Close the YouTube browser tab or window."""
    try:
        # Adding a small delay to ensure proper execution
        time.sleep(2)

        # Close the currently focused browser tab
        pyautogui.hotkey('ctrl', 'w')

        # Randomize the response message
        responses = [
            "YouTube has been closed.",
            "Alright, closing YouTube now.",
            "Done! YouTube is no longer open.",
            "YouTube tab is closed. Let me know what’s next!",
        ]
        return random.choice(responses)
    except Exception as e:
        return f"An error occurred while trying to close YouTube: {e}"






# Function to set the voice based on user choice
def set_voice(choice):
    if choice.lower() == "male":
        for voice in voices:
            if "male" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                return True
        print("No male voice found on this system.")
        return False
    elif choice.lower() == "female":
        for voice in voices:
            if "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                return True
        print("No female voice found on this system.")
        return False
    else:
        print("Invalid choice. Please choose 'male' or 'female'.")
        return False





def tell_joke():
    """Interactive joke-telling function with categories and repeat options."""
    categories = {
        "programming": "Programming jokes",
        "neutral": "Neutral jokes",
        "chuck norris": "Chuck Norris jokes"
    }

    speak("Let's have some fun! I can tell jokes about programming, neutral topics, or Chuck Norris. Which one would you like?")
    print("Available categories: programming, neutral, chuck norris")

    category = None
    while not category:
        response = takecommand()
        if response:
            for key in categories:
                if key in response.lower():
                    category = key
                    speak(f"Great choice! {categories[key]} it is.")
                    break
            else:
                speak("Sorry, I didn't catch that. Please choose from programming, neutral, or Chuck Norris.")






    while True:
        # Fetch and speak a joke
        if category == "chuck norris":
            joke = pyjokes.get_joke(language="en", category="chuck")
        else:
            joke = pyjokes.get_joke(language="en", category=category)
        speak(joke)
        print(joke)

        # Ask for next action
        speak("Would you like another joke, switch the category, or stop?")
        print("Options: 'another joke', 'switch category', 'stop'")

        response = takecommand()
        if not response:
            speak("I didn't hear that clearly. Let me know what you'd like.")
            continue

        if "another" in response.lower():
            continue  # Fetch another joke
        elif "switch" in response.lower():
            category = None  # Break out of category loop to restart category selection
            speak("Alright, let's pick a new category!")
            return tell_joke()  # Restart the joke flow
        elif "stop" in response.lower() or "exit" in response.lower():
            speak("Alright, no more jokes for now. Let me know if you'd like to hear some later!")
            break
        else:
            speak("I didn't catch that. Please say 'another joke', 'switch category', or 'stop'.")






# Get the current time in a specified time zone
def get_time_in_timezone(timezone: str) -> None:
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz)
        current_time = local_time.strftime("%I:%M:%S %p")
        speak(f"The current time in {timezone} is {current_time}")
        print(f"The current time in {timezone} is {current_time}")
    except pytz.UnknownTimeZoneError:
        speak("Sorry, I don't recognize that timezone.")
        print("Unknown Timezone.")
    except Exception as e:
        speak(f"An error occurred: {e}")
        print(f"Error: {e}")






# Get the current time in the user's local time zone
def get_local_time() -> None:
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    speak(f"The current time is {current_time}")
    print(f"The current time is {current_time}")





# Get the current date
def get_date() -> None:
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    speak(f"Today's date is {current_date}")
    print(f"Today's date is {current_date}")





# Ask user if they want to know the time in a different time zone
def time_with_timezone() -> None:
    speak("Would you like to know the time in a specific timezone?")
    print("Would you like to know the time in a specific timezone?")
    response = takecommand()


    if response:
        if "yes" in response.lower():
            speak("Please tell me the name of the timezone.")
            print("Please tell me the name of the timezone.")
            timezone = takecommand()

            if timezone:
                get_time_in_timezone(timezone)
            else:
                speak("I couldn't understand the timezone name.")
        else:
            speak("Alright, I will stick to the local time.")
            get_local_time()






# Set an alarm for a specific time
def set_alarm() -> None:
    speak("Please tell me the time to set the alarm. Use the format hour:minute AM/PM.")
    print("Please tell me the time to set the alarm. Use the format hour:minute AM/PM.")
    alarm_time = takecommand()

    if alarm_time:
        try:
            alarm_time = datetime.datetime.strptime(alarm_time, "%I:%M %p")
            current_time = datetime.datetime.now()
            time_diff = alarm_time - current_time

            if time_diff.total_seconds() > 0:
                speak(f"Setting the alarm for {alarm_time.strftime('%I:%M %p')}.")
                print(f"Setting the alarm for {alarm_time.strftime('%I:%M %p')}.")
                time.sleep(time_diff.total_seconds())  # Wait until the alarm time
                speak("Wake up! The alarm time has arrived!")
                print("Wake up! The alarm time has arrived!")
            else:
                speak("The time you set has already passed. Please choose a future time.")
        except ValueError:
            speak("Sorry, I couldn't understand the time format. Please try again.")
            print("Invalid time format.")





# Get fun facts about time
def time_fun_facts() -> None:
    fun_facts = [
        "The longest time zone difference is 26 hours, which happens in some parts of Nepal.",
        "The concept of time zones was first proposed by Sir Sandford Fleming in 1878.",
        "The shortest day of the year, the winter solstice, usually falls around December 21st or 22nd.",
        "There is no time zone in the middle of the Atlantic Ocean, making it a place without a defined time zone."
    ]
    fact = random.choice(fun_facts)
    speak(f"Here's a fun fact about time: {fact}")
    print(f"Fun fact: {fact}")





# Advanced time-related function with multiple user options
def advanced_time_functionality() -> None:
    speak("Would you like to know the current time, the current date, the time in another timezone, or perhaps set an alarm?")
    print("Options: 'current time', 'current date', 'time in timezone', 'set alarm', or 'fun facts about time'")

    while True:
        response = takecommand()

        if response:


            if "current time" in response.lower():

                get_local_time()

            elif "current date" in response.lower():
                get_date()

            elif "time in timezone" in response.lower():



                time_with_timezone()

            elif "set alarm" in response.lower():


                set_alarm()



            elif "fun facts about time" in response.lower():
                time_fun_facts()




            elif "exit" in response.lower() or "no" in response.lower():
                speak("Alright, I will stop talking about time. Let me know if you need anything else.")
                print("Exiting time functionality.")
                break




            else:
                speak("Sorry, I didn't understand that. Please try again.")
                print("I didn't understand that.")
        else:
            speak("I didn't hear anything. Please try again.")
            print("Listening again...")





def date():
    """Tells the current date and continues the conversation."""
    now = datetime.datetime.now()
    day = now.strftime("%A")  # Day of the week
    date = now.day
    month = now.strftime("%B")  # Month name
    year = now.year

    speak(f"Today is {day}, the {date}th of {month}, {year}.")
    print(f"Today is {day}, the {date}th of {month}, {year}.")

    
    
    
    speak("Would you like to know anything else, such as the time or the weather?")
    while True:
        response = takecommand()
        if response:
            if "time" in response:
                time()
                break
            elif "weather" in response:
                speak("I can't check the weather yet, but I'm working on it!")
                break
            elif "no" in response or "nothing" in response:
                speak("Alright, let me know if you need anything else!")
                break
            else:
                speak("I didn't catch that. Please say 'time,' 'weather,' or 'nothing.'")




def tell_joke():
    """Interactive joke-telling function with categories and repeat options."""
    categories = {
        "programming": "Programming jokes",
        "neutral": "Neutral jokes",
        "chuck norris": "Chuck Norris jokes"
    }

    speak("Let's have some fun! I can tell jokes about programming, neutral topics, or Chuck Norris. Which one would you like?")
    print("Available categories: programming, neutral, chuck norris")

    category = None
    while not category:
        response = takecommand()
        if response:
            for key in categories:
                if key in response.lower():
                    category = key
                    speak(f"Great choice! {categories[key]} it is.")
                    break
            else:
                speak("Sorry, I didn't catch that. Please choose from programming, neutral, or Chuck Norris.")



    while True:
        # Fetch and speak a joke
        if category == "chuck norris":
            joke = pyjokes.get_joke(language="en", category="chuck")
        else:
            joke = pyjokes.get_joke(language="en", category=category)
        speak(joke)
        print(joke)

        # Ask for next action
        speak("Would you like another joke, switch the category, or stop?")
        print("Options: 'another joke', 'switch category', 'stop'")

        response = takecommand()
        if not response:
            speak("I didn't hear that clearly. Let me know what you'd like.")
            continue

        if "another" in response.lower():
            continue  # Fetch another joke
        elif "switch" in response.lower():
            category = None  # Break out of category loop to restart category selection
            speak("Alright, let's pick a new category!")
            return tell_joke()  # Restart the joke flow
        elif "stop" in response.lower() or "exit" in response.lower():
            speak("Alright, no more jokes for now. Let me know if you'd like to hear some later!")
            break
        else:
            speak("I didn't catch that. Please say 'another joke', 'switch category', or 'stop'.")






def wishme() -> None:
    """Greets the user based on the time of day and adds interactivity."""

    # Welcome message with added personalization
    speak("Welcome back, sir! I hope you had a great day so far. How are you feeling today? Is there anything exciting you'd like to share with me?")

    print("Welcome back, sir! How are you feeling today?")

    # Capture user’s response about their mood
    mood = takecommand()  # Assuming you have takecommand to capture speech

    if mood:
        speak(f"That's good to know, I hope your {mood} day gets even better!")
        print(f"That's good to know, I hope your {mood} day gets even better!")
    else:
        speak("It's alright, you can tell me later how you're feeling.")
        print("It's alright, you can tell me later how you're feeling.")

    # Time-based greetings
    hour = datetime.datetime.now().hour
    if 4 <= hour < 12:
        speak("Good morning!")
        print("Good morning!")
    elif 12 <= hour < 16:
        speak("Good afternoon!")
        print("Good afternoon!")
    elif 16 <= hour < 24:
        speak("Good evening!")
        print("Good evening!")
    else:
         speak("Good night, see you tomorrow.")
         print("Goodbye for now, talk to you soon!")


    # Ask for the assistant's name or load it if already set
    assistant_name = load_name()

    # Personalized call to action
    speak(f"{assistant_name} at your service. How may I assist you today?")
    print(f"{assistant_name} at your service. How may I assist you today?")





def screenshot() -> None:
    """Takes a screenshot and saves it with a dynamic filename."""
    # Get the current date and time to create a unique filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the folder to save the screenshot (Pictures directory)
    screenshot_folder = os.path.expanduser("~\\Pictures")
    # Define the screenshot file path with a dynamic name
    img_path = os.path.join(screenshot_folder, f"screenshot_{current_time}.png")

    # Take the screenshot
    img = pyautogui.screenshot()
    img.save(img_path)

    # Provide feedback to the user
    speak(f"Screenshot saved as {img_path}.")
    print(f"Screenshot saved as {img_path}.")




def takecommand() -> str:
    #Takes microphone input from the user and returns it as text.
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1

        try:
            audio = r.listen(source, timeout=5)  # Listen with a timeout
        except sr.WaitTimeoutError:
            speak("Timeout occurred. Please try again.")
            return None

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-in")
        print(query)
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None
    except Exception as e:
        speak(f"An error occurred: {e}")
        print(f"Error: {e}")
        return None

    except sr.UnknownValueError:
        print("Sorry, I didn't understand that. Can you say that again?")
        return None
    except sr.RequestError:
        print("The speech recognition service is unavailable. Please check your internet connection.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # print(f"Error: {e}")
        return None 










def play_music(song_name=None):
    """Plays music from the user's Music directory and Desktop Music directory."""
    
    user_music_dir = os.path.expanduser("~\\Music")
    desktop_music_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Music")
    all_songs = []

    if os.path.exists(user_music_dir):
        for root, dirs, files in os.walk(user_music_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    all_songs.append(os.path.join(root, file))

    if os.path.exists(desktop_music_dir):
        for root, dirs, files in os.walk(desktop_music_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    all_songs.append(os.path.join(root, file))

    if song_name:
        all_songs = [song for song in all_songs if song_name.lower() in os.path.basename(song).lower()]

    if all_songs:
        song = random.choice(all_songs)
        pygame.mixer.music.load(song)
        pygame.mixer.music.play()
        speak(f"Playing {os.path.basename(song)}.")
        print(f"Playing {os.path.basename(song)}.")
    else:
        speak("No song found.")
        print("No song found.")







def pause_music():
    """Pause the currently playing music."""
    pygame.mixer.music.pause()
    speak("Music paused.")
    print("Music paused.")






def unpause_music():
    """Unpause the currently paused music."""
    pygame.mixer.music.unpause()
    speak("Music resumed.")
    print("Music resumed.")





def stop_music():
    """Stop the currently playing music."""
    pygame.mixer.music.stop()
    speak("Music stopped.")
    print("Music stopped.")





def next_song(all_songs, current_song):
    """Play the next song in the list."""
    current_index = all_songs.index(current_song)
    next_index = (current_index + 1) % len(all_songs)
    next_song = all_songs[next_index]
    pygame.mixer.music.load(next_song)
    pygame.mixer.music.play()
    speak(f"Next song: {os.path.basename(next_song)}.")
    print(f"Next song: {os.path.basename(next_song)}.")




def previous_song(all_songs, current_song):
    """Play the previous song in the list."""
    current_index = all_songs.index(current_song)
    prev_index = (current_index - 1) % len(all_songs)
    prev_song = all_songs[prev_index]
    pygame.mixer.music.load(prev_song)
    pygame.mixer.music.play()
    speak(f"Previous song: {os.path.basename(prev_song)}.")
    print(f"Previous song: {os.path.basename(prev_song)}.")




def volume_up():
    """Increase the system volume."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, 1, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = min(current_volume + 0.1, 1.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    speak(f"Volume increased to {new_volume * 100:.0f}%.")
    print(f"Volume increased to {new_volume * 100:.0f}%.")





def volume_down():
    """Decrease the system volume."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, 1, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(current_volume - 0.1, 0.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    speak(f"Volume decreased to {new_volume * 100:.0f}%.")
    print(f"Volume decreased to {new_volume * 100:.0f}%.")






def handle_play_command():
    """Handle the 'play music' command."""
    # play_music()
    speak("Opening music")
    song_name = query.replace("play music", "").strip()
    play_music(song_name)





def handle_pause_command():
    """Handle the 'pause music' command."""
    pause_music()




def handle_unpause_command():
    """Handle the 'unpause music' or 'resume music' command."""
    unpause_music()




def handle_stop_command():
    """Handle the 'stop music' command."""
    stop_music()




def handle_next_command():
    """Handle the 'next song' command."""
    next_song([], "")  # Dummy call for now, implement song list properly




def handle_previous_command():
    """Handle the 'previous song' command."""
    previous_song([], "")  # Dummy call for now, implement song list properly



def handle_volume_up_command():
    """Handle the 'volume up' command."""
    volume_up()



def handle_volume_down_command():
    """Handle the 'volume down' command."""
    volume_down()



def handle_exit_command():
    """Handle the 'exit' command."""
    speak("Goodbye!")
    print("Goodbye!")
    return True  # Indicate that the program should exit





def handle_unknown_command():
    """Handle unknown commands."""
    speak("Sorry, I didn't understand that.")
    print("Sorry, I didn't understand that.")





def handle_voice_commands():
    """Handles voice commands to control music playback."""
    recognizer = sr.Recognizer()

    # Set up the microphone
    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        while True:
            try:
                # Listen for the command
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                if "play music" in command:
                    handle_play_command()
                elif "pause music" in command:
                    handle_pause_command()
                elif "unpause music" in command or "resume music" in command:
                    handle_unpause_command()
                elif "stop music" in command:
                    handle_stop_command()
                elif "next song" in command:
                    handle_next_command()
                elif "previous song" in command:
                    handle_previous_command()
                elif "volume up" in command:
                    handle_volume_up_command()
                elif "volume down" in command:
                    handle_volume_down_command()
                elif "exit" in command:
                    if handle_exit_command():
                        break  # Exit the listening loop
                else:
                    handle_unknown_command()
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")





# def play_music(song_name=None) -> None:
#     """Plays music from the user's Music directory and Desktop Music directory."""
    
#     # Get the path to the Music folder in the user's directory
#     user_music_dir = os.path.expanduser("~\\Music")
    
#     # Get the path to the Music folder on the Desktop (if it exists)
#     desktop_music_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Music")
    
#     # Create a list to store all music files found
#     all_songs = []
    
#     # Check if the Music directory in the user's folder exists and add songs from it
#     if os.path.exists(user_music_dir):
#         all_songs.extend(os.listdir(user_music_dir))
    
#     # Check if the Music directory on the Desktop exists and add songs from it
#     if os.path.exists(desktop_music_dir):
#         all_songs.extend(os.listdir(desktop_music_dir))

#     # Filter songs if song_name is provided
#     if song_name:
#         all_songs = [song for song in all_songs if song_name.lower() in song.lower()]

#     # If we have any songs, play one of them
#     if all_songs:
#         song = random.choice(all_songs)  # Pick a random song from the available list
#         # Find the full path to the song
#         if song in os.listdir(user_music_dir):
#             song_path = os.path.join(user_music_dir, song)
#         else:
#             song_path = os.path.join(desktop_music_dir, song)
        
#         os.startfile(song_path)  # Play the song
#         speak(f"Playing {song}.")
#         print(f"Playing {song}.")
#     else:
#         speak("No song found.")
#         print("No song found.")




# Weather function
def get_weather(city):
    api_key = "your_openweathermap_api_key"  # Replace with your API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + f"q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(complete_url)
        weather_data = response.json()
        if weather_data["cod"] != "404":
            main = weather_data["main"]
            weather = weather_data["weather"][0]["description"]
            temperature = main["temp"]
            humidity = main["humidity"]
            speak(
                f"The weather in {city} is {weather}. The temperature is {temperature} degrees Celsius with a humidity of {humidity}%.")
            print(f"Weather in {city}: {weather}, Temp: {temperature}°C, Humidity: {humidity}%")
        else:
            speak("City not found. Please try again.")
    except Exception as e:
        speak(f"Could not fetch weather data. Error: {e}")



# News function
def get_news():
    api_key = "your_news_api_key"  # Replace with your API key
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

    try:
        response = requests.get(url)
        news_data = response.json()
        if news_data["status"] == "ok":
            speak("Here are the top headlines:")
            for i, article in enumerate(news_data["articles"][:5], start=1):
                speak(f"Headline {i}: {article['title']}")
                print(f"Headline {i}: {article['title']}")
        else:
            speak("Could not fetch news at the moment.")
    except Exception as e:
        speak(f"Error fetching news: {e}")


# Task Manager
tasks = []

def manage_tasks():
    speak("Would you like to add a task, view tasks, or delete a task?")
    response = takecommand()

    if "add" in response:
        speak("What task should I add?")
        task = takecommand()
        tasks.append(task)
        speak(f"Added task: {task}")
    elif "view" in response:
        if tasks:
            speak("Here are your tasks:")
            for i, task in enumerate(tasks, start=1):
                speak(f"Task {i}: {task}")
        else:
            speak("You have no tasks.")
    elif "delete" in response:
        if tasks:
            speak("Which task number should I delete?")
            for i, task in enumerate(tasks, start=1):
                speak(f"Task {i}: {task}")
            try:
                task_num = int(takecommand())
                removed_task = tasks.pop(task_num - 1)
                speak(f"Removed task: {removed_task}")
            except (ValueError, IndexError):
                speak("Invalid task number.")
        else:
            speak("You have no tasks to delete.")





# Function to activate the microphone and recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Adjust for ambient noise using 'with' statement
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjusting for ambient noise after opening the source
        print("Listening for commands...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        # Recognize the speech using Google's recognition engine
        query = recognizer.recognize_google(audio)
        print(f"Command received: {query}")
        return query.lower()

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        print("Sorry, I'm having trouble connecting to the service.")
        return None



# Function to handle voice typing (typing text to a file or app)
def voice_typing():
    speak("You can start typing now. Speak your text after the beep.")
    time.sleep(2)  # Pause before starting voice typing
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Please start speaking.")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing your voice...")
        text = recognizer.recognize_google(audio)
        print(f"Text received: {text}")
        
        # You can either type in a file or an application
        with open("voice_typing_output.txt", "a") as file:
            file.write(text + "\n")  # Write the recognized text into a file

        speak(f"Text typed: {text}")
    
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand that.")
    except sr.RequestError:
        speak("Sorry, I'm having trouble with the speech service.")


# math calculation functions 

def calculate():
    """
    Perform math calculations based on voice input and respond using the `speak` function.
    """
    speak("What calculation would you like me to perform?")
    operation = takecommand().lower()  # Take voice input for the calculation

    try:
        # Handle basic arithmetic operations
        if 'add' in operation or '+' in operation:
            speak("Performing addition.")
            parts = operation.split()
            result = float(parts[0]) + float(parts[2])  # Example: 2 plus 3
            speak(f"The result is {result}")
            print(f"Result: {result}")
        
        elif 'subtract' in operation or '-' in operation:
            speak("Performing subtraction.")
            parts = operation.split()
            result = float(parts[0]) - float(parts[2])  # Example: 5 minus 3
            speak(f"The result is {result}")
            print(f"Result: {result}")
        
        elif 'multiply' in operation or '*' in operation:
            speak("Performing multiplication.")
            parts = operation.split()
            result = float(parts[0]) * float(parts[2])  # Example: 2 multiplied by 3
            speak(f"The result is {result}")
            print(f"Result: {result}")
        
        elif 'divide' in operation or '/' in operation:
            speak("Performing division.")
            parts = operation.split()
            # Prevent division by zero
            if float(parts[2]) == 0:
                speak("Error: Division by zero is not allowed.")
                print("Error: Division by zero.")
            else:
                result = float(parts[0]) / float(parts[2])  # Example: 6 divided by 3
                speak(f"The result is {result}")
                print(f"Result: {result}")

        # Handle complex numbers
        elif 'complex' in operation:
            speak("Performing complex number operation.")
            parts = operation.split()
            real = float(parts[1])
            imag = float(parts[2])
            complex_num = complex(real, imag)
            speak(f"The complex number is {complex_num}")
            print(f"Complex number: {complex_num}")
        
        # Matrix operations
        elif 'matrix' in operation:
            speak("Performing matrix operation.")
            parts = operation.split('matrix')[1].strip()
            matrix = np.array(eval(parts))  # Evaluating the string input as a matrix
            speak(f"Matrix entered: {matrix}")
            print(f"Matrix: \n{matrix}")
        
        # Percentage calculations
        elif 'percent' in operation:
            speak("Calculating percentage.")
            parts = operation.split()
            result = (float(parts[0]) / 100) * float(parts[2])
            speak(f"The percentage is {result}")
            print(f"Percentage: {result}")
        
        # Trigonometric Inverses: arcsin, arccos, arctan
        elif 'arcsin' in operation or 'asin' in operation:
            speak("Performing arcsine operation.")
            parts = operation.split()
            value = float(parts[2])
            result = math.degrees(math.asin(value))  # Convert from radians to degrees
            speak(f"The arcsine of {value} is {result} degrees")
            print(f"Arcsine of {value}: {result} degrees")
        
        elif 'arccos' in operation or 'acos' in operation:
            speak("Performing arccosine operation.")
            parts = operation.split()
            value = float(parts[2])
            result = math.degrees(math.acos(value))  # Convert from radians to degrees
            speak(f"The arccosine of {value} is {result} degrees")
            print(f"Arccosine of {value}: {result} degrees")
        
        elif 'arctan' in operation or 'atan' in operation:
            speak("Performing arctangent operation.")
            parts = operation.split()
            value = float(parts[2])
            result = math.degrees(math.atan(value))  # Convert from radians to degrees
            speak(f"The arctangent of {value} is {result} degrees")
            print(f"Arctangent of {value}: {result} degrees")
        
        # Solving equations (Linear and Quadratic)
        elif 'solve' in operation:
            speak("Solving equation.")
            # For quadratic equations like 'solve x^2 + 5x + 6 = 0'
            if 'quadratic' in operation:
                parts = operation.split('quadratic')[1].strip()
                equation = sp.sympify(parts)  # Using sympy to solve equations
                solutions = sp.solve(equation)
                speak(f"The solutions are {solutions}")
                print(f"Solutions: {solutions}")
            # For linear equations
            else:
                parts = operation.split('equation')[1].strip()
                equation = sp.sympify(parts)
                solution = sp.solve(equation)
                speak(f"The solution is {solution}")
                print(f"Solution: {solution}")
        
        # Factorial and modulus
        elif 'factorial' in operation:
            speak("Performing factorial operation.")
            parts = operation.split()
            number = int(parts[1])
            result = math.factorial(number)
            speak(f"The factorial of {number} is {result}")
            print(f"Factorial: {result}")
        
        elif 'modulus' in operation:
            speak("Calculating modulus.")
            parts = operation.split()
            result = int(parts[0]) % int(parts[2])
            speak(f"The modulus is {result}")
            print(f"Modulus: {result}")
        
        # Logging calculations to file
        elif 'log' in operation:
            speak("Logging the calculation to history.")
            with open('calc_history.txt', 'a') as log_file:
                log_file.write(f"Operation: {operation}\n")
            speak("The operation has been logged.")
            print(f"Logged: {operation}")
        
        else:
            # Default: Evaluate any general mathematical expression
            result = eval(operation)
            speak(f"The result is {result}")
            print(f"Result: {result}")
        
    except Exception as e:
        # Handle any errors
        speak("Sorry, I couldn't perform that calculation. Please try again.")
        print(f"Error: {e}")


# System Control
def control_system(query):
    """Handle system commands like shutdown, restart, lock, sleep, and exit."""
    speak("Would you like to shutdown, restart, lock, sleep, log off, or exit?")
    try:
        if "shutdown" in query:
            speak("Shutting down the system, goodbye!")
            os.system("shutdown /s /f /t 1")
        
        elif "restart" in query:
            speak("Restarting the system, please wait!")
            os.system("shutdown /r /f /t 1")
        
        elif "lock" in query:
            speak("Locking the system. See you soon!")
            ctypes.windll.user32.LockWorkStation()
        
        elif "sleep" in query:
            speak("Putting the system to sleep. Goodbye!")
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        
        elif "log off" in query or "logout" in query:
            speak("Logging out. See you next time!")
            os.system("shutdown -l")
        
        elif "exit" in query or "offline" in query:
            speak("Going offline. Have a great day!")
            exit()
        
        else:
            speak("I didn't understand that system command. Please try again.")
    except Exception as e:
        speak(f"An error occurred while executing the command: {e}")
        print(f"Error: {e}")


def motivational_quote():
    categories = {
        "general": [
        # General Motivational Quotes
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "Your limitation—it's only your imagination. - Anonymous",
        "Push yourself, because no one else is going to do it for you. - Anonymous",
        "Great things never come from comfort zones. - Anonymous",
        "Dream it. Wish it. Do it. - Anonymous",
        "The best way to predict the future is to create it. - Abraham Lincoln",
        "Success usually comes to those who are too busy to be looking for it. - Henry David Thoreau",
        "Don’t watch the clock; do what it does. Keep going. - Sam Levenson",
        "The harder you work for something, the greater you’ll feel when you achieve it. - Anonymous",
        "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
        "Success is walking from failure to failure with no loss of enthusiasm. - Winston Churchill",
        "It does not matter how slowly you go as long as you do not stop. - Confucius",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "Don’t wait. The time will never be just right. - Napoleon Hill",
        "It always seems impossible until it’s done. - Nelson Mandela",
        "Our greatest glory is not in never falling, but in rising every time we fall. - Confucius",
        "Act as if what you do makes a difference. It does. - William James",
        "Everything you’ve ever wanted is on the other side of fear. - George Addair",
        "Opportunities don't happen, you create them. - Chris Grosser",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "If you can dream it, you can do it. - Walt Disney",
        "The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt",
        "Don’t stop when you’re tired. Stop when you’re done. - Anonymous",
        "It always feels impossible until it’s done. - Nelson Mandela",
        "Believe you can and you're halfway there. - Theodore Roosevelt",   
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "Your limitation—it's only your imagination. - Anonymous",
        "Push yourself, because no one else is going to do it for you. - Anonymous",
        "Great things never come from comfort zones. - Anonymous",
        "Dream it. Wish it. Do it. - Anonymous",
        "The best way to predict the future is to create it. - Abraham Lincoln",
        "Success usually comes to those who are too busy to be looking for it. - Henry David Thoreau",
        "Don’t watch the clock; do what it does. Keep going. - Sam Levenson",
        "The harder you work for something, the greater you’ll feel when you achieve it. - Anonymous",
        "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
        "Success is walking from failure to failure with no loss of enthusiasm. - Winston Churchill",
        "It does not matter how slowly you go as long as you do not stop. - Confucius",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "Don’t wait. The time will never be just right. - Napoleon Hill",
        "It always seems impossible until it’s done. - Nelson Mandela",
        "Our greatest glory is not in never falling, but in rising every time we fall. - Confucius",
        "Act as if what you do makes a difference. It does. - William James",
        "Everything you’ve ever wanted is on the other side of fear. - George Addair",
        "Opportunities don't happen, you create them. - Chris Grosser",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "If you can dream it, you can do it. - Walt Disney",
        "The only limit to our realization of tomorrow is our doubts of today. - Franklin D. Roosevelt",
        "Don’t stop when you’re tired. Stop when you’re done. - Anonymous",
        "It always feels impossible until it’s done. - Nelson Mandela",

        ],
        "scientists": [
            "The important thing is not to stop questioning. Curiosity has its own reason for existing. - Albert Einstein",
            "Science is a way of thinking much more than it is a body of knowledge. - Carl Sagan",
            "What we know is a drop, what we don’t know is an ocean. - Isaac Newton",
            "In the middle of difficulty lies opportunity. - Albert Einstein",
            "Pure mathematics is, in its way, the poetry of logical ideas. - Albert Einstein",
            "We are made of star stuff. - Carl Sagan",
            "Intelligence is the ability to adapt to change. - Stephen Hawking",
            "Nothing in life is to be feared; it is only to be understood. - Marie Curie",
            "If I have seen further, it is by standing on the shoulders of giants. - Isaac Newton",
            "Genius is one percent inspiration and ninety-nine percent perspiration. - Thomas Edison",
            "Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world. - Albert Einstein",
            "However difficult life may seem, there is always something you can do and succeed at. - Stephen Hawking",
            "I would rather have questions that can’t be answered than answers that can’t be questioned. - Richard Feynman",
            "You cannot teach a man anything; you can only help him find it within himself. - Galileo Galilei",
            "Be alone, that is the secret of invention; be alone, that is when ideas are born. - Nikola Tesla",
            "It is not the strongest of the species that survive, nor the most intelligent, but the one most responsive to change. - Charles Darwin",
            "The love for all living creatures is the most noble attribute of man. - Charles Darwin",
            "Many of life's failures are people who did not realize how close they were to success when they gave up. - Thomas Edison",
            "Somewhere, something incredible is waiting to be known. - Carl Sagan",
            "The cosmos is within us. We are made of star stuff. We are a way for the universe to know itself. - Carl Sagan",
 # Quotes from Scientists
        "The important thing is not to stop questioning. Curiosity has its own reason for existing. - Albert Einstein",
        "Imagination is more important than knowledge. - Albert Einstein",
        "Science is a way of thinking much more than it is a body of knowledge. - Carl Sagan",
        "If you can't explain it simply, you don't understand it well enough. - Albert Einstein",
        "I have no special talent. I am only passionately curious. - Albert Einstein",
        "What we know is a drop, what we don’t know is an ocean. - Isaac Newton",
        "It is not that I'm so smart. But I stay with the questions much longer. - Albert Einstein",
        "In the middle of difficulty lies opportunity. - Albert Einstein",
        "The measure of intelligence is the ability to change. - Albert Einstein",
        "Look deep into nature, and then you will understand everything better. - Albert Einstein",
        "We cannot solve our problems with the same thinking we used when we created them. - Albert Einstein",
        "Pure mathematics is, in its way, the poetry of logical ideas. - Albert Einstein",
        "Science is the poetry of reality. - Richard Dawkins",
        "There are no shortcuts to any place worth going. - Beverly Sills",
        "The greatest wealth is to live content with little. - Plato",
        "I am not a teacher, but an awakener. - Robert Frost",
        "The important thing is to never stop questioning. - Albert Einstein",
        "Logic will get you from A to B. Imagination will take you everywhere. - Albert Einstein",
        "Success is how high you bounce when you hit bottom. - George S. Patton",

        # Philosophical and Spiritual Quotes
        "You must be the change you wish to see in the world. - Mahatma Gandhi",
        "Peace begins with a smile. - Mother Teresa",
        "Live as if you were to die tomorrow. Learn as if you were to live forever. - Mahatma Gandhi",
        "It’s not what happens to you, but how you react to it that matters. - Epictetus",
        "Do not go where the path may lead, go instead where there is no path and leave a trail. - Ralph Waldo Emerson",
        "Happiness depends upon ourselves. - Aristotle",
        "The mind is everything. What you think you become. - Buddha",
        "Injustice anywhere is a threat to justice everywhere. - Martin Luther King Jr.",
        "The only true wisdom is in knowing you know nothing. - Socrates",
        "Knowing others is intelligence; knowing yourself is true wisdom. Mastering others is strength; mastering yourself is true power. - Lao Tzu",
        "Do one thing every day that scares you. - Eleanor Roosevelt",

        # Leadership and Perseverance Quotes
        "To lead people, walk behind them. - Lao Tzu",
        "A leader is one who knows the way, goes the way, and shows the way. - John C. Maxwell",
        "Leadership is not about being in charge. It's about taking care of those in your charge. - Simon Sinek",
        "If you can’t handle stress, you’ll never be able to handle success. - Anonymous",
        "The best way to find yourself is to lose yourself in the service of others. - Mahatma Gandhi",
        "You can never cross the ocean until you have the courage to lose sight of the shore. - Christopher Columbus",

        # Quotes from Writers and Poets
        "Don’t aim for success if you want it; just do what you love and believe in, and it will come naturally. - David Frost",
        "Do not wait for leaders; do it alone, person to person. - Mother Teresa",
        "Don’t let yesterday take up too much of today. - Will Rogers",
        "It’s hard to beat a person who never gives up. - Babe Ruth",
        "Our lives begin to end the day we become silent about things that matter. - Martin Luther King Jr.",
        "Everything you can imagine is real. - Pablo Picasso",
        "Life is 10% what happens to us and 90% how we react to it. - Charles R. Swindoll",
        "Every moment is a fresh beginning. - T.S. Eliot",
        "Life isn't about finding yourself. Life is about creating yourself. - George Bernard Shaw",
        "Do not go where the path may lead, go instead where there is no path and leave a trail. - Ralph Waldo Emerson",

        # More Influential Figures
        "The greatest glory in living lies not in never falling, but in rising every time we fall. - Nelson Mandela",
        "Your time is limited, don’t waste it living someone else’s life. - Steve Jobs",
        "If you want to live a happy life, tie it to a goal, not to people or things. - Albert Einstein",
        "The best revenge is massive success. - Frank Sinatra",
        "What lies behind us and what lies before us are tiny matters compared to what lies within us. - Ralph Waldo Emerson",
        "Success is not how high you have climbed, but how you make a positive difference to the world. - Roy T. Bennett",
        "The harder you work, the luckier you get. - Gary Player"
        ],
        "philosophical": [
            "You must be the change you wish to see in the world. - Mahatma Gandhi",
            "Live as if you were to die tomorrow. Learn as if you were to live forever. - Mahatma Gandhi",
            "The mind is everything. What you think you become. - Buddha",
            "The only true wisdom is in knowing you know nothing. - Socrates",
            "Knowing yourself is the beginning of all wisdom. - Aristotle",
        ],
        "leadership": [
            "To lead people, walk behind them. - Lao Tzu",
            "A leader is one who knows the way, goes the way, and shows the way. - John C. Maxwell",
            "Leadership is not about being in charge. It's about taking care of those in your charge. - Simon Sinek",
            "If you can’t handle stress, you’ll never be able to handle success. - Anonymous",
            "The best way to find yourself is to lose yourself in the service of others. - Mahatma Gandhi",
        ],
        "random": []
    }

    # Combine all quotes into the 'random' category
    for quotes in categories.values():
        categories["random"].extend(quotes)

    # Prompt user for category choice
    print("\nChoose a category for your motivational quote:")
    print("1. General")
    print("2. Scientists")
    print("3. Philosophical")
    print("4. Leadership")
    print("5. Random")
    
    speak("Please choose a category for your motivational quote. Say one for General, two for Scientists, three for Philosophical, four for Leadership, or five for Random.")
    
    try:
        # Get user input
        user_choice = takecommand().lower()

        # Map choices to categories
        category_mapping = {
            "one": "general",
            "two": "scientists",
            "three": "philosophical",
            "four": "leadership",
            "five": "random"
        }

        # Determine selected category
        selected_category = category_mapping.get(user_choice, "random")
        
        # Select a quote
        quote = random.choice(categories[selected_category])

        # Speak and print the quote
        speak(quote)
        print(f"\n{quote}")
    except Exception as e:
        speak("I couldn't process your choice. Here's a random quote instead.")
        quote = random.choice(categories["random"])
        speak(quote)
        print(f"\n{quote}")

def open_website(query):
    """Open a website based on user query."""
    # Dictionary mapping commands to URLs
    websites = {
        "microsoft": "https://www.microsoft.com",
        "instagram": "https://www.instagram.com",
        "facebook": "https://www.facebook.com",
        "stackoverflow": "https://stackoverflow.com",
        "stack exchange": "https://stackexchange.com",
        "kaggle": "https://www.kaggle.com",
        "github": "https://www.github.com",
        "reddit": "https://www.reddit.com",
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "twitter": "https://www.twitter.com",
        "linkedin": "https://www.linkedin.com",
        "amazon": "https://www.amazon.com",
        "netflix": "https://www.netflix.com",
        "spotify": "https://www.spotify.com",
        "wikipedia": "https://www.wikipedia.org",
        "quora": "https://www.quora.com",
        "bing": "https://www.bing.com",
        "yahoo": "https://www.yahoo.com",
        "ebay": "https://www.ebay.com",
        "pinterest": "https://www.pinterest.com",
        "medium": "https://medium.com",
        "canva": "https://www.canva.com",
        "dribbble": "https://dribbble.com",
        "behance": "https://www.behance.net",
        "fiverr": "https://www.fiverr.com",
        "upwork": "https://www.upwork.com",
        "alibaba": "https://www.alibaba.com",
        "airbnb": "https://www.airbnb.com",
        "zoom": "https://www.zoom.us",
        "slack": "https://slack.com",
        "trello": "https://trello.com",
        "asana": "https://asana.com",
        "notion": "https://www.notion.so",
        "unsplash": "https://unsplash.com",
        "pexels": "https://www.pexels.com",
        "github docs": "https://docs.github.com",
        "coursera": "https://www.coursera.org",
        "udemy": "https://www.udemy.com",
        "edx": "https://www.edx.org",
        "duolingo": "https://www.duolingo.com",
        "discord": "https://discord.com",
        "hacker news": "https://news.ycombinator.com",
        "product hunt": "https://www.producthunt.com",
        "kickstarter": "https://www.kickstarter.com",
        "bbc news": "https://www.bbc.com/news",
        "cnn": "https://www.cnn.com",
        "national geographic": "https://www.nationalgeographic.com",
        "tesla": "https://www.tesla.com",
        "nasa": "https://www.nasa.gov",
        "spotify web": "https://open.spotify.com",
        "apple": "https://www.apple.com",
        "adobe": "https://www.adobe.com",
        "dropbox": "https://www.dropbox.com",
        "icloud": "https://www.icloud.com",
        "weather": "https://www.weather.com",
        "coinbase": "https://www.coinbase.com",
        "binance": "https://www.binance.com",
        "paypal": "https://www.paypal.com",
        "shopify": "https://www.shopify.com",
        "wordpress": "https://wordpress.com",
        "tumblr": "https://www.tumblr.com",
        "bbc sports": "https://www.bbc.com/sport",
        "skype": "https://www.skype.com",
        "zoominfo": "https://www.zoominfo.com",
        "yelp": "https://www.yelp.com",
        "tripadvisor": "https://www.tripadvisor.com",
        "booking": "https://www.booking.com",
        "hotels": "https://www.hotels.com",
        "expedia": "https://www.expedia.com",
        "agoda": "https://www.agoda.com",
        "uber": "https://www.uber.com",
        "lyft": "https://www.lyft.com",
        "foodpanda": "https://www.foodpanda.com",
        "ubereats": "https://www.ubereats.com",
        "zomato": "https://www.zomato.com",
        "grubhub": "https://www.grubhub.com",
        "doordash": "https://www.doordash.com",
        "swiggy": "https://www.swiggy.com",
        "bbc weather": "https://www.bbc.com/weather",
        "reuters": "https://www.reuters.com",
        "the guardian": "https://www.theguardian.com",
        "the verge": "https://www.theverge.com",
        "techcrunch": "https://techcrunch.com",
        "wired": "https://www.wired.com",
        "mashable": "https://mashable.com",
        "cnet": "https://www.cnet.com",
        "9to5mac": "https://9to5mac.com",
        "gsmarena": "https://www.gsmarena.com",
        "phonearena": "https://www.phonearena.com",
        "xda developers": "https://www.xda-developers.com",
        "howstuffworks": "https://www.howstuffworks.com",
        "ted": "https://www.ted.com",
        "courier": "https://www.courier.com",
        "fedex": "https://www.fedex.com",
        "appstore": "https://www.apple.com/app-store",
        "ups": "https://www.ups.com",
        "dhl": "https://www.dhl.com",
        "flipkart": "https://www.flipkart.com",
        "snapdeal": "https://www.snapdeal.com",
        "indiatimes": "https://www.indiatimes.com",
        "ndtv": "https://www.ndtv.com",
        "moneycontrol": "https://www.moneycontrol.com",
        "livemint": "https://www.livemint.com",
        "forbes": "https://www.forbes.com",
        "bloomberg": "https://www.bloomberg.com",
        "wall street journal": "https://www.wsj.com",
        "economist": "https://www.economist.com",
        "nytimes": "https://www.nytimes.com",
        "latimes": "https://www.latimes.com",
        "washington post": "https://www.washingtonpost.com",
        "guardian": "https://www.theguardian.com",
        "nature": "https://www.nature.com",
        "science": "https://www.sciencemag.org",
        "springer": "https://www.springer.com",
        "plos": "https://www.plos.org",
        "sciencedirect": "https://www.sciencedirect.com",
        "researchgate": "https://www.researchgate.net",
        "arxiv": "https://arxiv.org",
        "pubmed": "https://pubmed.ncbi.nlm.nih.gov",
        "deepseek": "https://www.deepseek.net",
        "bing news": "https://www.bing.com/news",
        "google news": "https://news.google.com",
        "yahoo news": "https://news.yahoo.com",
        "duckduckgo": "https://duckduckgo.com",
        "baidu": "https://www.baidu.com",
        "yandex": "https://www.yandex.com",
        "aol": "https://www.aol.com",
        "ask": "https://www.ask.com",
        "excite": "https://www.excite.com",
        "lycos": "https://www.lycos.com",
        "webcrawler": "https://www.webcrawler.com",
        "infospace": "https://www.infospace.com",
        "alibaba cloud": "https://www.alibabacloud.com",
        "aws": "https://aws.amazon.com",
        "azure": "https://azure.microsoft.com",
        "google cloud": "https://cloud.google.com",
        "ibm cloud": "https://www.ibm.com/cloud",
        "oracle cloud": "https://www.oracle.com/cloud",
        "salesforce": "https://www.salesforce.com",
        "sap": "https://www.sap.com",
        "vmware": "https://www.vmware.com",
        "red hat": "https://www.redhat.com",
        "cisco": "https://www.cisco.com",
        "hpe": "https://www.hpe.com",
        "dell": "https://www.dell.com",
        "lenovo": "https://www.lenovo.com",
        "asus": "https://www.asus.com",
        "acer": "https://www.acer.com",
        "msi": "https://www.msi.com",
        "nvidia": "https://www.nvidia.com",
        "amd": "https://www.amd.com",
        "intel": "https://www.intel.com",
        "qualcomm": "https://www.qualcomm.com",
        "mediatek": "https://www.mediatek.com",
        "samsung": "https://www.samsung.com",
        "huawei": "https://www.huawei.com",
        "xiaomi": "https://www.mi.com",
        "oppo": "https://www.oppo.com",
        "vivo": "https://www.vivo.com",
        "oneplus": "https://www.oneplus.com",
        "realme": "https://www.realme.com",
        "motorola": "https://www.motorola.com",
        "sony": "https://www.sony.com",
        "lg": "https://www.lg.com",
        "htc": "https://www.htc.com",
        "nokia": "https://www.nokia.com",
        "google pixel": "https://store.google.com",
        "iphone": "https://www.apple.com/iphone",
        "samsung galaxy": "https://www.samsung.com/galaxy",
        "oneplus nord": "https://www.oneplus.com/nord",
        "xiaomi mi": "https://www.mi.com/mi",
        "realme x": "https://www.realme.com/x",
        "motorola edge": "https://www.motorola.com/edge",
        "sony xperia": "https://www.sony.com/xperia",
        "lg velvet": "https://www.lg.com/velvet",
        "htc desire": "https://www.htc.com/desire",
        "nokia lumia": "https://www.nokia.com/lumia",
        "google pixelbook": "https://store.google.com/product/pixelbook",
        "macbook pro": "https://www.apple.com/macbook-pro",
        "dell xps": "https://www.dell.com/xps",
        "lenovo thinkpad": "https://www.lenovo.com/thinkpad",
        "asus zenbook": "https://www.asus.com/zenbook",
        "acer swift": "https://www.acer.com/swift",
        "msi prestige": "https://www.msi.com/prestige",
        "nvidia geforce": "https://www.nvidia.com/geforce",
        "amd radeon": "https://www.amd.com/radeon",
    }

    # Match user query with websites dictionary
    for key, url in websites.items():
        if key in query.lower():
            speak(f"Opening {key.capitalize()}.")
            wb.open(url)
            return

    # If no website matches
    speak("Sorry, I can't find the requested website in my database.")




# Email Sending
def send_email():
    try:
        speak("To whom should I send the email? Please provide the recipient's email address.")
        recipient = takecommand()
        speak("What should I say?")
        content = takecommand()
        # Configure your email
        sender_email = "your_email@gmail.com"
        sender_password = "your_password"  # Use app-specific password if enabled
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, content)
        speak("Email sent successfully.")
    except Exception as e:
        speak(f"Could not send the email. Error: {e}") 







""" 


def set_name() -> None:
    # Sets or changes the assistant's name with user confirmation and persistent storage.
    # Load the current name if it exists
    if os.path.exists("assistant_name.txt"):
        with open("assistant_name.txt", "r") as file:
            current_name = file.read().strip()
    else:
        current_name = "Assistant"

    # Inform the user about the current name
    speak(f"My current name is {current_name}. Would you like to give me a new name?")
    print(f"My current name is {current_name}. Would you like to give me a new name?")
    response = takecommand()

    if response and ("yes" in response or "change" in response):
        speak("What would you like to name me?")
        print("What would you like to name me?")
        new_name = takecommand()

        if new_name:
            # Confirm the new name
            speak(f"Did I hear that correctly? You want to name me {new_name}?")
            print(f"Did I hear that correctly? You want to name me {new_name}?")
            confirmation = takecommand()

            if confirmation and "yes" in confirmation:
                # Save the new name to a file
                with open("assistant_name.txt", "w") as file:
                    file.write(new_name)
                speak(f"Great! From now on, you can call me {new_name}.")
                print(f"Great! From now on, you can call me {new_name}.")
            else:
                speak("Alright, I won't change my name for now.")
                print("Name change aborted.")
        else:
            speak("I didn't catch the new name. Please try again later.")
            print("Name input was empty.")
    elif response and ("no" in response or "keep" in response):
        speak(f"Alright, I'll stay as {current_name}. Let me know if you change your mind.")
        print(f"No changes made. Current name remains {current_name}.")
    else:
        speak("I didn't understand your response. Please try again.")
        print("Unclear response. Try again later.")



 """





def set_name() -> None:
    """Sets a new name for the assistant."""
    speak("What would you like to name me?")
    name = takecommand()
    if name:
        with open("assistant_name.txt", "w") as file:
            file.write(name)
        speak(f"Alright, I will be called {name} from now on.")
    else:
        speak("Sorry, I couldn't catch that.")





def load_name() -> str:
    """Loads the assistant's name from a file, or uses a default name."""
    try:
        with open("assistant_name.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "Jarvis"  # Default name






def search_wikipedia(query):
    """Searches Wikipedia and returns a summary."""
    try:
        speak("Searching Wikipedia...")
        result = wikipedia.summary(query, sentences=2)
        speak(result)
        print(result)
    except wikipedia.exceptions.DisambiguationError:
        speak("Multiple results found. Please be more specific.")
    except Exception:
        speak("I couldn't find anything on Wikipedia.")
        



def authenticate_google_calendar():
    """
    Authenticate and return Google Calendar API service.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)
    return service





def set_reminder(reminder_time, reminder_message):
    """
    Set a reminder at a specific time.
    :param reminder_time: Time at which to show the reminder (24-hour format, e.g., '15:30')
    :param reminder_message: Message to display when the reminder goes off
    """
    while True:
        current_time = time.strftime("%H:%M")
        if current_time == reminder_time:
            logging.info(f"Reminder: {reminder_message}")
            speak(f"Reminder: {reminder_message}")
            break
        time.sleep(60)  # Check every minute
        





def current_datetime():
    """Get the current date and time."""
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time





def schedule_event(task, hour, minute):
    """Schedule a task at a specific time."""
    schedule_time = f'{hour}:{minute}'
    schedule.every().day.at(schedule_time).do(task)
    logging.info(f"Task scheduled at {schedule_time}")
    speak(f"Task scheduled at {schedule_time}")
    
    while True:
        schedule.run_pending()
        time.sleep(1)





def create_google_calendar_event(start_time, end_time, event_summary):
    """Create an event in Google Calendar."""
    service = authenticate_google_calendar()
    
    event = {
        'summary': event_summary,
        'start': {
            'dateTime': start_time,
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'America/Los_Angeles',
        },
    }

    event_result = service.events().insert(calendarId='primary', body=event).execute()
    logging.info(f"Event created: {event_result['summary']}")
    speak(f"Event created: {event_result['summary']}")






def get_todays_events():
    """Fetch today's events from Google Calendar."""
    service = authenticate_google_calendar()
    
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=10, singleEvents=True,
        orderBy='startTime').execute()
    
    events = events_result.get('items', [])
    
    if not events:
        logging.info("No upcoming events found.")
        speak("You have no upcoming events for today.")
    else:
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            logging.info(f"{event['summary']} at {start}")
            speak(f"{event['summary']} at {start}")




def parse_event_details(command):
    """Parse event details (start time, end time, description) from user input."""
    # Regular expression to match the time and description pattern
    match = re.search(r"start at (\d{1,2}:\d{2} [APap][Mm]), end at (\d{1,2}:\d{2} [APap][Mm]), (.+)", command)
    if match:
        start_time = match.group(1)
        end_time = match.group(2)
        event_description = match.group(3)
        return start_time, end_time, event_description
    else:
        speak("I couldn't parse the event details. Please provide the details in the correct format.")
        return None, None, None

def convert_time_to_iso(event_time):
    """Convert 12-hour format time (e.g., 2:30 PM) to ISO 8601 format."""
    try:
        time_obj = datetime.strptime(event_time, '%I:%M %p')  # Convert 12-hour time to datetime object
        iso_time = time_obj.strftime('%Y-%m-%dT%H:%M:%S')
        return iso_time
    except ValueError:
        speak(f"Invalid time format: {event_time}")
        return None




def process_command(command):
    """Process the voice command and trigger the corresponding function."""
    if 'what time is it' in command:
        speak("The current time is " + current_datetime())
    
    elif 'schedule' in command and 'meeting' in command:
        speak("When would you like to schedule the meeting? Please provide the hour and minute.")
        command = takecommand()
        if command:
            try:
                hour, minute = map(int, command.split()[0:2])
                schedule_event(my_task, hour, minute)
            except ValueError:
                speak("I couldn't understand the time.")
    
    elif 'create event' in command:
        speak("Please provide the start time, end time, and event description.")
        command = takecommand()
        if command:
            start_time, end_time, event_summary = parse_event_details(command)
            if start_time and end_time and event_summary:
                # Convert start and end times to ISO 8601 format
                start_time_iso = convert_time_to_iso(start_time)
                end_time_iso = convert_time_to_iso(end_time)
                if start_time_iso and end_time_iso:
                    create_google_calendar_event(start_time_iso, end_time_iso, event_summary)
    
    elif 'get my events' in command:
        get_todays_events()

    elif 'set reminder' in command:
        speak("Please say the time for the reminder (in 24-hour format) and the message.")
        command = takecommand()
        if command:
            parts = command.split(' ', 1)
            if len(parts) == 2:
                reminder_time, reminder_message = parts
                set_reminder(reminder_time, reminder_message)
            else:
                speak("I didn't understand the reminder time or message.")
    
    elif "calendar" in command:
        speak("Opening calendar")
        get_todays_events()








def my_task():
    """A task that needs to be scheduled."""
    logging.info("It's time to do the task!")
    speak("It's time to do the task!")







def process(query):
    """Process the voice command and trigger the corresponding function."""
    if query is None:
        return

    if "time" in query:
        speak("Fetching time...")
        advanced_time_functionality()

    elif "date" in query:
        speak("Fetching date...")
        date()


    elif "create event" in query:
        speak("Please provide the event details (start time, end time, and description).")
        command = takecommand()
        if command:
            start_time, end_time, event_summary = parse_event_details(command)
            if start_time and end_time and event_summary:
                start_time_iso = convert_time_to_iso(start_time)
                end_time_iso = convert_time_to_iso(end_time)
                if start_time_iso and end_time_iso:
                    create_google_calendar_event(start_time_iso, end_time_iso, event_summary)

    elif "get events" in query:
        get_todays_events()


    elif "play on youtube" in query:
        speak("What should I play on YouTube?")
        youtube_query = takecommand()
        if youtube_query:
            response = play_youtube(youtube_query)
            speak(response)

    elif "close youtube" in query:
        response = close_youtube()
        speak(response)

    elif "wikipedia" in query:
        query = query.replace("wikipedia", "").strip()
        search_wikipedia(query)

    elif "calculate" in query or "what is" in query or "math" in query:
        calculate()

    elif "play music" in query:
         handle_voice_commands()
        # speak("Opening music")
        # song_name = query.replace("play music", "").strip()
        # play_music(song_name)
        
    elif "open chat gpt" in query:
        speak("Opening ChatGPT.")
        wb.open("https://chat.openai.com")


    elif "open" in query:
        query = query.replace("open", "").strip()
        open_website(query)


    elif "start typing" in query:
        speak("Starting voice typing.")
        voice_typing()


    elif "close" in query:
        speak("Closing browser.")
        close_browser_website()

    

    elif "open google" in query:
        speak("Opening Google")
        wb.open("google.com")



    elif "change your name" in query:
        speak("Changing my name.")
        set_name()



    elif "screenshot" in query:
        screenshot()
        speak("I've taken a screenshot, please check it.")



    elif "news" in query:
        speak("Here are some top news headlines.")
        get_news()

  
  
  
    elif "manage" in query:
        speak("Opening task manager.")
        manage_tasks()



    elif "motivate" in query:
        speak("Opening motivational quote.")
        motivational_quote()




    elif "tell me a joke" in query:
        speak("Here's a joke for you.")
        tell_joke()




    elif "battery" in query:
        battery_status = get_battery_status()
        speak(battery_status)




    elif "control system" in query:
        speak("Starting system control. What would you like to do?")
        query = takecommand()
        control_system(query)  




if __name__ == "__main__":


    print("Welcome to Jarvis AI.")
    # print(logo)
    # Show the image
    image.show()
    speak("Initializing Jarvis AI...")
    print("Initializing Jarvis...")
    wishme()
    jarvis = Jarvis()

    while True:
        query = takecommand()  # Continuously listen for commands



        if query:
            query = query.lower()  # Convert the command to lowercase for uniformity





            # Exit or quit command
            if "exit" in query or "quit" in query:
                jarvis.speak("Goodbye! Have a great day.")
                break
            

            elif "search files" in query or "find files" in query:
                handle_file_search_command(query)




            elif "screen record" in query or "record screen" in query:
                speak("For how many seconds should I record the screen?")
                try:
                    duration_query = takecommand()
                    duration = int(duration_query.split()[0])  # Extract the duration in seconds
                    screen_recording(duration=duration)
                except ValueError:
                    speak("Please specify a valid duration in seconds.")




            elif "record audio" in query or "audio record" in query:
                speak("For how many seconds should I record the audio?")
                try:
                    duration_query = takecommand()
                    duration = int(duration_query.split()[0])  # Extract the duration in seconds
                    audio_recording(duration=duration)
                except ValueError:
                    speak("Please specify a valid duration in seconds.")






            elif "transcribe audio" in query or "audio transcription" in query:
                speak("Please provide the filename of the audio to transcribe.")
                filename_query = takecommand()


                if filename_query:
                    file_path = os.path.join(settings["output_dir"], "audio_recordings", f"{filename_query.strip()}.wav")
                    
                    
                    if os.path.exists(file_path):


                        transcription = transcribe_audio(file_path)


                        if transcription:
                            print(f"Transcription: {transcription}")
                    
                    
                    else:
                        speak("Sorry, I could not find the file. Please check the filename.")

            elif "background record" in query or "record background" in query:
                speak("For how many seconds should I record in the background?")



                try:
                    duration_query = takecommand()
                    duration = int(duration_query.split()[0])  # Extract the duration in seconds
                    background_recording(duration=duration, output_filename="background_recording")
                except ValueError:
                    speak("Please specify a valid duration in seconds.")




            elif "change settings" in query or "settings" in query:
                speak("What setting would you like to change?")
                setting_query = takecommand()



                if "output directory" in setting_query:
                    speak("Please specify the new output directory.")
                    new_dir = takecommand()
                    settings["output_dir"] = new_dir.strip()
                    save_settings(settings)
                    speak(f"Output directory changed to {new_dir}.")



                elif "default duration" in setting_query:
                    speak("Please specify the new default duration in seconds.")
                    try:
                        new_duration = int(takecommand().split()[0])
                        settings["default_duration"] = new_duration
                        save_settings(settings)
                        speak(f"Default duration changed to {new_duration} seconds.")
                    except ValueError:
                        speak("Please specify a valid duration in seconds.")




                elif "language" in setting_query:
                    speak("Please specify the new language code (e.g., en-US, es-ES).")
                    new_language = takecommand().strip()
                    settings["language"] = new_language
                    save_settings(settings)
                    speak(f"Language changed to {new_language}.")   
        




            elif "creator" in query or "who created you" in query:
                response = jarvis.get_creator_info()  # Call the method to fetch creator info
                print(f"Jarvis: {response}")
                jarvis.speak(response)
                




            # Shutdown the system
            elif "shutdown" in query:
                speak("Shutting down the system, goodbye!")
                os.system("shutdown /s /f /t 1")
                break




            # Restart the system
            elif "restart" in query:
                speak("Restarting the system, please wait!")
                os.system("shutdown /r /f /t 1")
                break




            # Search functionality  
            elif "ultron" in query:
                speak("What would you like me to search for?")
                print("What would you like me to search for?")
                search()  # Call your custom search function

            # File search functionality
            elif "file" in query:


                if "search functions" in query:
                    speak("Please provide the directory to search in.")
                    directory = takecommand()



                    if directory:

                        try:
                            functions = search_functions_in_directory(directory)
                            if functions:
                                response = "Found the following functions in the directory:\n"
                                for file, funcs in functions.items():
                                    response += f"In {file}, the functions are: {', '.join(funcs)}\n"
                                speak(response)
                            else:
                                speak("No functions found in the Python files.")
                        except Exception as e:
                            speak(f"An error occurred: {e}")
                    else:
                        speak("No directory specified.")



            # Calendar functionality
            elif "calendar" in query or "events" in query:
                speak("Do you want to see today's events or create a new event?")
                calendar_query = takecommand()



                if "today" in calendar_query:
                    get_todays_events()  # Call your custom event function




                elif "create" in calendar_query or "add" in calendar_query:
                    speak("Please provide the start time, end time, and description of the event.")
                    event_details = takecommand()
                    # Parse the event details as needed
                    create_google_calendar_event('2025-01-22T14:00:00', '2025-01-22T16:00:00', 'Meeting with John')




            elif "hey " in query:
                print("Yes, how can I help you?")
                run_assistants()



            else:
                process(query)
                search_functions_in_directory