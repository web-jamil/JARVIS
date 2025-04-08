import os
import re
import time
import json
import glob
import math
import cmath
import random
import logging
import datetime
import platform
import calendar
import threading
import subprocess
import numpy as np
import sympy as sp
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser as wb
import pyautogui
import pyjokes
import requests
import smtplib
import pytz
import psutil
import pygame
import cv2
import vlc
import yt_dlp as youtube_dl
from PIL import Image
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import wikipediaapi
import ctypes
import requests as Request
# Constants
GOOGLE_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_custom_search_engine_id"
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

class JarvisVoice:
    """Handles all voice-related functionality"""
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.set_voice('male')  # Default to male voice

    def set_voice(self, gender: str) -> bool:
        """Set the voice gender (male/female)"""
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if gender.lower() in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                return True
        return False

    def speak(self, text: str) -> None:
        """Convert text to speech"""
        logging.info(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def take_command(self) -> Optional[str]:
        """Listen for and recognize voice command"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                query = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized: {query}")
                return query
            except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError) as e:
                print(f"Voice recognition error: {e}")
                return None

class MediaPlayer:
    """Handles media playback functionality"""
    def __init__(self):
        self.instance = vlc.Instance('--no-xlib')
        self.player = self.instance.media_player_new()
        self.is_playing = False
        self.current_volume = 50
        self.playlist = []
        self.current_track_index = 0
        pygame.mixer.init()

    def play_youtube(self, song_name: str) -> bool:
        """Play audio from YouTube"""
        try:
            ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(f"ytsearch1:{song_name}", download=False)
                
                if not result or 'entries' not in result or not result['entries']:
                    return False
                
                video = result['entries'][0]
                media = self.instance.media_new(video['url'])
                self.player.set_media(media)
                self.player.audio_set_volume(self.current_volume)
                
                if self.player.play() == -1:
                    return False
                
                self.is_playing = True
                self.playlist.append(video['url'])
                self.current_track_index = len(self.playlist) - 1
                return True
        except Exception as e:
            logging.error(f"Error playing YouTube audio: {e}")
            return False

    def play_local(self, file_path: str) -> None:
        """Play local media file"""
        if os.path.exists(file_path):
            media = self.instance.media_new(file_path)
            self.player.set_media(media)
            self.player.audio_set_volume(self.current_volume)
            self.player.play()
            self.is_playing = True
            self.playlist.append(file_path)
            self.current_track_index = len(self.playlist) - 1

    def control_media(self, action: str) -> None:
        """Control media playback"""
        if action == "pause":
            if self.player.is_playing():
                self.player.pause()
                self.is_playing = False
        elif action == "resume":
            if not self.player.is_playing() and self.is_playing:
                self.player.play()
        elif action == "stop":
            self.player.stop()
            self.is_playing = False
            self.playlist = []
            self.current_track_index = 0
        elif action == "next":
            if self.current_track_index < len(self.playlist) - 1:
                self.current_track_index += 1
                self.play_current_track()
        elif action == "previous":
            if self.current_track_index > 0:
                self.current_track_index -= 1
                self.play_current_track()

    def play_current_track(self) -> None:
        """Play the current track in playlist"""
        media = self.instance.media_new(self.playlist[self.current_track_index])
        self.player.set_media(media)
        self.player.play()

    def set_volume(self, level: int) -> None:
        """Set volume level (0-100)"""
        if 0 <= level <= 100:
            self.current_volume = level
            self.player.audio_set_volume(level)

class FileManager:
    """Handles file and directory operations"""
    @staticmethod
    def search_files(directory: str, extension: str = '*') -> List[str]:
        """Search for files with specific extension"""
        search_pattern = os.path.join(directory, f'**/*{extension}')
        return glob.glob(search_pattern, recursive=True)

    @staticmethod
    def search_functions(file_path: str) -> List[str]:
        """Search for function definitions in a file"""
        functions = []
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                if match:
                    functions.append(match.group(1))
        return functions

    @staticmethod
    def search_functions_in_directory(directory: str) -> Dict[str, List[str]]:
        """Search for functions in all Python files in directory"""
        python_files = FileManager.search_files(directory, '.py')
        functions_in_files = {}
        
        for file in python_files:
            functions = FileManager.search_functions(file)
            if functions:
                functions_in_files[file] = functions
                
        return functions_in_files

    @staticmethod
    def get_directory_size(directory: str) -> float:
        """Get directory size in MB"""
        total_size = 0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return round(total_size / (1024 * 1024), 2)

class WebServices:
    """Handles web-related services"""
    def __init__(self):
        self.browser_path = self._get_default_browser()

    @staticmethod
    def _get_default_browser() -> str:
        """Get system's default browser path"""
        try:
            if platform.system() == 'Windows':
                import winreg
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'http\shell\open\command') as key:
                    cmd = winreg.QueryValue(key, None)
                    return cmd.split('"')[1]
            elif platform.system() == 'Darwin':  # macOS
                return '/usr/bin/open'
            else:  # Linux
                return '/usr/bin/xdg-open'
        except:
            return ''

    def open_website(self, url: str) -> None:
        """Open website in default browser"""
        try:
            if self.browser_path:
                subprocess.Popen([self.browser_path, url])
            else:
                wb.open(url)
        except Exception as e:
            logging.error(f"Error opening website: {e}")

    def search_google(self, query: str) -> None:
        """Perform Google search"""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        self.open_website(url)

    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia and return summary"""
        try:
            wiki = wikipediaapi.Wikipedia('en')
            page = wiki.page(query)
            return page.summary[:500] if page.exists() else "No Wikipedia article found."
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}")
            return "Sorry, I couldn't access Wikipedia."

class SystemControl:
    """Handles system control operations"""
    @staticmethod
    def shutdown() -> None:
        """Shutdown the system"""
        if platform.system() == 'Windows':
            os.system("shutdown /s /f /t 1")
        else:
            os.system("shutdown -h now")

    @staticmethod
    def restart() -> None:
        """Restart the system"""
        if platform.system() == 'Windows':
            os.system("shutdown /r /f /t 1")
        else:
            os.system("shutdown -r now")

    @staticmethod
    def lock() -> None:
        """Lock the system"""
        if platform.system() == 'Windows':
            ctypes.windll.user32.LockWorkStation()
        elif platform.system() == 'Darwin':
            os.system("pmset displaysleepnow")
        else:
            os.system("gnome-screensaver-command -l")

    @staticmethod
    def get_battery_status() -> str:
        """Get battery status"""
        battery = psutil.sensors_battery()
        if not battery:
            return "Battery status not available"
        
        percent = battery.percent
        charging = "charging" if battery.power_plugged else "not charging"
        return f"Battery is at {percent}% and {charging}."

class CalendarManager:
    """Handles calendar and scheduling"""
    def __init__(self):
        self.service = self._authenticate()
        self.calendar_enabled = self.service is not None

    def _authenticate(self):
        """Authenticate with Google Calendar API"""
        creds = None
        token_path = 'token.json'
        creds_path = 'credentials.json'
        
        if not os.path.exists(creds_path):
            logging.error("Google API credentials file not found")
            return None
            
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logging.error(f"Authentication failed: {e}")
                    return None

            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        return build('calendar', 'v3', credentials=creds) if creds else None

    def create_event(self, start_time: str, end_time: str, summary: str) -> None:
        """Create calendar event"""
        if not self.calendar_enabled:
            return
            
        event = {
            'summary': summary,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        self.service.events().insert(calendarId='primary', body=event).execute()

    def get_events(self, max_results=10) -> List[Dict]:
        """Get upcoming events"""
        if not self.calendar_enabled:
            return []
            
        now = datetime.utcnow().isoformat() + 'Z'
        events_result = self.service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=max_results, singleEvents=True,
            orderBy='startTime').execute()
        return events_result.get('items', [])

class JarvisAI:
    """Main Jarvis AI class"""
    def __init__(self, enable_calendar=False):
        self.voice = JarvisVoice()
        self.media = MediaPlayer()
        self.files = FileManager()
        self.web = WebServices()
        self.system = SystemControl()
        self.calendar = CalendarManager() if enable_calendar else None
        self.settings = self._load_settings()
        self.name = self._load_name()
        self.logo = self._load_logo()
        
    def _load_settings(self) -> Dict:
        """Load settings from JSON file"""
        default_settings = {
            "output_dir": "recordings",
            "default_duration": 10,
            "language": "en-US"
        }
        
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                return {**default_settings, **json.load(f)}
        return default_settings

    def _load_name(self) -> str:
        """Load assistant name from file"""
        try:
            with open("assistant_name.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "Jarvis"

    def _load_logo(self) -> str:
        """Load ASCII logo"""
        return """                                          
              ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
          ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
          ██║███████║██████╔╝██║   ██║██║███████╗
     ██   ██║██╔══██║██╔═══╝ ██║   ██║██║╚════██║
     ╚█████╔╝██║  ██║██║     ╚██████╔╝██║███████║
      ╚════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝╚══════╝
        """

    def greet(self) -> None:
        """Greet the user based on time of day"""
        hour = datetime.now().hour
        if 4 <= hour < 12:
            greeting = "Good morning!"
        elif 12 <= hour < 16:
            greeting = "Good afternoon!"
        elif 16 <= hour < 24:
            greeting = "Good evening!"
        else:
            greeting = "Good night!"
        
        self.voice.speak(f"{greeting} I am {self.name}. How may I assist you today?")
        print(self.logo)

    def process_command(self, command: str) -> bool:
        """Process user command and return whether to continue running"""
        if not command:
            return True
            
        command = command.lower()
        
        # System commands
        if "exit" in command or "quit" in command:
            self.voice.speak("Goodbye! Have a great day.")
            return False
            
        elif "shutdown" in command:
            self.voice.speak("Shutting down the system, goodbye!")
            self.system.shutdown()
            return False
            
        elif "restart" in command:
            self.voice.speak("Restarting the system, please wait!")
            self.system.restart()
            return False
            
        # Media commands
        elif "play" in command:
            if "on youtube" in command:
                song = command.replace("play", "").replace("on youtube", "").strip()
                if not self.media.play_youtube(song):
                    self.voice.speak("Could not play that song on YouTube")
            elif "on spotify" in command:
                song = command.replace("play", "").replace("on spotify", "").strip()
                self.web.open_website(f"https://open.spotify.com/search/{song.replace(' ', '%20')}")
            else:
                song = command.replace("play", "").strip()
                if not self.media.play_youtube(song):
                    self.voice.speak("Could not play that song")
                
        elif "pause" in command:
            self.media.control_media("pause")
            self.voice.speak("Media paused")
            
        elif "resume" in command or "continue" in command:
            self.media.control_media("resume")
            self.voice.speak("Resuming media")
            
        elif "stop" in command:
            self.media.control_media("stop")
            self.voice.speak("Media stopped")
            
        elif "volume up" in command:
            self.media.set_volume(min(self.media.current_volume + 10, 100))
            self.voice.speak(f"Volume increased to {self.media.current_volume}%")
            
        elif "volume down" in command:
            self.media.set_volume(max(self.media.current_volume - 10, 0))
            self.voice.speak(f"Volume decreased to {self.media.current_volume}%")
            
        elif "set volume to" in command:
            try:
                volume = int(command.split("set volume to")[1].strip())
                self.media.set_volume(volume)
                self.voice.speak(f"Volume set to {volume}%")
            except ValueError:
                self.voice.speak("Please specify a valid volume between 0 and 100")
            
        # Web services
        elif "search" in command:
            query = command.replace("search", "").strip()
            if query:
                self.web.search_google(query)
            else:
                self.voice.speak("What would you like me to search for?")
                query = self.voice.take_command()
                if query:
                    self.web.search_google(query)
                    
        elif "wikipedia" in command:
            query = command.replace("wikipedia", "").strip()
            if query:
                summary = self.web.search_wikipedia(query)
                self.voice.speak(summary)
            else:
                self.voice.speak("What would you like me to look up on Wikipedia?")
                query = self.voice.take_command()
                if query:
                    summary = self.web.search_wikipedia(query)
                    self.voice.speak(summary)
                    
        # File operations
        elif "search files" in command:
            if "functions" in command:
                self.voice.speak("Which directory should I search?")
                directory = self.voice.take_command()
                if directory and os.path.exists(directory):
                    functions = self.files.search_functions_in_directory(directory)
                    if functions:
                        response = "Found functions in:\n"
                        for file, funcs in functions.items():
                            response += f"{os.path.basename(file)}: {', '.join(funcs)}\n"
                        self.voice.speak(response)
                    else:
                        self.voice.speak("No functions found in Python files.")
                else:
                    self.voice.speak("Directory not found or not specified")
                    
        # System info
        elif "battery" in command:
            status = self.system.get_battery_status()
            self.voice.speak(status)
            
        elif "lock" in command:
            self.system.lock()
            self.voice.speak("System locked")
            
        # Calendar
        elif ("calendar" in command or "events" in command) and self.calendar:
            events = self.calendar.get_events()
            if events:
                self.voice.speak("Here are your upcoming events:")
                for event in events:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    self.voice.speak(f"{event['summary']} at {start}")
            else:
                self.voice.speak("No upcoming events found.")
                
        elif "create event" in command and self.calendar:
            self.voice.speak("Please provide the event details in this format: "
                           "Start time, End time, Event description")
            details = self.voice.take_command()
            if details:
                try:
                    start, end, desc = [part.strip() for part in details.split(',', 2)]
                    self.calendar.create_event(start, end, desc)
                    self.voice.speak(f"Event '{desc}' created successfully")
                except Exception as e:
                    logging.error(f"Error creating event: {e}")
                    self.voice.speak("Could not create event. Please try again with proper format")
                    
        elif not self.calendar and ("calendar" in command or "event" in command):
            self.voice.speak("Calendar features are not enabled")
            
        # Jokes and fun
        elif "tell me a joke" in command or "joke" in command:
            joke = pyjokes.get_joke()
            self.voice.speak(joke)
            print(joke)
            
        # Default response
        else:
            self.voice.speak("I didn't understand that command. Please try again.")
            
        return True

    def run(self) -> None:
        """Main execution loop"""
        print("Initializing Jarvis AI...")
        self.greet()
        
        while True:
            command = self.voice.take_command()
            if not self.process_command(command):
                break

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        filename='jarvis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize with calendar disabled by default to prevent credential errors
        jarvis = JarvisAI(enable_calendar=False)
        jarvis.run()
    except KeyboardInterrupt:
        print("\nJarvis is shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"An error occurred: {e}")