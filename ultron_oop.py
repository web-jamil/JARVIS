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
from google.auth.transport.requests import Request
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import wikipediaapi
import ctypes
import pyaudio
# Add these imports at the top
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import jwt
import hashlib
from cryptography.fernet import Fernet
import socket
import wakeonlan


# Constants
GOOGLE_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_custom_search_engine_id"
SCOPES = ['https://www.googleapis.com/auth/calendar.events']




class NLProcessor:
    """Advanced Natural Language Processor"""
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize)
        self.context = {}
        self.command_vectors = None
        self.command_responses = []
        self._initialize_command_vectors()
        
    def _tokenize(self, text):
        tokens = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def _initialize_command_vectors(self):
        """Predefined commands with responses"""
        base_commands = [
            ("play music", "media"),
            ("search for", "web"),
            ("set reminder", "reminder"),
            ("send email", "email"),
            ("system status", "system"),
            ("smart home", "smarthome")
        ]
        texts = [cmd[0] for cmd in base_commands]
        self.command_vectors = self.vectorizer.fit_transform(texts)
        self.command_responses = [cmd[1] for cmd in base_commands]
    
    def process_command(self, text, user_id="default"):
        """Enhanced command processing with context"""
        # Update context with previous interaction
        prev_context = self.context.get(user_id, {})
        
        # Handle follow-up questions
        if prev_context.get('awaiting_response'):
            return self._handle_followup(text, user_id)
            
        # Vectorize input
        input_vec = self.vectorizer.transform([text])
        
        # Find most similar command
        similarities = cosine_similarity(input_vec, self.command_vectors)
        best_match_idx = similarities.argmax()
        
        if similarities[0, best_match_idx] > 0.6:  # Similarity threshold
            return self.command_responses[best_match_idx], {}
        
        # If no clear match, use keyword spotting
        return self._keyword_spotting(text), {}

    def _handle_followup(self, text, user_id):
        """Process follow-up responses"""
        context = self.context[user_id]
        response_type = context['awaiting_response']
        
        # Clear context first
        self.context[user_id] = {}
        
        if response_type == 'email_recipient':
            return "email", {'action': 'compose', 'recipient': text}
        elif response_type == 'reminder_time':
            try:
                time = self._parse_time(text)
                return "reminder", {'time': time, 'text': context['reminder_text']}
            except ValueError as e:
                return "error", {'message': str(e)}
        # Add more follow-up handlers
        
        return "unknown", {}

    def _keyword_spotting(self, text):
        """Fallback keyword-based processing"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['play', 'music', 'song']):
            return "media"
        elif any(word in text_lower for word in ['search', 'look up', 'find']):
            return "web"
        # Add more keyword mappings
        return "unknown"

class TaskManager:
    """Advanced Task Management System"""
    def __init__(self):
        self.tasks = []
        self.task_lock = threading.Lock()
        self.load_tasks()
        
    def add_task(self, description, due_date=None, priority=2, tags=None):
        """Add a new task with metadata"""
        if tags is None:
            tags = []
        task = {
            'id': len(self.tasks) + 1,
            'description': description,
            'due_date': due_date,
            'priority': priority,  # 1=high, 2=medium, 3=low
            'tags': tags,
            'completed': False,
            'created_at': datetime.now()
        }
        with self.task_lock:
            self.tasks.append(task)
        self.save_tasks()
        return task
    
    def complete_task(self, task_id):
        """Mark task as completed"""
        with self.task_lock:
            for task in self.tasks:
                if task['id'] == task_id:
                    task['completed'] = True
                    task['completed_at'] = datetime.now()
                    self.save_tasks()
                    return True
        return False
    
    def get_tasks(self, filter_type="active"):
        """Get tasks based on filter"""
        with self.task_lock:
            if filter_type == "active":
                return [t for t in self.tasks if not t['completed']]
            elif filter_type == "completed":
                return [t for t in self.tasks if t['completed']]
            elif filter_type == "all":
                return self.tasks.copy()
            elif filter_type.startswith("tag:"):
                tag = filter_type[4:]
                return [t for t in self.tasks if tag in t['tags']]
            else:
                return []
    
    def save_tasks(self):
        """Save tasks to file"""
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f, default=str)
    
    def load_tasks(self):
        """Load tasks from file"""
        try:
            with open('tasks.json', 'r') as f:
                self.tasks = json.load(f)
                # Convert string dates back to datetime objects
                for task in self.tasks:
                    for field in ['due_date', 'created_at', 'completed_at']:
                        if field in task and task[field]:
                            task[field] = datetime.fromisoformat(task[field])
        except FileNotFoundError:
            self.tasks = []

class SecurityManager:
    """Enhanced Security and Privacy System"""
    def __init__(self):
        self.encryption_key = self._get_encryption_key()
        self.voice_profiles = {}
        self.load_voice_profiles()
        
    def _get_encryption_key(self):
        """Get or generate encryption key"""
        key_file = 'encryption.key'
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        fernet = Fernet(self.encryption_key)
        if isinstance(data, str):
            data = data.encode()
        return fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data).decode()
    
    def create_voice_profile(self, user_id, audio_samples):
        """Create a voice biometric profile"""
        # In a real implementation, we'd extract voice features
        profile = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'sample_count': len(audio_samples)
        }
        self.voice_profiles[user_id] = profile
        self.save_voice_profiles()
        return profile
    
    def authenticate_user(self, audio_sample):
        """Authenticate user via voice"""
        # Placeholder - real implementation would compare voice features
        if self.voice_profiles:
            return list(self.voice_profiles.keys())[0]  # Return first user
        return None
    
    def save_voice_profiles(self):
        """Save voice profiles to file"""
        with open('voice_profiles.json', 'w') as f:
            json.dump(self.voice_profiles, f, default=str)
    
    def load_voice_profiles(self):
        """Load voice profiles from file"""
        try:
            with open('voice_profiles.json', 'r') as f:
                self.voice_profiles = json.load(f)
        except FileNotFoundError:
            self.voice_profiles = {}

class IoTManager:
    """Expanded IoT and Smart Home Control"""
    def __init__(self):
        self.devices = {
            # Existing devices
            'living room light': {'state': False, 'type': 'switch', 'ip': '192.168.1.100'},
            'thermostat': {'state': 72, 'type': 'thermostat', 'unit': '°F', 'ip': '192.168.1.101'},
            # New devices
            'bedroom light': {'state': False, 'type': 'switch', 'ip': '192.168.1.102'},
            'garage door': {'state': False, 'type': 'door', 'ip': '192.168.1.103'},
            'security system': {'state': 'armed', 'type': 'security', 'ip': '192.168.1.104'}
        }
        self.scenes = {
            'good morning': [
                {'device': 'living room light', 'action': 'on'},
                {'device': 'thermostat', 'action': 'set 72'}
            ],
            'good night': [
                {'device': 'living room light', 'action': 'off'},
                {'device': 'bedroom light', 'action': 'off'},
                {'device': 'thermostat', 'action': 'set 68'},
                {'device': 'security system', 'action': 'arm'}
            ]
        }
        self.device_lock = threading.Lock()
    
    def wake_on_lan(self, mac_address):
        """Wake device using Wake-on-LAN"""
        try:
            wakeonlan.send_magic_packet(mac_address)
            return True
        except Exception as e:
            logging.error(f"WOL failed: {e}")
            return False
    
    def execute_scene(self, scene_name):
        """Execute a predefined scene"""
        scene = self.scenes.get(scene_name.lower())
        if not scene:
            return False
            
        results = []
        with self.device_lock:
            for action in scene:
                device = self.devices.get(action['device'])
                if device:
                    if action['action'] == 'on':
                        device['state'] = True
                        results.append(f"{action['device']} turned on")
                    elif action['action'] == 'off':
                        device['state'] = False
                        results.append(f"{action['device']} turned off")
                    elif action['action'].startswith('set'):
                        temp = int(re.search(r'\d+', action['action']).group())
                        device['state'] = temp
                        results.append(f"{action['device']} set to {temp}{device.get('unit', '')}")
        
        return results if results else False

class FocusManager:
    """Focus Mode and Distraction Blocker"""
    def __init__(self):
        self.focus_mode = False
        self.blocked_items = {
            'websites': ['facebook.com', 'twitter.com', 'reddit.com'],
            'applications': ['steam.exe', 'discord.exe']
        }
        self.focus_sessions = []
        self.hosts_file = r'C:\Windows\System32\drivers\etc\hosts' if platform.system() == 'Windows' else '/etc/hosts'
        self.original_hosts = self._read_hosts()
    
    def enable_focus_mode(self, duration_minutes=60):
        """Enable focus mode with optional duration"""
        if self.focus_mode:
            return False
            
        self.focus_mode = True
        session = {
            'start': datetime.now(),
            'end': datetime.now() + timedelta(minutes=duration_minutes),
            'duration': duration_minutes,
            'blocked_items': []
        }
        
        # Block websites by modifying hosts file
        try:
            with open(self.hosts_file, 'a') as f:
                for site in self.blocked_items['websites']:
                    f.write(f"\n127.0.0.1 {site}\n127.0.0.1 www.{site}\n")
                    session['blocked_items'].append(site)
        except PermissionError:
            logging.error("Need admin rights to modify hosts file")
            return False
        
        # Block applications (Windows only)
        if platform.system() == 'Windows':
            for app in self.blocked_items['applications']:
                try:
                    subprocess.run(['taskkill', '/F', '/IM', app], check=True)
                    session['blocked_items'].append(app)
                except subprocess.CalledProcessError:
                    continue
        
        self.focus_sessions.append(session)
        return True
    
    def disable_focus_mode(self):
        """Disable focus mode and restore settings"""
        if not self.focus_mode:
            return False
            
        # Restore original hosts file
        try:
            with open(self.hosts_file, 'w') as f:
                f.write(self.original_hosts)
        except PermissionError:
            logging.error("Need admin rights to modify hosts file")
            return False
        
        self.focus_mode = False
        if self.focus_sessions:
            self.focus_sessions[-1]['end'] = datetime.now()
        return True
    
    def _read_hosts(self):
        """Read current hosts file content"""
        try:
            with open(self.hosts_file, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Couldn't read hosts file: {e}")
            return ""

class JarvisAI:
    """Enhanced Jarvis AI with new functionality"""
    def __init__(self, enable_calendar=False):
        # Existing initialization
        self.voice = JarvisVoice()
        self.media = MediaPlayer()
        self.files = FileManager()
        self.web = WebServices()
        self.system = SystemControl()
        self.calendar = CalendarManager() if enable_calendar else None
        self.smarthome = SmartHomeControl()
        self.email = EmailManager()
        self.news = NewsReader()
        self.health = HealthMonitor()
        self.learning = LearningModule()
        
        # New components
        self.nlp = NLProcessor()
        self.tasks = TaskManager()
        self.security = SecurityManager()
        self.iot = IoTManager()
        self.focus = FocusManager()
        
        # Enhanced settings
        self.settings = self._load_settings()
        self.name = self._load_name()
        self.logo = self._load_logo()
        
        # Thread-safe operations
        self.reminders = []
        self.timers = {}
        self.reminder_lock = threading.Lock()
        self.timer_lock = threading.Lock()
        
        # System status
        self.running = True
        self.current_user = None
        self.conversation_context = {}
        
        logging.info("Enhanced Jarvis initialized successfully")

    # Add these new methods to the JarvisAI class
    def authenticate_user(self):
        """Voice authentication for security"""
        self.voice.speak("Please say your authentication phrase")
        audio = self._record_audio_sample()
        user_id = self.security.authenticate_user(audio)
        
        if user_id:
            self.current_user = user_id
            self.voice.speak(f"Welcome back, {user_id}")
            return True
        else:
            self.voice.speak("Authentication failed")
            return False

    def _record_audio_sample(self, duration=3):
        """Record audio sample for authentication"""
        with sr.Microphone() as source:
            self.voice.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.voice.recognizer.listen(source, timeout=duration)
                return audio
            except Exception as e:
                logging.error(f"Audio recording failed: {e}")
                return None

    def handle_task_commands(self, command, params):
        """Process task-related commands"""
        action = params.get('action', 'list')
        
        if action == 'add':
            self.voice.speak("What task would you like to add?")
            task_desc = self.voice.take_command()
            if task_desc:
                task = self.tasks.add_task(task_desc)
                self.voice.speak(f"Added task: {task['description']}")
                
        elif action == 'complete':
            tasks = self.tasks.get_tasks()
            if tasks:
                self.voice.speak("Which task did you complete?")
                for i, task in enumerate(tasks[:5], 1):
                    self.voice.speak(f"{i}. {task['description']}")
                
                choice = self.voice.take_command()
                try:
                    idx = int(choice.split()[0]) - 1
                    if 0 <= idx < len(tasks):
                        self.tasks.complete_task(tasks[idx]['id'])
                        self.voice.speak("Task marked as completed")
                except:
                    self.voice.speak("Couldn't understand which task")
            else:
                self.voice.speak("No active tasks found")
                
        else:  # List tasks
            tasks = self.tasks.get_tasks()
            if tasks:
                self.voice.speak("Here are your current tasks:")
                for i, task in enumerate(tasks[:5], 1):
                    status = " (due soon)" if task.get('due_date') else ""
                    self.voice.speak(f"{i}. {task['description']}{status}")
            else:
                self.voice.speak("No active tasks")

    def handle_iot_commands(self, command, params):
        """Process advanced IoT commands"""
        if 'scene' in command:
            scene = command.replace("scene", "").strip()
            results = self.iot.execute_scene(scene)
            if results:
                self.voice.speak(f"Executed {scene} scene:")
                for result in results:
                    self.voice.speak(result)
            else:
                self.voice.speak(f"No scene named {scene} found")
        else:
            # Fall back to basic smart home control
            self._handle_smarthome_commands(command)

    def handle_focus_commands(self, command):
        """Process focus mode commands"""
        if 'enable' in command or 'start' in command:
            duration = 60  # Default 60 minutes
            if 'for' in command:
                try:
                    duration = int(re.search(r'\d+', command.split('for')[1]).group())
                except:
                    pass
                    
            if self.focus.enable_focus_mode(duration):
                self.voice.speak(f"Focus mode enabled for {duration} minutes")
            else:
                self.voice.speak("Couldn't enable focus mode")
                
        elif 'disable' in command or 'stop' in command:
            if self.focus.disable_focus_mode():
                self.voice.speak("Focus mode disabled")
            else:
                self.voice.speak("Focus mode wasn't active")

    def process_command(self, command: str) -> bool:
        """Enhanced command processing with NLP"""
        if not command or not self.running:
            return False
            
        # Use NLP processor to understand command
        intent, params = self.nlp.process_command(command, self.current_user)
        
        # Handle based on detected intent
        if intent == "media":
            self._handle_media_commands(command)
        elif intent == "task":
            self.handle_task_commands(command, params)
        elif intent == "iot":
            self.handle_iot_commands(command, params)
        elif intent == "focus":
            self.handle_focus_commands(command)
        # Add other intent handlers
        
        return True

    # Add to run() method after initialization
    def run(self):
        """Main execution loop with authentication"""
        print("Initializing Enhanced Jarvis AI...")
        self.greet()
        
        # Authenticate user
        if self.settings.get('require_authentication', False):
            if not self.authenticate_user():
                self.running = False
                return
        
        # Start background threads
        threading.Thread(target=self._monitor_reminders, daemon=True).start()
        threading.Thread(target=self._monitor_timers, daemon=True).start()
        threading.Thread(target=self._monitor_system, daemon=True).start()
        
        while self.running:
            command = self.voice.take_command()
            if command:
                self.process_command(command)

    def _monitor_system(self):
        """Background system monitoring"""
        while self.running:
            # Check for focus mode expiration
            if self.focus.focus_mode and self.focus.focus_sessions:
                last_session = self.focus.focus_sessions[-1]
                if datetime.now() >= last_session['end']:
                    self.focus.disable_focus_mode()
                    self.voice.speak("Focus mode session has ended")
            
            # Check system health periodically
            health = self.health.check_system_health()
            if float(health.get('cpu', '0').strip('%')) > 90:
                self.voice.speak("Warning: CPU usage is very high")
            
            time.sleep(60)  # Check every minute

class JarvisVoice:
    """Enhanced voice handling with better error recovery"""
    def __init__(self):
        self.engine = None
        self.recognizer = None
        self.initialize_voice()
        
    def initialize_voice(self):
        """Initialize voice components with error handling"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.set_voice('male')
            
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        except Exception as e:
            logging.error(f"Voice initialization failed: {e}")
            raise RuntimeError("Voice system initialization failed")

    def set_voice(self, gender: str) -> bool:
        """Set the voice gender (male/female) with fallback"""
        try:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if gender.lower() in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    return True
            # Fallback to first available voice
            if voices:
                self.engine.setProperty('voice', voices[0].id)
                return True
            return False
        except Exception as e:
            logging.error(f"Voice setting failed: {e}")
            return False

    def speak(self, text: str) -> None:
        """Convert text to speech with error handling"""
        try:
            if not self.engine:
                self.initialize_voice()
            logging.info(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Speech failed: {e}")
            print(f"JARVIS: {text}")  # Fallback to text output

    def take_command(self) -> Optional[str]:
        """Robust voice command recognition"""
        if not self.recognizer:
            self.initialize_voice()
            
        with sr.Microphone() as source:
            print("Listening...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                query = self.recognizer.recognize_google(audio, language="en-US").lower()
                print(f"Recognized: {query}")
                return query
            except sr.WaitTimeoutError:
                print("Listening timed out...")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None
            except Exception as e:
                logging.error(f"Voice recognition error: {e}")
                return None

class MediaPlayer:
    """Thread-safe media playback with enhanced error handling"""
    def __init__(self):
        self.instance = None
        self.player = None
        self.is_playing = False
        self.current_volume = 50
        self.playlist = []
        self.current_track_index = 0
        self.media_lock = threading.Lock()
        self.initialize_player()
        
        pygame.mixer.init()
        
    def initialize_player(self):
        """Initialize VLC player with error handling"""
        try:
            self.instance = vlc.Instance('--no-xlib')
            self.player = self.instance.media_player_new()
        except Exception as e:
            logging.error(f"Media player initialization failed: {e}")
            raise RuntimeError("Media system initialization failed")

    def play_youtube(self, song_name: str) -> bool:
        """Robust YouTube playback with error recovery"""
        with self.media_lock:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'quiet': True,
                    'extract_flat': True,
                    'noplaylist': True,
                    'socket_timeout': 10,
                    'retries': 3
                }
                
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(f"ytsearch1:{song_name}", download=False)
                    
                    if not result or 'entries' not in result or not result['entries']:
                        logging.error("No YouTube results found")
                        return False
                    
                    video = result['entries'][0]
                    media = self.instance.media_new(video['url'])
                    self.player.set_media(media)
                    self.player.audio_set_volume(self.current_volume)
                    
                    if self.player.play() == -1:
                        logging.error("VLC playback failed")
                        return False
                    
                    self.is_playing = True
                    self.playlist.append(video['url'])
                    self.current_track_index = len(self.playlist) - 1
                    return True
            except youtube_dl.DownloadError as e:
                logging.error(f"YouTube download error: {e}")
                return False
            except Exception as e:
                logging.error(f"Media playback error: {e}")
                return False

    def play_local(self, file_path: str) -> bool:
        """Safe local file playback"""
        with self.media_lock:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False
                
            try:
                media = self.instance.media_new(file_path)
                self.player.set_media(media)
                self.player.audio_set_volume(self.current_volume)
                if self.player.play() == -1:
                    return False
                
                self.is_playing = True
                self.playlist.append(file_path)
                self.current_track_index = len(self.playlist) - 1
                return True
            except Exception as e:
                logging.error(f"Local playback error: {file_path}: {e}")
                return False

    def control_media(self, action: str) -> bool:
        """Thread-safe media control"""
        with self.media_lock:
            try:
                if action == "pause":
                    if self.player.is_playing():
                        self.player.pause()
                        self.is_playing = False
                        return True
                elif action == "resume":
                    if not self.player.is_playing() and self.is_playing:
                        self.player.play()
                        return True
                elif action == "stop":
                    self.player.stop()
                    self.is_playing = False
                    self.playlist = []
                    self.current_track_index = 0
                    return True
                elif action == "next":
                    if self.current_track_index < len(self.playlist) - 1:
                        self.current_track_index += 1
                        return self.play_current_track()
                elif action == "previous":
                    if self.current_track_index > 0:
                        self.current_track_index -= 1
                        return self.play_current_track()
                return False
            except Exception as e:
                logging.error(f"Media control error: {action}: {e}")
                return False

    def play_current_track(self) -> bool:
        """Play current track safely"""
        try:
            media = self.instance.media_new(self.playlist[self.current_track_index])
            self.player.set_media(media)
            return self.player.play() != -1
        except Exception as e:
            logging.error(f"Track playback error: {e}")
            return False

    def set_volume(self, level: int) -> bool:
        """Safe volume adjustment"""
        with self.media_lock:
            try:
                if 0 <= level <= 100:
                    self.current_volume = level
                    self.player.audio_set_volume(level)
                    return True
                return False
            except Exception as e:
                logging.error(f"Volume set error: {e}")
                return False

class FileManager:
    """Enhanced file operations with validation"""
    @staticmethod
    def search_files(directory: str, extension: str = '*') -> List[str]:
        """Safe file searching with validation"""
        if not os.path.isdir(directory):
            logging.error(f"Invalid directory: {directory}")
            return []
            
        try:
            search_pattern = os.path.join(directory, f'**/*{extension}')
            return glob.glob(search_pattern, recursive=True)
        except Exception as e:
            logging.error(f"File search error: {e}")
            return []

    @staticmethod
    def search_functions(file_path: str) -> List[str]:
        """Robust function parsing"""
        functions = []
        if not os.path.isfile(file_path):
            return functions
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.match(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                    if match:
                        functions.append(match.group(1))
        except Exception as e:
            logging.error(f"Function search error: {file_path}: {e}")
        return functions

    @staticmethod
    def search_functions_in_directory(directory: str) -> Dict[str, List[str]]:
        """Safe directory function scanning"""
        functions_in_files = {}
        python_files = FileManager.search_files(directory, '.py')
        
        for file in python_files:
            functions = FileManager.search_functions(file)
            if functions:
                functions_in_files[file] = functions
                
        return functions_in_files

    @staticmethod
    def get_directory_size(directory: str) -> float:
        """Safe directory size calculation"""
        total_size = 0
        if not os.path.isdir(directory):
            return 0.0
            
        try:
            for dirpath, _, filenames in os.walk(directory):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        total_size += os.path.getsize(fp)
            return round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logging.error(f"Directory size error: {e}")
            return 0.0

class WebServices:
    """Robust web services with timeout handling"""
    def __init__(self):
        self.browser_path = self._get_default_browser()
        self.session = requests.Session()
        self.session.timeout = 10  # 10 second timeout

    @staticmethod
    def _get_default_browser() -> str:
        """Cross-platform browser detection"""
        try:
            if platform.system() == 'Windows':
                import winreg
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'http\shell\open\command') as key:
                    cmd = winreg.QueryValue(key, None)
                    return cmd.split('"')[1]
            elif platform.system() == 'Darwin':
                return '/usr/bin/open'
            else:  # Linux
                return '/usr/bin/xdg-open'
        except Exception as e:
            logging.error(f"Browser detection failed: {e}")
            return ''

    def open_website(self, url: str) -> bool:
        """Safe website opening"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            if self.browser_path:
                subprocess.Popen([self.browser_path, url], stderr=subprocess.DEVNULL)
            else:
                wb.open(url, new=2)
            return True
        except Exception as e:
            logging.error(f"Website open failed: {url}: {e}")
            return False

    def search_google(self, query: str) -> bool:
        """Safe Google search"""
        try:
            url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            return self.open_website(url)
        except Exception as e:
            logging.error(f"Google search failed: {e}")
            return False

    def search_wikipedia(self, query: str) -> str:
        """Robust Wikipedia search"""
        try:
            wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                timeout=10
            )
            page = wiki.page(query)
            
            if not page.exists():
                return "No Wikipedia article found."
                
            summary = page.summary[:500]
            return summary if summary else "Article has no summary."
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}")
            return "Sorry, I couldn't access Wikipedia right now."

class SystemControl:
    """Comprehensive system control with validation"""
    @staticmethod
    def shutdown() -> bool:
        """Safe system shutdown"""
        try:
            if platform.system() == 'Windows':
                os.system("shutdown /s /f /t 1")
            else:
                os.system("shutdown -h now")
            return True
        except Exception as e:
            logging.error(f"Shutdown failed: {e}")
            return False

    @staticmethod
    def restart() -> bool:
        """Safe system restart"""
        try:
            if platform.system() == 'Windows':
                os.system("shutdown /r /f /t 1")
            else:
                os.system("shutdown -r now")
            return True
        except Exception as e:
            logging.error(f"Restart failed: {e}")
            return False

    @staticmethod
    def lock() -> bool:
        """Safe system lock"""
        try:
            if platform.system() == 'Windows':
                ctypes.windll.user32.LockWorkStation()
            elif platform.system() == 'Darwin':
                os.system("pmset displaysleepnow")
            else:
                os.system("gnome-screensaver-command -l")
            return True
        except Exception as e:
            logging.error(f"Lock failed: {e}")
            return False

    @staticmethod
    def get_battery_status() -> str:
        """Robust battery status"""
        try:
            battery = psutil.sensors_battery()
            if not battery:
                return "Battery status not available"
            
            percent = battery.percent
            charging = "charging" if battery.power_plugged else "not charging"
            time_left = battery.secsleft if hasattr(battery, 'secsleft') else None
            
            status = f"Battery is at {percent}% and {charging}."
            if time_left and time_left != psutil.POWER_TIME_UNLIMITED:
                mins = time_left // 60
                status += f" About {mins} minutes remaining."
            return status
        except Exception as e:
            logging.error(f"Battery check failed: {e}")
            return "Could not check battery status"

class CalendarManager:
    """Robust calendar integration with OAuth"""
    def __init__(self):
        self.service = None
        self.calendar_enabled = False
        self.initialize_calendar()
        
    def initialize_calendar(self):
        """Initialize calendar service with error handling"""
        try:
            self.service = self._authenticate()
            self.calendar_enabled = self.service is not None
        except Exception as e:
            logging.error(f"Calendar initialization failed: {e}")
            self.calendar_enabled = False

    def _authenticate(self):
        """Safe OAuth authentication"""
        creds = None
        token_path = 'token.json'
        creds_path = 'credentials.json'
        
        if not os.path.exists(creds_path):
            logging.error("Google API credentials file not found")
            return None
            
        try:
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                    creds = flow.run_local_server(port=0)

                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            return build('calendar', 'v3', credentials=creds) if creds else None
        except Exception as e:
            logging.error(f"Calendar auth failed: {e}")
            return None

    def create_event(self, start_time: str, end_time: str, summary: str) -> bool:
        """Safe event creation"""
        if not self.calendar_enabled:
            return False
            
        try:
            event = {
                'summary': summary,
                'start': {'dateTime': start_time, 'timeZone': 'UTC'},
                'end': {'dateTime': end_time, 'timeZone': 'UTC'},
            }
            self.service.events().insert(calendarId='primary', body=event).execute()
            return True
        except Exception as e:
            logging.error(f"Event creation failed: {e}")
            return False

    def get_events(self, max_results=10) -> List[Dict]:
        """Safe event retrieval"""
        if not self.calendar_enabled:
            return []
            
        try:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            return events_result.get('items', [])
        except Exception as e:
            logging.error(f"Event retrieval failed: {e}")
            return []

class SmartHomeControl:
    """Enhanced smart home control with validation"""
    def __init__(self):
        self.devices = {
            'living room light': {'state': False, 'type': 'switch'},
            'thermostat': {'state': 72, 'type': 'thermostat', 'unit': '°F'},
            'security camera': {'state': False, 'type': 'camera'},
            'coffee maker': {'state': False, 'type': 'appliance'}
        }
        self.device_lock = threading.Lock()
    
    def control_device(self, device: str, action: str) -> str:
        """Thread-safe device control"""
        device = device.lower()
        
        with self.device_lock:
            # Find best matching device
            matched_device = None
            for dev in self.devices:
                if dev in device:
                    matched_device = dev
                    break
                    
            if not matched_device:
                return f"Device {device} not found"
                
            dev_info = self.devices[matched_device]
            
            if action == "on":
                if dev_info['type'] == 'thermostat':
                    return "Use 'set temperature' for thermostats"
                dev_info['state'] = True
                return f"Turned {matched_device} on"
                
            elif action == "off":
                if dev_info['type'] == 'thermostat':
                    return "Use 'set temperature' for thermostats"
                dev_info['state'] = False
                return f"Turned {matched_device} off"
                
            elif action.startswith("set") and dev_info['type'] == 'thermostat':
                try:
                    temp = int(re.search(r'\d+', action).group())
                    dev_info['state'] = temp
                    return f"Set {matched_device} to {temp}{dev_info['unit']}"
                except:
                    return f"Please specify a valid temperature for {matched_device}"
                    
            else:  # Status check
                state = dev_info['state']
                if dev_info['type'] == 'thermostat':
                    return f"{matched_device} is set to {state}{dev_info['unit']}"
                else:
                    return f"{matched_device} is {'on' if state else 'off'}"

class EmailManager:
    """Secure email handling with validation"""
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email = "your_email@gmail.com"  # Should be loaded from config
        self.password = "your_app_password"  # Should be loaded from config
        
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Safe email sending with validation"""
        if not all([to, subject, body]):
            return False
            
        if "@" not in to or "." not in to.split("@")[-1]:
            return False
            
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(self.email, to, message)
            return True
        except Exception as e:
            logging.error(f"Email failed to {to}: {e}")
            return False

class NewsReader:
    """Robust news reader with error handling"""
    def __init__(self):
        self.api_key = "your_newsapi_key"  # Should be loaded from config
        self.base_url = "https://newsapi.org/v2/top-headlines"
        self.session = requests.Session()
        self.session.timeout = 10
        
    def get_headlines(self, category: str = "general", country: str = "us") -> List[str]:
        """Safe headline retrieval"""
        if not self.api_key or self.api_key == "your_newsapi_key":
            return ["News API not configured"]
            
        try:
            params = {
                'category': category,
                'country': country,
                'apiKey': self.api_key
            }
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            headlines = []
            for article in articles[:5]:  # Limit to 5 headlines
                title = article.get('title', '').split(' - ')[0]
                if title:
                    headlines.append(title)
            return headlines if headlines else ["No headlines found"]
        except Exception as e:
            logging.error(f"News fetch error: {e}")
            return ["Could not fetch news headlines"]

class HealthMonitor:
    """Comprehensive system health monitoring"""
    @staticmethod
    def check_system_health() -> Dict:
        """Safe system health check"""
        health = {}
        try:
            # CPU
            cpu = psutil.cpu_percent(interval=1)
            health['cpu'] = f"{cpu}%"
            
            # Memory
            mem = psutil.virtual_memory()
            health['memory'] = (
                f"{mem.percent}% "
                f"({round(mem.used/1e9, 1)}GB/{round(mem.total/1e9, 1)}GB)"
            )
            
            # Disk
            disk = psutil.disk_usage('/')
            health['disk'] = f"{disk.percent}% used"
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    health['temperature'] = f"{temps['coretemp'][0].current}°C"
            except:
                pass
                
            # Network
            net = psutil.net_io_counters()
            health['network'] = (
                f"Sent: {round(net.bytes_sent/1e6, 1)}MB | "
                f"Recv: {round(net.bytes_recv/1e6, 1)}MB"
            )
            
            return health
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return {'error': 'Could not check system health'}

    @staticmethod
    def get_health_advice() -> str:
        """Get system health recommendations"""
        try:
            advice = []
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            if cpu > 90:
                advice.append("CPU critically high - close applications immediately")
            elif cpu > 80:
                advice.append("CPU very high - close some applications")
            elif cpu > 70:
                advice.append("CPU high - consider closing unused apps")
                
            if mem > 90:
                advice.append("Memory critically low - close applications")
            elif mem > 80:
                advice.append("Memory low - consider closing some apps")
                
            if disk > 90:
                advice.append("Disk space critically low - free up space")
            elif disk > 80:
                advice.append("Disk space low - consider cleaning up")
                
            return " | ".join(advice) if advice else "System health is good"
        except Exception as e:
            logging.error(f"Health advice failed: {e}")
            return "Could not assess system health"

class LearningModule:
    """Learning resources with safe URL handling"""
    def __init__(self):
        self.resources = {
            'python': {
                'url': "https://docs.python.org/3/tutorial/",
                'description': "Official Python tutorial"
            },
            'machine learning': {
                'url': "https://www.coursera.org/learn/machine-learning",
                'description': "Andrew Ng's ML course"
            },
            'web development': {
                'url': "https://developer.mozilla.org/en-US/",
                'description': "MDN Web Docs"
            },
            'data science': {
                'url': "https://www.kaggle.com/learn",
                'description': "Kaggle learning resources"
            }
        }
        
    def get_resource(self, topic: str) -> Dict[str, str]:
        """Safe resource lookup"""
        topic = topic.lower()
        for key in self.resources:
            if topic in key:
                return self.resources[key]
        return {
            'url': "",
            'description': f"No resources found for {topic}"
        }

class JarvisAI:
    """Complete Jarvis AI system with all features"""
    def __init__(self, enable_calendar=False):
        # Initialize all components with error handling
        try:
            self.voice = JarvisVoice()
            self.media = MediaPlayer()
            self.files = FileManager()
            self.web = WebServices()
            self.system = SystemControl()
            self.calendar = CalendarManager() if enable_calendar else None
            self.smarthome = SmartHomeControl()
            self.email = EmailManager()
            self.news = NewsReader()
            self.health = HealthMonitor()
            self.learning = LearningModule()
            
            self.settings = self._load_settings()
            self.name = self._load_name()
            self.logo = self._load_logo()
            
            # Thread-safe reminders and timers
            self.reminders = []
            self.timers = {}
            self.reminder_lock = threading.Lock()
            self.timer_lock = threading.Lock()
            
            # System status
            self.running = True
            logging.info("Jarvis initialized successfully")
        except Exception as e:
            logging.error(f"Jarvis initialization failed: {e}")
            raise RuntimeError("Failed to initialize Jarvis")

    def _load_settings(self) -> Dict:
        """Safe settings loading"""
        default_settings = {
            "output_dir": "recordings",
            "default_duration": 10,
            "language": "en-US",
            "voice": "male",
            "volume": 50
        }
        
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    loaded = json.load(f)
                    # Validate loaded settings
                    if isinstance(loaded, dict):
                        return {**default_settings, **loaded}
            return default_settings
        except Exception as e:
            logging.error(f"Settings load failed: {e}")
            return default_settings

    def _load_name(self) -> str:
        """Safe name loading"""
        try:
            if os.path.exists("assistant_name.txt"):
                with open("assistant_name.txt", "r") as f:
                    name = f.read().strip()
                    if name:
                        return name
            return "Jarvis"
        except Exception as e:
            logging.error(f"Name load failed: {e}")
            return "Jarvis"

    def _load_logo(self) -> str:
        """Logo with fallback"""
        try:
            return """                                          
                  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
              ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
              ██║███████║██████╔╝██║   ██║██║███████╗
         ██   ██║██╔══██║██╔═══╝ ██║   ██║██║╚════██║
         ╚█████╔╝██║  ██║██║     ╚██████╔╝██║███████║
          ╚════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝╚══════╝
            """
        except:
            return "JARVIS AI"

    def greet(self) -> None:
        """Personalized greeting"""
        try:
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
        except Exception as e:
            logging.error(f"Greeting failed: {e}")
            print("JARVIS AI initialized")

    def process_command(self, command: str) -> bool:
        """Main command processor with error handling"""
        if not command or not self.running:
            return False
            
        command = command.lower()
        
        try:
            # System commands
            if "exit" in command or "quit" in command:
                self.voice.speak("Goodbye! Have a great day.")
                self.running = False
                return False
                
            elif "shutdown" in command:
                if self.system.shutdown():
                    self.voice.speak("Shutting down the system, goodbye!")
                    self.running = False
                    return False
                else:
                    self.voice.speak("Failed to shutdown system")
                    
            elif "restart" in command:
                if self.system.restart():
                    self.voice.speak("Restarting the system, please wait!")
                    self.running = False
                    return False
                else:
                    self.voice.speak("Failed to restart system")
                    
            elif "lock" in command:
                if self.system.lock():
                    self.voice.speak("System locked")
                else:
                    self.voice.speak("Failed to lock system")
            
            # Media commands
            elif "play" in command:
                self._handle_media_commands(command)
            
            # Web services
            elif "search" in command:
                self._handle_search_commands(command)
                
            # File operations
            elif "search files" in command:
                self._handle_file_commands(command)
            
            # System info
            elif "battery" in command:
                status = self.system.get_battery_status()
                self.voice.speak(status)
            
            # Calendar
            elif ("calendar" in command or "events" in command) and self.calendar:
                self._handle_calendar_commands(command)
                
            # Smart home
            elif "smart home" in command or any(device in command for device in self.smarthome.devices):
                self._handle_smarthome_commands(command)
            
            # Email
            elif "send email" in command:
                self._handle_email_commands()
            
            # News
            elif "news" in command:
                self._handle_news_commands(command)
            
            # System health
            elif "system health" in command or "system status" in command:
                self._handle_health_commands()
            
            # Learning
            elif "learn" in command or "resources" in command:
                self._handle_learning_commands(command)
            
            # Reminders
            elif "set reminder" in command:
                self._handle_reminder_commands()
            
            # Timers
            elif "set timer" in command:
                self._handle_timer_commands(command)
            
            # Jokes
            elif "tell me a joke" in command or "joke" in command:
                joke = pyjokes.get_joke()
                self.voice.speak(joke)
                print(joke)
            
            # Default
            else:
                self.voice.speak("I didn't understand that command. Please try again.")
                
            return True
            
        except Exception as e:
            logging.error(f"Command processing failed: {command}: {e}")
            self.voice.speak("Sorry, I encountered an error processing that command")
            return True

    def _handle_media_commands(self, command: str) -> None:
        """Process media-related commands"""
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
                
        if "pause" in command:
            self.media.control_media("pause")
            self.voice.speak("Media paused")
        elif "resume" in command:
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
                if self.media.set_volume(volume):
                    self.voice.speak(f"Volume set to {volume}%")
                else:
                    self.voice.speak("Invalid volume level")
            except:
                self.voice.speak("Please specify a valid volume between 0 and 100")

    def _handle_search_commands(self, command: str) -> None:
        """Process search commands"""
        query = command.replace("search", "").strip()
        if query:
            if "wikipedia" in command:
                summary = self.web.search_wikipedia(query)
                self.voice.speak(summary)
            else:
                if not self.web.search_google(query):
                    self.voice.speak("Failed to perform search")
        else:
            self.voice.speak("What would you like me to search for?")
            query = self.voice.take_command()
            if query:
                self.web.search_google(query)

    def _handle_file_commands(self, command: str) -> None:
        """Process file-related commands"""
        if "functions" in command:
            self.voice.speak("Which directory should I search for functions?")
            directory = self.voice.take_command()
            if directory:
                functions = self.files.search_functions_in_directory(directory)
                if functions:
                    response = "Found functions in:\n"
                    for file, funcs in functions.items():
                        response += f"{os.path.basename(file)}: {', '.join(funcs[:5])}"
                        if len(funcs) > 5:
                            response += " and more"
                        response += "\n"
                    self.voice.speak(response)
                else:
                    self.voice.speak("No functions found in that directory")
        else:
            self.voice.speak("What type of files should I look for?")
            file_type = self.voice.take_command()
            if file_type:
                self.voice.speak("In which directory?")
                directory = self.voice.take_command()
                if directory:
                    files = self.files.search_files(directory, file_type)
                    if files:
                        self.voice.speak(f"Found {len(files)} files")
                        for i, file in enumerate(files[:3], 1):
                            self.voice.speak(f"File {i}: {os.path.basename(file)}")
                    else:
                        self.voice.speak("No files found matching your criteria")

    def _handle_calendar_commands(self, command: str) -> None:
        """Process calendar-related commands"""
        if "create" in command or "add" in command:
            self.voice.speak("What's the event title?")
            title = self.voice.take_command()
            if title:
                self.voice.speak("When does it start? (e.g., tomorrow 2pm)")
                start = self.voice.take_command()
                if start:
                    try:
                        parsed_start = self._parse_time_expression(start)
                        self.voice.speak("When does it end? (e.g., tomorrow 3pm)")
                        end = self.voice.take_command()
                        if end:
                            parsed_end = self._parse_time_expression(end)
                            if self.calendar.create_event(
                                parsed_start.isoformat(),
                                parsed_end.isoformat(),
                                title
                            ):
                                self.voice.speak("Event created successfully")
                            else:
                                self.voice.speak("Failed to create event")
                    except ValueError as e:
                        self.voice.speak(f"Couldn't understand the time: {str(e)}")
        else:
            events = self.calendar.get_events()
            if events:
                self.voice.speak("Here are your upcoming events:")
                for event in events[:3]:  # Limit to 3 events
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    summary = event.get('summary', 'No title')
                    self.voice.speak(f"{summary} at {self._format_calendar_time(start)}")
            else:
                self.voice.speak("No upcoming events found")

    def _parse_time_expression(self, time_str: str) -> datetime:
        """Convert natural language time to datetime"""
        now = datetime.now()
        
        if "tomorrow" in time_str:
            now += timedelta(days=1)
            time_str = time_str.replace("tomorrow", "").strip()
        
        if "am" in time_str or "pm" in time_str:
            time_format = "%I %p"  # 2 PM format
        else:
            time_format = "%H:%M"  # 14:00 format
            
        try:
            parsed_time = datetime.strptime(time_str, time_format)
            return datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=parsed_time.hour,
                minute=parsed_time.minute
            )
        except ValueError:
            raise ValueError("Please specify time in a format like '2pm' or '14:00'")

    def _format_calendar_time(self, time_str: str) -> str:
        """Format calendar time for speaking"""
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.strftime("%A %B %d at %I:%M %p")
        except:
            return time_str

    def _handle_smarthome_commands(self, command: str) -> None:
        """Process smart home commands"""
        for device in self.smarthome.devices:
            if device in command:
                if "on" in command:
                    response = self.smarthome.control_device(device, "on")
                elif "off" in command:
                    response = self.smarthome.control_device(device, "off")
                elif "set" in command and "temperature" in command:
                    response = self.smarthome.control_device(device, command)
                else:
                    response = self.smarthome.control_device(device, "status")
                self.voice.speak(response)
                return
        
        self.voice.speak("Available smart home devices: " + 
                        ", ".join(self.smarthome.devices.keys()))

    def _handle_email_commands(self) -> None:
        """Process email commands"""
        self.voice.speak("Who should I send the email to?")
        recipient = self.voice.take_command()
        if recipient:
            self.voice.speak("What's the subject?")
            subject = self.voice.take_command()
            if subject:
                self.voice.speak("What's the message?")
                body = self.voice.take_command()
                if body:
                    if self.email.send_email(recipient, subject, body):
                        self.voice.speak("Email sent successfully")
                    else:
                        self.voice.speak("Failed to send email")

    def _handle_news_commands(self, command: str) -> None:
        """Process news commands"""
        category = "general"
        if "sports" in command:
            category = "sports"
        elif "technology" in command:
            category = "technology"
        elif "business" in command:
            category = "business"
        elif "science" in command:
            category = "science"
            
        headlines = self.news.get_headlines(category)
        if headlines:
            self.voice.speak(f"Here are the latest {category} news headlines")
            for i, headline in enumerate(headlines[:3], 1):  # Limit to 3 headlines
                self.voice.speak(f"Headline {i}: {headline}")
        else:
            self.voice.speak(f"Could not retrieve {category} news")

    def _handle_health_commands(self) -> None:
        """Process system health commands"""
        health = self.health.check_system_health()
        if 'error' in health:
            self.voice.speak(health['error'])
        else:
            response = "System status: "
            for metric, value in health.items():
                response += f"{metric}: {value}. "
            self.voice.speak(response)
            
            advice = self.health.get_health_advice()
            if advice:
                self.voice.speak(f"Recommendation: {advice}")

    def _handle_learning_commands(self, command: str) -> None:
        """Process learning resource commands"""
        topic = command.replace("learn", "").replace("resources", "").strip()
        if not topic:
            self.voice.speak("What topic would you like to learn about?")
            topic = self.voice.take_command()
            
        if topic:
            resource = self.learning.get_resource(topic)
            if resource['url']:
                self.voice.speak(f"Here's a resource for {topic}: {resource['description']}")
                self.web.open_website(resource['url'])
            else:
                self.voice.speak(resource['description'])

    def _handle_reminder_commands(self) -> None:
        """Process reminder commands"""
        self.voice.speak("What should I remind you about?")
        reminder_text = self.voice.take_command()
        if reminder_text:
            self.voice.speak("When should I remind you? (e.g., in 30 minutes or tomorrow at 9am)")
            time_str = self.voice.take_command()
            if time_str:
                try:
                    reminder_time = self._parse_reminder_time(time_str)
                    with self.reminder_lock:
                        self.reminders.append((reminder_time, reminder_text))
                    self.voice.speak(f"Okay, I'll remind you at {reminder_time.strftime('%I:%M %p')}")
                    
                    # Start reminder thread
                    threading.Thread(
                        target=self._check_reminders,
                        daemon=True
                    ).start()
                except ValueError as e:
                    self.voice.speak(str(e))

    def _parse_reminder_time(self, time_str: str) -> datetime:
        """Parse natural language reminder time"""
        now = datetime.now()
        
        # Handle "in X minutes/hours"
        if "in " in time_str:
            if "minute" in time_str:
                minutes = int(re.search(r'\d+', time_str).group())
                return now + timedelta(minutes=minutes)
            elif "hour" in time_str:
                hours = int(re.search(r'\d+', time_str).group())
                return now + timedelta(hours=hours)
        
        # Handle specific times
        return self._parse_time_expression(time_str)

    def _check_reminders(self) -> None:
        """Background thread to check reminders"""
        while self.running:
            with self.reminder_lock:
                now = datetime.now()
                for reminder in self.reminders[:]:  # Create a copy for iteration
                    if now >= reminder[0]:
                        self.voice.speak(f"Reminder: {reminder[1]}")
                        self.reminders.remove(reminder)
            time.sleep(30)  # Check every 30 seconds

    def _handle_timer_commands(self, command: str) -> None:
        """Process timer commands"""
        try:
            # Extract duration from command
            duration = 0
            if "minute" in command:
                duration = int(re.search(r'\d+', command).group()) * 60
            elif "second" in command:
                duration = int(re.search(r'\d+', command).group())
            elif "hour" in command:
                duration = int(re.search(r'\d+', command).group()) * 3600
            else:
                self.voice.speak("Please specify a duration like '5 minutes'")
                return
                
            timer_name = f"timer_{len(self.timers) + 1}"
            end_time = datetime.now() + timedelta(seconds=duration)
            
            with self.timer_lock:
                self.timers[timer_name] = end_time
                
            self.voice.speak(f"Timer set for {duration} seconds")
            
            # Start timer thread
            threading.Thread(
                target=self._check_timers,
                args=(timer_name,),
                daemon=True
            ).start()
        except:
            self.voice.speak("Could not set timer. Please try again.")

    def _check_timers(self, timer_name: str) -> None:
        """Background thread to check timers"""
        with self.timer_lock:
            end_time = self.timers.get(timer_name)
            if not end_time:
                return
                
        while datetime.now() < end_time and self.running:
            time.sleep(1)
            
        with self.timer_lock:
            if timer_name in self.timers:
                self.voice.speak("Timer complete!")
                del self.timers[timer_name]

    def run(self) -> None:
        """Main execution loop"""
        print("Initializing Jarvis AI...")
        self.greet()
        
        # Start background threads
        threading.Thread(target=self._monitor_reminders, daemon=True).start()
        threading.Thread(target=self._monitor_timers, daemon=True).start()
        
        while self.running:
            command = self.voice.take_command()
            if command:
                self.process_command(command)

    def _monitor_reminders(self) -> None:
        """Background reminder monitoring"""
        while self.running:
            now = datetime.now()
            with self.reminder_lock:
                for reminder in self.reminders[:]:
                    if now >= reminder[0]:
                        self.voice.speak(f"Reminder: {reminder[1]}")
                        self.reminders.remove(reminder)
            time.sleep(30)

    def _monitor_timers(self) -> None:
        """Background timer monitoring"""
        while self.running:
            now = datetime.now()
            with self.timer_lock:
                for timer_name, end_time in list(self.timers.items()):
                    if now >= end_time:
                        self.voice.speak("Timer complete!")
                        del self.timers[timer_name]
            time.sleep(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        filename='jarvis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        jarvis = JarvisAI(enable_calendar=True)
        jarvis.run()
    except KeyboardInterrupt:
        print("\nJarvis is shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"An error occurred: {e}")