import os
import re
import time
import json
import glob
import math
import random
import logging
import datetime
import platform
import threading
import subprocess
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser
import pyautogui
import pyjokes
import requests
import smtplib
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
import socket
import wakeonlan
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from cryptography.fernet import Fernet
import requests
import platform
import subprocess
from typing import List, Dict, Optional
from dataclasses import dataclass

import yt_dlp as youtube_dl

from dataclasses import dataclass
from typing import List, Optional, Dict,Tuple




import fnmatch
from dataclasses import dataclass
import subprocess



import yt_dlp as youtube_dl
from dataclasses import dataclass
from typing import Optional, List
from queue import Queue

# Constants
GOOGLE_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_custom_search_engine_id"
SCOPES = ['https://www.googleapis.com/auth/calendar.events']
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
NEWS_API_KEY = "your_newsapi_key"


from dataclasses import dataclass
from typing import Dict, List, Optional
import wakeonlan


import hashlib
import cryptography
from cryptography.fernet import Fernet
import scapy.all as scapy
from typing import Dict, List, Optional, Tuple
import subprocess

import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import speech_recognition as sr

import numpy as np
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import pytz


import statistics

from typing import Union, List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import sympy as sp
from scipy import integrate, optimize


import shutil

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import mimetypes

DEFAULT_CREDENTIALS = {
    "assistant_name": "Ultron",
    "voice": "male",
    "volume": 70,
    "hotwords": ["hey assistant", "computer"]
}

def create_credentials_file(file_path="credentials.json"):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(DEFAULT_CREDENTIALS, f, indent=4)
        # print(f"Created {file_path} with default values")
    else:
        ...
create_credentials_file()

import os
import json
import base64
import secrets
import string
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import getpass

@dataclass
class PasswordEntry:
    """
    Represents a single password entry with metadata
    """
    service: str
    username: str
    password: str  # Encrypted
    url: Optional[str] = None
    notes: Optional[str] = None
    last_updated: Optional[str] = None
    category: Optional[str] = None
    strength: Optional[int] = None

class PasswordManager:
    """
    Advanced password manager with secure encryption and password generation
    """
    def __init__(self, master_password: str, db_file: str = "passwords.enc"):
        """
        Initialize the password manager with a master password
        
        Args:
            master_password: The master password used to encrypt/decrypt the database
            db_file: File to store encrypted passwords
        """
        self.db_file = db_file
        self.master_password = master_password.encode()
        self.key = self._derive_key(self.master_password)
        self.cipher = Fernet(self.key)
        self.entries: Dict[str, PasswordEntry] = {}
        self.loaded = False
        
        # Security settings
        self.min_password_length = 12
        self.max_password_length = 32
        self.default_password_length = 16
        self.iterations = 100000  # For key derivation
        
    def _derive_key(self, password: bytes, salt: bytes = None) -> bytes:
        """
        Derive a cryptographic key from the master password using PBKDF2
        
        Args:
            password: The master password as bytes
            salt: Optional salt (generates new if None)
            
        Returns:
            Derived key as bytes
        """
        if salt is None:
            salt = os.urandom(16)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string using Fernet encryption
        
        Args:
            plaintext: Text to encrypt
            
        Returns:
            Encrypted string
        """
        return self.cipher.encrypt(plaintext.encode()).decode()
    
    def _decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string using Fernet encryption
        
        Args:
            ciphertext: Text to decrypt
            
        Returns:
            Decrypted string
        """
        return self.cipher.decrypt(ciphertext.encode()).decode()
    
    def _calculate_password_strength(self, password: str) -> int:
        """
        Calculate password strength score (0-100)
        
        Args:
            password: Password to evaluate
            
        Returns:
            Strength score (0-100)
        """
        length = len(password)
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in string.punctuation for c in password)
        
        # Simple scoring system
        score = 0
        score += min(length, 20) * 3  # Max 60 points for length
        score += 10 if has_upper else 0
        score += 10 if has_lower else 0
        score += 10 if has_digit else 0
        score += 10 if has_special else 0
        
        return min(score, 100)
    
    def generate_password(self, length: int = None, 
                         include_upper: bool = True,
                         include_lower: bool = True,
                         include_digits: bool = True,
                         include_special: bool = True) -> str:
        """
        Generate a secure random password
        
        Args:
            length: Length of password (defaults to default_password_length)
            include_upper: Include uppercase letters
            include_lower: Include lowercase letters
            include_digits: Include digits
            include_special: Include special characters
            
        Returns:
            Generated password
        """
        if length is None:
            length = self.default_password_length
            
        length = max(self.min_password_length, min(length, self.max_password_length))
        chars = []
        
        if include_upper:
            chars.extend(string.ascii_uppercase)
        if include_lower:
            chars.extend(string.ascii_lowercase)
        if include_digits:
            chars.extend(string.digits)
        if include_special:
            chars.extend(string.punctuation)
            
        if not chars:
            raise ValueError("At least one character set must be included")
            
        while True:
            password = ''.join(secrets.choice(chars) for _ in range(length))
            # Ensure we meet complexity requirements
            if (not include_upper or any(c.isupper() for c in password)) and \
               (not include_lower or any(c.islower() for c in password)) and \
               (not include_digits or any(c.isdigit() for c in password)) and \
               (not include_special or any(c in string.punctuation for c in password)):
                return password
    
    def add_entry(self, service: str, username: str, password: str, 
                 url: str = None, notes: str = None, category: str = None) -> PasswordEntry:
        """
        Add a new password entry
        
        Args:
            service: Service/website name
            username: Username/email
            password: Plaintext password
            url: Optional URL
            notes: Optional notes
            category: Optional category
            
        Returns:
            The created PasswordEntry
        """
        if service in self.entries:
            raise ValueError(f"Entry for {service} already exists")
            
        encrypted_password = self._encrypt(password)
        strength = self._calculate_password_strength(password)
        entry = PasswordEntry(
            service=service,
            username=username,
            password=encrypted_password,
            url=url,
            notes=notes,
            last_updated=self._get_current_timestamp(),
            category=category,
            strength=strength
        )
        self.entries[service] = entry
        return entry
    
    def update_entry(self, service: str, **kwargs) -> PasswordEntry:
        """
        Update an existing password entry
        
        Args:
            service: Service name to update
            **kwargs: Fields to update (username, password, url, notes, category)
            
        Returns:
            Updated PasswordEntry
        """
        if service not in self.entries:
            raise ValueError(f"No entry found for {service}")
            
        entry = self.entries[service]
        
        if 'password' in kwargs:
            kwargs['password'] = self._encrypt(kwargs['password'])
            kwargs['strength'] = self._calculate_password_strength(kwargs['password'])
            
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
                
        entry.last_updated = self._get_current_timestamp()
        return entry
    
    def get_entry(self, service: str, decrypt_password: bool = True) -> PasswordEntry:
        """
        Get a password entry
        
        Args:
            service: Service name
            decrypt_password: Whether to decrypt the password
            
        Returns:
            PasswordEntry (with decrypted password if requested)
        """
        if service not in self.entries:
            raise ValueError(f"No entry found for {service}")
            
        entry = self.entries[service]
        
        if decrypt_password:
            decrypted_entry = PasswordEntry(
                service=entry.service,
                username=entry.username,
                password=self._decrypt(entry.password),
                url=entry.url,
                notes=entry.notes,
                last_updated=entry.last_updated,
                category=entry.category,
                strength=entry.strength
            )
            return decrypted_entry
            
        return entry
    
    def delete_entry(self, service: str) -> None:
        """
        Delete a password entry
        
        Args:
            service: Service name to delete
        """
        if service in self.entries:
            del self.entries[service]
    
    def search_entries(self, query: str, search_fields: List[str] = None) -> List[PasswordEntry]:
        """
        Search password entries
        
        Args:
            query: Search query
            search_fields: Fields to search (defaults to service, username, url, notes)
            
        Returns:
            List of matching PasswordEntry objects
        """
        if search_fields is None:
            search_fields = ['service', 'username', 'url', 'notes', 'category']
            
        results = []
        query = query.lower()
        
        for entry in self.entries.values():
            for field in search_fields:
                value = getattr(entry, field, "")
                if value and query in str(value).lower():
                    results.append(entry)
                    break
                    
        return results
    
    def get_all_entries(self) -> List[PasswordEntry]:
        """
        Get all password entries (without decrypting passwords)
        
        Returns:
            List of all PasswordEntry objects
        """
        return list(self.entries.values())
    
    def export_entries(self, file_path: str, format: str = 'json') -> None:
        """
        Export password entries to a file
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        entries = [vars(entry) for entry in self.get_all_entries()]
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(entries, f, indent=2)
        elif format == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                writer.writeheader()
                writer.writerows(entries)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_entries(self, file_path: str, format: str = 'json') -> None:
        """
        Import password entries from a file
        
        Args:
            file_path: Path to import file
            format: Import format ('json' or 'csv')
        """
        if format == 'json':
            with open(file_path, 'r') as f:
                entries = json.load(f)
        elif format == 'csv':
            import csv
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        for entry_data in entries:
            # Skip if service already exists
            if entry_data['service'] in self.entries:
                continue
                
            # Create new entry
            entry = PasswordEntry(
                service=entry_data['service'],
                username=entry_data['username'],
                password=entry_data['password'],  # Should already be encrypted
                url=entry_data.get('url'),
                notes=entry_data.get('notes'),
                last_updated=entry_data.get('last_updated'),
                category=entry_data.get('category'),
                strength=entry_data.get('strength')
            )
            self.entries[entry.service] = entry
    
    def change_master_password(self, new_password: str) -> None:
        """
        Change the master password and re-encrypt all passwords
        
        Args:
            new_password: New master password
        """
        # Decrypt all passwords with old key
        decrypted_entries = []
        for service, entry in self.entries.items():
            decrypted = self.get_entry(service, decrypt_password=True)
            decrypted_entries.append(decrypted)
        
        # Change master password and derive new key
        self.master_password = new_password.encode()
        self.key = self._derive_key(self.master_password)
        self.cipher = Fernet(self.key)
        
        # Re-encrypt all passwords with new key
        self.entries = {}
        for entry in decrypted_entries:
            self.add_entry(
                service=entry.service,
                username=entry.username,
                password=entry.password,
                url=entry.url,
                notes=entry.notes,
                category=entry.category
            )
    
    def save(self) -> None:
        """
        Save encrypted password database to file
        """
        # Convert entries to dict and encrypt sensitive data
        data = {
            'entries': [],
            'metadata': {
                'version': '1.0',
                'created_at': self._get_current_timestamp()
            }
        }
        
        for entry in self.entries.values():
            entry_data = vars(entry)
            data['entries'].append(entry_data)
        
        # Encrypt the entire database
        encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
        
        with open(self.db_file, 'wb') as f:
            f.write(encrypted_data)
    
    def load(self) -> bool:
        """
        Load encrypted password database from file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.db_file):
            return False
            
        try:
            with open(self.db_file, 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            self.entries = {}
            for entry_data in data.get('entries', []):
                entry = PasswordEntry(
                    service=entry_data['service'],
                    username=entry_data['username'],
                    password=entry_data['password'],
                    url=entry_data.get('url'),
                    notes=entry_data.get('notes'),
                    last_updated=entry_data.get('last_updated'),
                    category=entry_data.get('category'),
                    strength=entry_data.get('strength')
                )
                self.entries[entry.service] = entry
                
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading password database: {e}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        
        Returns:
            Current timestamp as string
        """
        return datetime.now().isoformat()
    
    def get_password_strength_report(self) -> Dict:
        """
        Generate a report on password strengths
        
        Returns:
            Dictionary with strength statistics
        """
        strengths = []
        weak_passwords = []
        
        for entry in self.entries.values():
            strength = entry.strength or self._calculate_password_strength(self._decrypt(entry.password))
            strengths.append(strength)
            if strength < 70:  # Considered weak
                weak_passwords.append({
                    'service': entry.service,
                    'username': entry.username,
                    'strength': strength
                })
        
        if not strengths:
            return {}
            
        return {
            'average_strength': sum(strengths) / len(strengths),
            'min_strength': min(strengths),
            'max_strength': max(strengths),
            'weak_passwords': sorted(weak_passwords, key=lambda x: x['strength']),
            'total_entries': len(strengths),
            'strong_passwords': len([s for s in strengths if s >= 80]),
            'medium_passwords': len([s for s in strengths if 50 <= s < 80]),
            'weak_passwords_count': len([s for s in strengths if s < 50])
        }
    
    def get_duplicate_passwords(self) -> List[Dict]:
        """
        Find duplicate passwords across entries
        
        Returns:
            List of duplicate password groups
        """
        password_map = {}
        
        for entry in self.entries.values():
            decrypted = self._decrypt(entry.password)
            if decrypted not in password_map:
                password_map[decrypted] = []
            password_map[decrypted].append({
                'service': entry.service,
                'username': entry.username
            })
            
        return [entries for entries in password_map.values() if len(entries) > 1]
    
    def get_old_passwords(self, days_threshold: int = 180) -> List[Dict]:
        """
        Find passwords that haven't been updated in a while
        
        Args:
            days_threshold: Number of days to consider a password "old"
            
        Returns:
            List of old password entries
        """
        old_entries = []
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        for entry in self.entries.values():
            if entry.last_updated:
                updated_date = datetime.fromisoformat(entry.last_updated)
                if updated_date < threshold_date:
                    old_entries.append({
                        'service': entry.service,
                        'username': entry.username,
                        'last_updated': entry.last_updated,
                        'days_old': (datetime.now() - updated_date).days
                    })
                    
        return sorted(old_entries, key=lambda x: x['days_old'], reverse=True)


class VoiceAssistantConfig:
    def __init__(self, config_file: str = "assistant_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file or create default"""
        default_config = {
            "name": "Ultron",
            "voice": "male",
            "volume": 70,
            "hotwords": ["hey assistant", "computer"]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**default_config, **loaded_config}
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def change_name(self, new_name: str) -> bool:
        """Change the assistant's name"""
        if not new_name.strip():
            return False
            
        self.config["name"] = new_name.strip()
        return self.save_config()
    
    def get_name(self) -> str:
        """Get current assistant name"""
        return self.config["name"]
    
    def get_all_config(self) -> Dict:
        """Return complete configuration"""
        return self.config
    
    def update_config(self, new_config: Dict) -> bool:
        """Update multiple settings at once"""
        self.config.update(new_config)
        return self.save_config()





@dataclass
class FileInfo:
    name: str
    path: str
    size: int  # in bytes
    modified: datetime
    file_type: str
    md5_hash: Optional[str] = None

class FileManager:
    def __init__(self, default_dir: str = None):
        self.current_dir = default_dir or os.getcwd()
        self.history: List[Dict] = []
        self.bookmarks: Dict[str, str] = {}
        
    def _log_operation(self, operation: str, path: str) -> None:
        """Log file operations to history"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'path': path,
            'directory': self.current_dir
        })
    
    def change_directory(self, path: str) -> bool:
        """Change current working directory"""
        try:
            if os.path.isdir(path):
                self.current_dir = os.path.abspath(path)
                return True
            elif os.path.isdir(os.path.join(self.current_dir, path)):
                self.current_dir = os.path.abspath(os.path.join(self.current_dir, path))
                return True
            return False
        except Exception as e:
            logging.error(f"Directory change failed: {e}")
            return False
    
    def create_file(self, filename: str, content: str = "") -> bool:
        """Create a new text file"""
        try:
            full_path = os.path.join(self.current_dir, filename)
            with open(full_path, 'w') as f:
                f.write(content)
            self._log_operation('create', full_path)
            return True
        except Exception as e:
            logging.error(f"File creation failed: {e}")
            return False
    
    def create_directory(self, dirname: str) -> bool:
        """Create a new directory"""
        try:
            full_path = os.path.join(self.current_dir, dirname)
            os.makedirs(full_path, exist_ok=True)
            self._log_operation('create_dir', full_path)
            return True
        except Exception as e:
            logging.error(f"Directory creation failed: {e}")
            return False
    
    def search_files(self, pattern: str, recursive: bool = True) -> List[FileInfo]:
        """Search for files matching pattern"""
        try:
            search_path = os.path.join(self.current_dir, '**' if recursive else '', pattern)
            files = []
            for filepath in glob.glob(search_path, recursive=recursive):
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append(self._get_file_info(filepath))
            return files
        except Exception as e:
            logging.error(f"File search failed: {e}")
            return []
    
    def _get_file_info(self, filepath: str) -> FileInfo:
        """Get detailed information about a file"""
        stat = os.stat(filepath)
        mime_type, _ = mimetypes.guess_type(filepath)
        return FileInfo(
            name=os.path.basename(filepath),
            path=filepath,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            file_type=mime_type or 'unknown'
        )
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file"""
        try:
            full_path = os.path.join(self.current_dir, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)
                self._log_operation('delete', full_path)
                return True
            return False
        except Exception as e:
            logging.error(f"File deletion failed: {e}")
            return False
    
    def delete_directory(self, dirname: str) -> bool:
        """Delete a directory (recursively)"""
        try:
            full_path = os.path.join(self.current_dir, dirname)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                self._log_operation('delete_dir', full_path)
                return True
            return False
        except Exception as e:
            logging.error(f"Directory deletion failed: {e}")
            return False
    
    def get_directory_contents(self) -> Tuple[List[FileInfo], List[FileInfo]]:
        """Get files and subdirectories in current directory"""
        try:
            files = []
            dirs = []
            for item in os.listdir(self.current_dir):
                full_path = os.path.join(self.current_dir, item)
                if os.path.isfile(full_path):
                    files.append(self._get_file_info(full_path))
                elif os.path.isdir(full_path):
                    dirs.append(self._get_file_info(full_path))
            return files, dirs
        except Exception as e:
            logging.error(f"Directory listing failed: {e}")
            return [], []
    
    def calculate_md5(self, filename: str) -> Optional[str]:
        """Calculate MD5 hash of a file"""
        try:
            full_path = os.path.join(self.current_dir, filename)
            if os.path.isfile(full_path):
                hash_md5 = hashlib.md5()
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            return None
        except Exception as e:
            logging.error(f"MD5 calculation failed: {e}")
            return None
    
    def add_bookmark(self, name: str, path: str) -> bool:
        """Bookmark a directory for quick access"""
        try:
            if os.path.isdir(path):
                self.bookmarks[name] = os.path.abspath(path)
                return True
            return False
        except Exception as e:
            logging.error(f"Bookmark failed: {e}")
            return False
    
    def goto_bookmark(self, name: str) -> bool:
        """Navigate to a bookmarked directory"""
        if name in self.bookmarks:
            return self.change_directory(self.bookmarks[name])
        return False
    
    def get_file_content(self, filename: str, lines: int = None) -> Optional[List[str]]:
        """Read file content (optionally limited to first N lines)"""
        try:
            full_path = os.path.join(self.current_dir, filename)
            if os.path.isfile(full_path):
                with open(full_path, 'r') as f:
                    if lines:
                        return [next(f) for _ in range(lines)]
                    return f.readlines()
            return None
        except Exception as e:
            logging.error(f"File reading failed: {e}")
            return None




class MathOperation(Enum):
    BASIC = auto()
    SCIENTIFIC = auto()
    STATISTICAL = auto()
    MATRIX = auto()
    CALCULUS = auto()

@dataclass
class CalculationResult:
    value: Union[float, List, Dict]
    operation_type: MathOperation
    steps: Optional[List[str]] = None
    error: Optional[str] = None

class MathCalculatorAdvance:
    def __init__(self):
        self.history: List[CalculationResult] = []
        self.variables: Dict[str, float] = {'ans': 0.0}
        self.angle_mode = 'radians'  # or 'degrees'

    def calculate(self, expression: str) -> CalculationResult:
        """Evaluate a mathematical expression with extended capabilities"""
        try:
            # Handle special commands
            if expression.lower().startswith('solve '):
                return self._solve_equation(expression[6:])
            elif expression.lower().startswith('integrate '):
                return self._perform_integration(expression[9:])
            elif '=' in expression:
                return self._handle_variable_assignment(expression)
            
            # Normal calculation
            result = self._safe_eval(expression)
            self.variables['ans'] = result
            self.history.append(CalculationResult(result, MathOperation.BASIC))
            return CalculationResult(result, MathOperation.BASIC)
        except Exception as e:
            logging.error(f"Calculation error: {e}")
            return CalculationResult(float('nan'), MathOperation.BASIC, error=str(e))

    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate mathematical expressions"""
        # Convert trigonometric functions based on angle mode
        if self.angle_mode == 'degrees':
            expr = self._convert_degrees_to_radians(expr)
        
        allowed_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'factorial': math.factorial,
            'pi': math.pi, 'e': math.e, 'tau': math.tau,
            'radians': math.radians, 'degrees': math.degrees,
            'ceil': math.ceil, 'floor': math.floor,
            'gcd': math.gcd, 'lcm': self._lcm,
            'abs': abs, 'round': round
        }
        
        # Add variables to the allowed functions
        allowed_functions.update(self.variables)
        
        return eval(expr, {'__builtins__': None}, allowed_functions)

    def _convert_degrees_to_radians(self, expr: str) -> str:
        """Convert trigonometric functions to use degrees"""
        trig_functions = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']
        for func in trig_functions:
            expr = re.sub(fr'{func}\((.+?)\)', fr'math.{func}(math.radians(\1))', expr)
        return expr

    def _handle_variable_assignment(self, expr: str) -> CalculationResult:
        """Handle variable assignments like 'x = 5 + 3'"""
        var_name, calculation = expr.split('=', 1)
        var_name = var_name.strip()
        result = self._safe_eval(calculation.strip())
        self.variables[var_name] = result
        return CalculationResult(result, MathOperation.BASIC, 
                               steps=[f"Assigned {var_name} = {result}"])

    def _solve_equation(self, equation: str) -> CalculationResult:
        """Solve algebraic equations symbolically"""
        try:
            x = sp.symbols('x')
            solution = sp.solve(equation, x)
            steps = [f"Solved equation: {equation}", f"Solution: {solution}"]
            return CalculationResult(float(solution[0]) if len(solution) == 1 else solution,
                                   MathOperation.SCIENTIFIC, steps)
        except Exception as e:
            return CalculationResult(float('nan'), MathOperation.SCIENTIFIC, error=str(e))

    def _perform_integration(self, expression: str) -> CalculationResult:
        """Perform numerical integration"""
        try:
            # Format: "integrate x^2 from 0 to 1"
            parts = re.split(r' from | to ', expression)
            func_str = parts[0]
            a = float(parts[1])
            b = float(parts[2])
            
            # Convert x^2 to x**2
            func_str = func_str.replace('^', '**')
            func = lambda x: eval(func_str, {'__builtins__': None}, {'x': x})
            
            result = integrate.quad(func, a, b)[0]
            steps = [f"Integrated {func_str} from {a} to {b}", f"Result: {result}"]
            return CalculationResult(result, MathOperation.CALCULUS, steps)
        except Exception as e:
            return CalculationResult(float('nan'), MathOperation.CALCULUS, error=str(e))

    def statistical_analysis(self, data: List[float]) -> CalculationResult:
        """Perform statistical analysis on a dataset"""
        try:
            results = {
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'mode': statistics.mode(data),
                'stdev': statistics.stdev(data),
                'variance': statistics.variance(data),
                'min': min(data),
                'max': max(data),
                'sum': sum(data)
            }
            return CalculationResult(results, MathOperation.STATISTICAL)
        except Exception as e:
            return CalculationResult(float('nan'), MathOperation.STATISTICAL, error=str(e))

    def matrix_operations(self, operation: str, matrix1: List[List[float]], 
                         matrix2: Optional[List[List[float]]] = None) -> CalculationResult:
        """Perform matrix operations"""
        try:
            np_matrix1 = np.array(matrix1)
            np_matrix2 = np.array(matrix2) if matrix2 else None
            
            if operation == 'add':
                result = np_matrix1 + np_matrix2
            elif operation == 'multiply':
                result = np_matrix1 @ np_matrix2
            elif operation == 'inverse':
                result = np.linalg.inv(np_matrix1)
            elif operation == 'determinant':
                result = np.linalg.det(np_matrix1)
            elif operation == 'transpose':
                result = np_matrix1.T
            else:
                raise ValueError("Unsupported matrix operation")
                
            return CalculationResult(result.tolist(), MathOperation.MATRIX)
        except Exception as e:
            return CalculationResult(float('nan'), MathOperation.MATRIX, error=str(e))

    def set_angle_mode(self, mode: str) -> None:
        """Set angle mode to degrees or radians"""
        if mode.lower() in ['degrees', 'radians']:
            self.angle_mode = mode.lower()
        else:
            raise ValueError("Angle mode must be 'degrees' or 'radians'")

    def get_history(self, n: int = 5) -> List[CalculationResult]:
        """Get last n calculations from history"""
        return self.history[-n:]

    def clear_history(self) -> None:
        """Clear calculation history"""
        self.history = []

    @staticmethod
    def _lcm(*args: int) -> int:
        """Calculate least common multiple"""
        return math.lcm(*args)

    def plot_function(self, func_str: str, x_range: Tuple[float, float]) -> None:
        """Plot a mathematical function (would use matplotlib in full implementation)"""
        print(f"Plotting {func_str} from {x_range[0]} to {x_range[1]}")
        # In a real implementation, this would use matplotlib to show the graph





class MeetingStatus(Enum):
    SCHEDULED = auto()
    ONGOING = auto()
    ENDED = auto()
    CANCELLED = auto()

@dataclass
class MeetingParticipant:
    name: str
    email: str
    join_time: Optional[str] = None
    leave_time: Optional[str] = None
    is_host: bool = False

@dataclass
class Meeting:
    id: str
    title: str
    start_time: str
    end_time: str
    participants: List[MeetingParticipant]
    status: MeetingStatus
    recording_path: Optional[str] = None
    transcript: Optional[str] = None

class JarvisMeet:
    def __init__(self, api_key: str = None):
        self.meetings: Dict[str, Meeting] = {}
        self.current_meeting: Optional[Meeting] = None
        self.recognizer = sr.Recognizer()
        self.calendar_service = self._authenticate_google_calendar()
        self.api_key = api_key
        self.is_recording = False
        self.is_transcribing = False
        self.transcription_thread = None
        self.virtual_bg_path = "backgrounds/default.jpg"
        self.noise_cancellation = True
        self.meeting_analytics = {}

    def _authenticate_google_calendar(self) -> Optional[build]:
        """Authenticate with Google Calendar API"""
        try:
            creds = None
            token_path = 'token.json'
            creds_path = 'credentials.json'
            
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/calendar'])
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_path, ['https://www.googleapis.com/auth/calendar'])
                    creds = flow.run_local_server(port=0)
                
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            return build('calendar', 'v3', credentials=creds)
        except Exception as e:
            print(f"Calendar auth failed: {e}")
            return None

    def schedule_meeting(
        self,
        title: str,
        start_time: str,
        duration_min: int,
        participants: List[str],
        description: str = ""
    ) -> Optional[Meeting]:
        """Schedule a new meeting via Google Calendar"""
        try:
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (datetime.fromisoformat(start_time) + timedelta(minutes=duration_min)).isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in participants],
                'conferenceData': {
                    'createRequest': {
                        'requestId': f"jarvis-{time.time()}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
            }

            event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()

            meeting = Meeting(
                id=event['id'],
                title=title,
                start_time=start_time,
                end_time=event['end']['dateTime'],
                participants=[MeetingParticipant(email=p['email'], name=p.get('displayName', 'Guest')) for p in event.get('attendees', [])],
                status=MeetingStatus.SCHEDULED
            )
            self.meetings[meeting.id] = meeting
            return meeting
        except Exception as e:
            print(f"Failed to schedule meeting: {e}")
            return None

    def start_meeting(self, meeting_id: str) -> bool:
        """Start a scheduled meeting (opens browser)"""
        if meeting_id not in self.meetings:
            return False
        
        meeting = self.meetings[meeting_id]
        meeting.status = MeetingStatus.ONGOING
        self.current_meeting = meeting
        
        # Open Google Meet in browser
        webbrowser.open(f"https://meet.google.com/{meeting.id}")
        
        # Start recording if enabled
        if self.is_recording:
            self._start_recording()
        
        # Start transcription if enabled
        if self.is_transcribing:
            self.transcription_thread = threading.Thread(target=self._transcribe_meeting, daemon=True)
            self.transcription_thread.start()
        
        return True

    def end_meeting(self) -> bool:
        """End the current meeting"""
        if not self.current_meeting:
            return False
        
        self.current_meeting.status = MeetingStatus.ENDED
        
        # Stop recording & transcription
        if self.is_recording:
            self._stop_recording()
        
        if self.is_transcribing:
            self.is_transcribing = False
            if self.transcription_thread:
                self.transcription_thread.join()
        
        # Generate meeting summary
        self._generate_meeting_summary()
        
        self.current_meeting = None
        return True

    def _start_recording(self) -> None:
        """Record meeting (screen + audio)"""
        self.is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recording = cv2.VideoWriter(
            f"meeting_{self.current_meeting.id}.avi",
            fourcc,
            20.0,
            (1920, 1080))
        
        print("Recording started...")

    def _stop_recording(self) -> None:
        """Stop recording and save file"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.recording.release()
        self.current_meeting.recording_path = f"meeting_{self.current_meeting.id}.avi"
        print("Recording saved.")

    def _transcribe_meeting(self) -> None:
        """Real-time meeting transcription"""
        self.is_transcribing = True
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Transcription started...")
            
            while self.is_transcribing:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio)
                    if self.current_meeting:
                        self.current_meeting.transcript = (self.current_meeting.transcript or "") + "\n" + text
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Transcription error: {e}")
                    break

    def _generate_meeting_summary(self) -> None:
        """Generate AI-powered meeting summary"""
        if not self.current_meeting or not self.current_meeting.transcript:
            return
        
        # Use OpenAI API (or any NLP service) to summarize
        summary = "Meeting Summary:\n- Discussed project updates\n- Assigned tasks\n- Next meeting in 1 week"
        self.current_meeting.transcript += f"\n\nSUMMARY:\n{summary}"

    def toggle_virtual_background(self, image_path: Optional[str] = None) -> bool:
        """Change virtual background (requires OpenCV)"""
        if image_path:
            self.virtual_bg_path = image_path
        print(f"Virtual background set to {self.virtual_bg_path}")
        return True

    def toggle_noise_cancellation(self, enable: bool) -> None:
        """Enable/disable noise cancellation"""
        self.noise_cancellation = enable
        print(f"Noise cancellation {'ON' if enable else 'OFF'}")

    def voice_command(self, command: str) -> str:
        """Process voice commands during meeting"""
        command = command.lower()
        
        if "mute" in command:
            pyautogui.hotkey('ctrl', 'd')  # Google Meet mute shortcut
            return "Microphone muted"
        
        elif "unmute" in command:
            pyautogui.hotkey('ctrl', 'd')
            return "Microphone unmuted"
        
        elif "turn on camera" in command:
            pyautogui.hotkey('ctrl', 'e')
            return "Camera turned on"
        
        elif "turn off camera" in command:
            pyautogui.hotkey('ctrl', 'e')
            return "Camera turned off"
        
        elif "start recording" in command:
            self.is_recording = True
            self._start_recording()
            return "Recording started"
        
        elif "stop recording" in command:
            self._stop_recording()
            return "Recording stopped"
        
        elif "end meeting" in command:
            self.end_meeting()
            return "Meeting ended"
        
        else:
            return "Command not recognized"

    def get_meeting_analytics(self) -> Dict:
        """Generate meeting engagement analytics"""
        if not self.current_meeting:
            return {}
        
        return {
            "duration": (datetime.fromisoformat(self.current_meeting.end_time) - datetime.fromisoformat(self.current_meeting.start_time)).seconds // 60,
            "participants": len(self.current_meeting.participants),
            "transcript_length": len(self.current_meeting.transcript.split()) if self.current_meeting.transcript else 0,
            "recording_available": bool(self.current_meeting.recording_path)
        }






class ThreatLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class SecurityEvent(Enum):
    INTRUSION_DETECTED = auto()
    MALWARE_FOUND = auto()
    BRUTE_FORCE_ATTEMPT = auto()
    DATA_LEAK = auto()
    UNAUTHORIZED_ACCESS = auto()

@dataclass
class SecurityAlert:
    timestamp: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    target: Optional[str] = None

class SecurityManager:
    def __init__(self):
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.security_logs: List[SecurityAlert] = []
        self.suspicious_ips: Dict[str, int] = {}  # IP: threat_score
        self.monitoring_active = False
        self._load_security_policies()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate or load encryption key"""
        key_file = 'security.key'
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _load_security_policies(self) -> None:
        """Load security policies from config"""
        self.policies = {
            'failed_login_threshold': 5,
            'port_scan_detection': True,
            'malware_scan_interval': 3600,  # 1 hour
            'network_monitoring': True,
            'auto_block_suspicious': False
        }
    
    def encrypt_file(self, file_path: str) -> bool:
        """Encrypt a file with AES encryption"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            encrypted_data = self.cipher_suite.encrypt(file_data)
            with open(file_path + '.enc', 'wb') as f:
                f.write(encrypted_data)
            os.remove(file_path)
            return True
        except Exception as e:
            logging.error(f"File encryption failed: {e}")
            return False
    
    def decrypt_file(self, encrypted_path: str) -> bool:
        """Decrypt an encrypted file"""
        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            original_path = encrypted_path[:-4]  # Remove .enc
            with open(original_path, 'wb') as f:
                f.write(decrypted_data)
            return True
        except Exception as e:
            logging.error(f"File decryption failed: {e}")
            return False
    
    def start_network_monitoring(self) -> None:
        """Start monitoring network traffic for anomalies"""
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitor_network, daemon=True)
        monitoring_thread.start()
    
    def _monitor_network(self) -> None:
        """Background network monitoring thread"""
        while self.monitoring_active:
            try:
                # Analyze ARP traffic
                packets = scapy.sniff(filter="arp", count=50, timeout=10)
                self._detect_arp_spoofing(packets)
                
                # Analyze TCP traffic
                packets = scapy.sniff(filter="tcp", count=100, timeout=10)
                self._detect_port_scans(packets)
                self._detect_brute_force(packets)
                
                time.sleep(5)
            except Exception as e:
                logging.error(f"Network monitoring error: {e}")
    
    def _detect_arp_spoofing(self, packets) -> None:
        """Detect ARP spoofing attempts"""
        arp_table = {}
        for packet in packets:
            if packet.haslayer(scapy.ARP):
                ip = packet.psrc
                mac = packet.hwsrc
                if ip in arp_table and arp_table[ip] != mac:
                    self._log_security_event(
                        SecurityEvent.INTRUSION_DETECTED,
                        ThreatLevel.HIGH,
                        f"ARP spoofing detected: {ip} is claiming to be {mac}",
                        source_ip=ip
                    )
                arp_table[ip] = mac
    
    def _detect_port_scans(self, packets) -> None:
        """Detect port scanning activity"""
        if not self.policies['port_scan_detection']:
            return
            
        scan_threshold = 20  # Number of ports to trigger detection
        src_ports = {}
        
        for packet in packets:
            if packet.haslayer(scapy.TCP):
                src_ip = packet[scapy.IP].src
                dst_port = packet[scapy.TCP].dport
                
                if src_ip not in src_ports:
                    src_ports[src_ip] = set()
                src_ports[src_ip].add(dst_port)
                
                if len(src_ports[src_ip]) > scan_threshold:
                    self._log_security_event(
                        SecurityEvent.INTRUSION_DETECTED,
                        ThreatLevel.MEDIUM,
                        f"Port scan detected from {src_ip}",
                        source_ip=src_ip
                    )
                    self._update_threat_score(src_ip, 10)
    
    def _detect_brute_force(self, packets) -> None:
        """Detect brute force attempts"""
        failed_attempts = {}
        
        for packet in packets:
            if packet.haslayer(scapy.TCP) and packet[scapy.TCP].flags == 'R':
                src_ip = packet[scapy.IP].src
                dst_port = packet[scapy.TCP].dport
                
                if dst_port in [22, 3389, 21]:  # SSH, RDP, FTP
                    if src_ip not in failed_attempts:
                        failed_attempts[src_ip] = 0
                    failed_attempts[src_ip] += 1
                    
                    if failed_attempts[src_ip] > self.policies['failed_login_threshold']:
                        self._log_security_event(
                            SecurityEvent.BRUTE_FORCE_ATTEMPT,
                            ThreatLevel.HIGH,
                            f"Brute force attempt detected from {src_ip} on port {dst_port}",
                            source_ip=src_ip
                        )
                        self._update_threat_score(src_ip, 15)
    
    def _update_threat_score(self, ip: str, score: int) -> None:
        """Update threat score for suspicious IP"""
        if ip not in self.suspicious_ips:
            self.suspicious_ips[ip] = 0
        self.suspicious_ips[ip] += score
        
        if self.policies['auto_block_suspicious'] and self.suspicious_ips[ip] > 30:
            self.block_ip(ip)
    
    def _log_security_event(self, 
                          event_type: SecurityEvent, 
                          threat_level: ThreatLevel,
                          description: str,
                          source_ip: Optional[str] = None,
                          target: Optional[str] = None) -> None:
        """Log a security event"""
        alert = SecurityAlert(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            source_ip=source_ip,
            target=target
        )
        self.security_logs.append(alert)
        logging.warning(f"Security Alert: {description}")
    
    def scan_for_malware(self, directory: str = "/") -> List[Dict]:
        """Scan filesystem for potential malware"""
        malware_signatures = [
            (rb"eval\(base64_decode\(", "PHP base64 encoded malware"),
            (rb"powershell -nop -w hidden -c", "Suspicious PowerShell command"),
            (rb"document.write\(\"<script src=\"", "JavaScript injection"),
            (rb"\/bin\/bash", "Suspicious bash command")
        ]
        
        suspicious_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read(4096)  # Read first 4KB
                        for signature, description in malware_signatures:
                            if re.search(signature, content):
                                suspicious_files.append({
                                    'path': file_path,
                                    'threat': description,
                                    'timestamp': datetime.now().isoformat()
                                })
                                self._log_security_event(
                                    SecurityEvent.MALWARE_FOUND,
                                    ThreatLevel.HIGH,
                                    f"Potential malware found: {file_path}",
                                    target=file_path
                                )
                                break
                except (IOError, PermissionError):
                    continue
        
        return suspicious_files
    
    def block_ip(self, ip: str) -> bool:
        """Block an IP address using system firewall"""
        try:
            if platform.system() == "Linux":
                subprocess.run(["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule",
                               f"name=Block {ip}", "dir=in", "action=block",
                               f"remoteip={ip}"], check=True)
            self._log_security_event(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                ThreatLevel.MEDIUM,
                f"Blocked suspicious IP: {ip}",
                source_ip=ip
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to block IP {ip}: {e}")
            return False
    
    def check_system_vulnerabilities(self) -> Dict:
        """Check for common system vulnerabilities"""
        vulns = {
            'unpatched_software': [],
            'weak_permissions': [],
            'exposed_services': []
        }
        
        # Check for outdated software
        try:
            if platform.system() == "Linux":
                result = subprocess.run(["apt-get", "update"], capture_output=True, text=True)
                outdated = subprocess.run(["apt-get", "upgrade", "--simulate"], 
                                        capture_output=True, text=True)
                if "The following packages will be upgraded" in outdated.stdout:
                    vulns['unpatched_software'] = re.findall(r"^  (\S+)", outdated.stdout, re.M)
        except Exception as e:
            logging.error(f"Vulnerability check failed: {e}")
        
        # Check for world-writable files
        try:
            sensitive_dirs = ['/etc', '/var', '/usr/local']
            for directory in sensitive_dirs:
                if os.path.exists(directory):
                    result = subprocess.run(["find", directory, "-perm", "-o=w", "-type", "f"],
                                          capture_output=True, text=True)
                    if result.stdout:
                        vulns['weak_permissions'] = result.stdout.splitlines()
        except Exception:
            pass
        
        # Check for exposed services
        try:
            netstat = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True)
            exposed = []
            for line in netstat.stdout.splitlines():
                if "LISTEN" in line and ("0.0.0.0" in line or ":::" in line):
                    exposed.append(line.strip())
            vulns['exposed_services'] = exposed
        except Exception:
            pass
        
        return vulns
    
    def get_security_alerts(self, min_threat: ThreatLevel = ThreatLevel.MEDIUM) -> List[SecurityAlert]:
        """Get security alerts filtered by threat level"""
        return [alert for alert in self.security_logs 
                if alert.threat_level.value >= min_threat.value]
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'alerts': len(self.security_logs),
            'suspicious_ips': len(self.suspicious_ips),
            'recent_alerts': [vars(a) for a in self.security_logs[-5:]],
            'top_threats': sorted(self.suspicious_ips.items(), 
                                 key=lambda x: x[1], reverse=True)[:5],
            'system_vulnerabilities': self.check_system_vulnerabilities(),
            'malware_scan': self.scan_for_malware("/tmp")  # Scan temp dir by default
        }





class DeviceType(Enum):
    LIGHT = auto()
    THERMOSTAT = auto()
    SECURITY_CAMERA = auto()
    SMART_PLUG = auto()
    TV = auto()
    AUDIO_SYSTEM = auto()



class DeviceStatus(Enum):
    ON = auto()
    OFF = auto()
    STANDBY = auto()
    UNAVAILABLE = auto()





@dataclass
class SmartDevice:
    id: str
    name: str
    type: DeviceType
    ip: str
    mac: str
    manufacturer: str
    status: DeviceStatus
    last_seen: float
    capabilities: List[str]





class SmartHomeControl:

    def __init__(self, config_file: str = "smarthome_config.json"):
        self.devices: Dict[str, SmartDevice] = {}
        self.scenes: Dict[str, List[Dict]] = {}
        self.routines: Dict[str, Dict] = {}
        self.load_config(config_file)
        self.discovery_thread = threading.Thread(target=self._device_discovery, daemon=True)
        self.discovery_thread.start()
        self._setup_virtual_assistant_integration()




    def load_config(self, config_file: str) -> bool:
        """Load smart home configuration from JSON file"""
        try:
            with open(config_file) as f:
                config = json.load(f)
                self.devices = {d['id']: SmartDevice(**d) for d in config.get('devices', [])}
                self.scenes = config.get('scenes', {})
                self.routines = config.get('routines', {})
            return True
        except Exception as e:
            print(f"Config load error: {e}")
            return False




    def save_config(self, config_file: str) -> bool:
        """Save current configuration to file"""
        try:
            with open(config_file, 'w') as f:
                json.dump({
                    'devices': [vars(d) for d in self.devices.values()],
                    'scenes': self.scenes,
                    'routines': self.routines
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Config save error: {e}")
            return False




    def _device_discovery(self) -> None:
        """Background thread for continuous device discovery"""
        while True:
            self._discover_devices()
            time.sleep(300)  # Check every 5 minutes




    def _discover_devices(self) -> None:
        """Discover devices on local network"""
        # Implement SSDP/UPnP discovery or manufacturer-specific discovery
        print("Running device discovery...")
        # Placeholder - actual implementation would use network scanning
        for device in self.devices.values():
            if self._ping_device(device.ip):
                device.status = DeviceStatus.ON
                device.last_seen = time.time()
            else:
                device.status = DeviceStatus.OFF




    def _ping_device(self, ip: str) -> bool:
        """Check if device is reachable"""
        try:
            socket.create_connection((ip, 80), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False




    def _setup_virtual_assistant_integration(self) -> None:
        """Setup webhooks for virtual assistant integration"""
        # Placeholder for Google Assistant/Alexa integration
        pass




    def control_device(self, device_id: str, action: str, value: Optional[str] = None) -> bool:
        """Control a smart device"""
        device = self.devices.get(device_id)
        if not device:
            return False

        try:
            if device.type == DeviceType.LIGHT:
                return self._control_light(device, action, value)
            elif device.type == DeviceType.THERMOSTAT:
                return self._control_thermostat(device, action, value)
            # Add other device types...
            return True
        except Exception as e:
            print(f"Device control error: {e}")
            return False




    def _control_light(self, device: SmartDevice, action: str, value: str) -> bool:
        """Control smart light"""
        if action == "turn_on":
            return self._send_device_command(device, {"power": "on"})
        elif action == "turn_off":
            return self._send_device_command(device, {"power": "off"})
        elif action == "set_brightness":
            return self._send_device_command(device, {"brightness": int(value)})
        elif action == "set_color":
            return self._send_device_command(device, {"color": value})
        return False




    def _control_thermostat(self, device: SmartDevice, action: str, value: str) -> bool:
        """Control smart thermostat"""
        if action == "set_temperature":
            return self._send_device_command(device, {"temperature": float(value)})
        elif action == "set_mode":
            return self._send_device_command(device, {"mode": value})
        return False





    def _send_device_command(self, device: SmartDevice, payload: Dict) -> bool:
        """Send command to device (placeholder implementation)"""
        try:
            # Actual implementation would use device-specific API
            print(f"Sending command to {device.name}: {payload}")
            return True
        except Exception as e:
            print(f"Command failed: {e}")
            return False




    def wake_on_lan(self, device_id: str) -> bool:
        """Wake device using Wake-on-LAN"""
        device = self.devices.get(device_id)
        if not device or not device.mac:
            return False
        try:
            wakeonlan.send_magic_packet(device.mac)
            return True
        except Exception as e:
            print(f"WOL failed: {e}")
            return False





    def activate_scene(self, scene_name: str) -> bool:
        """Activate a predefined scene"""
        scene = self.scenes.get(scene_name)
        if not scene:
            return False

        results = []
        for action in scene:
            device = self.devices.get(action['device_id'])
            if device:
                results.append(self.control_device(
                    device.id,
                    action['action'],
                    action.get('value')
                ))
        return all(results)





    def create_scene(self, name: str, actions: List[Dict]) -> bool:
        """Create a new scene"""
        self.scenes[name] = actions
        return self.save_config()




    def start_routine(self, routine_name: str) -> bool:
        """Start a scheduled routine"""
        routine = self.routines.get(routine_name)
        if not routine:
            return False

        if routine.get('type') == 'scheduled':
            if not self._is_scheduled_time(routine['schedule']):
                return False

        return self.activate_scene(routine['scene'])




    def _is_scheduled_time(self, schedule: Dict) -> bool:
        """Check if current time matches schedule"""
        # Implement time-based checking
        return True





    def get_device_status(self, device_id: str) -> Optional[Dict]:
        """Get detailed device status"""
        device = self.devices.get(device_id)
        if not device:
            return None

        return {
            'name': device.name,
            'type': device.type.name,
            'status': device.status.name,
            'last_seen': time.ctime(device.last_seen),
            'capabilities': device.capabilities
        }




    def voice_command_handler(self, command: str) -> str:
        """Handle natural language voice commands"""
        command = command.lower()
        
        # Simple voice command parsing (would use NLP in real implementation)
        if "turn on" in command and "light" in command:
            device = self._find_device_by_name(command.split("light")[-1].strip())
            if device:
                self.control_device(device.id, "turn_on")
                return f"Turned on {device.name}"
        
        # Add more voice command parsing...
        
        return "Sorry, I didn't understand that command"







class GoogleChromeSearcher:
    def __init__(self, chrome_path: Optional[str] = None):
        """
        Initialize the Chrome searcher.
        Args:
            chrome_path (str): Optional custom path to Chrome executable.
        """
        self.chrome_path = chrome_path
        self.browser_name = "google-chrome"  # Default (Linux/Mac). Windows uses "chrome".

    def search(self, query: str, new_tab: bool = True) -> bool:
        """
        Search a query on Google Chrome.
        Args:
            query (str): Search term (e.g., "Python tutorials").
            new_tab (bool): Open in new tab if True.
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            # Open Chrome (use custom path if specified)
            if self.chrome_path:
                webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(self.chrome_path))
                webbrowser.get('chrome').open(url, new=int(new_tab))
            else:
                webbrowser.get(self.browser_name).open(url, new=int(new_tab))
            
            time.sleep(1)  # Wait for browser to open
            pyautogui.hotkey('ctrl', 'l')  # Focus address bar (Windows/Linux)
            return True
        
        except Exception as e:
            print(f"Chrome search error: {e}")
            return False



@dataclass
class SearchResult:
    file_path: str
    line_number: int
    line_content: str
    score: float

class VSCodeFileSearcher:
    def __init__(self, workspace_root: str = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.ignore_dirs = ['.git', 'node_modules', '__pycache__', '.vscode']
        self.ignore_extensions = ['.exe', '.dll', '.png', '.jpg']
        self.search_history = []
        
    def set_workspace(self, path: str) -> bool:
        """Set the workspace root directory"""
        if os.path.isdir(path):
            self.workspace_root = path
            return True
        return False
    
    def find_files(
        self,
        pattern: str,
        content_search: str = None,
        file_ext: str = None,
        case_sensitive: bool = False
    ) -> List[SearchResult]:
        """
        Search for files with name matching pattern and optionally content
        Args:
            pattern: Filename pattern (supports * and ? wildcards)
            content_search: String to search within files
            file_ext: Filter by file extension
            case_sensitive: Whether search should be case sensitive
        Returns:
            List of SearchResult objects
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for root, dirs, files in os.walk(self.workspace_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                # Skip ignored extensions
                if any(file.endswith(ext) for ext in self.ignore_extensions):
                    continue
                
                # Match filename pattern
                if not fnmatch.fnmatch(file, pattern):
                    continue
                    
                # Match file extension if specified
                if file_ext and not file.endswith(file_ext):
                    continue
                    
                full_path = os.path.join(root, file)
                
                if content_search:
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if re.search(content_search, line, flags):
                                    score = self._calculate_score(pattern, content_search, file, line)
                                    results.append(SearchResult(
                                        file_path=full_path,
                                        line_number=line_num,
                                        line_content=line.strip(),
                                        score=score
                                    ))
                    except UnicodeDecodeError:
                        continue
                else:
                    score = self._calculate_score(pattern, None, file, None)
                    results.append(SearchResult(
                        file_path=full_path,
                        line_number=0,
                        line_content='',
                        score=score
                    ))
        
        # Sort by score (best matches first)
        results.sort(key=lambda x: x.score, reverse=True)
        self.search_history.extend(results)
        return results
    
    def _calculate_score(
        self,
        pattern: str,
        content: Optional[str],
        filename: str,
        line: Optional[str]
    ) -> float:
        """Calculate match score (0-1)"""
        score = 0.0
        
        # Filename match score
        if '*' not in pattern and '?' not in pattern:
            # Exact match
            if pattern.lower() == filename.lower():
                score += 0.5
            elif pattern.lower() in filename.lower():
                score += 0.3
        else:
            # Wildcard match
            score += 0.2
            
        # Content match score
        if content and line:
            if content.lower() in line.lower():
                score += 0.5 * (len(content) / len(line))
                
        return min(1.0, score)
    
    def open_in_vscode(self, file_path: str, line_number: int = 0) -> bool:
        """Open file in VS Code at specific line"""
        try:
            subprocess.run([
                'code',
                '--goto',
                f'{file_path}:{line_number}'
            ], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback for Windows
                subprocess.run([
                    'code.cmd',
                    '--goto',
                    f'{file_path}:{line_number}'
                ], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def search_project_symbols(self, symbol: str) -> List[Dict]:
        """Search for symbols in project (classes, functions, etc.)"""
        try:
            result = subprocess.run([
                'code',
                '--command',
                f'vscode.executeWorkspaceSymbolProvider {symbol}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return []
    
    def get_recent_files(self, limit: int = 10) -> List[str]:
        """Get recently opened files in VS Code"""
        try:
            # This requires VS Code's internal command
            result = subprocess.run([
                'code',
                '--command',
                'workbench.action.openRecent'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.splitlines()[:limit]
        except Exception:
            pass
        return []
    
    def create_search_snapshot(self) -> Dict:
        """Save current search state"""
        return {
            'workspace': self.workspace_root,
            'history': [{
                'file_path': r.file_path,
                'line_number': r.line_number,
                'query': self.search_history[-1] if self.search_history else ''
            } for r in self.search_history[-10:]]  # Keep last 10 searches
        }
    






class MusicSource(Enum):
    YOUTUBE = auto()
    SPOTIFY = auto()
    SOUNDCLOUD = auto()
    JIO_SAVAN = auto()
    LOCAL = auto()

@dataclass
class Song:
    title: str
    artist: str
    duration: int  # in seconds
    source: MusicSource
    url: str
    thumbnail: Optional[str] = None
    album: Optional[str] = None

class WebMusicPlayer:
    def __init__(self):
        self.instance = vlc.Instance('--no-xlib')
        self.player = self.instance.media_player_new()
        self.playlist: List[Song] = []
        self.current_index = -1
        self.volume = 70
        self.is_playing = False
        self.last_update = 0
        self.playback_speed = 1.0
        
        # Initialize with default volume
        self.set_volume(self.volume)
        
    def add_to_queue(self, song: Song) -> None:
        """Add a song to the playback queue"""
        self.playlist.append(song)
        if self.current_index == -1:
            self.current_index = 0
            
    def play(self, song: Optional[Song] = None) -> bool:
        """Play a specific song or resume playback"""
        if song:
            self.playlist.insert(self.current_index + 1, song)
            self.current_index += 1
            
        if not self.playlist:
            return False
            
        current_song = self.playlist[self.current_index]
        media = self.instance.media_new(current_song.url)
        self.player.set_media(media)
        
        if self.player.play() == -1:
            return False
            
        self.is_playing = True
        self.player.set_rate(self.playback_speed)
        return True
        
    def pause(self) -> None:
        """Pause the current playback"""
        if self.is_playing:
            self.player.pause()
            self.is_playing = False
            
    def stop(self) -> None:
        """Stop playback and clear queue"""
        self.player.stop()
        self.is_playing = False
        self.playlist = []
        self.current_index = -1
        
    def next_track(self) -> bool:
        """Skip to the next track in queue"""
        if self.current_index < len(self.playlist) - 1:
            self.current_index += 1
            return self.play()
        return False
        
    def previous_track(self) -> bool:
        """Go back to previous track"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.play()
        return False
        
    def set_volume(self, level: int) -> None:
        """Set volume level (0-100)"""
        self.volume = max(0, min(100, level))
        self.player.audio_set_volume(self.volume)
        
    def get_current_position(self) -> float:
        """Get current playback position in seconds"""
        return self.player.get_time() / 1000
        
    def set_position(self, seconds: float) -> None:
        """Seek to specific position in track"""
        self.player.set_time(int(seconds * 1000))
        
    def get_current_song(self) -> Optional[Song]:
        """Get currently playing song info"""
        if 0 <= self.current_index < len(self.playlist):
            return self.playlist[self.current_index]
        return None
        
    def shuffle_queue(self) -> None:
        """Shuffle the playback queue"""
        if len(self.playlist) > 1:
            current_song = self.playlist.pop(self.current_index)
            random.shuffle(self.playlist)
            self.playlist.insert(0, current_song)
            self.current_index = 0
            
    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed (0.5-2.0)"""
        self.playback_speed = max(0.5, min(2.0, speed))
        self.player.set_rate(self.playback_speed)
        
    def search_youtube(self, query: str) -> Optional[Song]:
        """Search YouTube for a song"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'extract_flat': True,
            'default_search': 'ytsearch1'
        }
        
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if not info or 'entries' not in info or not info['entries']:
                    return None
                    
                entry = info['entries'][0]
                return Song(
                    title=entry.get('title', 'Unknown Track'),
                    artist=entry.get('uploader', 'Unknown Artist'),
                    duration=entry.get('duration', 0),
                    source=MusicSource.YOUTUBE,
                    url=entry['url'],
                    thumbnail=entry.get('thumbnail', '')
                )
        except Exception:
            return None
            
    def create_radio_station(self, artist: str) -> bool:
        """Create a radio station based on artist"""
        # Implementation would depend on music service API
        pass
        
    def save_playlist(self, name: str) -> bool:
        """Save current queue as a playlist"""
        # Implementation would need storage solution
        pass
        
    def load_playlist(self, name: str) -> bool:
        """Load a saved playlist"""
        # Implementation would need storage solution
        pass


@dataclass
class RunningProcess:
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float

@dataclass
class SystemTask:
    id: int
    name: str
    priority: str  # "low", "normal", "high"
    created_at: float  # timestamp

class SystemControl:
    def __init__(self):
        self.tasks: List[SystemTask] = []
        self.next_task_id = 1
        self.os_type = platform.system()
    
    # System Monitoring
    def get_system_stats(self) -> Dict:
        """Return comprehensive system statistics"""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None
            },
            "memory": {
                "total": round(psutil.virtual_memory().total / (1024**3), 2),
                "available": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": round(psutil.disk_usage('/').total / (1024**3), 2),
                "used": round(psutil.disk_usage('/').used / (1024**3), 2),
                "free": round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            "os": {
                "name": platform.system(),
                "version": platform.version(),
                "architecture": platform.architecture()[0]
            }
        }

    # Process Management
    def list_processes(self) -> List[RunningProcess]:
        """Get all running processes with resource usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(RunningProcess(
                    pid=proc.info['pid'],
                    name=proc.info['name'],
                    cpu_percent=proc.info['cpu_percent'],
                    memory_percent=proc.info['memory_percent']
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes

    def kill_process(self, pid: int) -> bool:
        """Terminate a process by PID"""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except psutil.NoSuchProcess:
            return False

    # Task Management
    def add_task(self, name: str, priority: str = "normal") -> SystemTask:
        """Add a new system task"""
        task = SystemTask(
            id=self.next_task_id,
            name=name,
            priority=priority,
            created_at=time.time()
        )
        self.tasks.append(task)
        self.next_task_id += 1
        return task

    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed"""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                return True
        return False

    def list_tasks(self) -> List[SystemTask]:
        """Get all pending tasks"""
        return self.tasks

    # System Operations
    def shutdown(self, delay_seconds: int = 60) -> bool:
        """Initiate system shutdown"""
        try:
            if self.os_type == "Windows":
                subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)])
            else:
                subprocess.run(["shutdown", "-h", "+" + str(delay_seconds//60)])
            return True
        except subprocess.SubprocessError:
            return False

    def restart(self, delay_seconds: int = 60) -> bool:
        """Initiate system restart"""
        try:
            if self.os_type == "Windows":
                subprocess.run(["shutdown", "/r", "/t", str(delay_seconds)])
            else:
                subprocess.run(["shutdown", "-r", "+" + str(delay_seconds//60)])
            return True
        except subprocess.SubprocessError:
            return False

    def sleep(self) -> bool:
        """Put system to sleep"""
        try:
            if self.os_type == "Windows":
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            elif self.os_type == "Darwin":
                subprocess.run(["pmset", "sleepnow"])
            else:  # Linux
                subprocess.run(["systemctl", "suspend"])
            return True
        except subprocess.SubprocessError:
            return False

class SearchEngine(Enum):
    GOOGLE = auto()
    BING = auto()
    YOUTUBE = auto()
    WIKIPEDIA = auto()
    STACKOVERFLOW = auto()
    GITHUB = auto()

class MusicSource(Enum):
    YOUTUBE = auto()
    SPOTIFY = auto()
    LOCAL = auto()
    SOUNDCLOUD = auto()

@dataclass
class SearchResult:
    Title: str
    Url: str
    Description: str
    Source: str

@dataclass
class Song:
    Title: str
    Artist: str
    Duration: int
    Source: MusicSource
    Url: str
    Thumbnail: Optional[str] = None

class VoiceAssistant:
    def __init__(self):
        self.Engine = pyttsx3.init()
        self.Engine.setProperty('rate', 150)
        self.Recognizer = sr.Recognizer()
        self.Recognizer.energy_threshold = 300
        self.SetVoice('male')
        # Adjust these parameters in VoiceAssistant init
        self.Recognizer.energy_threshold = 4000  # Louder activation
        self.Recognizer.dynamic_energy_threshold = False
        self.Recognizer.pause_threshold = 0.8  # Faster response
    def SetVoice(self, gender: str) -> bool:
        voices = self.Engine.getProperty('voices')
        for voice in voices:
            if gender.lower() in voice.name.lower():
                self.Engine.setProperty('voice', voice.id)
                return True
        return False

    def Speak(self, text: str) -> None:
        logging.info(f"Speaking: {text}")
        self.Engine.say(text)
        self.Engine.runAndWait()

    def TakeCommand(self) -> Optional[str]:
        with sr.Microphone() as source:
            print("Listening...")
            self.Recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.Recognizer.listen(source, timeout=5)
                query = self.Recognizer.recognize_google(audio).lower()
                print(f"Recognized: {query}")
                return query
            except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError) as e:
                print(f"Voice recognition error: {e}")
                return None

class MediaPlayer:
    def __init__(self):
        self.Instance = vlc.Instance('--no-xlib')
        self.Player = self.Instance.media_player_new()
        self.IsPlaying = False
        self.CurrentVolume = 50
        self.Playlist = []
        self.CurrentTrackIndex = 0
        pygame.mixer.init()

    def PlayYouTube(self, songName: str) -> bool:
        try:
            ydlOpts = {'format': 'bestaudio/best', 'quiet': True}
            with youtube_dl.YoutubeDL(ydlOpts) as ydl:
                result = ydl.extract_info(f"ytsearch1:{songName}", download=False)
                if not result or 'entries' not in result or not result['entries']:
                    return False
                
                video = result['entries'][0]
                media = self.Instance.media_new(video['url'])
                self.Player.set_media(media)
                self.Player.audio_set_volume(self.CurrentVolume)
                
                if self.Player.play() == -1:
                    return False
                
                self.IsPlaying = True
                self.Playlist.append(video['url'])
                self.CurrentTrackIndex = len(self.Playlist) - 1
                return True
        except Exception as e:
            logging.error(f"Error playing YouTube audio: {e}")
            return False

    def PlayLocal(self, filePath: str) -> None:
        if os.path.exists(filePath):
            media = self.Instance.media_new(filePath)
            self.Player.set_media(media)
            self.Player.audio_set_volume(self.CurrentVolume)
            self.Player.play()
            self.IsPlaying = True
            self.Playlist.append(filePath)
            self.CurrentTrackIndex = len(self.Playlist) - 1

    def ControlMedia(self, action: str) -> None:
        if action == "pause":
            if self.Player.is_playing():
                self.Player.pause()
                self.IsPlaying = False
        elif action == "resume":
            if not self.Player.is_playing() and self.IsPlaying:
                self.Player.play()
        elif action == "stop":
            self.Player.stop()
            self.IsPlaying = False
            self.Playlist = []
            self.CurrentTrackIndex = 0
        elif action == "next":
            if self.CurrentTrackIndex < len(self.Playlist) - 1:
                self.CurrentTrackIndex += 1
                self.PlayCurrentTrack()
        elif action == "previous":
            if self.CurrentTrackIndex > 0:
                self.CurrentTrackIndex -= 1
                self.PlayCurrentTrack()

    def PlayCurrentTrack(self) -> None:
        media = self.Instance.media_new(self.Playlist[self.CurrentTrackIndex])
        self.Player.set_media(media)
        self.Player.play()

    def SetVolume(self, level: int) -> None:
        if 0 <= level <= 100:
            self.CurrentVolume = level
            self.Player.audio_set_volume(level)

class FileManager:
    @staticmethod
    def SearchFiles(directory: str, extension: str = '*') -> List[str]:
        searchPattern = os.path.join(directory, f'**/*{extension}')
        return glob.glob(searchPattern, recursive=True)

    @staticmethod
    def SearchFunctions(filePath: str) -> List[str]:
        functions = []
        with open(filePath, 'r') as file:
            for line in file:
                match = re.match(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                if match:
                    functions.append(match.group(1))
        return functions

    @staticmethod
    def SearchFunctionsInDirectory(directory: str) -> Dict[str, List[str]]:
        pythonFiles = FileManager.SearchFiles(directory, '.py')
        functionsInFiles = {}
        
        for file in pythonFiles:
            functions = FileManager.SearchFunctions(file)
            if functions:
                functionsInFiles[file] = functions
                
        return functionsInFiles

    @staticmethod
    def GetDirectorySize(directory: str) -> float:
        totalSize = 0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                totalSize += os.path.getsize(fp)
        return round(totalSize / (1024 * 1024), 2)

class WebServices:
    def __init__(self):
        self.BrowserPath = self.GetDefaultBrowser()

    @staticmethod
    def GetDefaultBrowser() -> str:
        try:
            if platform.system() == 'Windows':
                import winreg
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'http\shell\open\command') as key:
                    cmd = winreg.QueryValue(key, None)
                    return cmd.split('"')[1]
            elif platform.system() == 'Darwin':
                return '/usr/bin/open'
            else:
                return '/usr/bin/xdg-open'
        except:
            return ''

    def OpenWebsite(self, url: str) -> None:
        try:
            if self.BrowserPath:
                subprocess.Popen([self.BrowserPath, url])
            else:
                webbrowser.open(url)
        except Exception as e:
            logging.error(f"Error opening website: {e}")

    def SearchGoogle(self, query: str) -> None:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        self.OpenWebsite(url)

    def SearchWikipedia(self, query: str) -> str:
        try:
            wiki = wikipediaapi.Wikipedia('en')
            page = wiki.page(query)
            return page.summary[:500] if page.exists() else "No Wikipedia article found."
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}")
            return "Sorry, I couldn't access Wikipedia."

class CalendarManager:
    def __init__(self):
        self.Service = self.Authenticate()
        self.CalendarEnabled = self.Service is not None

    def Authenticate(self):
        creds = None
        tokenPath = 'token.json'
        credsPath = 'credentials.json'
        
        if not os.path.exists(credsPath):
            logging.error("Google API credentials file not found")
            return None
            
        if os.path.exists(tokenPath):
            creds = Credentials.from_authorized_user_file(tokenPath, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(credsPath, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logging.error(f"Authentication failed: {e}")
                    return None

            with open(tokenPath, 'w') as token:
                token.write(creds.to_json())

        return build('calendar', 'v3', credentials=creds) if creds else None

    def CreateEvent(self, startTime: str, endTime: str, summary: str) -> None:
        if not self.CalendarEnabled:
            return
            
        event = {
            'summary': summary,
            'start': {'dateTime': startTime, 'timeZone': 'UTC'},
            'end': {'dateTime': endTime, 'timeZone': 'UTC'},
        }
        self.Service.events().insert(calendarId='primary', body=event).execute()

    def GetEvents(self, maxResults=10) -> List[Dict]:
        if not self.CalendarEnabled:
            return []
            
        now = datetime.utcnow().isoformat() + 'Z'
        eventsResult = self.Service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=maxResults, singleEvents=True,
            orderBy='startTime').execute()
        return eventsResult.get('items', [])

# class MusicPlayer:
#     def __init__(self):
#         self.Instance = vlc.Instance('--no-xlib')
#         self.Player = self.Instance.media_player_new()
#         self.CurrentVolume = 50
#         self.CurrentTrack = None

#     def Play(self, source: MusicSource, query: str) -> bool:
#         if source == MusicSource.YOUTUBE:
#             return self.PlayYouTube(query)
#         elif source == MusicSource.SPOTIFY:
#             return self.PlaySpotify(query)
#         elif source == MusicSource.LOCAL:
#             return self.PlayLocal(query)
#         return False

#     def PlayYouTube(self, query: str) -> bool:
#         try:
#             ydlOpts = {'format': 'bestaudio/best', 'quiet': True}
#             with youtube_dl.YoutubeDL(ydlOpts) as ydl:
#                 info = ydl.extract_info(f"ytsearch1:{query}", download=False)
#                 if not info or 'entries' not in info or not info['entries']:
#                     return False
                
#                 video = info['entries'][0]
#                 self.CurrentTrack = Song(
#                     Title=video.get('title', 'Unknown'),
#                     Artist=video.get('uploader', 'Unknown'),
#                     Duration=video.get('duration', 0),
#                     Source=MusicSource.YOUTUBE,
#                     Url=video['url'],
#                     Thumbnail=video.get('thumbnail', '')
#                 )
                
#                 media = self.Instance.media_new(video['url'])
#                 self.Player.set_media(media)
#                 self.Player.audio_set_volume(self.CurrentVolume)
#                 return self.Player.play() != -1
#         except Exception as e:
#             logging.error(f"YouTube playback error: {e}")
#             return False

#     def PlaySpotify(self, query: str) -> bool:
#         print("Spotify integration not yet implemented")
#         return False

#     def PlayLocal(self, filePath: str) -> bool:
#         if os.path.exists(filePath):
#             self.CurrentTrack = Song(
#                 Title=os.path.basename(filePath),
#                 Artist="Local",
#                 Duration=0,
#                 Source=MusicSource.LOCAL,
#                 Url=filePath
#             )
#             media = self.Instance.media_new(filePath)
#             self.Player.set_media(media)
#             self.Player.audio_set_volume(self.CurrentVolume)
#             return self.Player.play() != -1
#         return False

#     def SetVolume(self, level: int) -> None:
#         if 0 <= level <= 100:
#             self.CurrentVolume = level
#             self.Player.audio_set_volume(level)

#     def Pause(self) -> None:
#         self.Player.pause()

#     def Resume(self) -> None:
#         self.Player.play()

#     def Stop(self) -> None:
#         self.Player.stop()
#         self.CurrentTrack = None


class MusicPlayer:
    def __init__(self):
        self.Instance = vlc.Instance('--no-xlib')
        self.Player = self.Instance.media_player_new()
        self.CurrentVolume = 50
        self.CurrentTrack = None
        self.is_playing = False  # Track playback state
        self.is_paused = False   # Track pause state

    def Play(self, source: MusicSource, query: str) -> bool:
        if source == MusicSource.YOUTUBE:
            return self.PlayYouTube(query)
        elif source == MusicSource.SPOTIFY:
            return self.PlaySpotify(query)
        elif source == MusicSource.LOCAL:
            return self.PlayLocal(query)
        return False

    def PlayYouTube(self, query: str) -> bool:
        try:
            ydlOpts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'noplaylist': True,
                'extract_flat': True
            }
            with youtube_dl.YoutubeDL(ydlOpts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if not info or 'entries' not in info or not info['entries']:
                    return False
                
                video = info['entries'][0]
                self.CurrentTrack = Song(
                    Title=video.get('title', 'Unknown'),
                    Artist=video.get('uploader', 'Unknown'),
                    Duration=video.get('duration', 0),
                    Source=MusicSource.YOUTUBE,
                    Url=video['url'],
                    Thumbnail=video.get('thumbnail', '')
                )
                
                media = self.Instance.media_new(video['url'])
                self.Player.set_media(media)
                self.Player.audio_set_volume(self.CurrentVolume)
                if self.Player.play() == -1:
                    return False
                self.is_playing = True
                self.is_paused = False
                return True
        except Exception as e:
            logging.error(f"YouTube playback error: {e}")
            return False

    def PlayLocal(self, filePath: str) -> bool:
        if os.path.exists(filePath):
            self.CurrentTrack = Song(
                Title=os.path.basename(filePath),
                Artist="Local",
                Duration=0,
                Source=MusicSource.LOCAL,
                Url=filePath
            )
            media = self.Instance.media_new(filePath)
            self.Player.set_media(media)
            self.Player.audio_set_volume(self.CurrentVolume)
            if self.Player.play() == -1:
                return False
            self.is_playing = True
            self.is_paused = False
            return True
        return False

    def SetVolume(self, level: int) -> bool:
        """Set volume with validation"""
        if 0 <= level <= 100:
            self.CurrentVolume = level
            return self.Player.audio_set_volume(level) == 0
        return False

    def Pause(self) -> bool:
        """Pause playback if currently playing"""
        if self.is_playing and not self.is_paused:
            self.Player.pause()
            self.is_paused = True
            return True
        return False

    def Resume(self) -> bool:
        """Resume playback if paused"""
        if self.is_playing and self.is_paused:
            self.Player.play()
            self.is_paused = False
            return True
        return False

    def Stop(self) -> bool:
        """Stop playback completely"""
        if self.is_playing:
            self.Player.stop()
            self.is_playing = False
            self.is_paused = False
            self.CurrentTrack = None
            return True
        return False

    def GetState(self) -> Dict:
        """Return current player state"""
        return {
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'volume': self.CurrentVolume,
            'current_track': self.CurrentTrack
        }
class LearningModule:
    def __init__(self):
        self.Resources = {
            'python': "https://docs.python.org/3/tutorial/",
            'machine learning': "https://www.coursera.org/learn/machine-learning",
            'web development': "https://developer.mozilla.org/en-US/",
            'data science': "https://www.kaggle.com/learn"
        }

    def GetResource(self, topic: str) -> Optional[str]:
        return self.Resources.get(topic.lower())



class LearningAdvanceModule:
    def __init__(self):
        self.resources: Dict[str, List[Dict[str, str]]] = {
            'programming': [
                {'name': 'Python Official Docs', 'url': 'https://docs.python.org/3/tutorial/', 'level': 'Beginner-Advanced'},
                {'name': 'freeCodeCamp', 'url': 'https://www.freecodecamp.org/', 'level': 'Beginner-Intermediate'},
                {'name': 'Codecademy', 'url': 'https://www.codecademy.com/', 'level': 'Beginner-Intermediate'},
                {'name': 'The Odin Project', 'url': 'https://www.theodinproject.com/', 'level': 'Beginner-Advanced'},
                {'name': 'LeetCode', 'url': 'https://leetcode.com/', 'level': 'Intermediate-Advanced'},
            ],
            'data science': [
                {'name': 'Kaggle Learn', 'url': 'https://www.kaggle.com/learn', 'level': 'Beginner-Intermediate'},
                {'name': 'DataCamp', 'url': 'https://www.datacamp.com/', 'level': 'Beginner-Advanced'},
                {'name': 'Fast.ai', 'url': 'https://course.fast.ai/', 'level': 'Intermediate-Advanced'},
                {'name': 'Google AI Education', 'url': 'https://ai.google/education/', 'level': 'All Levels'},
            ],
            'computer science': [
                {'name': 'MIT OpenCourseWare', 'url': 'https://ocw.mit.edu/', 'level': 'Intermediate-Advanced'},
                {'name': 'CS50 by Harvard', 'url': 'https://cs50.harvard.edu/', 'level': 'Beginner-Intermediate'},
                {'name': 'Teach Yourself CS', 'url': 'https://teachyourselfcs.com/', 'level': 'Intermediate-Advanced'},
            ],
            'mathematics': [
                {'name': 'Khan Academy Math', 'url': 'https://www.khanacademy.org/math', 'level': 'Beginner-Advanced'},
                {'name': '3Blue1Brown', 'url': 'https://www.3blue1brown.com/', 'level': 'Intermediate-Advanced'},
                {'name': 'Paul\'s Online Math Notes', 'url': 'https://tutorial.math.lamar.edu/', 'level': 'Intermediate'},
            ],
            'languages': [
                {'name': 'Duolingo', 'url': 'https://www.duolingo.com/', 'level': 'Beginner-Intermediate'},
                {'name': 'BBC Languages', 'url': 'https://www.bbc.co.uk/languages/', 'level': 'Beginner'},
                {'name': 'Memrise', 'url': 'https://www.memrise.com/', 'level': 'Beginner-Intermediate'},
            ],
            'science': [
                {'name': 'Khan Academy Science', 'url': 'https://www.khanacademy.org/science', 'level': 'Beginner-Advanced'},
                {'name': 'NASA STEM Engagement', 'url': 'https://www.nasa.gov/stem', 'level': 'All Levels'},
                {'name': 'Coursera Science', 'url': 'https://www.coursera.org/browse/physical-science-and-engineering', 'level': 'Intermediate-Advanced'},
            ],
            'business': [
                {'name': 'Coursera Business', 'url': 'https://www.coursera.org/browse/business', 'level': 'All Levels'},
                {'name': 'edX Business Courses', 'url': 'https://www.edx.org/course/subject/business-management', 'level': 'All Levels'},
                {'name': 'Harvard Business School Online', 'url': 'https://online.hbs.edu/', 'level': 'Intermediate-Advanced'},
            ],
            'creative arts': [
                {'name': 'Skillshare', 'url': 'https://www.skillshare.com/', 'level': 'Beginner-Intermediate'},
                {'name': 'CreativeLive', 'url': 'https://www.creativelive.com/', 'level': 'Beginner-Advanced'},
                {'name': 'Drawabox', 'url': 'https://drawabox.com/', 'level': 'Beginner'},
            ]
        }

    def get_resource(self, topic: str) -> Optional[Dict[str, str]]:
        """Get a random resource for a given topic"""
        topic = topic.lower()
        for category, resources in self.resources.items():
            if topic in category.lower():
                return random.choice(resources)
        return None

    def open_resource(self, topic: str) -> bool:
        """Open a learning resource in the default browser"""
        resource = self.get_resource(topic)
        if resource:
            webbrowser.open(resource['url'])
            return True
        return False

    def list_categories(self) -> List[str]:
        """Get all available learning categories"""
        return list(self.resources.keys())

    def get_resources_by_category(self, category: str) -> Optional[List[Dict[str, str]]]:
        """Get all resources for a specific category"""
        category = category.lower()
        for cat, resources in self.resources.items():
            if category in cat.lower():
                return resources
        return None

    def add_custom_resource(self, category: str, name: str, url: str, level: str = 'All Levels') -> None:
        """Add a custom learning resource"""
        if category.lower() not in (c.lower() for c in self.resources.keys()):
            self.resources[category] = []
        self.resources[category].append({
            'name': name,
            'url': url,
            'level': level
        })

class HealthMonitor:
    @staticmethod
    def CheckSystemHealth() -> Dict:
        health = {}
        try:
            # CPU
            cpu = psutil.cpu_percent(interval=1)
            health['cpu'] = f"{cpu}%"
            
            # Memory
            mem = psutil.virtual_memory()
            health['memory'] = f"{mem.percent}% used"
            
            # Disk
            disk = psutil.disk_usage('/')
            health['disk'] = f"{disk.percent}% used"
            
            return health
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return {'error': str(e)}

class NewsReader:
    def __init__(self):
        self.ApiKey = NEWS_API_KEY
        self.BaseUrl = "https://newsapi.org/v2/top-headlines"

    def GetHeadlines(self, category: str = "general", country: str = "us") -> List[str]:
        if not self.ApiKey:
            return ["News API not configured"]
            
        try:
            params = {
                'category': category,
                'country': country,
                'apiKey': self.ApiKey
            }
            response = requests.get(self.BaseUrl, params=params)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            return [article['title'] for article in articles[:5]]
        except Exception as e:
            logging.error(f"News fetch error: {e}")
            return ["Could not fetch news headlines"]

class EmailManager:
    def __init__(self):
        self.SmtpServer = "smtp.gmail.com"
        self.SmtpPort = 587
        self.Email = "your_email@gmail.com"
        self.Password = "your_app_password"

    def SendEmail(self, to: str, subject: str, body: str) -> bool:
        try:
            with smtplib.SMTP(self.SmtpServer, self.SmtpPort) as server:
                server.starttls()
                server.login(self.Email, self.Password)
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(self.Email, to, message)
            return True
        except Exception as e:
            logging.error(f"Email failed: {e}")
            return False


@dataclass
class Song:
    title: str
    artist: str
    duration: int  # in seconds
    url: str
    thumbnail: Optional[str] = None

class YouTubeMusicPlayer:
    def __init__(self):
        self.instance = vlc.Instance('--no-xlib')
        self.player = self.instance.media_player_new()
        self.playlist: List[Song] = []
        self.current_index = 0
        self.volume = 50
        self.is_playing = False
        self.keep_playing = True
        self.command_queue = Queue()
        self.player.event_manager().event_attach(
            vlc.EventType.MediaPlayerEndReached, 
            self._track_ended
        )
        
        # Start command processing thread
        self.command_thread = threading.Thread(
            target=self._process_commands, 
            daemon=True
        )
        self.command_thread.start()
        
        self.set_volume(self.volume)

    def _track_ended(self, event):
        """Automatically play next track when current ends"""
        if self.keep_playing and self.is_playing:
            self.next_track()

    def _process_commands(self):
        """Process commands from queue in background"""
        while True:
            command = self.command_queue.get()
            if command == "stop":
                self.stop()
            elif command == "pause":
                self.pause()
            elif command == "resume":
                self.resume()
            elif command == "next":
                self.next_track()
            elif command == "previous":
                self.previous_track()
            elif command.startswith("volume"):
                level = int(command.split()[1])
                self.set_volume(level)
            self.command_queue.task_done()

    def search_and_add(self, query: str) -> Optional[Song]:
        """Search YouTube and add to playlist"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'extract_flat': True,
            'default_search': 'ytsearch1'
        }
        
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if not info or 'entries' not in info or not info['entries']:
                    return None
                    
                entry = info['entries'][0]
                song = Song(
                    title=entry.get('title', 'Unknown Track'),
                    artist=entry.get('uploader', 'Unknown Artist'),
                    duration=entry.get('duration', 0),
                    url=entry['url'],
                    thumbnail=entry.get('thumbnail', '')
                )
                self.playlist.append(song)
                return song
        except Exception as e:
            print(f"Search error: {e}")
            return None

    def play(self, query: Optional[str] = None) -> bool:
        """Start playback (optionally with new search)"""
        if query:
            self.search_and_add(query)
            
        if not self.playlist:
            return False
            
        self.keep_playing = True
        self.current_index = max(0, min(self.current_index, len(self.playlist)-1))
        current_song = self.playlist[self.current_index]
        
        media = self.instance.media_new(current_song.url)
        self.player.set_media(media)
        
        if self.player.play() == -1:
            return False
            
        self.is_playing = True
        return True

    def pause(self) -> None:
        """Pause playback"""
        if self.is_playing:
            self.player.pause()
            self.is_playing = False

    def resume(self) -> None:
        """Resume playback"""
        if not self.is_playing and self.playlist:
            self.player.play()
            self.is_playing = True

    def stop(self) -> None:
        """Stop playback and clear playlist"""
        self.player.stop()
        self.is_playing = False
        self.keep_playing = False
        self.playlist = []
        self.current_index = 0

    def next_track(self) -> bool:
        """Skip to next track"""
        if self.current_index < len(self.playlist) - 1:
            self.current_index += 1
            return self.play()
        return False

    def previous_track(self) -> bool:
        """Go to previous track"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.play()
        return False

    def set_volume(self, level: int) -> None:
        """Set volume (0-100)"""
        self.volume = max(0, min(100, level))
        self.player.audio_set_volume(self.volume)

    def volume_up(self, increment: int = 10) -> None:
        """Increase volume"""
        self.set_volume(self.volume + increment)

    def volume_down(self, decrement: int = 10) -> None:
        """Decrease volume"""
        self.set_volume(self.volume - decrement)

    def get_current_song(self) -> Optional[Song]:
        """Get currently playing song info"""
        if 0 <= self.current_index < len(self.playlist):
            return self.playlist[self.current_index]
        return None

    def add_voice_command(self, command: str) -> None:
        """Add voice command to processing queue"""
        self.command_queue.put(command.lower())

    def run_continuous_playback(self, initial_query: str = None):
        """Start continuous playback until stopped"""
        if initial_query:
            self.search_and_add(initial_query)
        self.play()
        
        # Keep running until stopped
        while self.keep_playing:
            time.sleep(1)


class YouTubeMusic:
    def __init__(self):
        self.Player = vlc.Instance('--no-xlib').media_player_new()
        self.CurrentVolume = 50

    def Play(self, query: str) -> bool:
        try:
            ydlOpts = {'format': 'bestaudio/best', 'quiet': True}
            with youtube_dl.YoutubeDL(ydlOpts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if not info or 'entries' not in info or not info['entries']:
                    return False
                
                video = info['entries'][0]
                media = vlc.Instance().media_new(video['url'])
                self.Player.set_media(media)
                self.Player.audio_set_volume(self.CurrentVolume)
                return self.Player.play() != -1
        except Exception as e:
            logging.error(f"YouTube music error: {e}")
            return False

    def SetVolume(self, level: int) -> None:
        if 0 <= level <= 100:
            self.CurrentVolume = level
            self.Player.audio_set_volume(level)

class SpotifyMusic:
    def __init__(self):
        self.ClientId = SPOTIFY_CLIENT_ID
        self.ClientSecret = SPOTIFY_CLIENT_SECRET
        self.AccessToken = None
        self.Player = None

    def Authenticate(self) -> bool:
        return False

    def Play(self, query: str) -> bool:
        print("Spotify integration not yet implemented")
        return False

    def SetVolume(self, level: int) -> None:
        pass

class FileSearch:
    @staticmethod
    def SearchFiles(directory: str, pattern: str) -> List[str]:
        return [f for f in glob.glob(os.path.join(directory, f'**/*{pattern}*'), recursive=True)]

class MathCalculator:
    @staticmethod
    def Calculate(expression: str) -> float:
        try:
            return eval(expression, {'__builtins__': None}, {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'sqrt': math.sqrt,
                'log': math.log,
                'pi': math.pi,
                'e': math.e
            })
        except Exception as e:
            logging.error(f"Calculation error: {e}")
            return float('nan')

class ScreenCapture:
    @staticmethod
    def TakeScreenshot(filePath: str = "screenshot.png") -> bool:
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filePath)
            return True
        except Exception as e:
            logging.error(f"Screenshot error: {e}")
            return False

    @staticmethod
    def StartRecording(filePath: str = "recording.avi") -> bool:
        try:
            screenSize = pyautogui.size()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(filePath, fourcc, 20.0, screenSize)
            return out.isOpened()
        except Exception as e:
            logging.error(f"Recording error: {e}")
            return False




@dataclass
class CreatorInfo:
    """
    A class that provides information about the creators and designers of the program.
    Uses PascalCase naming convention as requested.
    """
    
    ProgramName: str = "Ultron AI Assistant"
    Version: str = "1.1.0"
    LeadCreator: str = "Jamil"
    LeadDesigner: str = "Your Design Team"
    Contributors: List[str] = None
    ContactEmail: str = "jamiluddin3282003@gmail.com"
    CreationYear: int = 2023
    TechnologiesUsed: List[str] = None
    License: str = "MIT License"
    
    def __post_init__(self):
        """Initialize default lists if None"""
        if self.Contributors is None:
            self.Contributors = ["Contributor 1", "Contributor 2"]
        if self.TechnologiesUsed is None:
            self.TechnologiesUsed = ["Python", "PyTorch", "VLC", "SpeechRecognition"]
    
    def GetCreatorInfo(self) -> Dict[str, str]:
        """Return basic creator information as a dictionary"""
        return {
            "Program": self.ProgramName,
            "Version": self.Version,
            "Creator": self.LeadCreator,
            "Designer": self.LeadDesigner,
            "Contact": self.ContactEmail
        }
    
    def GetDetailedInfo(self) -> Dict:
        """Return complete information about the project"""
        return {
            "ProgramName": self.ProgramName,
            "Version": self.Version,
            "LeadCreator": self.LeadCreator,
            "LeadDesigner": self.LeadDesigner,
            "Contributors": self.Contributors,
            "ContactEmail": self.ContactEmail,
            "CreationYear": self.CreationYear,
            "TechnologiesUsed": self.TechnologiesUsed,
            "License": self.License
        }
    
    def DisplayCredits(self) -> None:
        """Print formatted credits information"""
        credits = f"""
        {self.ProgramName} (v{self.Version})
        Created by: {self.LeadCreator}
        Designed by: {self.LeadDesigner}
        Contributors: {', '.join(self.Contributors)}
        Year: {self.CreationYear}
        Technologies: {', '.join(self.TechnologiesUsed)}
        License: {self.License}
        Contact: {self.ContactEmail}
        """
        print(credits)





@dataclass
class Movie:
    title: str
    year: int
    genre: str
    rating: float

@dataclass
class Joke:
    content: str
    category: str

class EntertainmentManager:
    def __init__(self):
        self.movie_db = self._load_movie_database()
        self.joke_categories = ['programming', 'pun', 'knock-knock']
        self.joke_db = self._load_joke_database()

    def _load_movie_database(self) -> List[Movie]:
        """Load movie database with more comprehensive examples"""
        return [
            Movie("Inception", 2010, "Sci-Fi", 8.8),
            Movie("The Shawshank Redemption", 1994, "Drama", 9.3),
            Movie("The Dark Knight", 2008, "Action", 9.0),
            Movie("Pulp Fiction", 1994, "Crime", 8.9),
            Movie("Parasite", 2019, "Thriller", 8.6)
        ]

    def _load_joke_database(self) -> Dict[str, List[str]]:
        """Pre-load all jokes to avoid recreation each time"""
        return {
        'programming': [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why do Java developers wear glasses? Because they can't C#.",
            "Why did the programmer quit their job? They didn't get arrays.",
            "How many programmers does it take to change a light bulb? None – it's a hardware problem.",
            "Why do Python programmers have curly hair? Because they love indentation.",
            "Why was the JavaScript developer sad? Because they didn't know how to 'null' their feelings.",
            "Why did the developer go broke? Because they used up all their cache.",
            "What's a programmer's favorite drink? Java.",
            "Why did the React component break up with the Angular component? It couldn't handle the binding.",
            "Why don't programmers like nature? It has too many bugs.",
            "Why did the developer get stuck in the shower? The instructions said 'lather, rinse, repeat.'",
            "What do you call a programmer from Finland? Nerdic.",
            "Why did the database administrator leave their wife? Because she had one-to-many relationships.",
            "How do you tell an introverted programmer from an extroverted one? The extrovert looks at YOUR shoes.",
            "Why did the programmer get kicked out of school? Because they refused to go to class (their methods were static).",
            "What's the object-oriented way to become wealthy? Inheritance.",
            "Why did the programmer get a dog? To bark their errors.",
            "Why do programmers always mix up Christmas and Halloween? Because Oct 31 == Dec 25.",
            "What's a programmer's favorite place to hang out? Foo Bar.",
            "Why did the programmer get fired from the zoo? They fed the ducks and dereferenced NULL.",
            "Why did the developer hate their keyboard? It had too many 'esc' keys.",
            "What's a programmer's favorite snake? Python... or maybe a boa constrictor (they love compression).",
            "Why did the frontend developer die? They didn't know their margins.",
            "Why did the developer go to the beach? To catch some async rays.",
            "What's a programmer's favorite exercise? Looping.",
            "Why did the developer get arrested? They committed a class felony.",
            "Why did the programmer get kicked out of the bar? They kept throwing exceptions.",
            "What do you call a programmer who doesn't comment their code? A future job applicant.",
            "Why did the developer cross the road? To git to the other side.",
            "Why was the developer bad at hide and seek? Because they always left their console.log().",
            "Why did the developer go broke? They spent all their cache on cookies.",
            "What's a programmer's favorite musical note? The C#.",
            "Why did the developer get lost in the forest? They took a wrong tree.",
            "Why did the developer bring a ladder to the bar? They heard the drinks were on the house.",
            "Why did the developer get a pet snake? To handle their Python exceptions.",
            "Why did the developer refuse to play cards? They were afraid of stack overflow.",
            "What's a programmer's favorite type of dog? A watchdog (for monitoring).",
            "Why did the developer get kicked out of the bakery? They kept kneading the dough (null).",
            "Why did the developer bring a pencil to the fight? To draw their weapon.",
            "Why did the developer get a job at the bakery? They kneaded the dough.",
            "Why did the developer hate their garden? Too many root issues.",
            "Why did the developer get a job at the space station? They wanted to work in a void.",
            "Why did the developer get fired from the circus? They kept juggling too many threads.",
            "Why did the developer get a job at the zoo? They were good at handling pandas.",
            "Why did the developer refuse to play chess? They couldn't find the knight moves.",
            "Why did the developer get a job at the post office? They loved sorting algorithms.",
            "Why did the developer get kicked out of the band? They kept breaking the rhythm.",
            "Why did the developer get a job at the farm? They loved planting seeds (for random numbers).",
            "Why did the developer get a job at the gym? They loved lifting weights (in their code).",
            "Why did the developer get a job at the bakery? They loved rolling in dough (null)."
        ],
        'pun': [
            "I told my computer I needed a break... now it won't stop sending me Kit-Kats.",
            "How do you comfort a JavaScript bug? You console it.",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a fake noodle? An impasta!",
            "How do you organize a space party? You planet!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "How do you make a tissue dance? Put a little boogie in it!",
            "Why did the math book look sad? Because it had too many problems!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What's brown and sticky? A stick!",
            "How do you catch a squirrel? Climb a tree and act like a nut!",
            "Why can't you explain puns to kleptomaniacs? They always take things literally!",
            "What's the best time to go to the dentist? Tooth-hurty!",
            "Why did the bicycle fall over? Because it was two-tired!",
            "What do you call cheese that isn't yours? Nacho cheese!",
            "Why did the golfer bring two pairs of pants? In case he got a hole in one!",
            "What do you call a snowman with a six-pack? An abdominal snowman!",
            "How does a penguin build its house? Igloos it together!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What did the buffalo say to his son when he left for college? Bison!",
            "Why did the tomato turn red? Because it saw the salad dressing!",
            "What do you call a fish wearing a bowtie? Sofishticated!",
            "How do you make a lemon drop? Just let it fall!",
            "What did one wall say to the other wall? I'll meet you at the corner!",
            "Why did the cookie go to the doctor? Because it was feeling crumbly!",
            "What do you call a dinosaur with an extensive vocabulary? A thesaurus!",
            "Why don't oysters donate to charity? Because they're shellfish!",
            "What did the janitor say when he jumped out of the closet? Supplies!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a snowman with a six-pack? An abdominal snowman!",
            "How do you organize a space party? You planet!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "How do you make a tissue dance? Put a little boogie in it!",
            "Why did the math book look sad? Because it had too many problems!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What's brown and sticky? A stick!",
            "How do you catch a squirrel? Climb a tree and act like a nut!",
            "Why can't you explain puns to kleptomaniacs? They always take things literally!",
            "What's the best time to go to the dentist? Tooth-hurty!",
            "Why did the bicycle fall over? Because it was two-tired!",
            "What do you call cheese that isn't yours? Nacho cheese!",
            "Why did the golfer bring two pairs of pants? In case he got a hole in one!",
            "What do you call a snowman with a six-pack? An abdominal snowman!",
            "How does a penguin build its house? Igloos it together!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What did the buffalo say to his son when he left for college? Bison!",
            "Why did the tomato turn red? Because it saw the salad dressing!",
            "What do you call a fish wearing a bowtie? Sofishticated!",
            "How do you make a lemon drop? Just let it fall!",
            "What did one wall say to the other wall? I'll meet you at the corner!",
            "Why did the cookie go to the doctor? Because it was feeling crumbly!",
            "What do you call a dinosaur with an extensive vocabulary? A thesaurus!",
            "Why don't oysters donate to charity? Because they're shellfish!",
            "What did the janitor say when he jumped out of the closet? Supplies!"
        ],
        'knock-knock': [
            "Knock knock. Who's there? Art. Art who? R2D2!",
            "Knock knock. Who's there? HTML. HTML who? Did you forget to close the tag?",
            "Knock knock. Who's there? CSS. CSS who? CSS-lee, I don't know who's there!",
            "Knock knock. Who's there? JavaScript. JavaScript who? JavaScript the doorbell, I'm not coming in!",
            "Knock knock. Who's there? PHP. PHP who? PHP-fully, you don't know who I am!",
            "Knock knock. Who's there? SQL. SQL who? Sequel the jokes, I'm running out!",
            "Knock knock. Who's there? API. API who? API the door and find out!",
            "Knock knock. Who's there? Linux. Linux who? Linux the door, it's cold outside!",
            "Knock knock. Who's there? Git. Git who? Git the door, it's freezing!",
            "Knock knock. Who's there? Python. Python who? Python the door, I'm a snake!",
            "Knock knock. Who's there? Java. Java who? Java nice day, don't you think?",
            "Knock knock. Who's there? C++. C++ who? C++ the door, I'm freezing!",
            "Knock knock. Who's there? Ruby. Ruby who? Ruby the doorbell, I'm not coming in!",
            "Knock knock. Who's there? Perl. Perl who? Perl the door, I'm a script!",
            "Knock knock. Who's there? Swift. Swift who? Swift the door, I'm in a hurry!",
            "Knock knock. Who's there? Kotlin. Kotlin who? Kotlin the door, I'm a new language!",
            "Knock knock. Who's there? Rust. Rust who? Rust the door, I'm a systems language!",
            "Knock knock. Who's there? Go. Go who? Go away, I'm busy!",
            "Knock knock. Who's there? Docker. Docker who? Docker the door, I'm a container!",
            "Knock knock. Who's there? Kubernetes. Kubernetes who? Kubernetes the door, I'm a cluster!",
            "Knock knock. Who's there? AWS. AWS who? AWS-ome, you finally opened the door!",
            "Knock knock. Who's there? Azure. Azure who? Azure the door, I'm a cloud!",
            "Knock knock. Who's there? GCP. GCP who? GCP the door, I'm a cloud too!",
            "Knock knock. Who's there? AI. AI who? AI the door, I'm a robot!",
            "Knock knock. Who's there? ML. ML who? ML the door, I'm learning!",
            "Knock knock. Who's there? VR. VR who? VR the door, I'm virtual!",
            "Knock knock. Who's there? AR. AR who? AR the door, I'm augmented!",
            "Knock knock. Who's there? IoT. IoT who? IoT the door, I'm connected!",
            "Knock knock. Who's there? 5G. 5G who? 5G the door, I'm fast!",
            "Knock knock. Who's there? WiFi. WiFi who? WiFi the door, I'm wireless!",
            "Knock knock. Who's there? Bluetooth. Bluetooth who? Bluetooth the door, I'm pairing!",
            "Knock knock. Who's there? USB. USB who? USB the door, I'm a connector!",
            "Knock knock. Who's there? HDMI. HDMI who? HDMI the door, I'm a cable!",
            "Knock knock. Who's there? CPU. CPU who? CPU the door, I'm processing!",
            "Knock knock. Who's there? GPU. GPU who? GPU the door, I'm rendering!",
            "Knock knock. Who's there? RAM. RAM who? RAM the door, I'm memory!",
            "Knock knock. Who's there? SSD. SSD who? SSD the door, I'm fast storage!",
            "Knock knock. Who's there? HDD. HDD who? HDD the door, I'm slow storage!",
            "Knock knock. Who's there? BIOS. BIOS who? BIOS the door, I'm firmware!",
            "Knock knock. Who's there? OS. OS who? OS the door, I'm an operating system!",
            "Knock knock. Who's there? GUI. GUI who? GUI the door, I'm graphical!",
            "Knock knock. Who's there? CLI. CLI who? CLI the door, I'm command line!",
            "Knock knock. Who's there? IDE. IDE who? IDE the door, I'm an editor!",
            "Knock knock. Who's there? VIM. VIM who? VIM the door, I'm a text editor!",
            "Knock knock. Who's there? Emacs. Emacs who? Emacs the door, I'm another editor!",
            "Knock knock. Who's there? Terminal. Terminal who? Terminal the door, I'm a shell!",
            "Knock knock. Who's there? Bash. Bash who? Bash the door, I'm a shell too!",
            "Knock knock. Who's there? Zsh. Zsh who? Zsh the door, I'm a better shell!",
            "Knock knock. Who's there? Fish. Fish who? Fish the door, I'm a friendly shell!",
            "Knock knock. Who's there? PowerShell. PowerShell who? PowerShell the door, I'm a Microsoft shell!",
            "Knock knock. Who's there? CMD. CMD who? CMD the door, I'm a Windows shell!",
            "Knock knock. Who's there? Batch. Batch who? Batch the door, I'm a script!",
            "Knock knock. Who's there? Cron. Cron who? Cron the door, I'm a scheduler!",
            "Knock knock. Who's there? Daemon. Daemon who? Daemon the door, I'm a background process!",
            "Knock knock. Who's there? Thread. Thread who? Thread the door, I'm a lightweight process!",
            "Knock knock. Who's there? Mutex. Mutex who? Mutex the door, I'm a lock!",
            "Knock knock. Who's there? Semaphore. Semaphore who? Semaphore the door, I'm a signal!",
            "Knock knock. Who's there? Deadlock. Deadlock who? Deadlock the door, I'm stuck!",
            "Knock knock. Who's there? Race. Race who? Race the door, I'm a condition!",
            "Knock knock. Who's there? Leak. Leak who? Leak the door, I'm a memory issue!",
            "Knock knock. Who's there? Bug. Bug who? Bug the door, I'm an issue!",
            "Knock knock. Who's there? Debug. Debug who? Debug the door, I'm fixing it!",
            "Knock knock. Who's there? Patch. Patch who? Patch the door, I'm an update!",
            "Knock knock. Who's there? Hotfix. Hotfix who? Hotfix the door, I'm urgent!",
            "Knock knock. Who's there? Feature. Feature who? Feature the door, I'm new!",
            "Knock knock. Who's there? Refactor. Refactor who? Refactor the door, I'm improving it!",
            "Knock knock. Who's there? Merge. Merge who? Merge the door, I'm combining!",
            "Knock knock. Who's there? Rebase. Rebase who? Rebase the door, I'm rewriting history!",
            "Knock knock. Who's there? Commit. Commit who? Commit the door, I'm saving!",
            "Knock knock. Who's there? Push. Push who? Push the door, I'm uploading!",
            "Knock knock. Who's there? Pull. Pull who? Pull the door, I'm downloading!",
            "Knock knock. Who's there? Fork. Fork who? Fork the door, I'm copying!",
            "Knock knock. Who's there? Clone. Clone who? Clone the door, I'm duplicating!",
            "Knock knock. Who's there? Branch. Branch who? Branch the door, I'm diverging!",
            "Knock knock. Who's there? Tag. Tag who? Tag the door, I'm labeling!",
            "Knock knock. Who's there? Stash. Stash who? Stash the door, I'm hiding!",
            "Knock knock. Who's there? Log. Log who? Log the door, I'm recording!",
            "Knock knock. Who's there? Diff. Diff who? Diff the door, I'm comparing!",
            "Knock knock. Who's there? Blame. Blame who? Blame the door, I'm accusing!",
            "Knock knock. Who's there? Cherry. Cherry who? Cherry-pick the door, I'm selecting!",
            "Knock knock. Who's there? Squash. Squash who? Squash the door, I'm combining commits!",
            "Knock knock. Who's there? Revert. Revert who? Revert the door, I'm undoing!",
            "Knock knock. Who's there? Reset. Reset who? Reset the door, I'm starting over!",
            "Knock knock. Who's there? Checkout. Checkout who? Checkout the door, I'm switching!",
            "Knock knock. Who's there? Fetch. Fetch who? Fetch the door, I'm retrieving!",
            "Knock knock. Who's there? Remote. Remote who? Remote the door, I'm distant!",
            "Knock knock. Who's there? Origin. Origin who? Origin the door, I'm the source!",
            "Knock knock. Who's there? Upstream. Upstream who? Upstream the door, I'm the main repo!",
            "Knock knock. Who's there? Downstream. Downstream who? Downstream the door, I'm a fork!",
            "Knock knock. Who's there? HEAD. HEAD who? HEAD the door, I'm the current commit!",
            "Knock knock. Who's there? Master. Master who? Master the door, I'm the main branch!",
            "Knock knock. Who's there? Main. Main who? Main the door, I'm the new default branch!",
            "Knock knock. Who's there? Develop. Develop who? Develop the door, I'm the working branch!",
            "Knock knock. Who's there? Release. Release who? Release the door, I'm the stable branch!",
            "Knock knock. Who's there? Hotfix. Hotfix who? Hotfix the door, I'm the urgent branch!",
            "Knock knock. Who's there? Feature. Feature who? Feature the door, I'm the new branch!",
            "Knock knock. Who's there? Bugfix. Bugfix who? Bugfix the door, I'm the fix branch!",
            "Knock knock. Who's there? Support. Support who? Support the door, I'm the legacy branch!",
            "Knock knock. Who's there? Archive. Archive who? Archive the door, I'm the old branch!",
            "Knock knock. Who's there? Trunk. Trunk who? Trunk the door, I'm the SVN branch!",
            "Knock knock. Who's there? Vendor. Vendor who? Vendor the door, I'm the third-party code!",
            "Knock knock. Who's there? Submodule. Submodule who? Submodule the door, I'm the nested repo!",
            "Knock knock. Who's there? Subtree. Subtree who? Subtree the door, I'm the merged repo!",
            "Knock knock. Who's there? LFS. LFS who? LFS the door, I'm the large file storage!",
            "Knock knock. Who's there? Hook. Hook who? Hook the door, I'm the script runner!",
            "Knock knock. Who's there? CI. CI who? CI the door, I'm the continuous integration!",
            "Knock knock. Who's there? CD. CD who? CD the door, I'm the continuous delivery!",
            "Knock knock. Who's there? DevOps. DevOps who? DevOps the door, I'm the culture!",
            "Knock knock. Who's there? SRE. SRE who? SRE the door, I'm the site reliability!",
            "Knock knock. Who's there? QA. QA who? QA the door, I'm the quality assurance!",
            "Knock knock. Who's there? UX. UX who? UX the door, I'm the user experience!",
            "Knock knock. Who's there? UI. UI who? UI the door, I'm the user interface!",
            "Knock knock. Who's there? API. API who? API the door, I'm the application interface!",
            "Knock knock. Who's there? SDK. SDK who? SDK the door, I'm the software development kit!",
            "Knock knock. Who's there? IDE. IDE who? IDE the door, I'm the integrated development environment!",
            "Knock knock. Who's there? CLI. CLI who? CLI the door, I'm the command line interface!",
            "Knock knock. Who's there? GUI. GUI who? GUI the door, I'm the graphical user interface!",
            "Knock knock. Who's there? TUI. TUI who? TUI the door, I'm the text user interface!",
            "Knock knock. Who's there? VUI. VUI who? VUI the door, I'm the voice user interface!",
            "Knock knock. Who's there? HUD. HUD who? HUD the door, I'm the heads-up display!",
            "Knock knock. Who's there? AR. AR who? AR the door, I'm the augmented reality!",
            "Knock knock. Who's there? VR. VR who? VR the door, I'm the virtual reality!",
            "Knock knock. Who's there? MR. MR who? MR the door, I'm the mixed reality!",
            "Knock knock. Who's there? XR. XR who? XR the door, I'm the extended reality!",
            "Knock knock. Who's there? AI. AI who? AI the door, I'm the artificial intelligence!",
            "Knock knock. Who's there? ML. ML who? ML the door, I'm the machine learning!",
            "Knock knock. Who's there? DL. DL who? DL the door, I'm the deep learning!",
            "Knock knock. Who's there? NN. NN who? NN the door, I'm the neural network!",
            "Knock knock. Who's there? CNN. CNN who? CNN the door, I'm the convolutional neural network!",
            "Knock knock. Who's there? RNN. RNN who? RNN the door, I'm the recurrent neural network!",
            "Knock knock. Who's there? GAN. GAN who? GAN the door, I'm the generative adversarial network!",
            "Knock knock. Who's there? NLP. NLP who? NLP the door, I'm the natural language processing!",
            "Knock knock. Who's there? CV. CV who? CV the door, I'm the computer vision!",
            "Knock knock. Who's there? RL. RL who? RL the door, I'm the reinforcement learning!",
            "Knock knock. Who's there? SL. SL who? SL the door, I'm the supervised learning!",
            "Knock knock. Who's there? UL. UL who? UL the door, I'm the unsupervised learning!",
            "Knock knock. Who's there? SSL. SSL who? SSL the door, I'm the semi-supervised learning!",
            "Knock knock. Who's there? TL. TL who? TL the door, I'm the transfer learning!",
            "Knock knock. Who's there? FL. FL who? FL the door, I'm the federated learning!",
            "Knock knock. Who's there? AL. AL who? AL the door, I'm the active learning!",
            "Knock knock. Who's there? QL. QL who? QL the door, I'm the quantum learning!",
            "Knock knock. Who's there? XAI. XAI who? XAI the door, I'm the explainable AI!",
            "Knock knock. Who's there? AGI. AGI who? AGI the door, I'm the artificial general intelligence!",
            "Knock knock. Who's there? ASI. ASI who? ASI the door, I'm the artificial superintelligence!",
            "Knock knock. Who's there? IoT. IoT who? IoT the door, I'm the internet of things!",
            "Knock knock. Who's there? IIoT. IIoT who? IIoT the door, I'm the industrial internet of things!",
            "Knock knock. Who's there? AIoT. AIoT who? AIoT the door, I'm the AI + IoT!",
            "Knock knock. Who's there? 5G. 5G who? 5G the door, I'm the fifth generation!",
            "Knock knock. Who's there? 6G. 6G who? 6G the door, I'm the sixth generation!",
            "Knock knock. Who's there? WiFi. WiFi who? WiFi the door, I'm the wireless fidelity!",
            "Knock knock. Who's there? LiFi. LiFi who? LiFi the door, I'm the light fidelity!",
            "Knock knock. Who's there? Bluetooth. Bluetooth who? Bluetooth the door, I'm the wireless personal area network!",
            "Knock knock. Who's there? Zigbee. Zigbee who? Zigbee the door, I'm the low-power wireless mesh network!",
            "Knock knock. Who's there? Z-Wave. Z-Wave who? Z-Wave the door, I'm the home automation protocol!",
            "Knock knock. Who's there? Thread. Thread who? Thread the door, I'm the IoT networking protocol!",
            "Knock knock. Who's there? Matter. Matter who? Matter the door, I'm the new smart home standard!",
            "Knock knock. Who's there? HomeKit. HomeKit who? HomeKit the door, I'm the Apple smart home framework!",
            "Knock knock. Who's there? Alexa. Alexa who? Alexa the door, I'm the Amazon voice assistant!",
            "Knock knock. Who's there? Google. Google who? Google the door, I'm the Google voice assistant!",
            "Knock knock. Who's there? Siri. Siri who? Siri the door, I'm the Apple voice assistant!",
            "Knock knock. Who's there? Cortana. Cortana who? Cortana the door, I'm the Microsoft voice assistant!",
            "Knock knock. Who's there? Bixby. Bixby who? Bixby the door, I'm the Samsung voice assistant!",
            "Knock knock. Who's there? Mycroft. Mycroft who? Mycroft the door, I'm the open-source voice assistant!",
            "Knock knock. Who's there? Jovo. Jovo who? Jovo the door, I'm the voice app framework!",
            "Knock knock. Who's there? Rasa. Rasa who? Rasa the door, I'm the conversational AI framework!",
            "Knock knock. Who's there? Dialogflow. Dialogflow who? Dialogflow the door, I'm the Google conversational AI!",
            "Knock knock. Who's there? LUIS. LUIS who? LUIS the door, I'm the Microsoft conversational AI!",
            "Knock knock. Who's there? Watson. Watson who? Watson the door, I'm the IBM conversational AI!",
            "Knock knock. Who's there? Lex. Lex who? Lex the door, I'm the Amazon conversational AI!",
            "Knock knock. Who's there? Wit. Wit who? Wit the door, I'm the Facebook conversational AI!",
            "Knock knock. Who's there? Snips. Snips who? Snips the door, I'm the privacy-focused voice assistant!",
            "Knock knock. Who's there? DeepPavlov. DeepPavlov who? DeepPavlov the door, I'm the Russian conversational AI!",
            "Knock knock. Who's there? HuggingFace. HuggingFace who? HuggingFace the door, I'm the open-source NLP!",
            "Knock knock. Who's there? SpaCy. SpaCy who? SpaCy the door, I'm the industrial-strength NLP!",
            "Knock knock. Who's there? NLTK. NLTK who? NLTK the door, I'm the natural language toolkit!",
            "Knock knock. Who's there? Gensim. Gensim who? Gensim the door, I'm the topic modeling library!",
            "Knock knock. Who's there? AllenNLP. AllenNLP who? AllenNLP the door, I'm the deep learning NLP!",
            "Knock knock. Who's there? Flair. Flair who? Flair the door, I'm the state-of-the-art NLP!",
            "Knock knock. Who's there? Stanza. Stanza who? Stanza the door, I'm the Stanford NLP!",
            "Knock knock. Who's there? CoreNLP. CoreNLP who? CoreNLP the door, I'm the Java NLP!",
            "Knock knock. Who's there? OpenNLP. OpenNLP who? OpenNLP the door, I'm the Apache NLP!",
            "Knock knock. Who's there? FastText. FastText who? FastText the door, I'm the efficient text classification!",
            "Knock knock. Who's there? Word2Vec. Word2Vec who? Word2Vec the door, I'm the word embeddings!",
            "Knock knock. Who's there? GloVe. GloVe who? GloVe the door, I'm the global vectors!",
            "Knock knock. Who's there? ELMo. ELMo who? ELMo the door, I'm the embeddings from language models!",
            "Knock knock. Who's there? BERT. BERT who? BERT the door, I'm the bidirectional encoder representations!",
            "Knock knock. Who's there? GPT. GPT who? GPT the door, I'm the generative pre-trained transformer!",
            "Knock knock. Who's there? XLNet. XLNet who? XLNet the door, I'm the generalized autoregressive pretraining!",
            "Knock knock. Who's there? RoBERTa. RoBERTa who? RoBERTa the door, I'm the robustly optimized BERT!",
            "Knock knock. Who's there? ALBERT. ALBERT who? ALBERT the door, I'm the lite BERT!",
            "Knock knock. Who's there? DistilBERT. DistilBERT who? DistilBERT the door, I'm the distilled BERT!",
            "Knock knock. Who's there? T5. T5 who? T5 the door, I'm the text-to-text transfer transformer!",
            "Knock knock. Who's there? Reformer. Reformer who? Reformer the door, I'm the efficient transformer!",
            "Knock knock. Who's there? Transformer. Transformer who? Transformer the door, I'm the attention is all you need!",
            "Knock knock. Who's there? Longformer. Longformer who? Longformer the door, I'm the long document transformer!",
            "Knock knock. Who's there? BigBird. BigBird who? BigBird the door, I'm the sparse transformer!",
            "Knock knock. Who's there? Pegasus. Pegasus who? Pegasus the door, I'm the pre-training with extracted gap-sentences!",
            "Knock knock. Who's there? BART. BART who? BART the door, I'm the denoising sequence-to-sequence pre-training!",
            "Knock knock. Who's there? ELECTRA. ELECTRA who? ELECTRA the door, I'm the efficiently learning an encoder!",
            "Knock knock. Who's there? DeBERTa. DeBERTa who? DeBERTa the door, I'm the decoding-enhanced BERT!",
            "Knock knock. Who's there? FNet. FNet who? FNet the door, I'm the Fourier transformer!",
            "Knock knock. Who's there? LUKE. LUKE who? LUKE the door, I'm the knowledge-enhanced transformer!",
            "Knock knock. Who's there? M2M. M2M who? M2M the door, I'm the massively multilingual machine translation!",
            "Knock knock. Who's there? mBART. mBART who? mBART the door, I'm the multilingual BART!",
            "Knock knock. Who's there? XLM. XLM who? XLM the door, I'm the cross-lingual language model!",
            "Knock knock. Who's there? XLM-R. XLM-R who? XLM-R the door, I'm the cross-lingual language model robust!",
            "Knock knock. Who's there? mT5. mT5 who? mT5 the door, I'm the multilingual T5!",
            "Knock knock. Who's there? ByT5. ByT5 who? ByT5 the door, I'm the byte-level T5!",
            "Knock knock. Who's there? Marian. Marian who? Marian the door, I'm the neural machine translation!",
            "Knock knock. Who's there? OPUS. OPUS who? OPUS the door, I'm the open parallel corpus!",
            "Knock knock. Who's there? FairSeq. FairSeq who? FairSeq the door, I'm the Facebook sequence modeling toolkit!",
            "Knock knock. Who's there? OpenNMT. OpenNMT who? OpenNMT the door, I'm the open neural machine translation!",
            "Knock knock. Who's there? JoeyNMT. JoeyNMT who? JoeyNMT the door, I'm the minimal NMT framework!",
            "Knock knock. Who's there? Sockeye. Sockeye who? Sockeye the door, I'm the Apache MXNet NMT!",
            "Knock knock. Who's there? Tensor2Tensor. Tensor2Tensor who? Tensor2Tensor the door, I'm the Google NMT!",
            "Knock knock. Who's there? TFSeq2Seq. TFSeq2Seq who? TFSeq2Seq the door, I'm the TensorFlow"
            ]
        }

    def get_movie_recommendation(self, genre: Optional[str] = None) -> Optional[Movie]:
        """Safer movie recommendation with better genre handling"""
        if genre:
            genre = genre.lower()
            movies = [m for m in self.movie_db if m.genre.lower() == genre]
            return random.choice(movies) if movies else None
        return random.choice(self.movie_db)

    def get_random_joke(self) -> Joke:
        """More efficient joke selection with pre-loaded database"""
        category = random.choice(self.joke_categories)
        return Joke(
            content=random.choice(self.joke_db[category]),
            category=category
        )

    # def get_tv_show_recommendation(self) -> Dict[str, str]:
    #     """TV show recommendation with more options"""
    #     shows = [
    #         # Sci-Fi/Fantasy (40 shows)
    #         {"title": "Stranger Things", "seasons": 4, "genre": "Sci-Fi", "rating": 8.7, "platform": "Netflix"},
    #         {"title": "The Mandalorian", "seasons": 3, "genre": "Sci-Fi", "rating": 8.8, "platform": "Disney+"},
    #         {"title": "Dark", "seasons": 3, "genre": "Sci-Fi", "rating": 8.8, "platform": "Netflix"},
    #         {"title": "The Expanse", "seasons": 6, "genre": "Sci-Fi", "rating": 8.5, "platform": "Prime Video"},
    #         {"title": "Westworld", "seasons": 4, "genre": "Sci-Fi", "rating": 8.5, "platform": "HBO Max"},
    #         {"title": "Black Mirror", "seasons": 5, "genre": "Sci-Fi", "rating": 8.8, "platform": "Netflix"},
    #         {"title": "Altered Carbon", "seasons": 2, "genre": "Sci-Fi", "rating": 8.0, "platform": "Netflix"},
    #         {"title": "The Witcher", "seasons": 3, "genre": "Fantasy", "rating": 8.2, "platform": "Netflix"},
    #         {"title": "Foundation", "seasons": 2, "genre": "Sci-Fi", "rating": 7.6, "platform": "Apple TV+"},
    #         {"title": "Raised by Wolves", "seasons": 2, "genre": "Sci-Fi", "rating": 7.5, "platform": "HBO Max"},
            
    #         # Drama (40 shows)
    #         {"title": "Breaking Bad", "seasons": 5, "genre": "Drama", "rating": 9.5, "platform": "Netflix"},
    #         {"title": "Better Call Saul", "seasons": 6, "genre": "Drama", "rating": 9.0, "platform": "Netflix"},
    #         {"title": "The Crown", "seasons": 5, "genre": "Drama", "rating": 8.7, "platform": "Netflix"},
    #         {"title": "Succession", "seasons": 4, "genre": "Drama", "rating": 8.9, "platform": "HBO Max"},
    #         {"title": "Mad Men", "seasons": 7, "genre": "Drama", "rating": 8.6, "platform": "Prime Video"},
    #         {"title": "The Sopranos", "seasons": 6, "genre": "Drama", "rating": 9.2, "platform": "HBO Max"},
    #         {"title": "House of the Dragon", "seasons": 1, "genre": "Drama", "rating": 8.5, "platform": "HBO Max"},
    #         {"title": "The Last of Us", "seasons": 1, "genre": "Drama", "rating": 8.9, "platform": "HBO Max"},
    #         {"title": "Pose", "seasons": 3, "genre": "Drama", "rating": 8.6, "platform": "Netflix"},
    #         {"title": "This Is Us", "seasons": 6, "genre": "Drama", "rating": 8.7, "platform": "Hulu"},
            
    #         # Crime/Thriller (40 shows)
    #         {"title": "True Detective", "seasons": 3, "genre": "Crime", "rating": 8.9, "platform": "HBO Max"},
    #         {"title": "Mindhunter", "seasons": 2, "genre": "Crime", "rating": 8.6, "platform": "Netflix"},
    #         {"title": "Narcos", "seasons": 3, "genre": "Crime", "rating": 8.8, "platform": "Netflix"},
    #         {"title": "Peaky Blinders", "seasons": 6, "genre": "Crime", "rating": 8.8, "platform": "Netflix"},
    #         {"title": "Ozark", "seasons": 4, "genre": "Crime", "rating": 8.5, "platform": "Netflix"},
    #         {"title": "Money Heist", "seasons": 5, "genre": "Crime", "rating": 8.3, "platform": "Netflix"},
    #         {"title": "The Wire", "seasons": 5, "genre": "Crime", "rating": 9.3, "platform": "HBO Max"},
    #         {"title": "Dexter", "seasons": 9, "genre": "Crime", "rating": 8.6, "platform": "Prime Video"},
    #         {"title": "Fargo", "seasons": 4, "genre": "Crime", "rating": 8.9, "platform": "Hulu"},
    #         {"title": "Killing Eve", "seasons": 4, "genre": "Crime", "rating": 8.3, "platform": "Hulu"},
            
    #         # Comedy (40 shows)
    #         {"title": "The Office (US)", "seasons": 9, "genre": "Comedy", "rating": 8.9, "platform": "Peacock"},
    #         {"title": "Friends", "seasons": 10, "genre": "Comedy", "rating": 8.9, "platform": "HBO Max"},
    #         {"title": "Brooklyn Nine-Nine", "seasons": 8, "genre": "Comedy", "rating": 8.4, "platform": "Peacock"},
    #         {"title": "Parks and Recreation", "seasons": 7, "genre": "Comedy", "rating": 8.6, "platform": "Peacock"},
    #         {"title": "The Good Place", "seasons": 4, "genre": "Comedy", "rating": 8.2, "platform": "Netflix"},
    #         {"title": "Ted Lasso", "seasons": 3, "genre": "Comedy", "rating": 8.8, "platform": "Apple TV+"},
    #         {"title": "Community", "seasons": 6, "genre": "Comedy", "rating": 8.5, "platform": "Netflix"},
    #         {"title": "Arrested Development", "seasons": 5, "genre": "Comedy", "rating": 8.7, "platform": "Netflix"},
    #         {"title": "Schitt's Creek", "seasons": 6, "genre": "Comedy", "rating": 8.5, "platform": "Netflix"},
    #         {"title": "The Marvelous Mrs. Maisel", "seasons": 5, "genre": "Comedy", "rating": 8.7, "platform": "Prime Video"},
            
    #         # Anime (20 shows)
    #         {"title": "Attack on Titan", "seasons": 4, "genre": "Anime", "rating": 9.0, "platform": "Crunchyroll"},
    #         {"title": "Death Note", "seasons": 1, "genre": "Anime", "rating": 9.0, "platform": "Netflix"},
    #         {"title": "Demon Slayer", "seasons": 3, "genre": "Anime", "rating": 8.7, "platform": "Crunchyroll"},
    #         {"title": "Fullmetal Alchemist: Brotherhood", "seasons": 1, "genre": "Anime", "rating": 9.1, "platform": "Crunchyroll"},
    #         {"title": "Cowboy Bebop", "seasons": 1, "genre": "Anime", "rating": 8.9, "platform": "Hulu"},
            
    #         # Reality TV (20 shows)
    #         {"title": "The Great British Bake Off", "seasons": 10, "genre": "Reality", "rating": 8.6, "platform": "Netflix"},
    #         {"title": "Queer Eye", "seasons": 7, "genre": "Reality", "rating": 8.6, "platform": "Netflix"},
    #         {"title": "RuPaul's Drag Race", "seasons": 15, "genre": "Reality", "rating": 8.3, "platform": "Paramount+"},
    #         {"title": "Survivor", "seasons": 43, "genre": "Reality", "rating": 7.5, "platform": "CBS"}]
        
    #     return random.choice(shows)


    def get_tv_show_recommendation(self) -> Dict[str, str]:
        """TV show recommendation with extensive options"""
        shows = [
            # Sci-Fi/Fantasy
            {"title": "Stranger Things", "seasons": 4, "genre": "Sci-Fi", "rating": 8.7},
            {"title": "The Mandalorian", "seasons": 3, "genre": "Sci-Fi", "rating": 8.7},
            {"title": "The Expanse", "seasons": 6, "genre": "Sci-Fi", "rating": 8.5},
            {"title": "Westworld", "seasons": 4, "genre": "Sci-Fi", "rating": 8.5},
            {"title": "Black Mirror", "seasons": 5, "genre": "Sci-Fi", "rating": 8.8},
            {"title": "Altered Carbon", "seasons": 2, "genre": "Sci-Fi", "rating": 8.0},
            {"title": "The Witcher", "seasons": 3, "genre": "Fantasy", "rating": 8.2},
            {"title": "Shadow and Bone", "seasons": 2, "genre": "Fantasy", "rating": 7.6},
            {"title": "The Boys", "seasons": 4, "genre": "Sci-Fi", "rating": 8.7},
            {"title": "Invincible", "seasons": 2, "genre": "Sci-Fi", "rating": 8.7},
            
            # Drama
            {"title": "The Crown", "seasons": 5, "genre": "Drama", "rating": 8.6},
            {"title": "Breaking Bad", "seasons": 5, "genre": "Drama", "rating": 9.5},
            {"title": "Better Call Saul", "seasons": 6, "genre": "Drama", "rating": 9.0},
            {"title": "Succession", "seasons": 4, "genre": "Drama", "rating": 8.8},
            {"title": "The Sopranos", "seasons": 6, "genre": "Drama", "rating": 9.2},
            {"title": "Mad Men", "seasons": 7, "genre": "Drama", "rating": 8.6},
            {"title": "The Wire", "seasons": 5, "genre": "Drama", "rating": 9.3},
            {"title": "House of Cards", "seasons": 6, "genre": "Drama", "rating": 8.7},
            {"title": "Ozark", "seasons": 4, "genre": "Drama", "rating": 8.5},
            {"title": "Peaky Blinders", "seasons": 6, "genre": "Drama", "rating": 8.8},
            
            # Crime/Thriller
            {"title": "Mindhunter", "seasons": 2, "genre": "Crime", "rating": 8.6},
            {"title": "True Detective", "seasons": 3, "genre": "Crime", "rating": 8.9},
            {"title": "Narcos", "seasons": 3, "genre": "Crime", "rating": 8.8},
            {"title": "Money Heist", "seasons": 5, "genre": "Crime", "rating": 8.3},
            {"title": "Dexter", "seasons": 8, "genre": "Crime", "rating": 8.6},
            {"title": "The Sinner", "seasons": 4, "genre": "Crime", "rating": 7.9},
            {"title": "Broadchurch", "seasons": 3, "genre": "Crime", "rating": 8.4},
            {"title": "Sherlock", "seasons": 4, "genre": "Crime", "rating": 9.1},
            {"title": "Luther", "seasons": 5, "genre": "Crime", "rating": 8.4},
            {"title": "Line of Duty", "seasons": 6, "genre": "Crime", "rating": 8.7},
            
            # Comedy
            {"title": "The Office (US)", "seasons": 9, "genre": "Comedy", "rating": 8.9},
            {"title": "Parks and Recreation", "seasons": 7, "genre": "Comedy", "rating": 8.6},
            {"title": "Brooklyn Nine-Nine", "seasons": 8, "genre": "Comedy", "rating": 8.4},
            {"title": "The Good Place", "seasons": 4, "genre": "Comedy", "rating": 8.2},
            {"title": "Arrested Development", "seasons": 5, "genre": "Comedy", "rating": 8.7},
            {"title": "Community", "seasons": 6, "genre": "Comedy", "rating": 8.5},
            {"title": "It's Always Sunny in Philadelphia", "seasons": 15, "genre": "Comedy", "rating": 8.8},
            {"title": "Ted Lasso", "seasons": 3, "genre": "Comedy", "rating": 8.8},
            {"title": "Schitt's Creek", "seasons": 6, "genre": "Comedy", "rating": 8.5},
            {"title": "The Marvelous Mrs. Maisel", "seasons": 5, "genre": "Comedy", "rating": 8.7},
            
            # Animated
            {"title": "Arcane", "seasons": 1, "genre": "Animated", "rating": 9.0},
            {"title": "Rick and Morty", "seasons": 6, "genre": "Animated", "rating": 9.1},
            {"title": "BoJack Horseman", "seasons": 6, "genre": "Animated", "rating": 8.8},
            {"title": "Avatar: The Last Airbender", "seasons": 3, "genre": "Animated", "rating": 9.3},
            {"title": "Attack on Titan", "seasons": 4, "genre": "Animated", "rating": 9.0},
            {"title": "Demon Slayer", "seasons": 3, "genre": "Animated", "rating": 8.7},
            {"title": "Castlevania", "seasons": 4, "genre": "Animated", "rating": 8.3},
            {"title": "Love, Death & Robots", "seasons": 3, "genre": "Animated", "rating": 8.5},
            {"title": "The Simpsons", "seasons": 34, "genre": "Animated", "rating": 8.6},
            {"title": "Family Guy", "seasons": 21, "genre": "Animated", "rating": 8.1},
            
            # Reality/Competition
            {"title": "Queer Eye", "seasons": 7, "genre": "Reality", "rating": 8.6},
            {"title": "The Great British Bake Off", "seasons": 13, "genre": "Reality", "rating": 8.6},
            {"title": "RuPaul's Drag Race", "seasons": 15, "genre": "Reality", "rating": 8.3},
            {"title": "MasterChef", "seasons": 12, "genre": "Reality", "rating": 7.8},
            {"title": "Survivor", "seasons": 44, "genre": "Reality", "rating": 7.5},
            {"title": "The Amazing Race", "seasons": 35, "genre": "Reality", "rating": 7.6},
            {"title": "Shark Tank", "seasons": 14, "genre": "Reality", "rating": 7.5},
            {"title": "Top Chef", "seasons": 20, "genre": "Reality", "rating": 7.9},
            {"title": "Nailed It!", "seasons": 7, "genre": "Reality", "rating": 7.4},
            {"title": "Making It", "seasons": 3, "genre": "Reality", "rating": 7.8},
            
            # Historical/Fantasy
            {"title": "Game of Thrones", "seasons": 8, "genre": "Fantasy", "rating": 9.2},
            {"title": "House of the Dragon", "seasons": 1, "genre": "Fantasy", "rating": 8.5},
            {"title": "The Last Kingdom", "seasons": 5, "genre": "Historical", "rating": 8.5},
            {"title": "Vikings", "seasons": 6, "genre": "Historical", "rating": 8.5},
            {"title": "Outlander", "seasons": 7, "genre": "Historical", "rating": 8.4},
            {"title": "The Tudors", "seasons": 4, "genre": "Historical", "rating": 8.1},
            {"title": "Rome", "seasons": 2, "genre": "Historical", "rating": 8.7},
            {"title": "Marco Polo", "seasons": 2, "genre": "Historical", "rating": 8.0},
            {"title": "Knightfall", "seasons": 2, "genre": "Historical", "rating": 6.9},
            {"title": "Barbarians", "seasons": 2, "genre": "Historical", "rating": 7.0},
            
            # Superhero
            {"title": "Daredevil", "seasons": 3, "genre": "Superhero", "rating": 8.6},
            {"title": "The Punisher", "seasons": 2, "genre": "Superhero", "rating": 8.5},
            {"title": "Jessica Jones", "seasons": 3, "genre": "Superhero", "rating": 8.0},
            {"title": "Luke Cage", "seasons": 2, "genre": "Superhero", "rating": 7.3},
            {"title": "WandaVision", "seasons": 1, "genre": "Superhero", "rating": 8.0},
            {"title": "The Falcon and the Winter Soldier", "seasons": 1, "genre": "Superhero", "rating": 7.3},
            {"title": "Loki", "seasons": 2, "genre": "Superhero", "rating": 8.2},
            {"title": "The Umbrella Academy", "seasons": 3, "genre": "Superhero", "rating": 8.0},
            {"title": "Doom Patrol", "seasons": 4, "genre": "Superhero", "rating": 7.9},
            {"title": "The Flash", "seasons": 9, "genre": "Superhero", "rating": 7.6},
            
            # Horror
            {"title": "The Haunting of Hill House", "seasons": 1, "genre": "Horror", "rating": 8.6},
            {"title": "The Haunting of Bly Manor", "seasons": 1, "genre": "Horror", "rating": 7.4},
            {"title": "Midnight Mass", "seasons": 1, "genre": "Horror", "rating": 7.7},
            {"title": "American Horror Story", "seasons": 12, "genre": "Horror", "rating": 8.0},
            {"title": "The Walking Dead", "seasons": 11, "genre": "Horror", "rating": 8.2},
            {"title": "Fear the Walking Dead", "seasons": 8, "genre": "Horror", "rating": 7.0},
            {"title": "Penny Dreadful", "seasons": 3, "genre": "Horror", "rating": 8.2},
            {"title": "Castle Rock", "seasons": 2, "genre": "Horror", "rating": 7.5},
            {"title": "The Terror", "seasons": 2, "genre": "Horror", "rating": 8.0},
            {"title": "Channel Zero", "seasons": 4, "genre": "Horror", "rating": 7.2},
            
            # International
            {"title": "Dark", "seasons": 3, "genre": "Sci-Fi", "rating": 8.8},
            {"title": "Money Heist", "seasons": 5, "genre": "Crime", "rating": 8.3},
            {"title": "Squid Game", "seasons": 1, "genre": "Thriller", "rating": 8.0},
            {"title": "Kingdom", "seasons": 2, "genre": "Horror", "rating": 8.4},
            {"title": "Elite", "seasons": 7, "genre": "Drama", "rating": 7.3},
            {"title": "Lupin", "seasons": 3, "genre": "Crime", "rating": 7.5},
            {"title": "Call My Agent!", "seasons": 4, "genre": "Comedy", "rating": 8.3},
            {"title": "Marianne", "seasons": 1, "genre": "Horror", "rating": 7.5},
            {"title": "The Rain", "seasons": 3, "genre": "Sci-Fi", "rating": 6.3},
            {"title": "Ragnarok", "seasons": 3, "genre": "Fantasy", "rating": 7.5}
        ]
        return random.choice(shows)
    

@dataclass
class GameNews:
    title: str
    source: str
    summary: str
    date: str

class GamingNewsManager:
    def __init__(self):
        self.sources = {
            'IGN': 'https://www.ign.com/articles',
            'Polygon': 'https://www.polygon.com/gaming'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        self.timeout = 10

    def fetch_latest_news(self, source: Optional[str] = None) -> List[GameNews]:
        """Fetch news with better error handling"""
        try:
            if source:
                return self._fetch_from_source(source)
            return [news for src in self.sources 
                    for news in self._fetch_from_source(src)]
        except Exception as e:
            print(f"Failed to fetch news: {e}")
            return []

    def _fetch_from_source(self, source: str) -> List[GameNews]:
        """Fetch with timeout and error handling"""
        try:
            if source == 'IGN':
                return self._parse_ign()
            elif source == 'Polygon':
                return self._parse_polygon()
            return []
        except Exception as e:
            print(f"Error with {source}: {e}")
            return []

    def _parse_ign(self) -> List[GameNews]:
        """More robust IGN parser"""
        try:
            response = requests.get(
                self.sources['IGN'],
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.select('article.listElmt') or soup.select('article.item-list')
            
            return [
                GameNews(
                    title=art.select_one('h3').get_text(strip=True) if art.select_one('h3') else "No title",
                    source='IGN',
                    summary=art.select_one('p').get_text(strip=True) if art.select_one('p') else "",
                    date=art.select_one('time')['datetime'] if art.select_one('time') else ""
                ) for art in articles[:3]
            ]
        except Exception as e:
            print(f"IGN parsing failed: {e}")
            return []

    def _parse_polygon(self) -> List[GameNews]:
        """More robust Polygon parser"""
        try:
            response = requests.get(
                self.sources['Polygon'],
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.select('article.c-entry-box--compact') or soup.select('article.entry')
            
            return [
                GameNews(
                    title=art.select_one('h2').get_text(strip=True) if art.select_one('h2') else "No title",
                    source='Polygon',
                    summary=art.select_one('p').get_text(strip=True) if art.select_one('p') else "",
                    date=art.select_one('time').get_text(strip=True) if art.select_one('time') else ""
                ) for art in articles[:3]
            ]
        except Exception as e:
            print(f"Polygon parsing failed: {e}")
            return []
        




class JokeTeller:
    @staticmethod
    def TellJoke() -> str:
        return pyjokes.get_joke()

class AlarmManager:
    def __init__(self):
        self.Alarms = []
        self.Running = False

    def SetAlarm(self, timeStr: str) -> bool:
        try:
            alarmTime = datetime.strptime(timeStr, "%H:%M")
            now = datetime.now()
            alarmTime = alarmTime.replace(year=now.year, month=now.month, day=now.day)
            
            if alarmTime < now:
                alarmTime += timedelta(days=1)
                
            self.Alarms.append(alarmTime)
            if not self.Running:
                self.Running = True
                threading.Thread(target=self.CheckAlarms, daemon=True).start()
            return True
        except ValueError:
            return False

    def CheckAlarms(self) -> None:
        while self.Running and self.Alarms:
            now = datetime.now()
            for alarm in self.Alarms[:]:
                if now >= alarm:
                    print("\a")  # System beep
                    self.Alarms.remove(alarm)
            time.sleep(30)

class DateTimeManager:
    @staticmethod
    def GetCurrentDate() -> str:
        return datetime.now().strftime("%A, %B %d, %Y")

    @staticmethod
    def GetCurrentTime() -> str:
        return datetime.now().strftime("%I:%M %p")

class VoiceWriter:
    def __init__(self):
        self.Recognizer = sr.Recognizer()

    def Transcribe(self, audioFile: str) -> Optional[str]:
        try:
            with sr.AudioFile(audioFile) as source:
                audio = self.Recognizer.record(source)
                return self.Recognizer.recognize_google(audio)
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None








class DCSeries:
    def __init__(self):
        self.movies = [
            {"title": "Man of Steel", "year": 2013, "director": "Zack Snyder", "rating": 7.0},
            {"title": "Batman v Superman: Dawn of Justice", "year": 2016, "director": "Zack Snyder", "rating": 6.4},
            {"title": "Wonder Woman", "year": 2017, "director": "Patty Jenkins", "rating": 7.4},
            {"title": "Justice League", "year": 2017, "director": "Zack Snyder", "rating": 6.1},
            {"title": "Aquaman", "year": 2018, "director": "James Wan", "rating": 6.8},
            {"title": "Shazam!", "year": 2019, "director": "David F. Sandberg", "rating": 7.0},
            {"title": "The Batman", "year": 2022, "director": "Matt Reeves", "rating": 7.9},
        ]
        
        self.tv_shows = [
            {"title": "Arrow", "seasons": 8, "years": "2012-2020", "rating": 7.5},
            {"title": "The Flash", "seasons": 9, "years": "2014-2023", "rating": 7.6},
            {"title": "Supergirl", "seasons": 6, "years": "2015-2021", "rating": 6.2},
            {"title": "DC's Legends of Tomorrow", "seasons": 7, "years": "2016-2022", "rating": 6.9},
            {"title": "Peacemaker", "seasons": 1, "years": "2022-present", "rating": 8.3},
        ]
        
        self.characters = [
            {"name": "Superman", "first_appearance": 1938, "powers": ["Super strength", "Flight", "Heat vision"]},
            {"name": "Batman", "first_appearance": 1939, "powers": ["Peak human condition", "Master detective", "Martial arts"]},
            {"name": "Wonder Woman", "first_appearance": 1941, "powers": ["Super strength", "Lasso of Truth", "Combat skills"]},
            {"name": "The Flash", "first_appearance": 1940, "powers": ["Super speed", "Time travel", "Phasing"]},
            {"name": "Aquaman", "first_appearance": 1941, "powers": ["Underwater breathing", "Telepathy with sea life", "Super strength"]},
        ]

    def get_movie_recommendation(self) -> dict:
        """Get a random DC movie recommendation"""
        return random.choice(self.movies)
    
    def get_tv_show_recommendation(self) -> dict:
        """Get a random DC TV show recommendation"""
        return random.choice(self.tv_shows)
    
    def get_character_info(self, name: str) -> Optional[dict]:
        """Get information about a DC character"""
        name = name.lower()
        for character in self.characters:
            if name in character["name"].lower():
                return character
        return None
    
    def search_content(self, query: str) -> List[dict]:
        """Search DC movies, shows, and characters"""
        query = query.lower()
        results = []
        
        # Search movies
        for movie in self.movies:
            if query in movie["title"].lower():
                results.append({"type": "movie", "data": movie})
        
        # Search TV shows
        for show in self.tv_shows:
            if query in show["title"].lower():
                results.append({"type": "tv_show", "data": show})
        
        # Search characters
        for character in self.characters:
            if query in character["name"].lower():
                results.append({"type": "character", "data": character})
        
        return results
    
class MarvelSeries:
    def __init__(self):
        self.movies = [
            {"title": "Iron Man", "year": 2008, "director": "Jon Favreau", "rating": 7.9},
            {"title": "The Incredible Hulk", "year": 2008, "director": "Louis Leterrier", "rating": 6.6},
            {"title": "Iron Man 2", "year": 2010, "director": "Jon Favreau", "rating": 7.0},
            {"title": "Thor", "year": 2011, "director": "Kenneth Branagh", "rating": 7.0},
            {"title": "Captain America: The First Avenger", "year": 2011, "director": "Joe Johnston", "rating": 6.9},
            {"title": "The Avengers", "year": 2012, "director": "Joss Whedon", "rating": 8.0},
            {"title": "Iron Man 3", "year": 2013, "director": "Shane Black", "rating": 7.1},
            {"title": "Thor: The Dark World", "year": 2013, "director": "Alan Taylor", "rating": 6.8},
            {"title": "Captain America: The Winter Soldier", "year": 2014, "director": "Anthony and Joe Russo", "rating": 7.8},
            {"title": "Guardians of the Galaxy", "year": 2014, "director": "James Gunn", "rating": 8.0},
            {"title": "Avengers: Age of Ultron", "year": 2015, "director": "Joss Whedon", "rating": 7.3},
            {"title": "Ant-Man", "year": 2015, "director": "Peyton Reed", "rating": 7.3},
            {"title": "Captain America: Civil War", "year": 2016, "director": "Anthony and Joe Russo", "rating": 7.8},
            {"title": "Doctor Strange", "year": 2016, "director": "Scott Derrickson", "rating": 7.5},
            {"title": "Guardians of the Galaxy Vol. 2", "year": 2017, "director": "James Gunn", "rating": 7.6},
            {"title": "Spider-Man: Homecoming", "year": 2017, "director": "Jon Watts", "rating": 7.4},
            {"title": "Thor: Ragnarok", "year": 2017, "director": "Taika Waititi", "rating": 7.9},
            {"title": "Black Panther", "year": 2018, "director": "Ryan Coogler", "rating": 7.3},
            {"title": "Avengers: Infinity War", "year": 2018, "director": "Anthony and Joe Russo", "rating": 8.4},
            {"title": "Ant-Man and the Wasp", "year": 2018, "director": "Peyton Reed", "rating": 7.0},
            {"title": "Captain Marvel", "year": 2019, "director": "Anna Boden and Ryan Fleck", "rating": 6.8},
            {"title": "Avengers: Endgame", "year": 2019, "director": "Anthony and Joe Russo", "rating": 8.4},
            {"title": "Spider-Man: Far From Home", "year": 2019, "director": "Jon Watts", "rating": 7.4},
            {"title": "Black Widow", "year": 2021, "director": "Cate Shortland", "rating": 6.7},
            {"title": "Shang-Chi and the Legend of the Ten Rings", "year": 2021, "director": "Destin Daniel Cretton", "rating": 7.5},
            {"title": "Eternals", "year": 2021, "director": "Chloé Zhao", "rating": 6.3},
            {"title": "Spider-Man: No Way Home", "year": 2021, "director": "Jon Watts", "rating": 8.3},
            {"title": "Doctor Strange in the Multiverse of Madness", "year": 2022, "director": "Sam Raimi", "rating": 6.9},
            {"title": "Thor: Love and Thunder", "year": 2022, "director": "Taika Waititi", "rating": 6.2},
            {"title": "Black Panther: Wakanda Forever", "year": 2022, "director": "Ryan Coogler", "rating": 7.0},
            {"title": "Ant-Man and the Wasp: Quantumania", "year": 2023, "director": "Peyton Reed", "rating": 6.1},
            {"title": "Guardians of the Galaxy Vol. 3", "year": 2023, "director": "James Gunn", "rating": 7.8},
            {"title": "The Marvels", "year": 2023, "director": "Nia DaCosta", "rating": 6.1},
            {"title": "Captain America: Brave New World", "year": 2025, "director": "Julius Onah", "rating": None}, # Assuming release date
            {"title": "Thunderbolts", "year": 2025, "director": "Jake Schreier", "rating": None}, # Assuming release date
            {"title": "Fantastic Four", "year": 2025, "director": "Matt Shakman", "rating": None}, # Assuming release date
            {"title": "Blade", "year": 2025, "director": "Yann Demange", "rating": None}, # Assuming release date
            {"title": "Avengers: The Kang Dynasty", "year": 2026, "director": "Destin Daniel Cretton", "rating": None}, # Assuming release date
            {"title": "Avengers: Secret Wars", "year": 2027, "director": None, "rating": None}, # Assuming release date
        ]

        self.tv_shows = [
            {"title": "Agents of S.H.I.E.L.D.", "seasons": 7, "years": "2013-2020", "rating": 7.5},
            {"title": "Agent Carter", "seasons": 2, "years": "2015-2016", "rating": 7.9},
            {"title": "Daredevil", "seasons": 3, "years": "2015-2018", "rating": 8.6},
            {"title": "Jessica Jones", "seasons": 3, "years": "2015-2019", "rating": 7.9},
            {"title": "Luke Cage", "seasons": 2, "years": "2016-2018", "rating": 7.3},
            {"title": "Iron Fist", "seasons": 2, "years": "2017-2018", "rating": 6.4},
            {"title": "The Defenders", "seasons": 1, "years": 2017, "rating": 7.2},
            {"title": "Inhumans", "seasons": 1, "years": 2017, "rating": 4.9},
            {"title": "The Punisher", "seasons": 2, "years": "2017-2019", "rating": 8.5},
            {"title": "Runaways", "seasons": 3, "years": "2017-2019", "rating": 7.1},
            {"title": "Cloak & Dagger", "seasons": 2, "years": "2018-2019", "rating": 6.8},
            {"title": "The Falcon and the Winter Soldier", "seasons": 1, "years": 2021, "rating": 7.2},
            {"title": "Loki", "seasons": 2, "years": "2021-present", "rating": 8.2},
            {"title": "WandaVision", "seasons": 1, "years": 2021, "rating": 7.9},
            {"title": "What If...?", "seasons": 2, "years": "2021-present", "rating": 7.4},
            {"title": "Hawkeye", "seasons": 1, "years": 2021, "rating": 7.5},
            {"title": "Moon Knight", "seasons": 1, "years": 2022, "rating": 7.3},
            {"title": "Ms. Marvel", "seasons": 1, "years": 2022, "rating": 6.2},
            {"title": "She-Hulk: Attorney at Law", "seasons": 1, "years": 2022, "rating": 5.1},
            {"title": "Secret Invasion", "seasons": 1, "years": 2023, "rating": 6.0},
            {"title": "Echo", "seasons": 1, "years": 2024, "rating": None}, # Assuming release date
            {"title": "Agatha: Darkhold Diaries", "seasons": 1, "years": 2024, "rating": None}, # Assuming release date
            {"title": "Daredevil: Born Again", "seasons": 1, "years": 2025, "rating": None}, # Assuming release date
            {"title": "Ironheart", "seasons": 1, "years": 2025, "rating": None}, # Assuming release date
        ]

        self.characters = [
            {"name": "Iron Man", "first_appearance": 1963, "powers": ["Powered armor", "Flight", "Energy blasts"]},
            {"name": "Captain America", "first_appearance": 1941, "powers": ["Super strength", "Agility", "Indestructible shield"]},
            {"name": "Thor", "first_appearance": 1962, "powers": ["Super strength", "Mjolnir", "Lightning manipulation"]},
            {"name": "Spider-Man", "first_appearance": 1962, "powers": ["Spider-sense", "Web-shooters", "Wall-crawling"]},
            {"name": "Hulk", "first_appearance": 1962, "powers": ["Super strength", "Durability", "Rage transformation"]},
            {"name": "Black Widow", "first_appearance": 1964, "powers": ["Master spy", "Martial arts", "Gadgets"]},
            {"name": "Hawkeye", "first_appearance": 1964, "powers": ["Master archer", "Expert marksman", "Combat skills"]},
            {"name": "Doctor Strange", "first_appearance": 1963, "powers": ["Sorcery", "Mystic arts", "Astral projection"]},
            {"name": "Black Panther", "first_appearance": 1966, "powers": ["Superhuman senses", "Agility", "Vibranium suit"]},
            {"name": "Captain Marvel", "first_appearance": 1967, "powers": ["Flight", "Energy absorption & projection", "Super strength"]},
            {"name": "Ant-Man", "first_appearance": 1962, "powers": ["Size manipulation", "Strength enhancement"]},
            {"name": "Wasp", "first_appearance": 1963, "powers": ["Flight", "Size manipulation", "Bio-electric blasts"]},
            {"name": "Guardians of the Galaxy", "first_appearance": 1969, "powers": ["Varied depending on member"]},
            {"name": "Daredevil", "first_appearance": 1964, "powers": ["Enhanced senses", "Radar sense", "Martial arts"]},
            {"name": "Jessica Jones", "first_appearance": 2001, "powers": ["Super strength", "Flight", "Durability"]},
            {"name": "Luke Cage", "first_appearance": 1972, "powers": ["Super strength", "Invulnerability"]},
            {"name": "Iron Fist", "first_appearance": 1974, "powers": ["Martial arts", "Chi manipulation"]},
            {"name": "Shang-Chi", "first_appearance": 1973, "powers": ["Master martial artist", "Chi manipulation"]},
            {"name": "Eternals", "first_appearance": 1976, "powers": ["Varied depending on member"]},
            {"name": "Moon Knight", "first_appearance": 1975, "powers": ["Varied depending on personality"]},
            {"name": "Ms. Marvel", "first_appearance": 2013, "powers": ["Shape-shifting", "Elongation", "Healing factor"]},
        ]

    def get_movie_recommendation(self) -> dict:
        """Get a random Marvel movie recommendation"""
        return random.choice(self.movies)

    def get_tv_show_recommendation(self) -> dict:
        """Get a random Marvel TV show recommendation"""
        return random.choice(self.tv_shows)

    def get_character_info(self, name: str) -> Optional[dict]:
        """Get information about a Marvel character"""
        name = name.lower()
        for character in self.characters:
            if name in character["name"].lower():
                return character
        return None

    def search_content(self, query: str) -> List[dict]:
        """Search Marvel movies, shows, and characters"""
        query = query.lower()
        results = []

        # Search movies
        for movie in self.movies:
            if query in movie["title"].lower():
                results.append({"type": "movie", "data": movie})

        # Search TV shows
        for show in self.tv_shows:
            if query in show["title"].lower():
                results.append({"type": "tv_show", "data": show})

        # Search characters
        for character in self.characters:
            if query in character["name"].lower():
                results.append({"type": "character", "data": character})

        return results
    


import requests
from bs4 import BeautifulSoup
import random
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EntertainmentNews:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the EntertainmentNews class.

        Args:
            api_key (Optional[str]): An API key for a news provider (if needed).
                                     Set to None if scraping is the primary method.
        """
        self.api_key = api_key
        self.base_urls = {
            "variety": "https://variety.com/",
            "hollywood_reporter": "https://www.hollywoodreporter.com/",
            "entertainment_weekly": "https://ew.com/",
            # Add more base URLs for other entertainment news sites
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def _fetch_html(self, url: str) -> Optional[str]:
        """
        Fetches the HTML content of a given URL.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The HTML content as a string, or None if an error occurred.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def scrape_latest_news(self, source: str = "random", num_articles: int = 3) -> List[Dict[str, str]]:
        """
        Scrapes the latest entertainment news from a specified or random source.

        Args:
            source (str): The news source to scrape from (e.g., "variety", "hollywood_reporter", "random").
                          Defaults to "random".
            num_articles (int): The number of articles to scrape. Defaults to 3.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary contains
                                   'title', 'link', and optionally 'summary'.
        """
        if source == "random":
            source = random.choice(list(self.base_urls.keys()))

        if source not in self.base_urls:
            logging.warning(f"Invalid news source '{source}'. Available sources: {list(self.base_urls.keys())}")
            return []

        url = self.base_urls[source]
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []

        try:
            if source == "variety":
                news_items = soup.find_all('article', class_='river-item')[:num_articles]
                for item in news_items:
                    title_tag = item.find('h3', class_='river-item-title')
                    link_tag = item.find('a', class_='u-display-block')
                    if title_tag and link_tag:
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + link_tag['href'].lstrip('/')
                        articles.append({'title': title, 'link': link})
            elif source == "hollywood_reporter":
                news_items = soup.find_all('div', class_='list-item__content')[:num_articles]
                for item in news_items:
                    title_tag = item.find('a', class_='list-item__title')
                    if title_tag and title_tag.has_attr('href'):
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + title_tag['href'].lstrip('/')
                        articles.append({'title': title, 'link': link})
            elif source == "entertainment_weekly":
                news_items = soup.find_all('div', class_='mntl-card-list-item')[:num_articles]
                for item in news_items:
                    title_tag = item.find('a', class_='mntl-card-list-item__link')
                    if title_tag and title_tag.has_attr('href'):
                        title = title_tag.find('span', class_='card__title-text').text.strip()
                        link = title_tag['href']
                        articles.append({'title': title, 'link': link})
            # Add more parsing logic for other news sources
            else:
                logging.warning(f"Scraping logic for '{source}' is not yet implemented.")
        except Exception as e:
            logging.error(f"Error parsing news from '{source}': {e}")

        return articles

    def get_random_news(self) -> Optional[Dict[str, str]]:
        """
        Gets a random entertainment news article from a random source.

        Returns:
            Optional[Dict[str, str]]: A dictionary containing 'title' and 'link' of a random article,
                                       or None if no articles could be retrieved.
        """
        latest_news = self.scrape_latest_news(source="random", num_articles=1)
        if latest_news:
            return latest_news[0]
        return None

import requests
from bs4 import BeautifulSoup
import random
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FootballData:
    def __init__(self):
        self.base_urls = {
            "goal_com": "https://www.goal.com/en",
            "espn_fc": "https://www.espn.com/soccer/",
            # Add more football news sources if needed
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def get_latest_football_news(self, source: str = "random", num_articles: int = 3) -> List[Dict[str, str]]:
        if source == "random":
            source = random.choice(list(self.base_urls.keys()))

        if source not in self.base_urls:
            logging.warning(f"Invalid football news source '{source}'. Available sources: {list(self.base_urls.keys())}")
            return []

        url = self.base_urls[source]
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []

        try:
            if source == "goal_com":
                news_items = soup.find_all('div', class_='item--headline')[:num_articles]
                for item in news_items:
                    link_tag = item.find('a')
                    title_tag = link_tag.find('span') if link_tag else None
                    if link_tag and title_tag:
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + link_tag['href']
                        articles.append({'title': title, 'link': link})
            elif source == "espn_fc":
                headlines = soup.select('div.contentItem__content a.contentItem__title')[:num_articles]
                links = [self.base_urls[source] + a['href'] for a in headlines if 'href' in a.attrs]
                titles = [a.text.strip() for a in headlines]
                articles = [{'title': t, 'link': l} for t, l in zip(titles, links)]
            # Add more parsing logic for other sources
            else:
                logging.warning(f"Scraping logic for '{source}' is not yet implemented.")
        except Exception as e:
            logging.error(f"Error parsing football news from '{source}': {e}")

        return articles

    def get_random_football_news(self) -> Optional[Dict[str, str]]:
        latest_news = self.get_latest_football_news(source="random", num_articles=1)
        if latest_news:
            return latest_news[0]
        return None

class FootballPlayer:
    def __init__(self, name: str, nationality: str, current_club: Optional[str] = None):
        self.name = name
        self.nationality = nationality
        self.current_club = current_club
        self.achievements = {
            "Ballon d'Or": 0,
            "Champions League": 0,
            "League Titles": 0,
            "Domestic Cups": 0,
            "International Trophies": 0,
            "Golden Boots": 0,
            # Add more relevant achievements
        }

    def add_achievement(self, achievement_type: str, count: int = 1):
        if achievement_type in self.achievements:
            self.achievements[achievement_type] += count
        else:
            logging.warning(f"Achievement type '{achievement_type}' not recognized for {self.name}.")

    def get_achievements(self) -> Dict[str, int]:
        return self.achievements

    def display_achievements(self):
        print(f"\nAchievements for {self.name}:")
        for trophy, count in self.achievements.items():
            if count > 0:
                print(f"- {trophy}: {count}")
        if all(count == 0 for count in self.achievements.values()):
            print(f"- No major achievements recorded yet.")

class CristianoRonaldo(FootballPlayer):
    def __init__(self):
        super().__init__(name="Cristiano Ronaldo", nationality="Portuguese", current_club="Al Nassr")
        self._load_achievements()

    def _load_achievements(self):
        self.add_achievement("Ballon d'Or", 5)
        self.add_achievement("Champions League", 5)
        self.add_achievement("League Titles", 7) # Portugal, England, Spain, Italy
        self.add_achievement("Domestic Cups", 12) # Across different leagues
        self.add_achievement("International Trophies", 2) # Euro, Nations League
        self.add_achievement("Golden Boots", 4) # Europe
        # Add more specific golden boots and other records

    def signature_move(self):
        return "Powerful shot and 'Siuuu' celebration!"

class LionelMessi(FootballPlayer):
    def __init__(self):
        super().__init__(name="Lionel Messi", nationality="Argentine", current_club="Inter Miami CF")
        self._load_achievements()

    def _load_achievements(self):
        self.add_achievement("Ballon d'Or", 8)
        self.add_achievement("Champions League", 4)
        self.add_achievement("League Titles", 12) # Spain, France
        self.add_achievement("Domestic Cups", 10) # Spain, France
        self.add_achievement("International Trophies", 3) # Copa America, Finalissima, World Cup
        self.add_achievement("Golden Boots", 6) # Europe
        # Add more specific golden boots and other records

    def signature_move(self):
        return "Dribbling past defenders and precise finishing."

class NeymarJr(FootballPlayer):
    def __init__(self):
        super().__init__(name="Neymar Jr", nationality="Brazilian", current_club="Al Hilal")
        self._load_achievements()

    def _load_achievements(self):
        self.add_achievement("Champions League", 1)
        self.add_achievement("League Titles", 6) # Brazil, Spain, France
        self.add_achievement("Domestic Cups", 10) # Brazil, Spain, France
        self.add_achievement("International Trophies", 1) # Confederations Cup
        self.add_achievement("Golden Boots", 0) # Major European Golden Boots
        # Add more specific achievements

    def signature_move(self):
        return "Skillful dribbling and flicks."

import requests
from bs4 import BeautifulSoup
import random
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GamingNews:
    def __init__(self):
        self.base_urls = {
            "ign": "https://www.ign.com/",
            "gamespot": "https://www.gamespot.com/",
            "polygon": "https://www.polygon.com/",
            # Add more gaming news sources if needed
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def get_latest_gaming_news(self, source: str = "random", num_articles: int = 3) -> List[Dict[str, str]]:
        if source == "random":
            source = random.choice(list(self.base_urls.keys()))

        if source not in self.base_urls:
            logging.warning(f"Invalid gaming news source '{source}'. Available sources: {list(self.base_urls.keys())}")
            return []

        url = self.base_urls[source]
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []

        try:
            if source == "ign":
                news_items = soup.select('div.list-item-default')[:num_articles]
                for item in news_items:
                    link_tag = item.find('a', class_='item-title')
                    if link_tag and 'href' in link_tag.attrs:
                        title = link_tag.text.strip()
                        link = self.base_urls[source] + link_tag['href'].lstrip('/')
                        articles.append({'title': title, 'link': link})
            elif source == "gamespot":
                news_items = soup.select('li.river-item')[:num_articles]
                for item in news_items:
                    link_tag = item.find('a', class_='js-event-tracking')
                    title_tag = item.find('h3', class_='media-title')
                    if link_tag and 'href' in link_tag.attrs and title_tag:
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + link_tag['href'].lstrip('/')
                        articles.append({'title': title, 'link': link})
            elif source == "polygon":
                news_items = soup.select('div.c-entry-box--compact__body')[:num_articles]
                for item in news_items:
                    link_tag = item.find('a', class_='c-entry-box--compact__title')
                    if link_tag and 'href' in link_tag.attrs:
                        title = link_tag.text.strip()
                        link = link_tag['href']
                        articles.append({'title': title, 'link': link})
            # Add more parsing logic for other sources
            else:
                logging.warning(f"Scraping logic for '{source}' is not yet implemented.")
        except Exception as e:
            logging.error(f"Error parsing gaming news from '{source}': {e}")

        return articles

    def get_random_gaming_news(self) -> Optional[Dict[str, str]]:
        latest_news = self.get_latest_gaming_news(source="random", num_articles=1)
        if latest_news:
            return latest_news[0]
        return None



import requests
from bs4 import BeautifulSoup
import random
from typing import List, Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FootballWorldCupData:
    def __init__(self):
        self.base_urls = {
            "fifa": "https://www.fifa.com/en/tournaments/mens/worldcup/",
            "goal_com": "https://www.goal.com/en/world-cup/",
            "espn_fc": "https://www.espn.com/soccer/world-cup/",
            # Add more World Cup specific news sources
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        self.next_world_cup_year = 2026
        self.next_world_cup_start_date = datetime(2026, 6, 11) # Placeholder, adjust actual date
        self.host_countries_2026 = ["Canada", "Mexico", "United States"]

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def get_world_cup_start_info(self) -> Dict[str, Union[int, datetime, List[str]]]:
        """Returns information about the next FIFA World Cup."""
        return {
            "year": self.next_world_cup_year,
            "start_date": self.next_world_cup_start_date.strftime("%Y-%m-%d"),
            "hosts": self.host_countries_2026
        }

    def get_latest_world_cup_news(self, source: str = "random", num_articles: int = 3) -> List[Dict[str, str]]:
        if source == "random":
            source = random.choice(list(self.base_urls.keys()))

        if source not in self.base_urls:
            logging.warning(f"Invalid World Cup news source '{source}'. Available sources: {list(self.base_urls.keys())}")
            return []

        url = self.base_urls[source]
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []

        try:
            if source == "fifa":
                news_items = soup.select('a.fc-item__link')[:num_articles]
                for item in news_items:
                    title_tag = item.find('h3', class_='fc-item__title')
                    if title_tag and 'href' in item.attrs:
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + item['href'].lstrip('/')
                        articles.append({'title': title, 'link': link})
            elif source == "goal_com":
                news_items = soup.find_all('div', class_='item--headline')[:num_articles]
                for item in news_items:
                    link_tag = item.find('a')
                    title_tag = link_tag.find('span') if link_tag else None
                    if link_tag and title_tag:
                        title = title_tag.text.strip()
                        link = self.base_urls[source] + link_tag['href']
                        articles.append({'title': title, 'link': link})
            elif source == "espn_fc":
                headlines = soup.select('div.contentItem__content a.contentItem__title')[:num_articles]
                links = [self.base_urls[source] + a['href'] for a in headlines if 'href' in a.attrs]
                titles = [a.text.strip() for a in headlines]
                articles = [{'title': t, 'link': l} for t, l in zip(titles, links)]
            # Add more parsing logic for other sources
            else:
                logging.warning(f"Scraping logic for '{source}' is not yet implemented.")
        except Exception as e:
            logging.error(f"Error parsing World Cup news from '{source}': {e}")

        return articles

    def get_random_world_cup_news(self) -> Optional[Dict[str, str]]:
        latest_news = self.get_latest_world_cup_news(source="random", num_articles=1)
        if latest_news:
            return latest_news[0]
        return None


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CricketData:
    def __init__(self):
        self.base_urls = {
            "cricbuzz": "https://www.cricbuzz.com/",
            "espncricinfo": "https://www.espncricinfo.com/",
            # Add more cricket news sources if needed
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def get_latest_cricket_news(self, source: str = "random", num_articles: int = 3) -> List[Dict[str, str]]:
        if source == "random":
            source = random.choice(list(self.base_urls.keys()))

        if source not in self.base_urls:
            logging.warning(f"Invalid cricket news source '{source}'. Available sources: {list(self.base_urls.keys())}")
            return []

        url = self.base_urls[source]
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []

        try:
            if source == "cricbuzz":
                headlines = soup.select('div#news-list ul li a')[:num_articles]
                for item in headlines:
                    title = item.text.strip()
                    link = self.base_urls[source] + item['href'].lstrip('/')
                    articles.append({'title': title, 'link': link})
            elif source == "espncricinfo":
                headlines = soup.select('div.news-story a.headline')[:num_articles]
                for item in headlines:
                    title = item.text.strip()
                    link = self.base_urls[source] + item['href'].lstrip('/')
                    articles.append({'title': title, 'link': link})
            # Add more parsing logic for other sources
            else:
                logging.warning(f"Scraping logic for '{source}' is not yet implemented.")
        except Exception as e:
            logging.error(f"Error parsing cricket news from '{source}': {e}")

        return articles

    def get_random_cricket_news(self) -> Optional[Dict[str, str]]:
        latest_news = self.get_latest_cricket_news(source="random", num_articles=1)
        if latest_news:
            return latest_news[0]
        return None

class Cricketer:
    def __init__(self, name: str, nationality: str, role: Optional[str] = None):
        self.name = name
        self.nationality = nationality
        self.role = role
        self.career_stats = {
            "Tests": {"matches": 0, "runs": 0, "wickets": 0, "highest_score": 0, "average": 0.0},
            "ODIs": {"matches": 0, "runs": 0, "wickets": 0, "highest_score": 0, "average": 0.0},
            "T20s": {"matches": 0, "runs": 0, "wickets": 0, "highest_score": 0, "average": 0.0},
            "IPL": {"matches": 0, "runs": 0, "wickets": 0, "highest_score": 0, "average": 0.0}, # Example for a major league
            # Add more relevant stats or leagues.  Added wickets here.
        }
        self.major_achievements = {
            "World Cups": 0,
            "Champions Trophies": 0,
            "T20 World Cups": 0,
            "Player of the Series Awards": 0,
            # Add more relevant achievements
        }

    def add_career_stats(self, format: str, matches: int = 0, runs: int = 0, wickets: int = 0, highest_score: int = 0, average: float = 0.0):
        if format in self.career_stats:
            self.career_stats[format]["matches"] += matches
            self.career_stats[format]["runs"] += runs
            self.career_stats[format]["wickets"] += wickets
            self.career_stats[format]["highest_score"] = max(self.career_stats[format]["highest_score"], highest_score)
            # Simple average update - might need more sophisticated handling for cumulative averages
            if self.career_stats[format]["matches"] > 0:
                self.career_stats[format]["average"] = (self.career_stats[format]["average"] * (self.career_stats[format]["matches"] - matches) + average * matches) / self.career_stats[format]["matches"]
            else:
                self.career_stats[format]["average"] = average
        else:
            logging.warning(f"Cricket format '{format}' not recognized for {self.name}.")

    def add_major_achievement(self, achievement: str, count: int = 1):
        if achievement in self.major_achievements:
            self.major_achievements[achievement] += count
        else:
            logging.warning(f"Major achievement '{achievement}' not recognized for {self.name}.")

    def display_career_stats(self):
        print(f"\nCareer Statistics for {self.name}:")
        for format, stats in self.career_stats.items():
            print(f"- {format}:")
            for key, value in stats.items():
                print(f"  - {key.replace('_', ' ').title()}: {value}")

    def display_major_achievements(self):
        print(f"\nMajor Achievements for {self.name}:")
        for trophy, count in self.major_achievements.items():
            if count > 0:
                print(f"- {trophy}: {count}")
        if all(count == 0 for count in self.major_achievements.values()):
            print(f"- No major achievements recorded yet.")

class SachinTendulkar(Cricketer):
    def __init__(self):
        super().__init__(name="Sachin Tendulkar", nationality="Indian", role="Batsman")
        self._load_achievements()
        self._load_stats()

    def _load_achievements(self):
        self.add_major_achievement("World Cups", 1)
        self.add_major_achievement("Player of the Series Awards", 7) # Across formats

    def _load_stats(self):
        self.add_career_stats("Tests", matches=200, runs=15921, wickets=46, highest_score=248, average=53.78)
        self.add_career_stats("ODIs", matches=463, runs=18426, wickets=154, highest_score=200, average=44.83)
        self.add_career_stats("T20s", matches=1, runs=10, highest_score=10, average=10.00)
        self.add_career_stats("IPL", matches=78, runs=2334, wickets=0, highest_score=100, average=34.84) # Added wickets=0
class ViratKohli(Cricketer):
    def __init__(self):
        super().__init__(name="Virat Kohli", nationality="Indian", role="Batsman")
        self._load_achievements()
        self._load_stats()

    def _load_achievements(self):
        self.add_major_achievement("World Cups", 1)
        self.add_major_achievement("Champions Trophies", 1)
        self.add_major_achievement("Player of the Series Awards", 11) # Across formats

    def _load_stats(self):
        self.add_career_stats("Tests", matches=113, runs=8848, wickets=0, highest_score=254, average=48.88)
        self.add_career_stats("ODIs", matches=292, runs=13848, wickets=4, highest_score=183, average=57.69)
        self.add_career_stats("T20s", matches=117, runs=4037, wickets=0, highest_score=122, average=51.75)
        self.add_career_stats("IPL", matches=243, runs=7624, wickets=0, highest_score=113, average=37.26)

class WasimAkram(Cricketer):
    def __init__(self):
        super().__init__(name="Wasim Akram", nationality="Pakistani", role="Bowler")
        self._load_achievements()
        self._load_stats()

    def _load_achievements(self):
        self.add_major_achievement("World Cups", 1)
        self.add_major_achievement("Player of the Series Awards", 5) # Across formats

    def _load_stats(self):
        self.add_career_stats("Tests", matches=104, runs=2898, wickets=414, highest_score=257, average=22.64)
        self.add_career_stats("ODIs", matches=356, runs=3717, wickets=502, highest_score=86, average=16.52)
        self.add_career_stats("T20s", matches=0) # No T20 internationals



# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IPLData:
    def __init__(self):
        self.base_url = "https://www.iplt20.com/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        self.teams = {
            "Chennai Super Kings": "CSK",
            "Delhi Capitals": "DC",
            "Gujarat Titans": "GT",
            "Kolkata Knight Riders": "KKR",
            "Lucknow Super Giants": "LSG",
            "Mumbai Indians": "MI",
            "Punjab Kings": "PBKS",
            "Rajasthan Royals": "RR",
            "Royal Challengers Bengaluru": "RCB",
            "Sunrisers Hyderabad": "SRH",
            # Add new teams as they join
        }
        self.cup_winners = { # Corrected dictionary structure.
            2008: "Rajasthan Royals",
            2009: "Deccan Chargers",
            2010: "Chennai Super Kings",
            2011: "Chennai Super Kings",
            2012: "Kolkata Knight Riders",
            2013: "Mumbai Indians",
            2014: "Kolkata Knight Riders",
            2015: "Mumbai Indians",
            2016: "Sunrisers Hyderabad",
            2017: "Mumbai Indians",
            2018: "Chennai Super Kings",
            2019: "Mumbai Indians",
            2020: "Mumbai Indians",
            2021: "Chennai Super Kings",
            2022: "Gujarat Titans",
            2023: "Chennai Super Kings"
        }

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL '{url}': {e}")
            return None

    def get_latest_news(self, num_articles: int = 5) -> List[Dict[str, str]]:
        """Fetches the latest IPL news."""
        url = self.base_url + "news"
        html_content = self._fetch_html(url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []
        try:
            # Adjust the selector as needed.  This one is more generic.
            headlines = soup.find_all('div', class_='media-block__content')[:num_articles]
            for item in headlines:
                title_element = item.find('h3', class_='media-block__title')
                link_element = item.find('a')
                if title_element and link_element: # check if elements are not None
                    title = title_element.text.strip()
                    link = self.base_url + link_element['href'].lstrip('/')
                    articles.append({'title': title, 'link': link})
        except Exception as e:
            logging.error(f"Error parsing IPL news: {e}")
            return []
        return articles

    def get_cup_winners(self) -> Dict[int, str]:
        """Returns a dictionary of IPL cup winners."""
        return self.cup_winners

    def get_teams(self) -> Dict[str, str]:
        """Returns the list of IPL teams."""
        return self.teams

    def display_teams(self) -> None:
        """Displays the IPL teams."""
        print("\nIPL Teams:")
        for team, short_name in self.teams.items():
            print(f"- {team} ({short_name})")

    def display_cup_winners(self) -> None:
        """Displays the IPL cup winners."""
        print("\nIPL Cup Winners:")
        for year, winner in self.cup_winners.items():
            print(f"- {year}: {winner}")

    def get_team_info(self, team_name: str) -> Optional[Dict[str, str]]:
        """Retrieves information about a specific IPL team.
        Args:
            team_name: The full name or a case-insensitive substring of the team.
        Returns:
            A dictionary containing the team name and a short name, or None if not found.
        """
        for full_name, short_name in self.teams.items():
            if team_name.lower() in full_name.lower():
                return {"full_name": full_name, "short_name": short_name}
        logging.warning(f"Team '{team_name}' not found.")
        return None


class CareerAdvisor:
    """Provides career advice based on professional experience"""
    def __init__(self):
        self.experience = self._load_experience()
        self.leetcode_problems = self._load_leetcode()
        self.tech_skills = self._load_tech_skills()
        
    def _load_experience(self) -> List[Dict]:
        return [
            {
                "company": "Microsoft",
                "role": "Software Engineer",
                "duration": "05/2018 - 04/2022",
                "achievements": [
                    "Led design of enterprise microservices driving $35.3B revenue",
                    "Developed license management systems handling millions of requests",
                    "Implemented telemetry and monitoring systems"
                ],
                "tech": ["C#", ".NET", "Azure", "Cosmos DB"]
            },
            {
                "company": "Amazon",
                "role": "Software Development Engineer",
                "duration": "04/2017 - 04/2018",
                "achievements": [
                    "Built Prime's Content Experiment Platforms",
                    "Automated marketing experiment systems",
                    "Integrated Selenium for UX testing"
                ],
                "tech": ["Java", "React", "AWS", "DynamoDB"]
            },
            {
                "company": "eBay Korea",
                "role": "Software Engineer",
                "duration": "12/2014 - 03/2017",
                "achievements": [
                    "Developed fintech apps processing $1B/month",
                    "Created executive dashboard for board decisions",
                    "Integrated Alipay increasing revenue by 23%"
                ],
                "tech": [".NET", "MSSQL", "Node.js", "React"]
            }
        ]
    
    def _load_leetcode(self) -> List[Dict]:
        return [
            {"id": 1, "title": "Two Sum", "difficulty": "Easy", "tags": ["Array", "Hash Table"]},
            {"id": 15, "title": "3Sum", "difficulty": "Medium", "tags": ["Array", "Two Pointers"]},
            {"id": 212, "title": "Word Search II", "difficulty": "Hard", "tags": ["Trie", "Backtracking"]}
        ]
    
    def _load_tech_skills(self) -> Dict[str, List[str]]:
        return {
            "Languages": ["C#", "Java", "JavaScript", "Python", "C++"],
            "Cloud": ["Azure", "AWS", "Serverless"],
            "Web": ["React", "Angular", "Node.js"],
            "Databases": ["Cosmos DB", "DynamoDB", "MSSQL"],
            "DevOps": ["CI/CD", "Monitoring", "Telemetry"]
        }
    
    def get_company_experience(self, company: str) -> Optional[Dict]:
        """Get details about experience at a specific company"""
        for exp in self.experience:
            if company.lower() in exp["company"].lower():
                return exp
        return None
    
    def recommend_leetcode(self, difficulty: str = None) -> Dict:
        """Recommend a LeetCode problem"""
        if difficulty:
            problems = [p for p in self.leetcode_problems if p["difficulty"].lower() == difficulty.lower()]
            return random.choice(problems) if problems else None
        return random.choice(self.leetcode_problems)
    
    def get_tech_stack(self) -> Dict[str, List[str]]:
        """Get categorized technical skills"""
        return self.tech_skills
    
    def generate_interview_question(self, topic: str) -> str:
        """Generate an interview question based on experience"""
        topics = {
            "system design": "How would you design a license management system handling millions of requests?",
            "behavioral": "Tell me about a time you had to work with multiple teams on a complex project",
            "coding": "Implement a concurrent request processor with rate limiting",
            "cloud": "How would you architect a scalable e-commerce platform on Azure?"
        }
        return topics.get(topic.lower(), "Explain the CAP theorem with examples")

class ContentCreator:
    """Handles YouTube content creation features"""
    def __init__(self):
        self.channels = {
            "PIRATE KING": {
                "niche": "Software Engineering & Entertainment",
                "content_types": ["SWE Skits", "Tech Life", "Coding Tutorials", "Career Advice"],
                "stats": {
                    "subscribers": "100K+",
                    "videos": "150+",
                    "start_date": "07/2021"
                }
            }
        }
        self.video_ideas = [
            "Day in the Life at Microsoft",
            "FAANG Interview Preparation Guide",
            "From Junior to Senior Engineer Journey",
            "Tech Salary Negotiation Tips"
        ]
    
    def get_channel_info(self) -> Dict:
        """Get information about YouTube channel"""
        return self.channels["PIRATE KING"]
    
    def generate_video_idea(self) -> str:
        """Generate a video idea"""
        return random.choice(self.video_ideas)
    
    def create_script_outline(self, topic: str) -> List[str]:
        """Create a basic script outline for a video"""
        outlines = {
            "interview preparation": [
                "Introduction to the company",
                "Interview process overview",
                "Technical preparation tips",
                "Behavioral questions advice",
                "Salary negotiation strategies",
                "Q&A from viewers"
            ],
            "coding tutorial": [
                "Problem statement",
                "Brute force approach",
                "Optimization process",
                "Final solution walkthrough",
                "Time complexity analysis",
                "Real-world applications"
            ]
        }
        return outlines.get(topic.lower(), ["Introduction", "Main Content", "Conclusion"])


class JurassicWorld:
    """Jurassic Park/World related information and utilities"""
    def __init__(self):
        self.dinosaurs = self._load_dinosaurs()
        self.movies = self._load_movies()
        self.parks = self._load_parks()

    def _load_dinosaurs(self) -> List[Dict]:
        return [
            {"name": "Tyrannosaurus Rex", "era": "Cretaceous", "diet": "Carnivore", "size": "40ft"},
            {"name": "Velociraptor", "era": "Cretaceous", "diet": "Carnivore", "size": "6ft"},
            {"name": "Brachiosaurus", "era": "Jurassic", "diet": "Herbivore", "size": "85ft"},
            {"name": "Triceratops", "era": "Cretaceous", "diet": "Herbivore", "size": "30ft"},
            {"name": "Mosasaurus", "era": "Cretaceous", "diet": "Carnivore", "size": "60ft"}
        ]

    def _load_movies(self) -> List[Dict]:
        return [
            {"title": "Jurassic Park", "year": 1993, "director": "Steven Spielberg"},
            {"title": "The Lost World: Jurassic Park", "year": 1997, "director": "Steven Spielberg"},
            {"title": "Jurassic Park III", "year": 2001, "director": "Joe Johnston"},
            {"title": "Jurassic World", "year": 2015, "director": "Colin Trevorrow"},
            {"title": "Jurassic World: Fallen Kingdom", "year": 2018, "director": "J.A. Bayona"},
            {"title": "Jurassic World Dominion", "year": 2022, "director": "Colin Trevorrow"}
        ]

    def _load_parks(self) -> List[Dict]:
        return [
            {"name": "Jurassic Park", "location": "Isla Nublar", "status": "Destroyed"},
            {"name": "Jurassic World", "location": "Isla Nublar", "status": "Destroyed"},
            {"name": "Lockwood Manor", "location": "California", "status": "Closed"},
            {"name": "Biosyn Sanctuary", "location": "Dolomites", "status": "Operational"}
        ]

    def get_dinosaur_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific dinosaur"""
        for dino in self.dinosaurs:
            if name.lower() in dino["name"].lower():
                return dino
        return None

    def get_random_dinosaur(self) -> Dict:
        """Get a random dinosaur fact"""
        return random.choice(self.dinosaurs)

    def get_movie_info(self, title: str) -> Optional[Dict]:
        """Get information about a specific movie"""
        for movie in self.movies:
            if title.lower() in movie["title"].lower():
                return movie
        return None

    def get_park_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific park"""
        for park in self.parks:
            if name.lower() in park["name"].lower():
                return park
        return None


class VoiceAssistantConfig:
    def __init__(self, config_file: str = "assistant_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file or create default"""
        default_config = {
            "name": "Ultron",
            "voice": "male",
            "volume": 70,
            "hotwords": ["hey assistant", "computer"]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**default_config, **loaded_config}
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def change_name(self, new_name: str) -> bool:
        """Change the assistant's name"""
        if not new_name.strip():
            return False
            
        self.config["name"] = new_name.strip()
        return self.save_config()
    
    def get_name(self) -> str:
        """Get current assistant name"""
        return self.config["name"]
    
    def get_all_config(self) -> Dict:
        """Return complete configuration"""
        return self.config
    
    def update_config(self, new_config: Dict) -> bool:
        """Update multiple settings at once"""
        self.config.update(new_config)
        return self.save_config()


class UltronAI:
    def __init__(self):
        self.Alarm = AlarmManager()
        self.Calendar = CalendarManager()
        self.Calculator = MathCalculator()
        self.ChromeSearcher = GoogleChromeSearcher()  # Add this line
        self.Config = VoiceAssistantConfig()  # Add this line
        self.ContentCreator = ContentCreator()  # Add this line
        self.CreatorInfo = CreatorInfo()
        self.CricketData = CricketData()
        self.DC = DCSeries()
        self.DateTime = DateTimeManager()
        self.Email = EmailManager()
        self.Entertainment = EntertainmentManager()
        self.EntertainmentNews = EntertainmentNews()
        self.FileSearch = FileSearch()
        self.FileManager = FileManager()
        self.FileSearcher = VSCodeFileSearcher()
        self.FootballData = FootballData()  # Initialize the FootballData class
        self.FootballWorldCupData = FootballWorldCupData()  # Initialize the FootballWorldCupData class
        self.GamingNews = GamingNews()  # Initialize the GamingNews class
        self.GamingNewsManager = GamingNewsManager()
        self.HealthMonitor = HealthMonitor()
        self.IPLData = IPLData()  # Add this line
        self.JarvisMeet = JarvisMeet(api_key="your_google_api_key")
        self.JokeTeller = JokeTeller()
        self.Jurassic = JurassicWorld()
        self.Learning = LearningModule()
        self.LearningAdvanceModule = LearningAdvanceModule()
        self.MathCalculator = MathCalculatorAdvance()  # Add this line
        self.Marvel = MarvelSeries()
        self.MediaPlayer = MediaPlayer()
        self.Messi = LionelMessi()
        self.Config = VoiceAssistantConfig()  # Add this line
        self.Name = self.Config.get_name()    # Set the name from config
        self.MusicPlayer = MusicPlayer()
        self.MusicPlayer = WebMusicPlayer()  # Add this line
        self.Name = self.Config.get_name()  # Set the name from config
        self.Name = "Ultron"
        self.Neymar = NeymarJr()
        self.NewsReader = NewsReader()
        self.Password_manager = None  # Will be initialized after master password is set
        self.Password_manager_file = "passwords.enc"
        self.Ronaldo = CristianoRonaldo()
        self.Sachin = SachinTendulkar()
        self.ScreenCapture = ScreenCapture()
        self.SecurityManager = SecurityManager()
        self.Settings = self.LoadSettings()
        self.SmartHome = SmartHomeControl()
        self.Spotify = SpotifyMusic()
        self.SystemControl = SystemControl()  # Add this line
        self.Voice = VoiceAssistant()
        self.VoiceWriter = VoiceWriter()
        self.Virat = ViratKohli()
        self.Wasim = WasimAkram()
        self.WebServices = WebServices()
        self.World_cup_data = FootballWorldCupData()
        self.YouTube = YouTubeMusic()
        self.YouTubePlayer = YouTubeMusicPlayer()  # Add this line
        self.angle_mode = 'degrees'
        self.Name = "Ultron"
        self.Logo = """
        ██╗   ██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗
        ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
        ██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║
        ██║   ██║██║     ██║   ██╔══██╗██║   ████║╚██╗██║
        ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║
         ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
        """



    def LoadSettings(self) -> Dict:
        defaultSettings = {
            'voice': 'male',
            'volume': 50,
            'language': 'en-US'
        }
        
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    return {**defaultSettings, **json.load(f)}
            return defaultSettings
        except Exception as e:
            logging.error(f"Settings load error: {e}")
            return defaultSettings


    def Greet(self) -> None:
        hour = datetime.now().hour
        if 4 <= hour < 12:
            greeting = "Good morning!"
        elif 12 <= hour < 16:
            greeting = "Good afternoon!"
        elif 16 <= hour < 24:
            greeting = "Good evening!"
        else:
            greeting = "Good night!"
        
        self.Voice.Speak(f"{greeting} I am {self.Name}. How may I assist you today?")
        print(self.Logo)

    def ProcessCommand(self, command: str) -> bool:
        if not command:
            return True
        try:     
            command = command.lower()
            
            # System commands
            if "exit" in command or "quit" in command:
                self.Voice.Speak("Goodbye! Have a great day.")
                return False
             # Jurassic commands
            elif any(cmd in command for cmd in ["jurassic", "dinosaur"]):
                self.HandleJurassicCommand(command)
            elif any(cmd in command for cmd in ["meeting", "schedule", "mute", "unmute", "record"]):
                self.HandleMeetingCommand(command)
            elif any(cmd in command for cmd in ["change your name", "what's your name", "reset your name"]):
                self.ConfigCommand(command)
            elif any(cmd in command for cmd in ["movie", "joke", "tv show", "entertainment"]):
                return self.HandleEntertainmentCommand(command)
            elif "gaming news" in command:
                return self.HandleGamingNewsCommand(command)
            # Media commands
            elif "play" in command:
                self.HandleMediaCommand(command)
             # Password manager commands
            elif any(cmd in command for cmd in ["password", "passwords", "credentials", "login"]):
                    self.HandlePasswordCommand(command)
            elif any(cmd in command for cmd in ["calculate", "solve", "integrate", "+", "-", "*", "/", "=", "statistics"]):
                self.HandleMathCommand(command)

            elif any(cmd in command for cmd in ["file", "directory", "folder", "create", "delete", "search", "list", "read", "bookmark", "cd", "ls"]):
                self.HandleFileCommand(command)
            
            

            elif any(phrase in command for phrase in ["latest entertainment news", "random entertainment news", "search entertainment news for"]):
                self.HandleEntertainmentNewsCommand(command)
            elif any(phrase in command for phrase in ["latest football news", "random football news", "ronaldo achievements", "messi achievements", "neymar achievements", "tell me about ronaldo", "tell me about messi", "tell me about neymar"]):
                self.HandleFootballCommand(command)
            elif any(phrase in command for phrase in ["latest cricket news", "random cricket news", "sachin tendulkar stats", "virat kohli stats", "wasim akram stats", "tell me about sachin tendulkar", "tell me about virat kohli", "tell me about wasim akram"]):
                self.HandleCricketCommand(command)

            # Web services
            elif "search" in command:
                self.HandleSearchCommand(command)
                
            # File operations
            elif "search files" in command:
                self.HandleFileCommand(command)
                
            # Calendar
            elif ("calendar" in command or "events" in command) and self.Calendar.CalendarEnabled:
                self.HandleCalendarCommand(command)
                
            # Email
            elif "send email" in command:
                self.HandleEmailCommand()
                
            # News
            elif "news" in command:
                self.HandleNewsCommand(command)
                
            # System health
            elif "system health" in command:
                self.HandleHealthCommand()
                
            # Learning
            elif "learn" in command or "resources" in command:
                self.HandleLearningCommand(command)
            # Learning
            elif "learning" in command or "resource" in command:
                self.HandleLearningAdvanceCommand(command)
                
            # Tasks
            elif "add task" in command:
                self.HandleTaskCommand(command)
            
            elif any(cmd in command for cmd in ["security", "malware", "scan", "block", "vulnerability"]):
                self.HandleSecurityCommand(command)
            # Jokes
            elif "tell me a joke" in command or "joke" in command:
                joke = self.JokeTeller.TellJoke()
                self.Voice.Speak(joke)
                print(joke)
                
            # Alarm
            elif "set alarm" in command:
                self.HandleAlarmCommand(command)
                
            # Date and time
            elif "what time is it" in command:
                self.Voice.Speak(f"The current time is {self.DateTime.GetCurrentTime()}")
                
            elif "what's today" in command or "what day is it" in command:
                self.Voice.Speak(f"Today is {self.DateTime.GetCurrentDate()}")
            
            elif any(cmd in command for cmd in ["turn on", "turn off", "light", "thermostat", "scene", "routine"]):
                self.HandleSmartHomeCommand(command)
            # Screen capture
            elif "take screenshot" in command:
                if self.ScreenCapture.TakeScreenshot():
                    self.Voice.Speak("Screenshot taken")
                else:
                    self.Voice.Speak("Failed to take screenshot")
            elif any(cmd in command for cmd in ["creator", "designer", "credits", "about", "version", "contact"]):
                self.HandleInfoCommand(command)     
            elif any(cmd in command for cmd in ["change your name", "what's your name", "reset your name"]):
                self.ConfigCommand(command)
            # Math
            elif "calculate" in command:
                self.HandleMathCommand(command)
            elif any(phrase in command for phrase in ["latest gaming news", "random gaming news", "search gaming news for"]):
                self.HandleGamingNewsCommand(command)
            elif any(phrase in command for phrase in ["world cup start", "latest world cup news", "random world cup news", "world cup hosts"]):
                self.HandleWorldCupCommand(command)
            elif "on chrome" in command:
                self.HandleChromeSearch(command)
            elif any(cmd in command for cmd in ["play", "pause", "resume", "next track", "volume"]):
                self.HandleMusicCommand(command)
            elif any(cmd in command for cmd in ["system", "shutdown", "restart", "task", "process"]):
                self.HandleSystemCommand(command)
              # Add this condition to the existing if-elif chain
            elif any(cmd in command for cmd in ["dc movie", "dc series", "dc character", "search dc"]):
                self.HandleDCCommand(command)
             # Add this condition for Marvel commands
            elif any(cmd in command for cmd in ["marvel movie", "marvel series", "marvel character", "search marvel"]):
                self.HandleMarvelCommand(command)
            elif any(cmd in command for cmd in ["play", "stop", "pause", "resume", "next", "volume", "music"]):
                self.HandleMusicCommand(command)
            elif any(cmd in command for cmd in ["search files", "find file", "recent files"]):
                self.HandleFileSearch(command)
            elif any(phrase in command for phrase in ["latest ipl news", "ipl cup winners", "ipl teams", "ipl team info"]): # Added IPL commands
                self.HandleIPLCommand(command)
            elif any(phrase in command for phrase in ["company experience", "recommend leetcode", "show tech stack", "generate interview question"]):
                self.HandleCareerCommand(command) # Add this line
            
            elif any(phrase in command for phrase in ["channel info", "generate video idea", "create script outline"]):
                self.HandleContentCommand(command) # Add this line

            # Default response
            else:
                self.Voice.Speak("I didn't understand that command. Please try again.")
                
            return True
            
        except Exception as e:
            logging.error(f"Command processing error: {e}")
            self.Voice.Speak("Sorry, I encountered an error. Please try again.")
            return True  # Continue after errors
    
    def ConfigCommand(self, command: str) -> None:
        """Handle configuration changes"""
        command = command.lower()
        
        try:
            if "change your name to" in command:
                new_name = command.split("change your name to")[-1].strip()
                if self.Config.change_name(new_name):
                    self.Name = new_name
                    self.Voice.Speak(f"Success! You can now call me {new_name}")
                else:
                    self.Voice.Speak("Sorry, I couldn't change my name")
            
            elif "what's your name" in command:
                self.Voice.Speak(f"My name is {self.Name}")
                
            elif "reset your name" in command:
                if self.Config.change_name("Ultron"):
                    self.Name = "Ultron"
                    self.Voice.Speak("My name has been reset to Ultron")
                else:
                    self.Voice.Speak("Failed to reset my name")
        
        except Exception as e:
            self.Voice.Speak("Configuration command failed")
            logging.error(f"Config command error: {e}")
    def HandleCareerCommand(self, command: str) -> None:
        """Handles career-related commands."""
        command = command.lower()
        
        try:
            if "company experience" in command:
                company_name = command.split("company experience at")[-1].strip()
                experience = self.career_advisor.get_company_experience(company_name)
                if experience:
                    self.Voice.Speak(f"Here's my experience at {experience['company']}:")
                    self.Voice.Speak(f"Role: {experience['role']}")
                    self.Voice.Speak(f"Duration: {experience['duration']}")
                    self.Voice.Speak("Achievements:")
                    for achievement in experience['achievements']:
                        self.Voice.Speak(f"- {achievement}")
                    self.Voice.Speak("Technologies Used:")
                    self.Voice.Speak(", ".join(experience['tech']))
                else:
                    self.Voice.Speak(f"Sorry, I don't have information about experience at {company_name}.")
            
            elif "recommend leetcode" in command:
                if "easy leetcode" in command:
                    difficulty = "easy"
                elif "medium leetcode" in command:
                    difficulty = "medium"
                elif "hard leetcode" in command:
                    difficulty = "hard"
                else:
                    difficulty = None
                
                problem = self.career_advisor.recommend_leetcode(difficulty)
                if problem:
                    self.Voice.Speak(f"I recommend this LeetCode problem:")
                    self.Voice.Speak(f"Title: {problem['title']}")
                    self.Voice.Speak(f"Difficulty: {problem['difficulty']}")
                    self.Voice.Speak(f"Tags: {', '.join(problem['tags'])}")
                else:
                    self.Voice.Speak(f"Sorry, I couldn't find a LeetCode problem with the specified difficulty.")

            elif "show tech stack" in command:
                tech_stack = self.career_advisor.get_tech_stack()
                self.Voice.Speak("Here's my technical skill set:")
                for category, skills in tech_stack.items():
                    self.Voice.Speak(f"{category}: {', '.join(skills)}")
            
            elif "generate interview question" in command:
                topic = command.split("generate interview question about")[-1].strip()
                question = self.career_advisor.generate_interview_question(topic)
                self.Voice.Speak(f"Here's an interview question for you: {question}")
            
            else:
                self.Voice.Speak("I can help you with career advice. Ask me about my company experience, LeetCode, tech skills, or interview questions.")
        except Exception as e:
            self.Voice.Speak("Failed to handle career command.")
            logging.error(f"Career command error: {e}")
    
    def HandleContentCommand(self, command: str) -> None:
        """Handles YouTube content creation commands."""
        command = command.lower()
        
        try:
            if "channel info" in command:
                channel_info = self.content_creator.get_channel_info()
                self.Voice.Speak("Here's information about my YouTube channel:")
                self.Voice.Speak(f"Channel Name: PIRATE KING")
                self.Voice.Speak(f"Niche: {channel_info['niche']}")
                self.Voice.Speak(f"Content Types: {', '.join(channel_info['content_types'])}")
                self.Voice.Speak(f"Subscribers: {channel_info['stats']['subscribers']}")
                self.Voice.Speak(f"Videos: {channel_info['stats']['videos']}")
                self.Voice.Speak(f"Start Date: {channel_info['stats']['start_date']}")
            
            elif "generate video idea" in command:
                video_idea = self.content_creator.generate_video_idea()
                self.Voice.Speak(f"Here's a video idea: {video_idea}")
            
            elif "create script outline" in command:
                topic = command.split("create script outline for")[-1].strip()
                outline = self.content_creator.create_script_outline(topic)
                self.Voice.Speak(f"Here's a script outline for {topic}:")
                for i, point in enumerate(outline):
                    self.Voice.Speak(f"{i+1}. {point}")
            
            else:
                self.Voice.Speak("I can help you with YouTube content creation. Ask me about channel info, video ideas, or script outlines.")
        except Exception as e:
            self.Voice.Speak("Failed to handle content command.")
            logging.error(f"Content command error: {e}")






    def HandleIPLCommand(self, command: str) -> None:
        """Handles IPL related commands."""
        command = command.lower()

        try:
            if "latest ipl news" in command:
                self.Voice.Speak("Fetching the latest IPL news...")
                news_articles = self.ipl_data.get_latest_news()
                if news_articles:
                    self.Voice.Speak("Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest IPL news.")

            elif "ipl cup winners" in command:
                self.Voice.Speak("Displaying IPL cup winners:")
                self.ipl_data.display_cup_winners()

            elif "ipl teams" in command:
                self.Voice.Speak("Displaying IPL teams:")
                self.ipl_data.display_teams()
            
            elif "ipl team info" in command:
                team_name = command.split("ipl team info about")[-1].strip() # Extract team name
                team_info = self.ipl_data.get_team_info(team_name)
                if team_info:
                    self.Voice.Speak(f"Here is the information about {team_info['full_name']}:")
                    self.Voice.Speak(f"Full Name: {team_info['full_name']}")
                    self.Voice.Speak(f"Short Name: {team_info['short_name']}")
                else:
                    self.Voice.Speak(f"Sorry, I could not find information about the team: {team_name}")

            # Add more IPL-related commands here

        except Exception as e:
            self.Voice.Speak("Failed to handle IPL command.")
            logging.error(f"IPL command error: {e}")



    def HandleCricketCommand(self, command: str) -> None:
        """Handles cricket related commands."""
        command = command.lower()

        try:
            if "latest cricket news" in command:
                source = "random"
                if "from cricbuzz" in command:
                    source = "cricbuzz"
                elif "from espncricinfo" in command:
                    source = "espncricinfo"

                self.Voice.Speak(f"Fetching the latest cricket news from {source.upper()}...")
                news_articles = self.cricket_data.get_latest_cricket_news(source=source)
                if news_articles:
                    self.Voice.Speak("Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest cricket news.")

            elif "random cricket news" in command:
                self.Voice.Speak("Fetching a random piece of cricket news...")
                news = self.cricket_data.get_random_cricket_news()
                if news:
                    self.Voice.Speak(f"Here's a random story: {news['title']}")
                    self.Voice.Speak(f"You can read more at: {news['link']}")
                else:
                    self.Voice.Speak("Could not retrieve a random cricket news story.")

            elif "sachin tendulkar stats" in command:
                self.Voice.Speak("Fetching Sachin Tendulkar's career stats...")
                self.sachin.display_career_stats()
                self.Voice.Speak("And his major achievements...")
                self.sachin.display_major_achievements()

            elif "virat kohli stats" in command:
                self.Voice.Speak("Fetching Virat Kohli's career stats...")
                self.virat.display_career_stats()
                self.Voice.Speak("And his major achievements...")
                self.virat.display_major_achievements()

            elif "wasim akram stats" in command:
                self.Voice.Speak("Fetching Wasim Akram's career stats...")
                self.wasim.display_career_stats()
                self.Voice.Speak("And his major achievements...")
                self.wasim.display_major_achievements()

            elif "tell me about" in command and "sachin tendulkar" in command:
                self.Voice.Speak("Sachin Tendulkar is widely regarded as one of the greatest batsmen in the history of cricket. He holds numerous records in both Test and One Day International cricket...") # Add more info
            elif "tell me about" in command and "virat kohli" in command:
                self.Voice.Speak("Virat Kohli is a modern-day great batsman from India, known for his aggressive style and consistent performance across all formats of the game...") # Add more info
            elif "tell me about" in command and "wasim akram" in command:
                self.Voice.Speak("Wasim Akram is considered one of the finest fast bowlers of all time. Known for his swing bowling and variations, he was a key player for Pakistan...") # Add more info

        except Exception as e:
            self.Voice.Speak("Failed to handle cricket command.")
            logging.error(f"Cricket command error: {e}")



    def HandleWorldCupCommand(self, command: str) -> None:
        """Handles FIFA World Cup related commands."""
        command = command.lower()

        try:
            if "world cup start" in command:
                start_info = self.world_cup_data.get_world_cup_start_info()
                self.Voice.Speak(f"The next FIFA World Cup is scheduled for {start_info['year']}.")
                self.Voice.Speak(f"The estimated start date is {start_info['start_date']}.")
                self.Voice.Speak(f"It will be hosted by {', '.join(start_info['hosts'])}.")

            elif "latest world cup news" in command:
                source = "random"
                if "from fifa" in command:
                    source = "fifa"
                elif "from goal" in command:
                    source = "goal_com"
                elif "from espn" in command:
                    source = "espn_fc"

                self.Voice.Speak(f"Fetching the latest World Cup news from {source.replace('_', '.').upper()}...")
                news_articles = self.world_cup_data.get_latest_world_cup_news(source=source)
                if news_articles:
                    self.Voice.Speak("Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest World Cup news.")

            elif "random world cup news" in command:
                self.Voice.Speak("Fetching a random piece of World Cup news...")
                news = self.world_cup_data.get_random_world_cup_news()
                if news:
                    self.Voice.Speak(f"Here's a random story: {news['title']}")
                    self.Voice.Speak(f"You can read more at: {news['link']}")
                else:
                    self.Voice.Speak("Could not retrieve a random World Cup news story.")

            elif "world cup hosts" in command:
                hosts = self.world_cup_data.get_world_cup_start_info()['hosts']
                self.Voice.Speak(f"The FIFA World Cup {self.world_cup_data.next_world_cup_year} will be hosted by {', '.join(hosts)}.")

        except Exception as e:
            self.Voice.Speak("Failed to handle World Cup command.")
            logging.error(f"World Cup command error: {e}")
    def HandleGamingNewsCommand(self, command: str) -> None:
        """Handles gaming news related commands."""
        command = command.lower()

        try:
            if "latest gaming news" in command:
                source = "random"
                if "from ign" in command:
                    source = "ign"
                elif "from gamespot" in command:
                    source = "gamespot"
                elif "from polygon" in command:
                    source = "polygon"

                self.Voice.Speak(f"Fetching the latest gaming news from {source.title()}...")
                news_articles = self.gaming_news.get_latest_gaming_news(source=source)
                if news_articles:
                    self.Voice.Speak("Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest gaming news.")

            elif "random gaming news" in command:
                self.Voice.Speak("Fetching a random piece of gaming news...")
                news = self.gaming_news.get_random_gaming_news()
                if news:
                    self.Voice.Speak(f"Here's a random story: {news['title']}")
                    self.Voice.Speak(f"You can read more at: {news['link']}")
                else:
                    self.Voice.Speak("Could not retrieve a random gaming news story.")

            elif "search gaming news for" in command:
                query = command.replace("search gaming news for", "").strip()
                self.Voice.Speak(f"Searching for '{query}' in gaming news...")
                # You would need to implement a search functionality, possibly using an API
                # or by scraping search results from news sites.
                self.Voice.Speak("Search functionality for specific gaming topics is not yet implemented.")

        except Exception as e:
            self.Voice.Speak("Failed to handle gaming news command.")
            logging.error(f"Gaming news command error: {e}")
        
    
    
    def HandleFootballCommand(self, command: str) -> None:
        """Handles football related commands."""
        command = command.lower()

        try:
            if "latest football news" in command:
                source = "random"
                if "from goal" in command:
                    source = "goal_com"
                elif "from espn" in command:
                    source = "espn_fc"

                self.Voice.Speak(f"Fetching the latest football news from {source.replace('_', '.').upper()}...")
                news_articles = self.football_data.get_latest_football_news(source=source)
                if news_articles:
                    self.Voice.Speak("Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest football news.")

            elif "random football news" in command:
                self.Voice.Speak("Fetching a random piece of football news...")
                news = self.football_data.get_random_football_news()
                if news:
                    self.Voice.Speak(f"Here's a random story: {news['title']}")
                    self.Voice.Speak(f"You can read more at: {news['link']}")
                else:
                    self.Voice.Speak("Could not retrieve a random football news story.")

            elif "ronaldo achievements" in command:
                self.Voice.Speak("Fetching Cristiano Ronaldo's achievements...")
                self.ronaldo.display_achievements()
                self.Voice.Speak(f"His signature move is: {self.ronaldo.signature_move()}")

            elif "messi achievements" in command:
                self.Voice.Speak("Fetching Lionel Messi's achievements...")
                self.messi.display_achievements()
                self.Voice.Speak(f"His signature move is: {self.messi.signature_move()}")

            elif "neymar achievements" in command:
                self.Voice.Speak("Fetching Neymar Junior's achievements...")
                self.neymar.display_achievements()
                self.Voice.Speak(f"His signature move is: {self.neymar.signature_move()}")

            elif "tell me about" in command and "ronaldo" in command:
                self.Voice.Speak(f"Cristiano Ronaldo, a Portuguese professional footballer who plays as a forward and captains the Portugal national team. He currently plays for Saudi Arabian club Al Nassr. Widely regarded as one of the greatest players of all time...") # Add more info
            elif "tell me about" in command and "messi" in command:
                self.Voice.Speak(f"Lionel Messi, an Argentine professional footballer who plays as a forward and captains both Spanish club Inter Miami CF and the Argentina national team. Often considered the best player in the world and widely regarded as one of the greatest players of all time...") # Add more info
            elif "tell me about" in command and "neymar" in command:
                self.Voice.Speak(f"Neymar da Silva Santos Júnior, commonly known as Neymar, is a Brazilian professional footballer who plays as a forward for Saudi Pro League club Al Hilal and the Brazil national team. He is widely regarded as one of the most talented and exciting players in the world...") # Add more info

        except Exception as e:
            self.Voice.Speak("Failed to handle football command.")
            logging.error(f"Football command error: {e}")





    def HandleEntertainmentNewsCommand(self, command: str) -> None:
        """Handles entertainment news related commands."""
        command = command.lower()

        try:
            if "latest entertainment news" in command:
                source = "random"
                if "from variety" in command:
                    source = "variety"
                elif "from hollywood reporter" in command:
                    source = "hollywood_reporter"
                elif "from entertainment weekly" in command:
                    source = "entertainment_weekly"

                self.Voice.Speak(f"Fetching the latest entertainment news from {source.replace('_', ' ').title()}...")
                news_articles = self.entertainment_news.scrape_latest_news(source=source)
                if news_articles:
                    self.Voice.Speak(f"Here are a few headlines:")
                    for i, article in enumerate(news_articles[:3]):  # Limit to 3 headlines
                        self.Voice.Speak(f"{i+1}. {article['title']}")
                        self.Voice.Speak(f"Link: {article['link']}")
                else:
                    self.Voice.Speak("Could not retrieve the latest entertainment news.")

            elif "random entertainment news" in command:
                self.Voice.Speak("Fetching a random piece of entertainment news...")
                news = self.entertainment_news.get_random_news()
                if news:
                    self.Voice.Speak(f"Here's a random story: {news['title']}")
                    self.Voice.Speak(f"You can read more at: {news['link']}")
                else:
                    self.Voice.Speak("Could not retrieve a random entertainment news story.")

            elif "search entertainment news for" in command:
                query = command.replace("search entertainment news for", "").strip()
                self.Voice.Speak(f"Searching for '{query}' in entertainment news...")
                # You would need to implement a search functionality, possibly using an API
                # or by scraping search results from news sites.
                self.Voice.Speak("Search functionality for specific topics is not yet implemented.")

        except Exception as e:
            self.Voice.Speak("Failed to handle entertainment news command.")
            logging.error(f"Entertainment news command error: {e}")
        
    
    
    
    
    def HandleDCCommand(self, command: str) -> None:
        """Handle DC-related commands"""
        command = command.lower()
        
        try:
            if "dc movie" in command:
                movie = self.DC.get_movie_recommendation()
                self.Voice.Speak(f"I recommend {movie['title']} from {movie['year']}, directed by {movie['director']} with rating {movie['rating']}/10")
                
            elif "dc tv show" in command or "dc series" in command:
                show = self.DC.get_tv_show_recommendation()
                self.Voice.Speak(f"I recommend {show['title']} with {show['seasons']} seasons from {show['years']}, rated {show['rating']}/10")
                
            elif "dc character" in command:
                char_name = command.replace("dc character", "").strip()
                if char_name:
                    character = self.DC.get_character_info(char_name)
                    if character:
                        powers = ", ".join(character["powers"])
                        self.Voice.Speak(f"{character['name']} first appeared in {character['first_appearance']}. Powers include: {powers}")
                    else:
                        self.Voice.Speak("Character not found")
                else:
                    self.Voice.Speak("Please specify a character name")
                    
            elif "search dc" in command:
                query = command.replace("search dc", "").strip()
                if query:
                    results = self.DC.search_content(query)
                    if results:
                        for result in results[:3]:  # Limit to top 3 results
                            if result["type"] == "movie":
                                self.Voice.Speak(f"Movie: {result['data']['title']} ({result['data']['year']})")
                            elif result["type"] == "tv_show":
                                self.Voice.Speak(f"TV Show: {result['data']['title']} ({result['data']['years']})")
                            elif result["type"] == "character":
                                self.Voice.Speak(f"Character: {result['data']['name']}")
                    else:
                        self.Voice.Speak("No DC content found matching your query")
                else:
                    self.Voice.Speak("Please specify a search query")
                    
        except Exception as e:
            self.Voice.Speak("DC command failed")
            logging.error(f"DC command error: {e}")


    def HandleMarvelCommand(self, command: str) -> None:
        """Handle Marvel-related commands"""
        command = command.lower()

        try:
            if "marvel movie" in command:
                movie = self.Marvel.get_movie_recommendation()
                self.Voice.Speak(f"I recommend {movie['title']} from {movie['year']}, directed by {movie['director']} with rating {movie['rating']}/10")

            elif "marvel tv show" in command or "marvel series" in command:
                show = self.Marvel.get_tv_show_recommendation()
                self.Voice.Speak(f"I recommend {show['title']} with {show['seasons']} seasons from {show['years']}, rated {show['rating']}/10")

            elif "marvel character" in command:
                char_name = command.replace("marvel character", "").strip()
                if char_name:
                    character = self.Marvel.get_character_info(char_name)
                    if character:
                        powers = ", ".join(character["powers"])
                        self.Voice.Speak(f"{character['name']} first appeared in {character['first_appearance']}. Powers include: {powers}")
                    else:
                        self.Voice.Speak("Character not found")
                else:
                    self.Voice.Speak("Please specify a character name")

            elif "search marvel" in command:
                query = command.replace("search marvel", "").strip()
                if query:
                    results = self.Marvel.search_content(query)
                    if results:
                        for result in results[:3]:  # Limit to top 3 results
                            if result["type"] == "movie":
                                self.Voice.Speak(f"Movie: {result['data']['title']} ({result['data']['year']})")
                            elif result["type"] == "tv_show":
                                self.Voice.Speak(f"TV Show: {result['data']['title']} ({result['data']['years']})")
                            elif result["type"] == "character":
                                self.Voice.Speak(f"Character: {result['data']['name']}")
                    else:
                        self.Voice.Speak("No Marvel content found matching your query")
                else:
                    self.Voice.Speak("Please specify a search query")

        except Exception as e:
            self.Voice.Speak("Marvel command failed")
            logging.error(f"Marvel command error: {e}")

    def HandlePasswordCommand(self, command: str) -> None:
        """Handle password manager commands with better error handling"""
        command = command.lower()
        
        try:
            # Initialize password manager if not already done
            if self.password_manager is None and "password manager" in command:
                try:
                    self.Voice.Speak("Please enter your master password")
                    try:
                        master_password = getpass.getpass("Master password: ")  # For secure input
                        if not master_password:
                            self.Voice.Speak("Master password cannot be empty")
                            return
                            
                        self.password_manager = PasswordManager(master_password, self.password_manager_file)
                        if self.password_manager.load():
                            self.Voice.Speak("Password manager unlocked and loaded")
                        else:
                            self.Voice.Speak("New password database created")
                    except KeyboardInterrupt:
                        self.Voice.Speak("Password input cancelled")
                        return
                    except Exception as e:
                        self.Voice.Speak("Failed to initialize password manager")
                        logging.error(f"Password manager init error: {e}")
                        return
                except Exception as e:
                    self.Voice.Speak("Error setting up password manager")
                    logging.error(f"Password manager setup error: {e}")
                    return
                return
            
            if self.password_manager is None:
                self.Voice.Speak("Please initialize the password manager first by saying 'password manager'")
                return
            
        #  # Add password
            if "add password" in command:
                self.Voice.Speak("Please provide the service name:")
                service = self.Voice.TakeCommand()
                self.Voice.Speak("Please provide the username:")
                username = self.Voice.TakeCommand()
                self.Voice.Speak("Would you like me to generate a secure password?")
                generate = self.Voice.TakeCommand()
                
                if generate and "yes" in generate.lower():
                    password = self.password_manager.generate_password()
                    self.Voice.Speak(f"Generated password: {password}")
                else:
                    self.Voice.Speak("Please provide the password:")
                    password = self.Voice.TakeCommand()
                    
                entry = self.password_manager.add_entry(
                    service=service,
                    username=username,
                    password=password
                )
                self.password_manager.save()
                self.Voice.Speak(f"Password for {service} added successfully")
                
            # Get password
            elif "get password" in command or "show password" in command:
                self.Voice.Speak("For which service?")
                service = self.Voice.TakeCommand()
                try:
                    entry = self.password_manager.get_entry(service)
                    self.Voice.Speak(f"Credentials for {service}: Username {entry.username}, Password {entry.password}")
                except ValueError:
                    self.Voice.Speak(f"No password found for {service}")
                    
            # Generate password
            elif "generate password" in command:
                length = None
                if "length" in command:
                    try:
                        length = int(''.join(filter(str.isdigit, command)))
                    except:
                        pass
                        
                password = self.password_manager.generate_password(length=length)
                self.Voice.Speak(f"Generated password: {password}")
                
            # Password strength report
            elif "password report" in command:
                report = self.password_manager.get_password_strength_report()
                if report:
                    self.Voice.Speak(f"Password strength report: Average strength is {report['average_strength']:.1f}%")
                    self.Voice.Speak(f"Found {report['weak_passwords_count']} weak passwords")
                else:
                    self.Voice.Speak("No passwords in database")
                    
            # Change master password
            elif "change master password" in command:
                self.Voice.Speak("Please enter your current master password")
                current = getpass.getpass("Current master password: ")
                if current.encode() != self.password_manager.master_password:
                    self.Voice.Speak("Incorrect master password")
                    return
                    
                self.Voice.Speak("Please enter new master password")
                new_password = getpass.getpass("New master password: ")
                self.Voice.Speak("Please confirm new master password")
                confirm = getpass.getpass("Confirm: ")
                
                if new_password == confirm:
                    self.password_manager.change_master_password(new_password)
                    self.password_manager.save()
                    self.Voice.Speak("Master password changed successfully")
                else:
                    self.Voice.Speak("Passwords did not match")
                    
            # Save passwords
            elif "save passwords" in command:
                self.password_manager.save()
                self.Voice.Speak("Passwords saved securely")
        
        except Exception as e:
            self.Voice.Speak("Password command failed")
            logging.error(f"Password command error: {e}")


    def HandleEntertainmentCommand(self, command: str) -> bool:
        """Handle entertainment-related commands"""
        command = command.lower()
        
        try:
            # Movie recommendations
            if "movie recommendation" in command:
                genre = None
                if "scientific move" in command:
                    genre = "Sci-Fi"
                elif "drama" in command:
                    genre = "Drama"
                    
                movie = self.Entertainment.get_movie_recommendation(genre)
                if movie:
                    self.Voice.Speak(f"I recommend {movie.title}, a {movie.genre} movie from {movie.year} with rating {movie.rating}/10 platform on IMDb")
                else:
                    self.Voice.Speak("No movies found in that genre")
                return True
                
            # Jokes
            elif "tell me a joke" in command or "joke" in command:
                joke = self.Entertainment.get_random_joke()
                self.Voice.Speak(joke.content)
                return True
                
            # TV shows
            elif "tv show recommendation" in command:
                show = self.Entertainment.get_tv_show_recommendation()
                self.Voice.Speak(f"How about {show['title']}? It's a {show['genre']} series with {show['seasons']} seasons")
                return True
                
        except Exception as e:
            self.Voice.Speak("Entertainment command failed")
            logging.error(f"Entertainment error: {e}")
            return False

    def HandleGamingNewsCommand(self, command: str) -> bool:
        """Handle gaming news commands"""
        command = command.lower()
        
        try:
            if "gaming news" in command:
                source = None
                if "from ign" in command:
                    source = 'IGN'
                elif "from polygon" in command:
                    source = 'Polygon'
                    
                news_items = self.GamingNews.fetch_latest_news(source)
                if not news_items:
                    self.Voice.Speak("Could not fetch gaming news")
                    return True
                    
                self.Voice.Speak(f"Latest gaming news from {news_items[0].source}:")
                for item in news_items[:2]:  # Read top 2 headlines
                    self.Voice.Speak(item.title)
                return True
                
        except Exception as e:
            self.Voice.Speak("Failed to get gaming news")
            logging.error(f"Gaming news error: {e}")
            return False


    def HandleInfoCommand(self, command: str) -> None:
        """Handle requests for creator/program information"""
        command = command.lower()
        
        if "creator" in command or "designer" in command:
            info = self.CreatorInfo.GetCreatorInfo()
            response = (f"This program {info['Program']} version {info['Version']} "
                    f"was created by {info['Creator']} and designed by {info['Designer']}")
            self.Voice.Speak(response)
        
        elif "credits" in command or "about" in command:
            info = self.CreatorInfo.GetDetailedInfo()
            response = (f"Detailed credits: {info['ProgramName']} version {info['Version']}, "
                    f"created in {info['CreationYear']}. Lead developer: {info['LeadCreator']}, "
                    f"Lead designer: {info['LeadDesigner']}. License: {info['License']}")
            self.Voice.Speak(response)
        
        elif "version" in command:
            self.Voice.Speak(f"Current version is {self.CreatorInfo.Version}")
        
        elif "contact" in command:
            self.Voice.Speak(f"For support, please contact {self.CreatorInfo.ContactEmail}")
    # Helper methods for simple responses
    def _handle_exit(self):
        self.Voice.Speak("Goodbye! Have a great day.")
        return False

    def _tell_joke(self):
        joke = self.JokeTeller.TellJoke()
        self.Voice.Speak(joke)
        print(joke)

    def _tell_time(self):
        self.Voice.Speak(f"The current time is {self.DateTime.GetCurrentTime()}")

    def _tell_date(self):
        self.Voice.Speak(f"Today is {self.DateTime.GetCurrentDate()}")

    def _take_screenshot(self):
        if self.ScreenCapture.TakeScreenshot():
            self.Voice.Speak("Screenshot taken")
        else:
            self.Voice.Speak("Failed to take screenshot")


    def HandleMediaCommand(self, command: str) -> None:
        """Handle all media playback commands"""
        try:
            # YouTube Music
            if "on youtube" in command:
                song = command.replace("play", "").replace("on youtube", "").strip()
                if self.YouTube.Play(song):
                    self.Voice.Speak(f"Playing {song} on YouTube")
                else:
                    self.Voice.Speak("Could not find that song on YouTube")

            # Spotify Music
            elif "on spotify" in command:
                song = command.replace("play", "").replace("on spotify", "").strip()
                if self.Spotify.Play(song):
                    self.Voice.Speak(f"Playing {song} on Spotify")
                else:
                    self.Voice.Speak("Could not connect to Spotify")

            # Local files
            elif "local file" in command or "my music" in command:
                song = command.replace("play", "").replace("local file", "").replace("my music", "").strip()
                if os.path.exists(song):
                    if self.MusicPlayer.PlayLocal(song):
                        self.Voice.Speak(f"Playing local file {os.path.basename(song)}")
                else:
                    self.Voice.Speak("File not found")

            # Media controls
            elif "pause" in command:
                self.MediaPlayer.ControlMedia("pause")
                self.Voice.Speak("Playback paused")
                
            elif "resume" in command or "continue" in command:
                self.MediaPlayer.ControlMedia("resume")
                self.Voice.Speak("Resuming playback")
                
            elif "stop" in command:
                self.MediaPlayer.ControlMedia("stop")
                self.Voice.Speak("Playback stopped")
                
            elif "next" in command:
                self.MediaPlayer.ControlMedia("next")
                self.Voice.Speak("Playing next track")
                
            elif "previous" in command:
                self.MediaPlayer.ControlMedia("previous")
                self.Voice.Speak("Playing previous track")
                
            elif "volume" in command:
                try:
                    vol = int(re.search(r'\d+', command).group())
                    if 0 <= vol <= 100:
                        self.MediaPlayer.SetVolume(vol)
                        self.Voice.Speak(f"Volume set to {vol} percent")
                    else:
                        self.Voice.Speak("Volume must be between 0 and 100")
                except:
                    self.Voice.Speak("Please specify a volume level between 0 and 100")

        except Exception as e:
            logging.error(f"Media command error: {e}")
            self.Voice.Speak("There was an error processing your media command")



    def HandleMusicCommand(self, command: str) -> None:
        """Handle music playback commands"""
        command = command.lower()
        
        try:
            if "play" in command and "on youtube" in command:
                query = command.replace("play", "").replace("on youtube", "").strip()
                song = self.MusicPlayer.search_youtube(query)
                if song:
                    if self.MusicPlayer.play(song):
                        self.Voice.Speak(f"Playing {song.title} by {song.artist}")
                    else:
                        self.Voice.Speak("Failed to start playback")
                else:
                    self.Voice.Speak("Song not found")
                    
            elif "pause music" in command:
                self.MusicPlayer.pause()
                self.Voice.Speak("Music paused")
                
            elif "resume music" in command:
                if self.MusicPlayer.play():
                    self.Voice.Speak("Resuming playback")
                    
            elif "next track" in command:
                if self.MusicPlayer.next_track():
                    song = self.MusicPlayer.get_current_song()
                    self.Voice.Speak(f"Playing {song.title}")
                else:
                    self.Voice.Speak("No next track available")
                    
            elif "previous track" in command:
                if self.MusicPlayer.previous_track():
                    song = self.MusicPlayer.get_current_song()
                    self.Voice.Speak(f"Playing {song.title}")
                else:
                    self.Voice.Speak("No previous track available")
                    
            elif "volume" in command:
                if "up" in command:
                    new_vol = min(100, self.MusicPlayer.volume + 20)
                    self.MusicPlayer.set_volume(new_vol)
                    self.Voice.Speak(f"Volume set to {new_vol}")
                elif "down" in command:
                    new_vol = max(0, self.MusicPlayer.volume - 20)
                    self.MusicPlayer.set_volume(new_vol)
                    self.Voice.Speak(f"Volume set to {new_vol}")
                elif any(char.isdigit() for char in command):
                    vol = int(''.join(filter(str.isdigit, command)))
                    self.MusicPlayer.set_volume(vol)
                    self.Voice.Speak(f"Volume set to {vol}")
                    
            elif "shuffle" in command:
                self.MusicPlayer.shuffle_queue()
                self.Voice.Speak("Queue shuffled")
                
        except Exception as e:
            self.Voice.Speak("Music command failed")
            logging.error(f"Music command error: {e}")

    def ConfigCommand(self, command: str) -> None:
        """Handle configuration changes"""
        command = command.lower()
        
        try:
            if "change your name to" in command:
                new_name = command.split("change your name to")[-1].strip()
                if self.Config.change_name(new_name):
                    self.Name = new_name
                    self.Voice.Speak(f"Success! You can now call me {new_name}")
                else:
                    self.Voice.Speak("Sorry, I couldn't change my name")
            
            elif "what's your name" in command:
                self.Voice.Speak(f"My name is {self.Name}")
                
            elif "reset your name" in command:
                if self.Config.change_name("Ultron"):
                    self.Name = "Ultron"
                    self.Voice.Speak("My name has been reset to Ultron")
                else:
                    self.Voice.Speak("Failed to reset my name")
        
        except Exception as e:
            self.Voice.Speak("Configuration command failed")
            logging.error(f"Config command error: {e}")




    
    def HandleJurassicCommand(self, command: str) -> None:
        """Handle Jurassic Park/World related commands"""
        try:
            if "dinosaur" in command:
                dino_name = command.split("dinosaur")[-1].strip()
                if dino_name:
                    dino = self.Jurassic.get_dinosaur_info(dino_name)
                    if dino:
                        self.Voice.Speak(f"{dino['name']} was a {dino['diet']} from the {dino['era']} period, growing up to {dino['size']}")
                    else:
                        self.Voice.Speak("Dinosaur not found in database")
                else:
                    dino = self.Jurassic.get_random_dinosaur()
                    self.Voice.Speak(f"Did you know about {dino['name']}? It was a {dino['diet']} from the {dino['era']} period")
                    
            elif "jurassic movie" in command:
                movie_name = command.split("movie")[-1].strip()
                if movie_name:
                    movie = self.Jurassic.get_movie_info(movie_name)
                    if movie:
                        self.Voice.Speak(f"{movie['title']} was released in {movie['year']} directed by {movie['director']}")
                else:
                    movie = random.choice(self.Jurassic.movies)
                    self.Voice.Speak(f"Recommend watching {movie['title']} from {movie['year']}")
                    
        except Exception as e:
            self.Voice.Speak("Failed to process Jurassic command")
            logging.error(f"Jurassic command error: {e}")

    def HandleFileCommand(self, command: str) -> None:
        """Handle file system operations"""
        command = command.lower()
        
        try:
            # Change directory
            if any(cmd in command for cmd in ["go to", "change to", "cd"]):
                dir_name = command.split("to")[-1].strip() if "to" in command else command.split("cd")[-1].strip()
                if self.FileManager.change_directory(dir_name):
                    self.Voice.Speak(f"Changed to directory {self.FileManager.current_dir}")
                else:
                    self.Voice.Speak("Directory not found")
            
            # Create file
            elif "create file" in command:
                filename = command.replace("create file", "").strip()
                if self.FileManager.create_file(filename):
                    self.Voice.Speak(f"Created file {filename}")
                else:
                    self.Voice.Speak("Failed to create file")
            
            # Create directory
            elif "create directory" in command or "mkdir" in command:
                dirname = command.replace("create directory", "").replace("mkdir", "").strip()
                if self.FileManager.create_directory(dirname):
                    self.Voice.Speak(f"Created directory {dirname}")
                else:
                    self.Voice.Speak("Failed to create directory")
            
            # Search files
            elif "search for" in command and "files" in command:
                pattern = command.split("search for")[-1].split("files")[0].strip()
                results = self.FileManager.search_files(pattern)
                if results:
                    self.Voice.Speak(f"Found {len(results)} files matching {pattern}")
                    for i, file in enumerate(results[:3], 1):
                        self.Voice.Speak(f"{i}. {file.name}")
                else:
                    self.Voice.Speak("No files found")
            
            # Delete file
            elif "delete file" in command:
                filename = command.replace("delete file", "").strip()
                if self.FileManager.delete_file(filename):
                    self.Voice.Speak(f"Deleted file {filename}")
                else:
                    self.Voice.Speak("File not found or couldn't be deleted")
            
            # List contents
            elif "list files" in command or "ls" in command:
                files, dirs = self.FileManager.get_directory_contents()
                self.Voice.Speak(f"Current directory: {self.FileManager.current_dir}")
                if dirs:
                    self.Voice.Speak(f"Subdirectories: {', '.join([d.name for d in dirs[:3]])}")
                if files:
                    self.Voice.Speak(f"Files: {', '.join([f.name for f in files[:3]])}")
            
            # Read file
            elif "read file" in command:
                filename = command.replace("read file", "").strip()
                content = self.FileManager.get_file_content(filename, lines=5)
                if content:
                    self.Voice.Speak(f"First lines of {filename}:")
                    for line in content[:3]:
                        self.Voice.Speak(line.strip())
                else:
                    self.Voice.Speak("File not found or couldn't be read")
            
            # Bookmark management
            elif "bookmark this" in command:
                name = command.replace("bookmark this as", "").strip()
                if self.FileManager.add_bookmark(name, self.FileManager.current_dir):
                    self.Voice.Speak(f"Bookmarked current directory as {name}")
                else:
                    self.Voice.Speak("Bookmark failed")
            
            elif "go to bookmark" in command:
                name = command.replace("go to bookmark", "").strip()
                if self.FileManager.goto_bookmark(name):
                    self.Voice.Speak(f"Changed to bookmarked directory {name}")
                else:
                    self.Voice.Speak("Bookmark not found")
        
        except Exception as e:
            self.Voice.Speak("File operation failed")
            logging.error(f"File command error: {e}")







    def HandleMathCommandAdvance(self, command: str) -> None:
        """Handle advanced math operations"""
        command = command.lower()
        
        try:
            # Angle mode switching
            if "switch to radians" in command:
                self.MathCalculator.set_angle_mode('radians')
                self.angle_mode = 'radians'
                self.Voice.Speak("Angle mode set to radians")
                return
            elif "switch to degrees" in command:
                self.MathCalculator.set_angle_mode('degrees')
                self.angle_mode = 'degrees'
                self.Voice.Speak("Angle mode set to degrees")
                return
            
            # Equation solving
            if "solve" in command:
                equation = command.replace("solve", "").strip()
                result = self.MathCalculator.calculate(f"solve {equation}")
                if result.error:
                    self.Voice.Speak(f"Could not solve equation: {result.error}")
                else:
                    self.Voice.Speak(f"The solution is: {result.value}")
                return
            
            # Integration
            if "integrate" in command:
                parts = command.replace("integrate", "").strip().split(" from ")
                if len(parts) == 2:
                    func = parts[0]
                    limits = parts[1].split(" to ")
                    if len(limits) == 2:
                        expr = f"integrate {func} from {limits[0]} to {limits[1]}"
                        result = self.MathCalculator.calculate(expr)
                        if result.error:
                            self.Voice.Speak(f"Integration failed: {result.error}")
                        else:
                            self.Voice.Speak(f"The integral is approximately {result.value:.3f}")
                        return
            
            # Variable assignment
            if "=" in command:
                result = self.MathCalculator.calculate(command)
                if result.error:
                    self.Voice.Speak(f"Assignment failed: {result.error}")
                else:
                    var_name = command.split("=")[0].strip()
                    self.Voice.Speak(f"Assigned {var_name} = {result.value}")
                return
            
            # Statistical analysis
            if "calculate statistics" in command:
                # In a real implementation, you'd get data from somewhere
                sample_data = [1, 2, 3, 4, 5]
                result = self.MathCalculator.statistical_analysis(sample_data)
                self.Voice.Speak(f"Statistical analysis results: Mean is {result.value['mean']}, Standard deviation is {result.value['stdev']:.2f}")
                return
            
            # Default calculation
            result = self.MathCalculator.calculate(command)
            if result.error:
                self.Voice.Speak(f"Calculation error: {result.error}")
            else:
                self.Voice.Speak(f"The result is {result.value}")
                
        except Exception as e:
            self.Voice.Speak("Math command failed")
            logging.error(f"Math command error: {e}")





    def HandleMeetingCommand(self, command: str) -> None:
        """Handle video conferencing commands"""
        command = command.lower()
        
        try:
            if "schedule meeting" in command:
                # Extract details from voice command
                title = re.search(r"schedule meeting (.+?) at", command).group(1)
                time_str = re.search(r"at (.+?) with", command).group(1)
                participants = re.search(r"with (.+)", command).group(1).split(" and ")
                
                meeting = self.JarvisMeet.schedule_meeting(
                    title=title,
                    start_time=time_str,
                    duration_min=60,
                    participants=participants
                )
                
                if meeting:
                    self.Voice.Speak(f"Meeting '{title}' scheduled at {time_str}")
                else:
                    self.Voice.Speak("Failed to schedule meeting")
                    
            elif "start meeting" in command:
                meeting_id = re.search(r"start meeting (.+)", command).group(1)
                if self.JarvisMeet.start_meeting(meeting_id):
                    self.Voice.Speak("Meeting started")
                else:
                    self.Voice.Speak("Meeting not found")
                    
            elif "end meeting" in command:
                if self.JarvisMeet.end_meeting():
                    self.Voice.Speak("Meeting ended")
                    
            elif "mute" in command:
                response = self.JarvisMeet.voice_command("mute")
                self.Voice.Speak(response)
                
            elif "unmute" in command:
                response = self.JarvisMeet.voice_command("unmute")
                self.Voice.Speak(response)
                
            elif "start recording" in command:
                self.JarvisMeet.is_recording = True
                self.JarvisMeet._start_recording()
                self.Voice.Speak("Recording started")
                
            elif "stop recording" in command:
                self.JarvisMeet._stop_recording()
                self.Voice.Speak("Recording saved")
                
        except Exception as e:
            self.Voice.Speak("Meeting command failed")
            logging.error(f"Meeting error: {e}")







    def HandleSecurityCommand(self, command: str) -> None:
        """Handle security-related voice commands"""
        command = command.lower()
        
        try:
            if "scan for malware" in command:
                directory = command.replace("scan for malware", "").strip() or "/"
                results = self.SecurityManager.scan_for_malware(directory)
                if results:
                    self.Voice.Speak(f"Found {len(results)} potential threats")
                    for i, result in enumerate(results[:3], 1):
                        self.Voice.Speak(f"Threat {i}: {result['threat']} in {os.path.basename(result['path'])}")
                else:
                    self.Voice.Speak("No malware detected")
                    
            elif "security report" in command:
                report = self.SecurityManager.generate_security_report()
                self.Voice.Speak(f"Security report generated with {report['alerts']} alerts")
                if report['suspicious_ips']:
                    self.Voice.Speak(f"Top threat IP: {report['top_threats'][0][0]}")
                    
            elif "start monitoring" in command:
                self.SecurityManager.start_network_monitoring()
                self.Voice.Speak("Network monitoring activated")
                
            elif "block ip" in command:
                ip = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', command)
                if ip:
                    if self.SecurityManager.block_ip(ip.group()):
                        self.Voice.Speak(f"Blocked IP {ip.group()}")
                    else:
                        self.Voice.Speak("Failed to block IP")
                else:
                    self.Voice.Speak("Please specify a valid IP address")
                    
            elif "check vulnerabilities" in command:
                vulns = self.SecurityManager.check_system_vulnerabilities()
                if vulns['unpatched_software']:
                    self.Voice.Speak(f"Found {len(vulns['unpatched_software'])} outdated packages")
                else:
                    self.Voice.Speak("No major vulnerabilities detected")
                    
        except Exception as e:
            self.Voice.Speak("Security command failed")
            logging.error(f"Security command error: {e}")






    def HandleFileSearch(self, command: str) -> None:
        """Handle file search commands"""
        command = command.lower()
        
        try:
            if "search files for" in command:
                # Extract search query
                parts = command.split("search files for", 1)
                query = parts[1].strip() if len(parts) > 1 else ""
                
                # Search with default parameters
                results = self.FileSearcher.find_files(
                    pattern="*",
                    content_search=query,
                    case_sensitive=False
                )
                
                if results:
                    self.Voice.Speak(f"Found {len(results)} matches")
                    for i, result in enumerate(results[:3], 1):  # Top 3 results
                        filename = os.path.basename(result.file_path)
                        self.Voice.Speak(f"Match {i} in {filename}, line {result.line_number}")
                        self.Voice.Speak(f"Content: {result.line_content[:50]}...")
                        
                        # Open the best match in VS Code
                        if i == 1:
                            self.FileSearcher.open_in_vscode(
                                result.file_path,
                                result.line_number
                            )

                else:
                    self.Voice.Speak("No matches found")

            elif "find file named" in command:
                filename = command.split("find file named", 1)[1].strip()
                results = self.FileSearcher.find_files(
                    pattern=f"*{filename}*",
                    case_sensitive=False
                )
                
                if results:
                    self.Voice.Speak(f"Found {len(results)} files")
                    for i, result in enumerate(results[:3], 1):
                        self.Voice.Speak(f"{i}. {os.path.basename(result.file_path)}")
                        if i == 1:
                            self.FileSearcher.open_in_vscode(result.file_path)
                else:
                    self.Voice.Speak("No files found with that name")
                    
            elif "recent files" in command:
                files = self.FileSearcher.get_recent_files(5)
                if files:
                    self.Voice.Speak("Recently opened files:")
                    for i, file in enumerate(files, 1):
                        self.Voice.Speak(f"{i}. {os.path.basename(file)}")
                else:
                    self.Voice.Speak("Could not retrieve recent files")
                    
        except Exception as e:
            self.Voice.Speak("File search failed")
            logging.error(f"File search error: {e}")




    def HandleSmartHomeCommand(self, command: str) -> None:
        """Handle smart home voice commands"""
        command = command.lower()
        
        try:
            # Device control
            if any(cmd in command for cmd in ["turn on", "turn off"]):
                device_type = None
                if "light" in command:
                    device_type = "light"
                elif "thermostat" in command:
                    device_type = "thermostat"
                
                if device_type:
                    action = "turn_on" if "turn on" in command else "turn_off"
                    device_name = command.split(device_type)[-1].strip()
                    response = self.SmartHome.voice_command_handler(command)
                    self.Voice.Speak(response)
            
            # Scene activation
            elif "activate scene" in command:
                scene_name = command.replace("activate scene", "").strip()
                if self.SmartHome.activate_scene(scene_name):
                    self.Voice.Speak(f"Activated {scene_name} scene")
                else:
                    self.Voice.Speak(f"Could not find {scene_name} scene")
            
            # Status check
            elif "status" in command and "device" in command:
                device_name = command.replace("status", "").replace("device", "").strip()
                device = self._find_smart_device(device_name)
                if device:
                    status = self.SmartHome.get_device_status(device.id)
                    self.Voice.Speak(f"{device.name} is {status['status'].lower()}")
                else:
                    self.Voice.Speak("Device not found")
            
            # Routine control
            elif "start routine" in command:
                routine_name = command.replace("start routine", "").strip()
                if self.SmartHome.start_routine(routine_name):
                    self.Voice.Speak(f"Started {routine_name} routine")
                else:
                    self.Voice.Speak(f"Could not start {routine_name} routine")
        
        except Exception as e:
            self.Voice.Speak("Smart home command failed")
            logging.error(f"Smart home error: {e}")

    def _find_smart_device(self, name: str) -> Optional[SmartDevice]:
        """Find device by approximate name matching"""
        for device in self.SmartHome.devices.values():
            if name.lower() in device.name.lower():
                return device
        return None



    def HandleLearningAdvanceCommand(self, command: str) -> None:
        """Handle advanced learning commands"""
        command = command.lower()
        
        try:
            if "list categories" in command:
                categories = self.LearningModule.list_categories()
                self.Voice.Speak("Available learning categories are: " + ", ".join(categories))
                
            elif "resources for" in command:
                category = command.replace("resources for", "").strip()
                resources = self.LearningModule.get_resources_by_category(category)
                if resources:
                    self.Voice.Speak(f"Top resources for {category}:")
                    for i, resource in enumerate(resources[:3], 1):
                        self.Voice.Speak(f"{i}. {resource['name']} ({resource['level']})")
                else:
                    self.Voice.Speak(f"No resources found for {category}")
                    
            elif "learn" in command or "resources" in command:
                topic = command.replace("learn", "").replace("resources", "").strip()
                if self.LearningModule.open_resource(topic):
                    self.Voice.Speak(f"Opening learning resources for {topic}")
                else:
                    self.Voice.Speak(f"Couldn't find resources for {topic}. Try specifying a category.")
                        
        except Exception as e:
            self.Voice.Speak("Failed to process learning request")
            logging.error(f"Learning command error: {e}")





    def HandleMusicAdvanceCommand(self, command: str) -> None:
        """Handle YouTube music commands"""
        command = command.lower()
        
        try:
            if "play" in command and ("on youtube" in command or "music" in command):
                query = command.replace("play", "").replace("on youtube", "").replace("music", "").strip()
                if query:
                    self.YouTubePlayer.run_continuous_playback(query)
                    song = self.YouTubePlayer.get_current_song()
                    if song:
                        self.Voice.Speak(f"Playing {song.title} by {song.artist}")
                    else:
                        self.Voice.Speak("Couldn't find that song")
                        
            elif any(cmd in command for cmd in ["stop music", "stop playback"]):
                self.YouTubePlayer.add_voice_command("stop")
                self.Voice.Speak("Music stopped")
                
            elif "pause music" in command:
                self.YouTubePlayer.add_voice_command("pause")
                self.Voice.Speak("Music paused")
                
            elif "resume music" in command:
                self.YouTubePlayer.add_voice_command("resume")
                self.Voice.Speak("Resuming music")
                
            elif "next track" in command:
                self.YouTubePlayer.add_voice_command("next")
                song = self.YouTubePlayer.get_current_song()
                if song:
                    self.Voice.Speak(f"Playing {song.title}")
                    
            elif "previous track" in command:
                self.YouTubePlayer.add_voice_command("previous")
                song = self.YouTubePlayer.get_current_song()
                if song:
                    self.Voice.Speak(f"Playing {song.title}")
                    
            elif "volume up" in command:
                self.YouTubePlayer.volume_up()
                self.Voice.Speak(f"Volume increased to {self.YouTubePlayer.volume}")
                
            elif "volume down" in command:
                self.YouTubePlayer.volume_down()
                self.Voice.Speak(f"Volume decreased to {self.YouTubePlayer.volume}")
                
            elif "set volume to" in command:
                vol = int(''.join(filter(str.isdigit, command)))
                self.YouTubePlayer.set_volume(vol)
                self.Voice.Speak(f"Volume set to {vol}")
                
        except Exception as e:
            self.Voice.Speak("Music command failed")
            logging.error(f"Music command error: {e}")



    def HandleSystemCommand(self, command: str) -> None:
        """Handle system control commands"""
        command = command.lower()
        
        try:
            if "system stats" in command:
                stats = self.SystemControl.get_system_stats()
                self.Voice.Speak(f"CPU usage: {stats['cpu']['percent']}%")
                self.Voice.Speak(f"Memory available: {stats['memory']['available']} GB")
                
            elif "list processes" in command:
                processes = self.SystemControl.list_processes()
                top_processes = sorted(processes, key=lambda x: x.memory_percent, reverse=True)[:3]
                self.Voice.Speak("Top processes by memory usage:")
                for i, proc in enumerate(top_processes, 1):
                    self.Voice.Speak(f"{i}. {proc.name}: {proc.memory_percent:.1f}% memory")
                    
            elif "add task" in command:
                task_name = command.replace("add task", "").strip()
                if task_name:
                    task = self.SystemControl.add_task(task_name)
                    self.Voice.Speak(f"Added task: {task.name} (ID: {task.id})")
                    
            elif "complete task" in command:
                task_id = int(re.search(r'\d+', command).group())
                if self.SystemControl.complete_task(task_id):
                    self.Voice.Speak(f"Completed task ID {task_id}")
                    
            elif "shutdown" in command:
                if "cancel" in command:
                    subprocess.run(["shutdown", "/a"]) if platform.system() == "Windows" else None
                    self.Voice.Speak("Shutdown cancelled")
                else:
                    delay = 60  # Default 1 minute
                    if "in" in command:
                        delay = int(re.search(r'in (\d+)', command).group(1)) * 60
                    self.SystemControl.shutdown(delay)
                    self.Voice.Speak(f"System will shutdown in {delay//60} minutes")
                    
            elif "restart" in command:
                delay = 60
                if "in" in command:
                    delay = int(re.search(r'in (\d+)', command).group(1)) * 60
                self.SystemControl.restart(delay)
                self.Voice.Speak(f"System will restart in {delay//60} minutes")
                
            elif "sleep" in command:
                if self.SystemControl.sleep():
                    self.Voice.Speak("Putting system to sleep")
                    
        except Exception as e:
            self.Voice.Speak("Failed to execute system command")
            logging.error(f"System command error: {e}")
        
    def HandleChromeSearch(self, command: str) -> None:
        """Handle Chrome-specific searches."""
        query = command.replace("search", "").replace("on chrome", "").strip()
        if query:
            if self.ChromeSearcher.search(query):
                self.Voice.Speak(f"Searching for {query} on Chrome")
            else:
                self.Voice.Speak("Failed to open Chrome")







    def HandleSearchCommand(self, command: str) -> None:
        try:
            query = command.replace("search", "").strip()
            if "wikipedia" in command:
                result = self.WebServices.SearchWikipedia(query)
                self.Voice.Speak(result)
            else:
                self.WebServices.SearchGoogle(query)
                self.Voice.Speak(f"Here are the search results for {query}")
        except Exception as e:
            logging.error(f"Search command error: {e}")
            self.Voice.Speak("There was an error processing your search request")





    def HandleFileCommand(self, command: str) -> None:
        try:
            query = command.replace("search files", "").strip()
            results = self.FileSearch.SearchFiles(os.getcwd(), query)
            if results:
                self.Voice.Speak(f"Found {len(results)} files matching your query")
                for i, result in enumerate(results[:3], 1):
                    self.Voice.Speak(f"File {i}: {os.path.basename(result)}")
            else:
                self.Voice.Speak("No files found matching your query")
        except Exception as e:
            logging.error(f"File command error: {e}")
            self.Voice.Speak("There was an error searching for files")





    def HandleCalendarCommand(self, command: str) -> None:
        try:
            if "create event" in command:
                # This would need more sophisticated parsing for real events
                self.Voice.Speak("Please specify event details")
            else:
                events = self.Calendar.GetEvents(3)
                if events:
                    self.Voice.Speak("Here are your upcoming events")
                    for event in events:
                        start = event['start'].get('dateTime', event['start'].get('date'))
                        self.Voice.Speak(f"{event['summary']} at {start}")
                else:
                    self.Voice.Speak("No upcoming events found")
        except Exception as e:
            logging.error(f"Calendar command error: {e}")
            self.Voice.Speak("There was an error accessing your calendar")




    def HandleEmailCommand(self) -> None:
        try:
            self.Voice.Speak("Please provide the email details")
            self.Voice.Speak("Recipient email address:")
            to = self.Voice.TakeCommand()
            self.Voice.Speak("Email subject:")
            subject = self.Voice.TakeCommand()
            self.Voice.Speak("Email body:")
            body = self.Voice.TakeCommand()
            
            if to and subject and body:
                if self.Email.SendEmail(to, subject, body):
                    self.Voice.Speak("Email sent successfully")
                else:
                    self.Voice.Speak("Failed to send email")
            else:
                self.Voice.Speak("Email details incomplete")
        except Exception as e:
            logging.error(f"Email command error: {e}")
            self.Voice.Speak("There was an error sending the email")




    def HandleNewsCommand(self, command: str) -> None:
        try:
            category = "general"
            if "sports" in command:
                category = "sports"
            elif "technology" in command:
                category = "technology"
            elif "business" in command:
                category = "business"
                
            headlines = self.NewsReader.GetHeadlines(category)
            self.Voice.Speak(f"Here are the latest {category} news headlines")
            for i, headline in enumerate(headlines[:3], 1):
                self.Voice.Speak(f"Headline {i}: {headline}")
        except Exception as e:
            logging.error(f"News command error: {e}")
            self.Voice.Speak("There was an error fetching news headlines")




    def HandleHealthCommand(self) -> None:
        try:
            health = self.HealthMonitor.CheckSystemHealth()
            if 'error' in health:
                self.Voice.Speak("Could not check system health")
            else:
                self.Voice.Speak("System health status:")
                for key, value in health.items():
                    self.Voice.Speak(f"{key}: {value}")
        except Exception as e:
            logging.error(f"Health command error: {e}")
            self.Voice.Speak("There was an error checking system health")





    def HandleLearningCommand(self, command: str) -> None:
        try:
            topic = command.replace("learn", "").replace("resources", "").strip()
            resource = self.Learning.GetResource(topic)
            if resource:
                self.Voice.Speak(f"Here's a resource for learning {topic}")
                self.WebServices.OpenWebsite(resource)
            else:
                self.Voice.Speak(f"Sorry, I don't have resources for {topic}")
        except Exception as e:
            logging.error(f"Learning command error: {e}")
            self.Voice.Speak("There was an error finding learning resources")





    def HandleTaskCommand(self, command: str) -> None:
        try:
            task = command.replace("add task", "").strip()
            if task:
                self.TaskManager.AddTask(task)
                self.Voice.Speak(f"Added task: {task}")
            else:
                self.Voice.Speak("Please specify a task description")
        except Exception as e:
            logging.error(f"Task command error: {e}")
            self.Voice.Speak("There was an error adding your task")




    def HandleAlarmCommand(self, command: str) -> None:
        try:
            timeStr = command.replace("set alarm", "").strip()
            if self.Alarm.SetAlarm(timeStr):
                self.Voice.Speak(f"Alarm set for {timeStr}")
            else:
                self.Voice.Speak("Please specify a valid time in HH:MM format")
        except Exception as e:
            logging.error(f"Alarm command error: {e}")
            self.Voice.Speak("There was an error setting the alarm")





    def HandleMathCommand(self, command: str) -> None:
        try:
            expression = command.replace("calculate", "").strip()
            result = self.Calculator.Calculate(expression)
            if not math.isnan(result):
                self.Voice.Speak(f"The result is {result}")
            else:
                self.Voice.Speak("Could not calculate that expression")
        except Exception as e:
            logging.error(f"Math command error: {e}")
            self.Voice.Speak("There was an error processing your calculation")



if __name__ == "__main__":
    assistant = UltronAI()
    assistant.Greet()
    
    running = True
    while running:
        command = assistant.Voice.TakeCommand()
        if command:
            running = assistant.ProcessCommand(command)
