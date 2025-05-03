
import sys
import os
import time
import json
import base64
import sqlite3
import secrets
import string
import hashlib
import pyperclip
import qrcode
import zxcvbn
from datetime import datetime, timedelta
from getpass import getpass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import argon2
import uuid
import socket
import platform
import pyotp
import logging
from logging.handlers import RotatingFileHandler
import signal
import atexit
import csv

# Windows-safe input handling
if sys.platform != "win32":
    try:
        import readline
    except ImportError:
        pass


# ==================== SECURITY CONFIGURATION ====================
class SecurityLevel(Enum):
    BASIC = auto()
    ENTERPRISE = auto()
    MILITARY = auto()


CURRENT_SECURITY_LEVEL = SecurityLevel.MILITARY

SECURITY_PARAMS = {
    SecurityLevel.BASIC: {
        "argon2": {"time_cost": 2, "memory_cost": 32768, "parallelism": 2},
        "pbkdf2": {"iterations": 100000, "hash": hashes.SHA256()},
        "min_length": 12,
        "max_attempts": 10,
        "lockout_time": 60,
        "session_timeout": 3600,
    },
    SecurityLevel.ENTERPRISE: {
        "argon2": {"time_cost": 3, "memory_cost": 65536, "parallelism": 4},
        "pbkdf2": {"iterations": 300000, "hash": hashes.SHA512()},
        "min_length": 16,
        "max_attempts": 5,
        "lockout_time": 300,
        "session_timeout": 1800,
    },
    SecurityLevel.MILITARY: {
        "argon2": {"time_cost": 4, "memory_cost": 131072, "parallelism": 8},
        "pbkdf2": {"iterations": 600000, "hash": hashes.BLAKE2b(64)},
        "min_length": 20,
        "max_attempts": 3,
        "lockout_time": 900,
        "session_timeout": 900,
    },
}

CONFIG = SECURITY_PARAMS[CURRENT_SECURITY_LEVEL]


# ==================== DATA STRUCTURES ====================
@dataclass
class PasswordEntry:
    id: str
    service: str
    username: str
    encrypted_password: bytes
    notes: str
    created_at: datetime
    last_used: Optional[datetime]
    strength_score: int
    length: int
    complexity: str
    tags: List[str]
    history: List[Tuple[datetime, bytes]]


@dataclass
class UserAccount:
    username: str
    encrypted_master: bytes
    salt: bytes
    last_login: Optional[datetime]
    failed_attempts: int
    locked_until: Optional[datetime]
    mfa_secret: Optional[str]
    permissions: List[str]


# ==================== CRYPTOGRAPHY SERVICE ====================
class CryptoService:
    @staticmethod
    def generate_key_from_password(password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=CONFIG["pbkdf2"]["hash"],
            length=32,
            salt=salt,
            iterations=CONFIG["pbkdf2"]["iterations"],
            backend=default_backend(),
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_data(data: str, key: bytes) -> bytes:
        f = Fernet(key)
        return f.encrypt(data.encode())

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> str:
        f = Fernet(key)
        try:
            return f.decrypt(encrypted_data).decode()
        except InvalidToken:
            raise DecryptionError("Failed to decrypt data - invalid token or key")

    @staticmethod
    def generate_secure_random(length: int = 32) -> bytes:
        return secrets.token_bytes(length)

    @staticmethod
    def hash_password(password: str, salt: bytes) -> str:
        hasher = argon2.PasswordHasher(
            time_cost=CONFIG["argon2"]["time_cost"],
            memory_cost=CONFIG["argon2"]["memory_cost"],
            parallelism=CONFIG["argon2"]["parallelism"],
            hash_len=32,
            salt_len=16,
        )
        return hasher.hash(password.encode() + salt)

    @staticmethod
    def verify_password(stored_hash: str, password: str, salt: bytes) -> bool:
        hasher = argon2.PasswordHasher()
        try:
            return hasher.verify(stored_hash, password.encode() + salt)
        except (argon2.exceptions.VerifyMismatchError, argon2.exceptions.InvalidHash):
            return False
        except Exception as e:
            raise SecurityError(f"Password verification failed: {str(e)}")

    @staticmethod
    def generate_mfa_secret() -> str:
        return pyotp.random_base32()

    @staticmethod
    def verify_mfa_code(secret: str, code: str) -> bool:
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)


# ==================== LOGGING SERVICE ====================
class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("PasswordManagerAudit")
        self.logger.setLevel(logging.INFO)

        os.makedirs("logs", exist_ok=True)

        handler = RotatingFileHandler(
            "logs/audit.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if "--debug" in sys.argv:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            self.logger.addHandler(console)

    def log_event(self, event: str, user: Optional[str] = None, level: str = "INFO"):
        metadata = {
            "user": user,
            "ip": self._get_ip_address(),
            "host": platform.node(),
            "os": platform.platform(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        log_entry = {"event": event, "metadata": metadata, "level": level}

        getattr(self.logger, level.lower())(json.dumps(log_entry))

    def _get_ip_address(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except:
            return "unknown"


# ==================== PASSWORD GENERATOR ====================
class AdvancedPasswordGenerator:
    @staticmethod
    def generate(
        length: int = 16,
        use_upper: bool = True,
        use_digits: bool = True,
        use_symbols: bool = True,
        avoid_ambiguous: bool = True,
        avoid_similar: bool = True,
        require_each_type: bool = True,
        custom_chars: str = "",
        passphrase: bool = False,
        word_count: int = 6,
        separator: str = "-",
    ) -> str:
        if passphrase:
            return AdvancedPasswordGenerator._generate_passphrase(word_count, separator)

        if not 8 <= length <= 256:
            raise ValueError("Password length must be between 8 and 256")

        chars = string.ascii_lowercase
        char_sets = []

        if use_upper:
            chars += string.ascii_uppercase
            char_sets.append(string.ascii_uppercase)
        if use_digits:
            chars += string.digits
            char_sets.append(string.digits)
        if use_symbols:
            chars += string.punctuation
            char_sets.append(string.punctuation)
        if custom_chars:
            chars += custom_chars
            char_sets.append(custom_chars)

        if avoid_ambiguous:
            ambiguous = "l1IoO0"
            chars = "".join(c for c in chars if c not in ambiguous)
            char_sets = ["".join(c for c in s if c not in ambiguous) for s in char_sets]

        if avoid_similar:
            similar = "ij1l|"
            chars = "".join(c for c in chars if c not in similar)
            char_sets = ["".join(c for c in s if c not in similar) for s in char_sets]

        for _ in range(100):
            password = "".join(secrets.choice(chars) for _ in range(length))

            if not require_each_type:
                return password

            checks_passed = True
            for char_set in char_sets:
                if not any(c in char_set for c in password):
                    checks_passed = False
                    break

            if checks_passed:
                return password

        raise RuntimeError("Failed to generate password meeting requirements")

    @staticmethod
    def _generate_passphrase(word_count: int = 6, separator: str = "-") -> str:
        try:
            wordlist = AdvancedPasswordGenerator._load_wordlist()
            words = [secrets.choice(wordlist) for _ in range(word_count)]
            return separator.join(words)
        except Exception as e:
            raise RuntimeError(f"Failed to generate passphrase: {str(e)}")

    @staticmethod
    def _load_wordlist() -> List[str]:
        try:
            with open("diceware.wordlist", "r", encoding="utf-8") as f:
                return [line.split()[1] for line in f if line.strip()]
        except FileNotFoundError:
            return [
                "correct",
                "horse",
                "battery",
                "staple",
                "secure",
                "password",
                "generator",
                "system",
                "privacy",
                "encryption",
                "security",
            ]


# ==================== PASSWORD MANAGER CORE ====================
class MilitaryGradePasswordManager:
    def __init__(self):
        self.db_name = "vault.db"
        self.key_file = "master.key"
        self.audit_log = AuditLogger()
        self.current_user = None
        self.user_key = None
        self.session_expiry = None

        self._initialize_keystore()
        self._initialize_database()

        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _initialize_keystore(self):
        if not os.path.exists(self.key_file):
            with open(self.key_file, "wb") as f:
                f.write(CryptoService.generate_secure_random(64))
            if os.name != "nt":
                os.chmod(self.key_file, 0o600)

    def _initialize_database(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        encrypted_master TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        last_login TEXT,
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until TEXT,
                        mfa_secret TEXT,
                        permissions TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS passwords (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        service TEXT NOT NULL,
                        username TEXT,
                        encrypted_password TEXT NOT NULL,
                        notes TEXT,
                        tags TEXT,
                        strength_score INTEGER,
                        length INTEGER,
                        complexity TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_used TEXT,
                        history TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_events (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        event_type TEXT NOT NULL,
                        details TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    def _cleanup(self):
        if self.user_key:
            if isinstance(self.user_key, bytes):
                for i in range(len(self.user_key)):
                    self.user_key = self.user_key[:i] + b"\x00" + self.user_key[i + 1 :]
            self.user_key = None

        self.current_user = None
        self.session_expiry = None

    def _handle_signal(self, signum, frame):
        self._cleanup()
        sys.exit(0)

    def _check_session(self):
        if not self.session_expiry or datetime.utcnow() > self.session_expiry:
            self.audit_log.log_event("Session expired", self.current_user)
            raise SessionExpiredError("Your session has expired. Please log in again.")

    def register_user(self, username: str, password: str, admin: bool = False):
        if not username or not password:
            raise ValueError("Username and password are required")

        if len(password) < CONFIG["min_length"]:
            raise PasswordPolicyError(
                {"min_length": CONFIG["min_length"]}, {"actual_length": len(password)}
            )

        salt = CryptoService.generate_secure_random(16)
        master_key = CryptoService.generate_key_from_password(password, salt)

        storage_key = self._get_storage_key()
        encrypted_master = CryptoService.encrypt_data(
            base64.b64encode(master_key).decode(), storage_key
        )

        mfa_secret = CryptoService.generate_mfa_secret()
        permissions = ["admin"] if admin else ["user"]

        user_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (
                        id, username, encrypted_master, salt, mfa_secret, permissions
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        username,
                        encrypted_master,
                        base64.b64encode(salt).decode(),
                        mfa_secret,
                        json.dumps(permissions),
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            raise IntegrityError(f"Username '{username}' already exists")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to register user: {str(e)}")

        self.audit_log.log_event("New user registered", username)
        return {
            "user_id": user_id,
            "mfa_secret": mfa_secret,
            "mfa_qr": self._generate_mfa_qr(username, mfa_secret),
        }

    def _get_storage_key(self) -> bytes:
        try:
            with open(self.key_file, "rb") as f:
                key_data = f.read()
            return key_data[:32]
        except IOError as e:
            raise EncryptionError(f"Failed to read key file: {str(e)}")

    def _generate_mfa_qr(self, username: str, secret: str) -> str:
        uri = pyotp.totp.TOTP(secret).provisioning_uri(
            username, issuer_name="MilitaryGradePasswordManager"
        )
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(uri)
        qr.make(fit=True)

        filename = f"mfa_{username}_{datetime.now().strftime('%Y%m%d')}.png"
        try:
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(filename)
            return filename
        except Exception as e:
            raise ConfigurationError(f"Failed to generate QR code: {str(e)}")

    def authenticate(
        self, username: str, password: str, mfa_code: Optional[str] = None
    ):
        user = self._get_user(username)
        if not user:
            raise AuthenticationError("Invalid username or password")

        if user.locked_until and datetime.utcnow() < datetime.fromisoformat(
            user.locked_until
        ):
            remaining = (
                datetime.fromisoformat(user.locked_until) - datetime.utcnow()
            ).seconds // 60
            raise AccountLockedError(
                f"Account locked. Try again in {remaining} minutes."
            )

        try:
            salt = base64.b64decode(user.salt)
            storage_key = self._get_storage_key()
            encrypted_master = base64.b64decode(user.encrypted_master)
            master_key_encoded = CryptoService.decrypt_data(
                encrypted_master, storage_key
            )
            master_key = base64.b64decode(master_key_encoded)
        except (InvalidToken, DecryptionError):
            self._record_failed_attempt(username)
            raise AuthenticationError("Invalid username or password")
        except Exception as e:
            raise EncryptionError(f"Authentication failed: {str(e)}")

        if user.mfa_secret:
            if not mfa_code:
                raise MFARequiredError()
            if not CryptoService.verify_mfa_code(user.mfa_secret, mfa_code):
                self._record_failed_attempt(username)
                raise MFAVerificationError()

        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE users 
                    SET last_login = ?, failed_attempts = 0, locked_until = NULL
                    WHERE username = ?
                """,
                    (datetime.utcnow().isoformat(), username),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update user record: {str(e)}")

        self.current_user = username
        self.user_key = master_key
        self.session_expiry = datetime.utcnow() + timedelta(
            seconds=CONFIG["session_timeout"]
        )

        self.audit_log.log_event("User logged in", username)
        return True

    def _get_user(self, username: str) -> Optional[UserAccount]:
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT username, encrypted_master, salt, last_login, 
                           failed_attempts, locked_until, mfa_secret, permissions
                    FROM users 
                    WHERE username = ?
                """,
                    (username,),
                )
                row = cursor.fetchone()

            if not row:
                return None

            return UserAccount(
                username=row[0],
                encrypted_master=row[1],
                salt=row[2],
                last_login=datetime.fromisoformat(row[3]) if row[3] else None,
                failed_attempts=row[4],
                locked_until=datetime.fromisoformat(row[5]) if row[5] else None,
                mfa_secret=row[6],
                permissions=json.loads(row[7]),
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve user: {str(e)}")
        except json.JSONDecodeError:
            raise DatabaseError("Corrupted user permissions data")

    def _record_failed_attempt(self, username: str):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE users 
                    SET failed_attempts = failed_attempts + 1
                    WHERE username = ?
                """,
                    (username,),
                )

                cursor.execute(
                    """
                    SELECT failed_attempts FROM users WHERE username = ?
                """,
                    (username,),
                )
                attempts = cursor.fetchone()[0]

                if attempts >= CONFIG["max_attempts"]:
                    lockout_time = datetime.utcnow() + timedelta(
                        seconds=CONFIG["lockout_time"]
                    )
                    cursor.execute(
                        """
                        UPDATE users 
                        SET locked_until = ?
                        WHERE username = ?
                    """,
                        (lockout_time.isoformat(), username),
                    )

                    self.audit_log.log_event(
                        "Account locked due to failed attempts", username, "WARNING"
                    )

                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to record failed attempt: {str(e)}")

    def generate_password(self, **kwargs) -> Dict:
        self._check_session()

        try:
            password = AdvancedPasswordGenerator.generate(**kwargs)
            analysis = self._analyze_password(password)

            return {"password": password, "analysis": analysis}
        except Exception as e:
            raise PasswordPolicyError(
                requirements={"length": kwargs.get("length", 16)},
                actual={"error": str(e)},
            )

    def _analyze_password(self, password: str) -> Dict:
        result = zxcvbn.zxcvbn(password)

        char_set = 0
        if any(c.islower() for c in password):
            char_set += 26
        if any(c.isupper() for c in password):
            char_set += 26
        if any(c.isdigit() for c in password):
            char_set += 10
        if any(c in string.punctuation for c in password):
            char_set += 32

        length = len(password)
        entropy = length * (char_set**0.5) if char_set > 0 else 0

        return {
            "length": length,
            "unique_chars": len(set(password)),
            "entropy": entropy,
            "strength": result["score"],
            "crack_time": result["crack_times_display"][
                "offline_slow_hashing_1e4_per_second"
            ],
            "feedback": result["feedback"],
        }

    def store_password(self, service: str, password: str, **metadata):
        self._check_session()

        if not service or not password:
            raise ValueError("Service and password are required")

        try:
            encrypted_pw = CryptoService.encrypt_data(password, self.user_key)
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt password: {str(e)}")

        entry_id = str(uuid.uuid4())
        analysis = self._analyze_password(password)

        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO passwords (
                        id, user_id, service, username, encrypted_password,
                        notes, tags, strength_score, length, complexity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry_id,
                        self._get_user_id(self.current_user),
                        service,
                        metadata.get("username", ""),
                        encrypted_pw,
                        metadata.get("notes", ""),
                        json.dumps(metadata.get("tags", [])),
                        analysis["strength"],
                        len(password),
                        self._get_complexity_description(password),
                    ),
                )

                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to store password: {str(e)}")

        self.audit_log.log_event(f"Password stored for {service}", self.current_user)
        return entry_id

    def _get_user_id(self, username: str) -> str:
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id FROM users WHERE username = ?
                """,
                    (username,),
                )
                result = cursor.fetchone()
                if not result:
                    raise NotFoundError("user", username)
                return result[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get user ID: {str(e)}")

    def _get_complexity_description(self, password: str) -> str:
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in string.punctuation for c in password)

        parts = []
        if has_upper and has_lower:
            parts.append("mixed case")
        elif has_upper:
            parts.append("uppercase")
        elif has_lower:
            parts.append("lowercase")

        if has_digit:
            parts.append("digits")
        if has_symbol:
            parts.append("symbols")

        return ", ".join(parts) if parts else "simple"

    def get_password(self, service: str) -> Optional[PasswordEntry]:
        self._check_session()

        user_id = self._get_user_id(self.current_user)

        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, service, username, encrypted_password, notes, tags,
                           strength_score, length, complexity, created_at, last_used, history
                    FROM passwords
                    WHERE user_id = ? AND service = ?
                    ORDER BY last_used DESC
                    LIMIT 1
                """,
                    (user_id, service),
                )

                row = cursor.fetchone()

                if not row:
                    return None

                cursor.execute(
                    """
                    UPDATE passwords
                    SET last_used = ?
                    WHERE id = ?
                """,
                    (datetime.utcnow().isoformat(), row[0]),
                )
                conn.commit()

                try:
                    password = CryptoService.decrypt_data(row[3], self.user_key)
                except InvalidToken:
                    raise DecryptionError(
                        "Failed to decrypt password - invalid key or corrupted data"
                    )

                return PasswordEntry(
                    id=row[0],
                    service=row[1],
                    username=row[2],
                    encrypted_password=password,
                    notes=row[4],
                    created_at=datetime.fromisoformat(row[9]),
                    last_used=datetime.fromisoformat(row[10]) if row[10] else None,
                    strength_score=row[6],
                    length=row[7],
                    complexity=row[8],
                    tags=json.loads(row[5]),
                    history=json.loads(row[11]) if row[11] else [],
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve password: {str(e)}")
        except json.JSONDecodeError:
            raise DatabaseError("Corrupted password entry data")

    def export_vault(self, format: str = "json", encrypt: bool = True) -> str:
        self._check_session()

        user_id = self._get_user_id(self.current_user)
        entries = []

        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT service, username, encrypted_password, notes, tags
                    FROM passwords
                    WHERE user_id = ?
                """,
                    (user_id,),
                )

                for row in cursor.fetchall():
                    try:
                        password = CryptoService.decrypt_data(row[2], self.user_key)
                        entries.append(
                            {
                                "service": row[0],
                                "username": row[1],
                                "password": password,
                                "notes": row[3],
                                "tags": json.loads(row[4]),
                            }
                        )
                    except (InvalidToken, DecryptionError):
                        continue
                    except json.JSONDecodeError:
                        entries.append(
                            {
                                "service": row[0],
                                "username": row[1],
                                "password": "[DECRYPTION FAILED]",
                                "notes": row[3],
                                "tags": [],
                            }
                        )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to export vault: {str(e)}")

        if not entries:
            raise NotFoundError("passwords", f"for user {self.current_user}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vault_export_{self.current_user}_{timestamp}"

        try:
            if format == "json":
                data = json.dumps(entries, indent=2)
                if encrypt:
                    data = CryptoService.encrypt_data(data, self.user_key)
                    filename += ".enc"
                else:
                    filename += ".json"

                with open(filename, "wb" if encrypt else "w") as f:
                    f.write(data if not encrypt else data.encode())

            elif format == "csv":
                filename += ".csv"

                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["service", "username", "password", "notes", "tags"],
                    )
                    writer.writeheader()

                    for entry in entries:
                        writer.writerow(
                            {
                                "service": entry["service"],
                                "username": entry["username"],
                                "password": entry["password"],
                                "notes": entry["notes"],
                                "tags": ",".join(entry["tags"]),
                            }
                        )
            else:
                raise ValidationError("Unsupported export format", {"format": format})
        except IOError as e:
            raise OperationNotAllowedError(
                "export", f"Failed to write export file: {str(e)}"
            )
        except Exception as e:
            raise OperationNotAllowedError("export", str(e))

        self.audit_log.log_event(f"Vault exported to {filename}", self.current_user)
        return filename


# ==================== ERROR CLASSES ====================
class PasswordManagerError(Exception):
    def __init__(self, message: str, code: int = 1000, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def __str__(self):
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> Dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class AuthenticationError(PasswordManagerError):
    def __init__(
        self, message: str = "Authentication failed", details: Optional[Dict] = None
    ):
        super().__init__(message, code=1100, details=details)


class AccountLockedError(AuthenticationError):
    def __init__(self, unlock_time: Optional[datetime] = None):
        details = {}
        if unlock_time:
            remaining = (unlock_time - datetime.utcnow()).seconds // 60
            details["unlock_in_minutes"] = remaining
            details["unlock_time"] = unlock_time.isoformat()

        super().__init__(
            message="Account temporarily locked due to too many failed attempts",
            code=1101,
            details=details,
        )


class MFARequiredError(AuthenticationError):
    def __init__(self, methods: List[str] = ["TOTP"]):
        super().__init__(
            message="Multi-factor authentication required",
            code=1102,
            details={"available_methods": methods},
        )


class MFAVerificationError(AuthenticationError):
    def __init__(self, message: str = "Invalid MFA code"):
        super().__init__(message, code=1103)


class SessionExpiredError(AuthenticationError):
    def __init__(self):
        super().__init__("Session expired. Please log in again", code=1104)


class AuthorizationError(PasswordManagerError):
    def __init__(
        self, message: str = "Authorization failed", details: Optional[Dict] = None
    ):
        super().__init__(message, code=1200, details=details)


class PermissionDeniedError(AuthorizationError):
    def __init__(self, required_permission: str, available_permissions: List[str]):
        super().__init__(
            message=f"Permission denied: {required_permission} required",
            code=1201,
            details={
                "required_permission": required_permission,
                "available_permissions": available_permissions,
            },
        )


class EncryptionError(PasswordManagerError):
    def __init__(self, message: str = "Encryption/decryption error"):
        super().__init__(message, code=1300)


class DecryptionError(EncryptionError):
    def __init__(self, item_type: str = "password"):
        super().__init__(
            message=f"Failed to decrypt {item_type}",
            code=1301,
            details={"item_type": item_type},
        )


class KeyDerivationError(EncryptionError):
    def __init__(self):
        super().__init__("Failed to derive encryption key", code=1302)


class DatabaseError(PasswordManagerError):
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, code=1400)


class IntegrityError(DatabaseError):
    def __init__(self, constraint: str):
        super().__init__(
            message=f"Data integrity violation: {constraint}",
            code=1401,
            details={"constraint": constraint},
        )


class NotFoundError(DatabaseError):
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code=1402,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(PasswordManagerError):
    def __init__(
        self, message: str = "Validation failed", fields: Optional[Dict] = None
    ):
        super().__init__(message, code=1500, details={"fields": fields or {}})


class PasswordPolicyError(ValidationError):
    def __init__(self, requirements: Dict, actual: Dict):
        super().__init__(
            message="Password does not meet policy requirements",
            code=1501,
            details={"requirements": requirements, "actual": actual},
        )


class OperationNotAllowedError(PasswordManagerError):
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Operation not allowed: {operation}",
            code=1600,
            details={"operation": operation, "reason": reason},
        )


class RateLimitError(PasswordManagerError):
    def __init__(self, retry_after: int):
        super().__init__(
            message="Too many requests. Please try again later.",
            code=1700,
            details={
                "retry_after_seconds": retry_after,
                "retry_time": (
                    datetime.utcnow() + timedelta(seconds=retry_after)
                ).isoformat(),
            },
        )


class ConfigurationError(PasswordManagerError):
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message, code=1800)


class SecurityError(PasswordManagerError):
    def __init__(self, message: str = "Security violation detected"):
        super().__init__(message, code=1900)


class AuditLogError(PasswordManagerError):
    def __init__(self, message: str = "Audit logging failed"):
        super().__init__(message, code=2000)


# ... [previous imports remain the same] ...


# ==================== COMMAND LINE INTERFACE ====================
class PasswordManagerCLI:
    def __init__(self):
        self.manager = MilitaryGradePasswordManager()
        self.current_user = None

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_banner(self):
        self.clear_screen()
        print(
            """
        ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗
        ██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║
        ██████╔╝██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║
        ██╔═══╝ ██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║
        ██║     ╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║
        ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝
        """
        )
        print("Military-Grade Password Manager")
        print("===============================")

    def main_menu(self):
        while True:
            self.print_banner()
            print("\nMain Menu:")
            print("1. Register")
            print("2. Login")
            print("3. Exit")

            choice = input("\nEnter your choice: ")

            if choice == "1":
                self.register()
            elif choice == "2":
                self.login()
            elif choice == "3":
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("\nInvalid choice. Please try again.")
                time.sleep(1)

    def register(self):
        self.print_banner()
        print("\nUser Registration:")
        username = input("Enter username: ")
        password = getpass("Enter password: ")
        confirm_password = getpass("Confirm password: ")

        if password != confirm_password:
            print("\nPasswords do not match!")
            time.sleep(1)
            return

        try:
            result = self.manager.register_user(username, password)
            print("\nRegistration successful!")
            print(f"MFA Secret: {result['mfa_secret']}")
            print(f"MFA QR code saved to: {result['mfa_qr']}")
            print("\nPlease scan the QR code with your authenticator app.")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nRegistration failed: {str(e)}")
            time.sleep(2)

    def login(self):
        self.print_banner()
        print("\nUser Login:")
        username = input("Enter username: ")
        password = getpass("Enter password: ")

        try:
            # First try without MFA
            if self.manager.authenticate(username, password):
                self.current_user = username
                self.vault_menu()
                return

            # If MFA is required
            mfa_code = input("Enter MFA code: ")
            if self.manager.authenticate(username, password, mfa_code):
                self.current_user = username
                self.vault_menu()
        except MFARequiredError:
            mfa_code = input("MFA required. Enter MFA code: ")
            try:
                if self.manager.authenticate(username, password, mfa_code):
                    self.current_user = username
                    self.vault_menu()
            except Exception as e:
                print(f"\nLogin failed: {str(e)}")
                time.sleep(2)
        except Exception as e:
            print(f"\nLogin failed: {str(e)}")
            time.sleep(2)

    def vault_menu(self):
        while True:
            self.print_banner()
            print(f"\nWelcome, {self.current_user}!")
            print("\nVault Menu:")
            print("1. Generate Password")
            print("2. Store Password")
            print("3. Retrieve Password")
            print("4. Export Vault")
            print("5. Logout")

            choice = input("\nEnter your choice: ")

            if choice == "1":
                self.generate_password()
            elif choice == "2":
                self.store_password()
            elif choice == "3":
                self.retrieve_password()
            elif choice == "4":
                self.export_vault()
            elif choice == "5":
                self.current_user = None
                return
            else:
                print("\nInvalid choice. Please try again.")
                time.sleep(1)

    def generate_password(self):
        self.print_banner()
        print("\nPassword Generator:")

        try:
            length = int(input("Length (default 16): ") or "16")
            use_upper = input("Include uppercase? (Y/n): ").lower() != "n"
            use_digits = input("Include digits? (Y/n): ").lower() != "n"
            use_symbols = input("Include symbols? (Y/n): ").lower() != "n"

            result = self.manager.generate_password(
                length=length,
                use_upper=use_upper,
                use_digits=use_digits,
                use_symbols=use_symbols,
            )

            print("\nGenerated Password:")
            print(f"Password: {result['password']}")
            print(f"Strength: {result['analysis']['strength']}/4")
            print(f"Estimated crack time: {result['analysis']['crack_time']}")

            if input("\nCopy to clipboard? (Y/n): ").lower() != "n":
                pyperclip.copy(result["password"])
                print("Password copied to clipboard!")

            if input("\nStore this password? (Y/n): ").lower() == "y":
                service = input("Service name: ")
                username = input("Username (optional): ")
                notes = input("Notes (optional): ")
                tags = input("Tags (comma separated, optional): ").split(",")

                metadata = {
                    "username": username,
                    "notes": notes,
                    "tags": [tag.strip() for tag in tags if tag.strip()],
                }

                self.manager.store_password(service, result["password"], **metadata)
                print("\nPassword stored successfully!")

            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            time.sleep(2)

    def store_password(self):
        self.print_banner()
        print("\nStore Password:")

        try:
            service = input("Service name: ")
            password = getpass("Password: ")
            username = input("Username (optional): ")
            notes = input("Notes (optional): ")
            tags = input("Tags (comma separated, optional): ").split(",")

            metadata = {
                "username": username,
                "notes": notes,
                "tags": [tag.strip() for tag in tags if tag.strip()],
            }

            self.manager.store_password(service, password, **metadata)
            print("\nPassword stored successfully!")
            time.sleep(1)
        except Exception as e:
            print(f"\nError: {str(e)}")
            time.sleep(2)

    def retrieve_password(self):
        self.print_banner()
        print("\nRetrieve Password:")

        try:
            service = input("Service name: ")
            entry = self.manager.get_password(service)

            if not entry:
                print("\nNo password found for this service!")
                time.sleep(1)
                return

            print("\nPassword Details:")
            print(f"Service: {entry.service}")
            print(f"Username: {entry.username}")
            print(f"Password: {entry.encrypted_password.decode()}")
            print(f"Created: {entry.created_at}")
            print(f"Last used: {entry.last_used}")
            print(f"Strength: {entry.strength_score}/4")
            print(f"Complexity: {entry.complexity}")

            if input("\nCopy password to clipboard? (Y/n): ").lower() != "n":
                pyperclip.copy(entry.encrypted_password.decode())
                print("Password copied to clipboard!")

            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            time.sleep(2)

    def export_vault(self):
        self.print_banner()
        print("\nExport Vault:")

        try:
            print("Export formats:")
            print("1. JSON (encrypted)")
            print("2. JSON (plaintext)")
            print("3. CSV (plaintext)")

            choice = input("\nChoose format: ")

            if choice == "1":
                filename = self.manager.export_vault("json", encrypt=True)
            elif choice == "2":
                filename = self.manager.export_vault("json", encrypt=False)
            elif choice == "3":
                filename = self.manager.export_vault("csv", encrypt=False)
            else:
                print("\nInvalid choice")
                time.sleep(1)
                return

            print(f"\nVault exported to: {filename}")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            time.sleep(2)


# ==================== MAIN APPLICATION ====================
if __name__ == "__main__":
    try:
        cli = PasswordManagerCLI()
        cli.main_menu()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)
