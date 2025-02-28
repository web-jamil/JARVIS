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
import cmath
import numpy as np
import sympy as sp

logo="""                                          
                                        
          ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██║██╔═══╝ ██║   ██║██║╚════██║
 ╚█████╔╝██║  ██║██║     ╚██████╔╝██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝╚══════╝

   """


# Initialize the engine
engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)


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

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()


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
        speak("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None
    except Exception as e:
        speak(f"An error occurred: {e}")
        print(f"Error: {e}")
        return None

    except sr.UnknownValueError:
        speak("Sorry, I didn't understand that. Can you say that again?")
        return None
    except sr.RequestError:
        speak("The speech recognition service is unavailable. Please check your internet connection.")
        return None
    except Exception as e:
        speak(f"An unexpected error occurred: {e}")
        print(f"Error: {e}")
        return None 


""" def play_music(song_name=None) -> None:
    #Plays music from the user's Music directory.
    song_dir = os.path.expanduser("~\\Music")
    songs = os.listdir(song_dir)

    if song_name:
        songs = [song for song in songs if song_name.lower() in song.lower()]

    if songs:
        song = random.choice(songs)
        os.startfile(os.path.join(song_dir, song))
        speak(f"Playing {song}.")
        print(f"Playing {song}.")
    else:
        speak("No song found.")
        print("No song found.")
 """
 
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



def play_music(song_name=None) -> None:
    """Plays music from the user's Music directory and Desktop Music directory."""
    
    # Get the path to the Music folder in the user's directory
    user_music_dir = os.path.expanduser("~\\Music")
    
    # Get the path to the Music folder on the Desktop (if it exists)
    desktop_music_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Music")
    
    # Create a list to store all music files found
    all_songs = []
    
    # Check if the Music directory in the user's folder exists and add songs from it
    if os.path.exists(user_music_dir):
        all_songs.extend(os.listdir(user_music_dir))
    
    # Check if the Music directory on the Desktop exists and add songs from it
    if os.path.exists(desktop_music_dir):
        all_songs.extend(os.listdir(desktop_music_dir))

    # Filter songs if song_name is provided
    if song_name:
        all_songs = [song for song in all_songs if song_name.lower() in song.lower()]

    # If we have any songs, play one of them
    if all_songs:
        song = random.choice(all_songs)  # Pick a random song from the available list
        # Find the full path to the song
        if song in os.listdir(user_music_dir):
            song_path = os.path.join(user_music_dir, song)
        else:
            song_path = os.path.join(desktop_music_dir, song)
        
        os.startfile(song_path)  # Play the song
        speak(f"Playing {song}.")
        print(f"Playing {song}.")
    else:
        speak("No song found.")
        print("No song found.")



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
    """
    Opens popular websites based on the user's query.
    """
    websites = {
        "microsoft": "https://www.microsoft.com",
        "instagram": "https://www.instagram.com",
        "facebook": "https://www.facebook.com",
        "stackoverflow": "https://stackoverflow.com",
        "stack exchange": "https://stackexchange.com",
        "kaggle": "https://www.kaggle.com",
        "github": "https://github.com",
        "reddit": "https://www.reddit.com",
        "youtube": "https://www.youtube.com",
        "chat gpt": "https://chat.openai.com",
        "google": "https://www.google.com",
        "hugging face": "https://huggingface.co",
        "leetcode": "https://leetcode.com",
        "geeks for geeks": "https://www.geeksforgeeks.org",
        "w3schools": "https://www.w3schools.com"
    }

    # Normalize query and search for a match in the dictionary
    for key, url in websites.items():
        if key in query.lower():
            speak(f"Opening {key}.")
            wb.open(url)
            return True

    speak("Sorry, I couldn't find a matching website.")
    return False
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
        
print("Welcome to jarvis AI.")

print(logo)
if __name__ == "__main__":
    #run the program
    # main()
    wishme()

    while True:
        query = takecommand()
        if not query:
            continue

        elif "time" in query:
            # time()
            advanced_time_functionality()

        elif "date" in query:
            date()

        elif "wikipedia" in query:
            query = query.replace("wikipedia", "").strip()
            search_wikipedia(query)
        elif "calculate" in query or "what is" in query or "math" in query:
            calculate()

        elif "play music" in query:
            speak("opening music")
            song_name = query.replace("play music", "").strip()
            play_music(song_name)

        elif "open microsoft" in query:
            speak("Opening Microsoft.")
            wb.open("https://www.microsoft.com")

        elif "open instagram" in query:
            speak("Opening Instagram.")
            wb.open("https://www.instagram.com")

        elif "open facebook" in query:
            speak("Opening Facebook.")
            wb.open("https://www.facebook.com")

        elif "open stackoverflow" in query:
            speak("Opening StackOverflow.")
            wb.open("https://stackoverflow.com")
            
        elif "open stack exchange" in query:
            speak("Opening Stack Exchange.")
            wb.open("https://stackexchange.com")
            
        elif "open kaggle" in query:
            speak("Opening Kaggle.")
            wb.open("https://www.kaggle.com")
            
        elif "open github" in query:
            speak("Opening GitHub.")
            wb.open("https://github.com")
            
        elif "open reddit" in query:
            speak("Opening Reddit.")
            wb.open("https://www.reddit.com")


        elif "open youtube" in query:
            speak("opening Youtube ")
            wb.open("youtube.com")
            
        elif "open chat gpt" in query:
            speak("Opening ChatGPT.")
            wb.open("https://chat.openai.com")
            
        elif "open google" in query:
            speak("opening Google")
            wb.open("google.com")

        elif "change your name" in query:
            speak("changing my name ")
            set_name()

        elif "screenshot" in query:
            screenshot()
            speak("I've taken screenshot, please check it")

        elif "news" in query:
            speak("Here are some top news headlines")
            get_news()
            

        elif "manage" in query:
            speak("opening task manager")
            manage_tasks()

        elif "calculate" in query:
            speak("opening calculator")
            calculate()

        elif "motivate" in query:
            speak("opening motivational quote")
            motivational_quote()

        elif "tell me a joke" in query:
            speak("tell me a joke")
            tell_joke()
      

        elif "control system" in query:
            speak("Starting system control. What would you like to do?")
            # Listen for the next command
            query = takecommand()
            # Pass the new query to control_system
            control_system(query)

        elif "shutdown" in query:
            speak("Shutting down the system, goodbye!")
            os.system("shutdown /s /f /t 1")
            break

        elif "restart" in query:
            speak("Restarting the system, please wait!")
            os.system("shutdown /r /f /t 1")
            break

        elif "offline" in query or "exit" in query:
            speak("Going offline. Have a good day!")
            break

