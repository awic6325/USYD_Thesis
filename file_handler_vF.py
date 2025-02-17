import datetime
import PyCapture2
import numpy as np
import os
from pathlib import Path

# Get current time
def get_timestamp():

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # up to milliseconds

    return timestamp

def generate_filename(prefix, pair_number, timestamp):

    pair_str = f"pair{str(pair_number).zfill(4)}"

    filename = f"{prefix}_{pair_str}_{timestamp}"

    return filename

# Function for saving spectrum
def save_spectrum(spectrum, save_dir, filename):
    
    np.savetxt(f"{save_dir}/{filename}.csv", spectrum, delimiter=",")

# Function for saving image
def save_image(img_data, save_dir, filename):

    img_data.save(f"{save_dir}/{filename}.png".encode('utf-8'), PyCapture2.IMAGE_FILE_FORMAT.PNG)

# Set the working directory to the directory of the current script
def set_working_directory_to_script_location():

    # Get the absolute path of the current script
    script_directory = Path(__file__).parent.resolve()
    
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    print(f"\nWorking directory changed to: {os.getcwd()}")


# Function for siloing experimental data
def create_new_session_folder(data_dir="../data", metadata=""):

    # Ensure the working directory is the script's directory
    set_working_directory_to_script_location()
    
    # Get the current date in the format YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # Get a list of existing directories under the data folder
    data_path = Path(data_dir).resolve()
    existing_sessions = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith(current_date + "_session")]

    # Extract the session numbers from existing directories
    session_numbers = [int(d.name.split("_session")[1].split("_")[0]) for d in existing_sessions if "_session" in d.name]

    # Determine the next session number (increment from the highest existing session number)
    next_session_number = max(session_numbers, default=0) + 1

    # Format metadata for inclusion in the folder name
    metadata_suffix = f"_{metadata}" if metadata else ""

    # Create the new session folder name with metadata
    new_session_folder = data_path / f"{current_date}_session{next_session_number}{metadata_suffix}"

    # Create the new directory
    new_session_folder.mkdir(parents=True, exist_ok=True)

    print(f"New session folder created: {new_session_folder}")

    #new_session_folder_str = str(new_session_folder)
    #return new_session_folder_str

    return str(new_session_folder.resolve())

def format_path(path):
    # Replace each backslash with a double backslash
    return path.replace("\\", "\\\\")