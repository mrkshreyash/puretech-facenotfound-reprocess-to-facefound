import face_recognition
import os
import shutil
import time
from colorama import Fore
import imghdr
from PIL import Image
import numpy as np
import re

# Program Initial Time
program_started = time.time()

print("System Started...\n")
# Function to check if a person's face matches the reference image
def is_match(reference_encoding, face_encoding, tolerance=0.6):
    return face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=tolerance)[0]

def sanitize_filename(filename):
    return re.sub(r'\x00', '', filename)

print("\nWait for some time.... Creating Encodings for Target Image....\n")

image_name = "TARGET.JPG"
# Load the reference image (the person you want to match)
reference_image = face_recognition.load_image_file(image_name)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Directory containing images
images_directory = "null_directory"

# Output directory for matched images
output_directory = "captured_from_null_directory"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

total_files, face_found, face_not_found = 0, 0, 0

for filename in os.listdir(images_directory):
    filename = sanitize_filename(filename)
    image_path = os.path.join(images_directory, filename)
    total_files += 1

    # Use imghdr to check if the file is an image
    image_type = imghdr.what(image_path)

    if image_type not in ['jpeg', 'png', 'gif', 'bmp', 'jpg']:
        print(f"{Fore.YELLOW}Skipped file : {filename} (Possibly Not an image)")
        print(Fore.RESET)
        continue

    try:
        # initial time
        t0 = time.time()   

        # Load the image
        current_image = face_recognition.load_image_file(image_path)

        # Print the shape of the original image
        print(f"\n==========> ACCESSED {filename} <==========")
        print(f"Original Image Shape: {current_image.shape}\n")

        # Resize the image
        percentage = 0.2
        resized_height = int(current_image.shape[0] * percentage)
        resized_width = int(current_image.shape[1] * percentage)
        resized_image = np.array(Image.fromarray(current_image).resize((resized_width, resized_height), Image.LANCZOS))

        # Print the shape of the resized image
        print(f"Resized Image Shape: {resized_image.shape}\n")

        # Face detection on resized image
        face_locations = face_recognition.face_locations(resized_image)
        face_encodings = face_recognition.face_encodings(resized_image, face_locations)

        # Print the number of face encodings detected
        print(f"Processing {filename}: {len(face_encodings)} face(s) detected\n")

        for i, (top, right, bottom, left) in enumerate(face_locations):
            confidence = face_recognition.face_distance([reference_encoding], face_encodings[i])[0]
            print(f"Face {i + 1}: Confidence = {confidence:.4f}")

        # Check if a face is found in the image
        if len(face_encodings) > 0:
            # Check if the face matches the reference image
            if is_match(reference_encoding, face_encodings[0]):
                print(f"{Fore.GREEN}Face match found in {filename}")
                print(Fore.RESET, end="")
                face_found += 1

                # final time
                t1 = time.time()

                print(f"{Fore.BLUE}Time Taken to detect and process : {t1-t0} sec.\n")
                print(Fore.RESET)

                # Copy the matched image to the output directory
                output_path = os.path.join(output_directory, filename)
                resized_image_dir = os.path.join(output_directory, "resized")
                resized_output_path = os.path.join(output_directory, "resized", filename)
                # if not os.path.exists(resized_image_dir):
                #     os.makedirs(resized_image_dir)
                # Image.fromarray(resized_image).save(resized_output_path)

            else:
                print(f"{Fore.RED}Face match not found in {filename}\n")
                print(Fore.RESET, end="")
                face_not_found += 1

        else:
            print(f"{Fore.YELLOW}No faces found in {filename}. Storing in null_directory.\n")
            print(Fore.RESET)
            null_directory = "null_directory"
            if not os.path.exists(null_directory):
                os.makedirs(null_directory)
            shutil.copy(image_path, os.path.join(null_directory, filename))

    except Exception as e:
        print(f"Error processing {filename}: {e}\n")

# ...

print("Matching process completed.")

print(f"""
TOTAL FILES : {total_files}
FACE FOUND  : {face_found}
FACE NOT FOUND : {face_not_found}
""")

# Program Ended time
program_end = time.time()

print(f"\nTotal Time Taken : {program_end - program_started}\n")
