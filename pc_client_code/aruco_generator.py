import cv2
import os
import numpy as np

def generate_markers():
    # --- CONFIGURATION ---
    # Must match the dictionary used in your tracker
    DICT_TYPE = cv2.aruco.DICT_4X4_50
    
    # Folder to save images
    OUTPUT_FOLDER = "tags"
    curr_dir = os.getcwd()
    OUTPUT_FOLDER = os.path.join(curr_dir, OUTPUT_FOLDER)
    
    # Size of the output image in pixels (high res for printing)
    SIZE_PX = 600
    
    # Which IDs do you want? (e.g., 0-3 for table corners)
    IDS_TO_GENERATE = list(np.arange(8))
    print(IDS_TO_GENERATE)

    # --- SETUP ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    
    print(f"Generating ArUco markers (DICT_4X4_50) into '{OUTPUT_FOLDER}'...")

    # --- GENERATION LOOP ---
    for marker_id in IDS_TO_GENERATE:
        # Generate the marker image
        # generateImageMarker(dictionary, id, size, borderBits)
        # borderBits=1 adds a thin black border, but we usually print on white paper anyway.
        tag_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, SIZE_PX)
        
        # Add a white margin for easier cutting/printing (Optional)
        # Creates a white canvas slightly larger and places tag in center
        margin = 50
        full_size = SIZE_PX + (margin * 2)
        canvas = np.ones((full_size, full_size), dtype=np.uint8) * 255
        canvas[margin:margin+SIZE_PX, margin:margin+SIZE_PX] = tag_img
        
        # Save
        filename = os.path.join(OUTPUT_FOLDER, f"id_{marker_id}.png")
        cv2.imwrite(filename, canvas)
        print(f" - Saved: {filename}")
        
    print("\nDone! Print these out and place them on your table.")
    print("Tip: Ensure the white border around the black square is visible when you cut them out.")

if __name__ == "__main__":
    generate_markers()