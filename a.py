import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Initialize global variables
rectangles = []
current_rect = None

def on_select(eclick, erelease):
    global current_rect
    start_point = (int(eclick.xdata), int(eclick.ydata))
    end_point = (int(erelease.xdata), int(erelease.ydata))
    current_rect = (start_point, end_point)

    # Draw the rectangle on the image
    rect_img = img.copy()
    cv2.rectangle(rect_img, start_point, end_point, (0, 255, 0), 2)
    
    # Add text annotation with coordinates
    text = f"Start: {start_point}, End: {end_point}"
    cv2.putText(rect_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Convert BGR to RGB for matplotlib display
    rect_img_rgb = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(rect_img_rgb)
    ax.set_title('Select Multiple ROIs')
    ax.axis('off')
    plt.draw()

def on_key(event):
    global current_rect
    if event.key == 'enter':
        if current_rect:
            rectangles.append(current_rect)
            print(f"Saved ROI: Start {current_rect[0]}, End {current_rect[1]}")
            current_rect = None  # Clear current rectangle selection
        else:
            print("No rectangle selected to save.")

def save_rectangles(file_path):
    with open(file_path, 'w') as f:
        for start, end in rectangles:
            f.write(f"Start: {start}, End: {end}\n")

# Load image
image_path = 'D:\\ftest - Copy\\image.jpg'  # Change this to your image path
img = cv2.imread(image_path)

if img is None:
    print(f"Error: The image file at {image_path} could not be opened.")
else:
    # Convert BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a figure and axis for displaying the image
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title('Select Multiple ROIs')
    ax.axis('off')

    # Create a RectangleSelector for selecting ROIs
    rectangle_selector = RectangleSelector(ax, on_select,
                                           button=[1],  # Left mouse button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    
    # Connect the key press event to the on_key function
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Display the image and wait for user interaction
    plt.show()

    # Save rectangles to file
    save_path = 'rectangles.txt'  # File where rectangles will be saved
    save_rectangles(save_path)
    print(f"Rectangles saved to {save_path}")
