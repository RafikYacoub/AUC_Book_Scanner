import tkinter as tk
import tkinter.ttk as ttk
import cv2 
from PIL import Image, ImageTk
import pytesseract
import re
import isbnlib
import numpy as np
import os
import imutils 
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from scipy.ndimage import rotate
from tkinter import messagebox
from pyzbar import pyzbar
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from tkinter import simpledialog, messagebox, Toplevel, Label,Button, Tk
from PIL import ImageTk, Image

# Parameter for comparing histograms
correl_threshold = 0.9

# Parameters for SSIM comparison
similarity_index_threshold = 0.0
ssim_matches_limit = 2

# Parameters for SIFT comparision
sift_features_limit = 1000
lowe_ratio = 0.75
predictions_count = 2

# Parameters to display results
query_image_number = 0
amazon_reviews_count = 3


def similarity_index(q_path,m_path):
    q_i = cv2.imread(q_path,0)
    q_i = cv2.resize(q_i,(8,8))
    m_i = cv2.imread(m_path,0)
    m_i = cv2.resize(m_i,(8,8))
    return ssim(q_i,m_i)


def gen_sift_features(image):
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


class CustomDialog:
    def __init__(self, parent, image):
        self.parent = parent
        self.image = image
        self.result = None

        self.top = tk.Toplevel(parent)
        self.top.title("Image Prediction")
        
        # Display the predicted image
        self.canvas = tk.Canvas(self.top, width=image.shape[1], height=image.shape[0])
        self.canvas.pack()
        self.display_image(image)

        # Create "Yes" and "No" buttons
        self.yes_button = tk.Button(self.top, text="Yes", command=self.set_result_true)
        self.no_button = tk.Button(self.top, text="No", command=self.set_result_false)
        
        self.yes_button.pack()
        self.no_button.pack()

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

    def display_image(self, image):
        # Convert BGR image to RGB for displaying in tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb , (400,400))

        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Create a PhotoImage from the PIL Image
        self.photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update the canvas image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_result_true(self):
        self.result = True
        self.top.destroy()

    def set_result_false(self):
        self.result = False
        self.top.destroy()

    def on_close(self):
        # Handle window close event (e.g., clicking the X button)
        self.result = False
        self.top.destroy()


class CameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("AUC Library | Book Scanner")
        self.window.configure(bg="#023047")
        self.window.resizable(True, True)

        # Open the video source
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        # Create a canvas for displaying the video
        self.canvas = tk.Canvas(window, bg="#023047") 
        # Canvas resizes with the window
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Button to capture ISBN
        self.capture_isbn = tk.Button(
            window, text="Capture ISBN", command=self.capture_ISBN, bg="#248415", fg="white")
        self.capture_isbn.pack()
        # Button to capture Barcode
        self.capture_barcode = tk.Button(
            window, text="Capture Barcode", command=self.capture_barcode, bg="#248415", fg="white")
        self.capture_barcode.pack()
        # Button to capture cover page
        self.capture_cover = tk.Button(
            window, text="Capture Cover Page", command=self.capture_cover, bg="#248415", fg="white")
        self.capture_cover.pack()

        self.stop_button = tk.Button(
            window, text="Stop Scanning", command=self.return_to_main_window, bg="red", fg="white")
        self.stop_button.pack()

        # After configuring the GUI elements, start the video playback
        self.update()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame to RGB and resize it to fit the canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get the current size of the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            # Resize the frame to fit the canvas
            frame_resized = cv2.resize(
                frame_rgb, (canvas_width, canvas_height))

            # Create an ImageTk object from the resized frame
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

        # Schedule the next update
        self.window.after(15, self.update)

    #############################################################################################################
    # ISBN Detection part
    #############################################################################################################

    def capture_ISBN(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Save the frame as an image file
            cv2.imwrite("query\captured_image.jpg", frame)
            print("Image captured!")

            # Call the backend function to process the captured image
            self.process_image_ISBN(frame)

    def process_image_ISBN(self, image):
        # Preprocess the image
        # Normlizing
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

        # noise
        new_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform text detection using pytesseract

        text = pytesseract.image_to_string(gray)

        print(text)
        # Extract ISBN-like patterns using regular expressions
        isbn_patterns = re.findall(
            r"\b(?:ISBN(?:-10)?:? )?((?:[\dX]+[- ]?){9}[\dX])\b", text)

        # Validate and print the extracted ISBN numbers
        for isbn in isbn_patterns:
            normalized_isbn = isbnlib.canonical(isbn)
            if isbnlib.is_isbn13(normalized_isbn) or isbnlib.is_isbn10(normalized_isbn):
                messagebox.showinfo("Valid ISBN:", isbn)
                self.check_isbn_duplicate(isbn)  # Pass single isbn value
                print("ISBN", isbn)
            else:
                messagebox.showinfo("Invalid ISBN:", isbn)
                print("ISBN", isbn)
        print("isbn_patterns", isbn_patterns)

        if len(isbn_patterns) == 0:
            messagebox.showinfo("ISBN Result", "Couldn't capture ISBN. Please, try again!")

        self.return_to_main_window()

    def check_isbn_duplicate(self, isbn):
        # Remove dashes from isbn
        isbn = isbn.replace('-', '')

        # Load the dataset
        data = pd.read_csv('books.csv')

        # Convert float representations to proper ISBN string
        data['isbn'] = data['isbn'].apply(lambda x: str(int(x)) if pd.notnull(x) else 'NaN')

        # Check if ISBN exists in the dataset
        print("Checking ISBN in dataset:")
        print(data['isbn'].values)
        print("ISBN to check: ", isbn)
        if isbn in data['isbn'].values:
            messagebox.showinfo("Search Result", "A duplicated book.")
        else:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            messagebox.showinfo("Search Result", "A new book. Added it to the datset.")
            # Add the new book to the dataset
            new_book = pd.DataFrame({'isbn': [isbn]}, columns=data.columns)
            new_book.fillna(np.nan, inplace=True)
            data = pd.concat([data, new_book])
            # Save the dataset
            print("Saving updated dataset.")
            data.to_csv('books.csv', index=False)
            new_books_data = pd.DataFrame({"time" : [timestamp],"ISBN" : [isbn]})
            new_books_data.to_csv("new_books.csv", mode = 'a', header = False, index = False)

            print("Updated dataset saved.")


    #############################################################################################################
    # Barcode Detection part
    #############################################################################################################
    def capture_barcode(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Save the frame as an image file
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured!")

            # Call the backend function to process the captured image
            self.process_image_barcode(frame)

    def process_image_barcode(self, image):

        # best approach
        show = 1
        # resize image
        image = cv2.resize(image, None, fx=0.7, fy=0.7,
                           interpolation=cv2.INTER_CUBIC)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # calculate x & y gradient
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blur the image
        blurred = cv2.blur(gradient, (3, 3))

        # threshold the image
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)


        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts, hierarchy = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        c1 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # draw a bounding box arounded the detected barcode and display the
        # image
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        # Crop the barcode region and convert it to grayscale for decoding
        (x, y, w, h) = cv2.boundingRect(c)
        barcode_roi = gray[y:y + h, x:x + w]


        # Decode the barcode using pyzbar
        decoded_objects = pyzbar.decode(barcode_roi)

        # Extract and display the barcode digits
        for obj in decoded_objects:
            barcode_data = obj.data.decode("utf-8")
            print("Barcode Data:", barcode_data)
            messagebox.showinfo("Barcode Result", barcode_data)
            if isbnlib.is_isbn13(barcode_data) or isbnlib.is_isbn10(barcode_data):
                print("Valid ISBN")
                self.check_isbn_duplicate(barcode_data)
            else:
                print("Not a valid ISBN")
            self.return_to_main_window()
        
        if len(decoded_objects) == 0:
            messagebox.showinfo("Barcode Result", "Couldn't capture the Barcode. Please, try again!")

        image = cv2.resize(image, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_CUBIC)


        self.return_to_main_window()
    


    #############################################################################################################
    # Cover page Detection part
    #############################################################################################################
    
    def capture_cover(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
           
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            file_name ='query\captured_image_'+str(timestamp) +'.jpg'
            # Save the frame as an image file
            cv2.imwrite(file_name, frame)
            print("Image captured!")
            captured_image = cv2.imread(file_name)
            # Call the backend function to process the captured image and get the path of the predicited matching image
            prediction_path = self.hist(frame)

            #show the predicted image and prompt the user to chose if this is the image or not
            print(prediction_path)
            if isinstance(prediction_path, str):
                predicted_image = cv2.imread(prediction_path)
                if predicted_image is not None:
                    response = CustomDialog(self.window, predicted_image)
                    self.window.wait_window(response.top)
                    if response.result:
                        print("Matching image found!")
                        messagebox.showinfo("Info", "Matching image found!")
                        self.return_to_main_window()
                    else:
                        with open('train_hist_data.pkl', 'ab') as f:
                            hist = cv2.calcHist([predicted_image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
                            hist = cv2.normalize(hist, None)
                            pickle.dump(hist, f)
                        messagebox.showinfo("Info", "A new image found, added to the dataset!")
                        self.return_to_main_window()
            else:
                with open('train_hist_data.pkl', 'ab') as f:
                    hist = cv2.calcHist([captured_image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, None)
                    pickle.dump(hist, f)
                    messagebox.showinfo("Info", "Couldn't find the image. Added to the dataset.")
                self.return_to_main_window()

    #image parameter can be removed but i am not sure yet
    def hist(self, image):
        query_path = "query"
        query_paths = [os.path.join(query_path, f) for f in os.listdir(query_path)]
        #print(len(query_paths))

        with open('train_hist_data.pkl', 'rb') as f:
            hist_train = pickle.load(f)

        hist_query = []
        for path in query_paths:
            image = cv2.imread(path)
            
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # extract a 3D RGB color histogram from the image,
            # using 8 bins per channel, normalize, and update the index
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, None)
            hist_query.append((path,hist))
            hist_matches = []


        for i in range(len(hist_query)):
            matches = []
            for j in range(len(hist_train)):
                cmp = cv2.compareHist(hist_query[i][1], hist_train[j][1], cv2.HISTCMP_CORREL)
                if cmp > correl_threshold:
                    matches.append((cmp,hist_train[j][0]))
            matches.sort(key=lambda x : x[0] , reverse = True)
            hist_matches.append((hist_query[i][0],matches))




        ssim_matches = []

        for i in range(len(hist_matches)):
            query_image_path = hist_matches[i][0]
            #print("1",query_image_path)
            matches = []
            for j in range(len(hist_matches[i][1])):
                match_image_path = hist_matches[i][1][j][1]
                #print("2",match_image_path)
                si = similarity_index(query_image_path,match_image_path)
                if si > similarity_index_threshold:
                    matches.append((si,match_image_path))
            matches.sort(key=lambda x : x[0] , reverse = True)
            ssim_matches.append((query_image_path,matches[:ssim_matches_limit]))
        
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)


        predictions = []
        for i in range(len(ssim_matches)):
            matches_flann = []
            # Reading query image
            q_path = ssim_matches[i][0]
            q_img = cv2.imread(q_path)
            if q_img is None:
                continue
            q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
            # Generating SIFT features for query image
            q_kp,q_des = gen_sift_features(q_img)
            if q_des is None:
                continue
            
            for j in range(len(ssim_matches[i][1])):
                matches_count = 0
                m_path = ssim_matches[i][1][j][1]
                m_img = cv2.imread(m_path)        
                if m_img is None:
                    continue
                m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
                # Generating SIFT features for predicted ssim images
                m_kp,m_des = gen_sift_features(m_img)
                if m_des is None:
                    continue
                # Calculating number of feature matches using FLANN
                matches = flann.knnMatch(q_des,m_des,k=2)
                #ratio query as per Lowe's paper
                matches_count = 0
                for x,(m,n) in enumerate(matches):
                    if m.distance < lowe_ratio*n.distance:
                        matches_count += 1
                matches_flann.append((matches_count,m_path))
            matches_flann.sort(key=lambda x : x[0] , reverse = True)
            predictions.append((q_path,matches_flann[:predictions_count]))

        #print("here",len(predictions))
        final_matches=[]
        for i in range(len(predictions)):
            for j in range(len(predictions[i][1])):
                final_matches.append(predictions[i][1][j][1].replace("\\\\",'\\'))



        #return the path of the 1st prediciton
        if len(final_matches) > 0:
            return final_matches[0]
        else:
            return False

    def return_to_camera_window(self):
        self.window.deiconify()  # Show the camera window
        self.canvas.focus_set()  # Set focus to canvas

    def return_to_main_window(self):
        # Close the camera window and return to the main window
        self.vid.release()
        self.window.destroy()

class MainWindow:
    def __init__(self, window):
        self.window = window
        self.window.title("AUC Library | Book Scanner")
        self.window.configure(bg="#023047")  # Set window background color

        # Welcome message label
        self.welcome_label = tk.Label(window, text="Welcome to AUC Library Book Scanner!", font=(
            "Arial", 40), bg="#023047", fg="white")  # Set label background color
        self.welcome_label.pack(pady=100)

        # Button to Scan ISBN
        self.scan = tk.Button(window, text="Start Scanning", command=self.open_camera_window, font=(
            "Arial", 24), bg="#248415", fg="white")  # Set button background and foreground colors
        self.scan.pack(pady=40)

    def open_camera_window(self):
        # Create the camera app window


        camera_window = tk.Toplevel(self.window)
        camera_window.title("Scanning Window")
        # Width x Height + X position + Y position
        camera_window.geometry("500x500+450+80")
        camera_window.resizable(False, False)  # Disable window resizing

        app = CameraApp(camera_window)

        


    def return_to_main_window(self):
        # Close the camera window
        self.window.quit()

# Main


# Create the tkinter main window
main_window = tk.Tk()

# Set the main window size and position
# Width x Height + X position + Y position
main_window.geometry("1280x720+120+50")
# main_window.resizable(False, False)  # Disable window resizing

# Create the main window
main_app = MainWindow(main_window)

# Start the tkinter event loop
main_window.mainloop()
