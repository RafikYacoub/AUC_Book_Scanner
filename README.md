# AUC_Book_Scanner
It's a tool developed by Rafik Yacoub, Yehia El Kassas, and Shahed Elmahallawy, for AUC Libraries to identify the new books whether they are duplicates or not. This is done by scanning their ISBN, Barcode, or Cover Pages.

# Executive Summary
The "Book Scanner" project was initiated to address a challenge faced by AUC Libraries. To elaborate more, every book has to be searched manually in the catalog, to ensure that holdings are not duplicated. This report outlines the development and implementation of an innovative desktop application designed to recognize books in the library's holdings using either ISBN numbers, barcodes or title pages. By leveraging Optical Character Recognition (OCR) technology and computer vision techniques, the app has significantly reduced the manual cataloging workload, leading to improved catalog accuracy.

The project successfully achieved its primary objectives, including the development of an app capable of recognizing both ISBN numbers and title pages. The app is equipped with a warning system that triggers alerts when a book is found to be already present in the library's collection. Additionally, an advanced training feature was added, allowing users to contribute title pages to the database and rate the similarity of scanned title pages.

# Introduction
# Background
AUC Libraries face an annual influx of thousands of book donations, resulting in a substantial challenge: the labor-intensive task of manually searching for duplicates within their collection. In response to this challenge, the "Book Cataloging" project was launched with the primary objective of developing an application to streamline this process and reduce the burden on library staff.
# Objectives
The central objectives of this project are twofold: firstly, to create an application capable of recognizing books within the library's holdings using either ISBN numbers, barcodes or title pages; and secondly, to implement a warning system that identifies and flags books already present in the catalog. In case the scanned book is not identified as a duplicate, it will be added to the database, and in a new dataset that stores new collections.

In this report, we provide a comprehensive overview of the methodology, results, and future recommendations of the "Book Cataloging" project, with the aim of providing insight into its impact on library operations and its potential for broader implementation.
# Methodology
# Data Collection
We used two datasets from Kaggle; the first dataset contains images of book covers and a CSV file that contains the ISBN of the books. All of those books are English and they are about 32K cover page. The second dataset has many Arabic documents, but we took only the cover pages of books and we add it tothe first dataset. There about 825 books.
https://www.kaggle.com/datasets/lukaanicin/book-covers-dataset
https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset 
# App Development
First and foremost, we performed image preprocessing (Normalize, Noise reduction, gray scale conversion, gaussian blurring) to enhance the quality of the image to capture more details that might have been blurred.
- ISBN detection: we implemented an OCR (Optical Character Recognition) that captures the numbers beside the word ISBN, and the program searches for the number in the CSV file to find if it already exists. If so, a warning is displayed for the user that a duplicate has been found; otherwise, the message is output to let the user know that a new book has been scanned and added to the database. The text detection is done using tesseract OCR. Then we extract ISBN like patterns using regular expressions
- Barcode detection:  we start by preprocessing through Resizing image, converting to grayscale, calculating gradient, gradient subtraction, converting gradient to absolute values, bluring the image, thresholding to create a binary image, morphological operations, erosion and dilation, finding contours, sorting and selecting largest contour, bounding box and drawing, crop and decode barcode, the message is output to let the user know that a new book has been scanned and added to the database (the same csv file mentioned before since the barcode is a graphical representation of the ISBN).
- Cover page detection: we Calculate histogram, calculate SSIM (Structural Similarity Index Measure), FLANN-Matcher using SIFT features”, FLANN is a library for performing fast approximate nearest neighbour searches in high dimensional spaces, search for closest matches of query book-cover from the matches. Display the top predicted image to the user and ask if it is the same book, if so, it is a duplicate; otherwise, new book added to the database (pkl file and dataset folder).

Hyperparameters for cover page training: correl threshold for comparing histogram, similarity index threshold for SSIM comparisons, limit = 100
#Hyper-Parameters for SIFT comparison
sift_features_limit = 1000
lowe_ratio = 0.75
predictions_count = 4
#Hyper-Parameters to display results
query_image_number = 1


# Results
# ISBN detection:
# Barcode detection:
# Cover Page detection:

# Discussion
One of the downsides of the app is the inaccuracy of edge detection, which might produce incorrect results for the cover page detection.

It’s suggested to have a deep learning CNN model to handle the cover page detection and matching but it requires machines with power processing power which weren’t accessible for us to implement. 

We also tried a script to cut the cover page, removing the background, to improve the accuracy but it didn’t give the intended results but we will upload it to the repo for further and future development.


# Recommendations
For future references, we recommend using deep learning to improve the results and the dataset each time the user
Invest in ongoing research and development to refine the similarity measurement algorithm for title page recognition.
Explore advanced computer vision techniques and machine learning models to improve accuracy.
Ensure robust data security and privacy measures to protect sensitive information within the app, particularly when handling user-contributed data.


# Conclusion
The "Book Cataloging" project has successfully addressed the challenge of identifying duplicate books in AUC Libraries' collection. The developed app, capable of recognizing ISBN numbers and title pages, has streamlined the cataloging process. With continued efforts to improve the database and algorithms, the system will become even more efficient and accurate in the future.


