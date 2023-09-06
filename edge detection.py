import cv2
import numpy as np

def is_rectangle(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return len(approx) == 4

image_path = "your_image_path.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
edges = cv2.Canny(blurred_image, threshold1=10, threshold2=250)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
book_contours = [cnt for cnt in contours if is_rectangle(cnt)]

if book_contours:
    largest_book_contour = max(book_contours, key=cv2.contourArea)
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [largest_book_contour], -1, (0, 255, 0), 4)
    cropped_book_cover = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Cropped Book Cover", cropped_book_cover)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No book cover found.")
