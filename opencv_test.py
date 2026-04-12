import cv2
img = cv2.imread('images/img.jpg')
cv2.imshow("Image", img)
cv2.waitKey(0) # Waits for a key press
cv2.destroyAllWindows()