import cv2

# Stop Sign Cascade Classifier xml
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

# Replace 'your_image_path.jpg' with the path to your image
img = cv2.imread('stop_landscape.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

# Detect the stop sign, x,y = origin points, w = width, h = height
for (x, y, w, h) in stop_sign_scaled:
    # Draw rectangle around the stop sign
    stop_sign_rectangle = cv2.rectangle(img, (x, y),
                                        (x + w, y + h),
                                        (0, 255, 0), 3)
    # Write "Stop sign" on the bottom of the rectangle
    stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                 text="",
                                 org=(x, y + h + 30),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1, color=(0, 0, 255),
                                 thickness=2, lineType=cv2.LINE_4)

cv2.imshow("Stop Sign Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()