import cv2
import cv2.aruco as aruco

# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters_create()

# Initialize the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Start the webcam feed
cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Outline the markers
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    # Display the resulting frame
    cv2.imshow('frame', frame_markers)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
