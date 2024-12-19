import threading

import cv2
from deepface import DeepFace

# defining video capture  
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) #0 is the  first camera thats conected, DSHOW is direct show  by msf for capturing,processing, rending video

# setting properties
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# We don't want to match every instance so do it once in a while:
counter = 0

faceMatch = False #this boolean is globally accessible

reference_img = cv2.imread("reference.jpg") #loading reference image

#function checks if the reference image face matches the face in the frame
def check_face(frame):
    global faceMatch
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            faceMatch = True
        else:
            faceMatch = False
    except ValueError:
        faceMatch = False


while True:
    ret, frame = capture.read() # we get the frame here from the camera without anyt text

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)) #we pass a copy of frame to check_face function
            except ValueError:
                pass #even if there is no return, keep running 

        counter += 1

##### we get the result and depending on the result we do either  ####


        if faceMatch:
            cv2.putText(frame,"MATCH!",(20,450),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),3) #writes in green color that there is a match
        else:
            cv2.putText(frame,"No Match!",(20,450),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,0,255),3) # writes in red color there is no match    

        cv2.imshow("screen", frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()