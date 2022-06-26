import cap as cap
from scipy.spatial import distance
# !pip install imutils
import imutils
from imutils import face_utils
import dlib
import cv2


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    L = (A + B + C) / 3
    D = distance.euclidean(mouth[0], mouth[6])
    mar = L / D
    return mar


def helper():
    # Eyes and mouth threshold value
    eyeThresh = 0.25
    mouthThresh = 0.60

    # frame to check
    frame_check_eye = 5
    frame_check_mouth = 5

    # Initializing the Face Detector object
    detect = dlib.get_frontal_face_detector()

    # Loading the trained model
    predict = dlib.shape_predictor("C:/Users/jhuro/Desktop/drowsiness/DrowsinessDetection-master/shape_predictor_68_face_landmarks.dat")

    # Getting the eyes and mouth index
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    # Initializing the Video capturing object
    cap = cv2.VideoCapture(0)

    # Initializing the flags for eyes and mouth
    flag_eye = 0
    flag_mouth = 0

    # Calculating the Euclidean distance between facial landmark points of eyes and mouth
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=800, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mar = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)

            # Drawing the overlay on the face
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth], -1, (255, 0, 0), 1)
            cv2.putText(frame, "Eye Aspect Ratio: {}".format(ear), (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Mouth Aspect Ratio: {}".format(mar), (5, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Comparing threshold value of Mouth Aspect Ratio (MAR)
            if mar > mouthThresh:
                flag_mouth += 1
                if flag_mouth >= frame_check_mouth:
                    cv2.putText(frame, "****************** SUBJECT IS YAWNING *******************", (10, 370),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag_mouth = 0

            # Comparing threshold value of Eye Aspect Ratio (EAR)
            if ear < eyeThresh:
                flag_eye += 1
                if flag_eye >= frame_check_eye:
                    cv2.putText(frame, "****************** SUBJECT IS SLEEPING *******************", (10, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag_eye = 0

        if flag_mouth == 0 & flag_eye == 0:
            cv2.putText(frame, "****************** SUBJECT IS ALERT *******************", (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Plotting the frame
            cv2.imshow("Frame", frame)

            # Waiting for exit key
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break


# Destroying all windows
cv2.destroyAllWindows()
# cap.stop()


def main():
    helper()


if __name__ == '__main__':
    main()

