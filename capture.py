import numpy as np
import cv2
import sys, errno
import time
import os

if len(sys.argv) < 5:
    print("usage: python3 capture.py [capture_frames] [delay_start] [folder_name] [frame_each]")
    sys.exit(errno.EINVAL)

capture_frames = int(sys.argv[1])
sleep_before = float(sys.argv[2])
folder = sys.argv[3]
frame_modulo = int(sys.argv[4])

if os.path.isdir(folder):
    print("The folder already exists.")
    sys.exit()

os.mkdir(folder)

left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(1)

frames_captured = 0
filenames_file = open(folder + '/filenames.txt', 'a')

print("Delaying before captuing... [" + str(sleep_before) + " secs]")
time.sleep(sleep_before)

print("Capturing to " + folder + "/...")

while frames_captured < capture_frames:
    # Capture frame-by-frame
    ret_l, frame_l = left_cam.read()
    ret_r, frame_r = right_cam.read()

    if ret_l and ret_r:
        cv2.imshow('Cam LEFT', frame_l)
        cv2.imshow('Cam RIGHT', frame_r)
        
        if frames_captured % frame_modulo == 0:
            cv2.imwrite(folder + '/frame_left_' + str(frames_captured) + '.jpg', frame_l)
            cv2.imwrite(folder + '/frame_right_' + str(frames_captured) + '.jpg', frame_r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frames_captured % frame_modulo == 0:
        print("Saved frame " + str(frames_captured))
        filenames_file.write('frame_left_' + str(frames_captured) + '.jpg frame_right_' + str(frames_captured) + '.jpg\n')

    frames_captured += 1

# When everything is done, release the capture
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
filenames_file.close()

print(str(frames_captured) + " frames captured.")