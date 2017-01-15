import sys

import dlib
from skimage import io

#Take the image file from the command prompt
inputFile = sys.argv[1]

#Create a HOG face detector using dlib
face_detector = dlib.get_frontal_face_detector()
window = dlib.image_window()

#Load the image into an array
image = io.imread(inputFile)

#Run the HOG on the image data
detected_faces = face_detector(image, 1)

print("{} faces detected".format(len(detected_faces)))

#Open a window on the desktop showing the image
window.set_image(image)

#Loop through each face found in the image
for i, face_rect in enumerate(detected_faces):

	#Detected faces are returned as an object with the coordinates
	#of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	#Draw a box around each face we found
	window.add_overlay(face_rect)

#Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()

if (len(sys.argv[1:]) > 0):
    img = io.imread(sys.argv[1])
    dets, scores, idx = face_detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
