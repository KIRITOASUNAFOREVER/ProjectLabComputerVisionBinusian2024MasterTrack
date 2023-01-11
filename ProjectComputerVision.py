import os
import cv2 as cv
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory

        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    ret_subdirectories = []
    train_subfolder = os.listdir(root_path)

    for subfolder in enumerate(train_subfolder):
        ret_subdirectories.append(subfolder)

    return ret_subdirectories

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    ret_images_path = []
    ret_classes_id = []

    for i, subfolder in train_names:
        full_path = root_path + '/' + subfolder
        image_list = os.listdir(full_path)

        for filename in image_list:
            img_path = full_path + '/' + filename

            ret_images_path.append(img_path)
            ret_classes_id.append(i)

    ret_train_images = []
    for path in ret_images_path:
        ret_train_images.append(cv.imread(path))
    
    return ret_train_images, ret_classes_id

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id

        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    ret_images_gray = []
    ret_rect_position = []
    ret_classes_id = []

    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    for i, img in enumerate(image_list):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=3)

        for x,y,w,h in detected_faces:
            curr_face = img_gray[y:y+h, x:x+w]

            ret_images_gray.append(curr_face)
            ret_rect_position.append((x, y, w, h))

            if image_classes_list != None:
                ret_classes_id.append(image_classes_list[i])
    
    return ret_images_gray, ret_rect_position, ret_classes_id

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id

        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory

        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    ret_test_images = []

    for img_loc in os.listdir(test_root_path):
        full_path = test_root_path + '/' + img_loc
        ret_test_images.append(cv.imread(full_path))

    return ret_test_images

def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    ret_prediction_res = []

    for img in test_faces_gray:
        idx, confidence = recognizer.predict(img)
        confidence = ((confidence * 100) / 100).__floor__()

        ret_prediction_res.append(idx)

    return ret_prediction_res

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
             final result
    '''
    verification_statuses = []
    unverified_agent = ['Riza', 'High T']

    for i, pred in enumerate(predict_results):
        name = train_names[pred]
        if name[1] in unverified_agent:
            verification_statuses.append([name, 0])
        else:
            verification_statuses.append([name, 1])
    
    drawn_imgs = []

    font = cv.FONT_HERSHEY_SIMPLEX
    for ver, img, (x, y, w, h) in zip(verification_statuses, test_image_list, test_faces_rects):
        if ver[1] == 1:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness =8)
            img = cv.putText(img,str(ver[0][1]), (x, y + h + 55), font, 1, (0, 0, 0), thickness =16)
            img = cv.putText(img,str(ver[0][1]), (x, y + h + 55), font, 1, (0, 255, 0), thickness =3)
        else:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness =8)
            img = cv.putText(img,str(ver[0][1])+ " (Fake)", (x, y + h + 55), font, 1, (0, 0, 0), thickness =16)
            img = cv.putText(img, str(ver[0][1])+ " (Fake)", (x, y + h + 55), font, 1, (0, 0, 255), thickness =3)
        img = cv.resize(img, (250, 250))
        drawn_imgs.append([img, ver[1]])
    return drawn_imgs

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    white_img = np.zeros([250, 250, 3], dtype=np.uint8)
    white_img.fill(255)

    first_row_odd = []
    second_row_even = []
    indexs = 1
    for img, ver in image_list:
        if (indexs % 2) == 1:
            first_row_odd.append(img)
        else:
            second_row_even.append(img)
        indexs = indexs + 1

    row2 = cv.hconcat(second_row_even)
    row1 = cv.hconcat(first_row_odd)
    
    result = cv.vconcat([row1, row2])
    
    cv.imshow("MIB: International", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == '__main__':
    
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = 'dataset/train'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = 'dataset/test'
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)