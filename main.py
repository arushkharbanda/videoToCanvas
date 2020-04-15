import cv2
import numpy as np
from lib.utils import Logger, mkdir
import os
import shutil
from datetime import datetime

logger = Logger()
path="data/sample_stitch.mp4"
project_root_dir = os.path.dirname(os.path.abspath(__file__))

delta_dict={}

#scroll direction
UP=1
DOWN=2
LEFT=3
RIGHT=4


#TODO - Capture configuration parameters as arguments

def extractScroll(sequence):
    # TODO - extract scroll direction and amount
    # TODO - for each frame check 40 pixels towards all 4 direction
    previous_frame=[]
    for frame in sequence:
        previous_frame=frame
    return  DOWN, 50



def isCut(s1,s2,threshold=1.8):
    f1 = cv2.resize(s1, (90, 160))
    f2 = cv2.resize(s1, (9, 16))
    g1 = cv2.resize(s2, (90, 160))
    g2 = cv2.resize(s2, (9, 16))
    e = np.mean(((f1 - g1) ** 2))
    e += np.mean(((f2 - g2) ** 2))
    e = np.sqrt(e/2.0)
    print(e)
    return e > threshold, e



# Returns threahold img in black and white
def preprocess(img):
    cv2.imshow("Input",img)
    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
    #cv2.imshow("Sobel",sobelx)
    #cv2.waitKey(0)
    ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("Threshold",threshold_img)
    #cv2.waitKey(0)
    return threshold_img

scene_count=0

def save_n_show_frame(frame, delta):
    global scene_count
    scene_count=scene_count+1
    cv2.imwrite("output/scene{}_{}.png".format(scene_count,delta ),frame)

def main():
    global project_root_dir, path
    all_frames=[]
    all_deltas=[]
    shutil.rmtree("{}/{}".format(project_root_dir,"output"))
    os.mkdir("{}/{}".format(project_root_dir,"output"))
    previous_frame=[]
    delta=0
    frame_count=0
    print("{}/{}".format(project_root_dir, path))
    video = cv2.VideoCapture("{}/{}".format(project_root_dir, path))

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    all_delta = np.empty((frameCount), np.dtype('single'))

    out = cv2.VideoWriter('output_{}.avi'.format(datetime.now()),cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    fc = 0
    ret = True
    scroll_sequences=[]

    current_sequence=[]
    complete_frames=[]
    start_frames=[]

    scrolling=False
    while (fc < frameCount  and ret):
        # 1. for delta above 1.5 scroll starts when scroll reaches below 1.5 scroll stops
        # 2. save the frame just before scroll starts for extraction
        # 3. save intermediatary frames to extract scroll direction
        delta=0
        ret, frame = video.read()
        if not ret:
            logger.warning("ret false")
            break
        if frame is None:
            logger.warning("frame drop")
            break
        if len(previous_frame):
            isACut,delta=isCut(previous_frame, frame, threshold=1)
            all_delta[fc]=delta
        if delta>1.5:
            current_sequence.append(frame)
            if not scrolling:
                #scroll_begins
                complete_frames.append(previous_frame)
                scrolling=True

        if delta<1.5:
            if scrolling:
                #check if below 1.5 for 10 previous frames
                if (all_delta[fc-10:fc]<1.5).all():
                    #scroll ends
                    scroll_sequences.append(current_sequence[0:len(current_sequence)-9])
                    current_sequence=[]
                    scrolling=False

        cv2.putText(frame, '{}'.format( scrolling), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 1)
        out.write(frame)
        cv2.imshow("Frame", frame)
        previous_frame=frame
        fc += 1

    for sequence in scroll_sequences:
        if len(sequence)>1:
            extractScroll(sequence)
            # TODO -  Crop area based on direction and amount, overwrite existing data if any

if __name__ == '__main__':
    main()
