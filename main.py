import pickle
import cv2
import numpy as np
from lib.utils import Logger, mkdir
import os
import shutil
import argparse
from datetime import datetime

logger = Logger()
#path="data/video5.mp4"
scroll_match_cutoff=1.6
scrolling_cutoff=1
project_root_dir = os.path.dirname(os.path.abspath(__file__))

delta_dict={}

#scroll direction
UP=1
DOWN=2
LEFT=3
RIGHT=4

def saveImage(end_frame,start_frame, low_direction, low_amount, lowest_delta, count, axis):
    vis2 = np.concatenate((start_frame,end_frame), axis=axis)
    vis = cv2.addWeighted(end_frame,0.4,start_frame,0.1,0)
    cv2.imwrite('output/out_{}_{}_{}_{}.png'.format(count,low_direction,low_amount,lowest_delta), vis)
    cv2.imwrite('output/out_{}_{}_{}_{}_2.png'.format(count,low_direction,low_amount,lowest_delta), vis2)

def checkScroll(end_frame, start_frame, count):
    x,y,_=end_frame.shape
    lowest_delta_x=float("inf")
    lowest_delta_y=float("inf")
    low_direction=UP
    low_amount=0

    #UP - if scrolled up then crop leave the top part of the end frame and bottom part of start frame and compare
    for i in range(1,(x)):
        _,delta=isCut(end_frame[i:x,0:y,], start_frame[0:x-i,0:y,])
        if delta< lowest_delta_x:
            saveImage(end_frame[i:x,0:y,], start_frame[0:x-i,0:y,], low_direction, low_amount, delta, count, axis=1)
            if i>150 and i< x-150:
                low_amount=i
                low_direction=UP
                lowest_delta_x=delta
    '''
    #DOWN - if scrolled down then leave the bottom part of the end frame and top part of the start frame and compare
    for i in range(1,(x)):
        if i>150:
            _,delta=isCut(end_frame[0:x-i,0:y,], start_frame[i:x,0:y,])
            if delta< lowest_delta_x:
                low_amount=i
                low_direction=DOWN
                saveImage(end_frame[0:x-i,0:y,], start_frame[i:x,0:y,], low_direction, low_amount, delta, count, axis=1)
                lowest_delta_x=delta
    #lowest_delta_y=lowest_delta_x*0.7
    '''
    #LEFT - if scrolled left then leave the left part of the end frame and right part of the start frame and compare
    '''
    for i in range(1,(y)):
        if y-i>200:
            _,delta=isCut(end_frame[0:x,i:y,], start_frame[0:x,0:y-i,])
            if delta< lowest_delta_y:
                low_amount=i
                low_direction=LEFT
                saveImage(end_frame[0:x,i:y,], start_frame[0:x,0:y-i,], low_direction, low_amount, delta, count, axis=0)
                lowest_delta_y=delta
    '''
    #RIGHT - if scrolled right then leave the right part of the end frame and the left part start frame and compare
    '''
    for i in range(1,(y)):
        if y-i>200:
            _,delta=isCut(end_frame[0:x,0:y-i,], start_frame[0:x,i:y,])
            if delta< lowest_delta_y:
                low_amount=i
                low_direction=RIGHT
                saveImage(end_frame[0:x,0:y-i,], start_frame[0:x,i:y,], low_direction, low_amount, delta, count, axis=0)
                lowest_delta_y=delta
    '''
    return low_direction, low_amount, lowest_delta_x



#TODO - Capture configuration parameters as arguments
def extractScroll(sequence, start_frame, end_frame, count):
    # extract scroll direction and amount
    previous_frame=[]
    x,y,_=end_frame.shape

    lowest_delta_x=float("inf")
    lowest_delta_y=float("inf")
    low_direction=0
    low_amount=0
    prev=end_frame
    '''
    for frame in sequence:
        dir, amo, delta=checkScroll(prev, frame)
        if low_direction==0:
            low_direction=dir
        elif dir==low_direction:
            low_amount=low_amount+amo
    '''
    final_direction, final_amount,lowest_delta=checkScroll(end_frame, start_frame, count)
    print("From Scroll {} {} from frames {} {} {}".format(low_direction, low_amount, final_direction, final_amount, lowest_delta))
    if lowest_delta <scroll_match_cutoff:
        return final_direction, final_amount,



def isCut(s1,s2,threshold=1.5):
    '''
    f1 = cv2.resize(s1, (90, 160))
    f2 = cv2.resize(s1, (9, 16))
    g1 = cv2.resize(s2, (90, 160))
    g2 = cv2.resize(s2, (9, 16))
    e = np.mean(((f1 - g1) ** 2))
    e += np.mean(((f2 - g2) ** 2))
    '''
    s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
    s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
    #s1=preprocess(s1)
    #s2=preprocess(s2)
    e=np.mean(((s1 - s2) ** 2))
    e = np.sqrt(e/2.0)
    #print(e)
    return e > threshold, e


def average( a ):

    # Find sum of array element
    sum = 0
    n=len(a)
    for i in range(n):
        sum += a[i]

    return sum/n


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
    global project_root_dir, scroll_match_cutoff
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/video5.mp4')
    parser.add_argument("--scroll_match_cutoff", type=float, default=1.6)
    #parser.add_argument("--scrolling_cutoff", type=float,default=1.6)

    args = parser.parse_args()

    path=args.path
    scroll_match_cutoff=args.scroll_match_cutoff
    #scrolling_cutoff=args.scrolling_cutoff

    shutil.rmtree("{}/{}".format(project_root_dir,"output"))
    shutil.rmtree("{}/{}".format(project_root_dir,"output2"))
    os.mkdir("{}/{}".format(project_root_dir,"output"))
    os.mkdir("{}/{}".format(project_root_dir,"output2"))
    previous_frame=[]
    delta=0
    frame_count=0
    #print("{}/{}".format(project_root_dir, path))
    video = cv2.VideoCapture("{}/{}".format(project_root_dir, path))

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    all_delta = np.empty((frameCount), np.dtype('single'))

    #out = cv2.VideoWriter('output_{}.avi'.format(datetime.now()),cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    fc = 0
    ret = True
    scroll_sequences=[]

    current_sequence=[]
    complete_frames=[]
    start_frames=[]

    scrolling=False
    while (fc < frameCount  and ret):
        # 1. for delta above scroll cuttoff starts when scroll reaches below 1.5 scroll stops
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
        if fc==0:
            start_frames.append(frame)

        if len(previous_frame):
            isACut,delta=isCut(previous_frame, frame, threshold=1)
            all_delta[fc]=delta
            if fc>10:
                moving_avg=average(all_delta[fc-10:fc])
            else:
                moving_avg=0
            scrolling_cutoff=moving_avg*3
            if delta>scrolling_cutoff:
                print("sequence delta high  - {} currently scrolling {}".format(delta, scrolling))
                if not scrolling:
                    #scroll_begins
                    complete_frames.append(previous_frame)
                    scrolling=True

            if scrolling:
                current_sequence.append(frame)

            if delta<scrolling_cutoff:
                if scrolling:
                    print("In sequence delta{}".format(delta))
                    #check if below 1 for 10 previous frames
                    if (all_delta[fc-10:fc]<scrolling_cutoff).all():

                        #scroll ends
                        scroll_sequences.append(current_sequence[0:len(current_sequence)-9])
                        start_frames.append(current_sequence[len(current_sequence)-10:len(current_sequence)-9][0])
                        current_sequence=[]
                        scrolling=False


            #cv2.putText(frame, '{}'.format( scrolling), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 255), 1)
            #out.write(frame)


            if fc==frameCount-1:
                complete_frames.append(frame)

        else:
            all_delta[fc]=0
        previous_frame=frame
        fc += 1


    with open("all_delta{}".format(path.split("/")[1].split('.')[0]),'wb') as f:
        pickle.dump(all_delta,f)
    with open("start_frames",'wb') as f:
        pickle.dump(start_frames,f)
    with open("complete_frames",'wb') as f:
        pickle.dump(complete_frames,f)
    with open("scroll_sequences",'wb') as f:
        pickle.dump(scroll_sequences,f)
    with open("start_frames",'rb') as f:
        start_frames=pickle.load(f)
    with open("complete_frames",'rb') as f:
        complete_frames=pickle.load(f)
    with open("scroll_sequences",'rb') as f:
        scroll_sequences=pickle.load(f)

    masterimage=complete_frames[0]
    cv2.imwrite('output_1.png', masterimage)
    for i,sequence in enumerate(scroll_sequences):
        if len(sequence)>1:
            result=extractScroll(sequence, start_frames[i+1], complete_frames[i], i)
            if result:
                direction, amount=result
                print(direction, amount)
                # Crop area based on direction and amount, overwrite existing data if any
                x,y,_= complete_frames[i].shape
                if direction==UP:
                    cropped=complete_frames[i][x-amount:x, 0:y,]
                    cv2.imwrite('output_cropped_{}.png'.format(i), cropped)
                    masterimage=np.concatenate((masterimage,cropped), axis=0)
    cv2.imwrite('output_final_{}.png'.format(path.split("/")[1].split('.')[0]), masterimage)



if __name__ == '__main__':
    main()
