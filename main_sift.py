import pickle
from lib.utils import Logger, mkdir
import os
import shutil
import argparse

import numpy as np
import cv2

print(cv2.__version__)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

#scroll direction
UP=1
DOWN=2
LEFT=3
RIGHT=4


logger = Logger()



def average( a ):
    # Find sum of array element
    sum = 0
    n=len(a)
    for i in range(n):
        sum += a[i]
    return sum/n

def isCut(s1,s2,threshold=1.5, verbose=False):
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
    if verbose:
        print(e)
    return e > threshold, e


def processScroll(sequence, start_frame, complete_frame, c, save_temp, file_name):
    # find the keypoints and descriptors with SIFT
    direction=DOWN
    amount=0
    kp1, des1 = sift.detectAndCompute(complete_frame,None)
    kp2, des2 = sift.detectAndCompute(start_frame,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #matches = flann.knnMatch(des1,des2,k=2)
    matches = flann.knnMatch(des2,des1,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(start_frame,kp2,complete_frame,kp1,matches,None,**draw_params)

    if save_temp:
        cv2.imwrite("output{}/temp_scroll_dist_{}.png".format(file_name,c),img3)

    points=[(kp2[matches[i][0].queryIdx].pt,kp1[ matches[i][0].trainIdx].pt) for i,x in enumerate(matchesMask) if x[0]==1]
    distances=[(int(point[1][0]-point[0][0]),int(point[1][1]-point[0][1]) ) for point in points]

    vote_x={}
    vote_y={}

    #voting
    for distance in distances:
        y,x=distance
        if x in vote_x.keys():
            vote_x[x]=vote_x[x]+1
        else:
            vote_x[x]=1

        if y in vote_y.keys():
            vote_y[y]=vote_y[y]+1
        else:
            vote_y[y]=1

    scroll_x=sorted(vote_x.items(), key=lambda x:x[1], reverse=True)[0][0]
    count_x=sorted(vote_x.items(), key=lambda x:x[1], reverse=True)[0][1]
    if scroll_x-1 in vote_x.keys():
        count_x=count_x+vote_x[scroll_x-1]
    if scroll_x+1 in vote_x.keys():
        count_x=count_x+vote_x[scroll_x+1]
    scroll_y=sorted(vote_y.items(), key=lambda x:x[1], reverse=True)[0][0]
    count_y=sorted(vote_y.items(), key=lambda x:x[1], reverse=True)[0][1]
    if scroll_y-1 in vote_y.keys():
        count_y=count_y+vote_y[scroll_y-1]
    if scroll_y+1 in vote_y.keys():
        count_y=count_y+vote_y[scroll_y+1]

    if count_x>(0.5*len(distances)) and count_y>(0.5*len(distances)):
        return scroll_x, scroll_y
    else:
        return None

def main():
    #parse argument
    # video-file
    # generate-intermediate
    # no of frames for avg
    # last m frames to check if scroll is still on
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='videos')
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--scroll_avg", type=int, default=5)
    parser.add_argument("--scrolling_check", type=int,default=10)
    parser.add_argument("--save_temp_files", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()


    input_path=args.input_path
    verbose=args.verbose
    save_temp_file=args.save_temp_files
    output_path=args.output_path
    scrolling_check=args.scrolling_check
    scrolling_avg=args.scroll_avg


    project_root_dir = os.path.dirname(os.path.abspath(__file__))

    files=os.listdir("{}/{}/".format(project_root_dir,input_path))

    #shutil.rmtree("{}/{}".format(project_root_dir,"output{}".format(file_name)), ignore_errors=True)
    full_output="{}/{}".format(project_root_dir,output_path)
    if not os.path.exists(full_output):
        os.makedirs(full_output)


    for file_name_w_ext in files:
        try:
            #get filename for saving files
            file_name=file_name_w_ext.split('.')[0]

            #read video
            video = cv2.VideoCapture("{}/{}/{}".format(project_root_dir, input_path,file_name_w_ext))
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            if int(major_ver)  < 3 :
                fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            else:
                fps = video.get(cv2.CAP_PROP_FPS)
                w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))



            # Initialize loop related variables
            previous_frame=[]
            frame_count=0
            scrolling=False
            fc = 0
            ret = True
            current_sequence=[]
            cutoff=3

            # initializing variables to be capurted in the processing
            all_sequences=[]
            all_complete_frames=[]
            all_start_frames=[]
            all_delta = np.empty((frameCount), np.dtype('single'))

            if not os.path.exists("{}/{}/complete_frames{}".format(project_root_dir,output_path,file_name)):
                while (fc < frameCount  and ret):
                    ret, frame = video.read()
                    if not ret:
                        logger.warning("ret false")
                        break
                    if frame is None:
                        logger.warning("frame drop")
                        break

                    # save first frame
                    if fc==0:
                        all_start_frames.append(frame)


                    if len(previous_frame):


                        _,delta= isCut(frame, previous_frame, verbose=verbose)
                        all_delta[fc]=delta

                        if not scrolling:
                            if fc>=scrolling_avg:
                                moving_avg=average(all_delta[fc-scrolling_avg:fc])
                                if moving_avg<0.3:
                                    moving_avg=0.3
                            else:
                                moving_avg=0.3
                                #moving_avg=average(all_delta)
                            cutoff=moving_avg*3
                        if not scrolling and delta > cutoff:
                            # start scrolling
                            current_sequence = []
                            current_sequence.append(frame)
                            all_complete_frames.append(previous_frame)
                            scrolling = True

                        elif scrolling and delta < cutoff:
                            if (all_delta[ fc-scrolling_check : fc ] < cutoff).all():
                                #stop scrolling
                                scrolling = False
                                all_sequences.append(current_sequence[0:len(current_sequence)-scrolling_check])
                                all_start_frames.append(current_sequence[len(current_sequence)-scrolling_check-1:len(current_sequence)-scrolling_check][0])
                            else:
                                current_sequence.append(frame)
                        elif scrolling and delta > cutoff:
                            current_sequence.append(frame)

                    previous_frame=frame
                    fc=fc+1
                all_complete_frames.append(previous_frame)


                if save_temp_file:
                    with open("{}/all_delta{}".format(output_path,file_name),'wb') as f:
                        pickle.dump(all_delta,f)
                    with open("{}/start_frames{}".format(output_path,file_name),'wb') as f:
                        pickle.dump(all_start_frames,f)
                    with open("{}/complete_frames{}".format(output_path,file_name),'wb') as f:
                        pickle.dump(all_complete_frames,f)
                    with open("{}/scroll_sequences{}".format(output_path,file_name),'wb') as f:
                        pickle.dump(all_sequences,f)
            else:
                with open("{}/start_frames{}".format(output_path,file_name),'rb') as f:
                    all_start_frames=pickle.load(f)
                with open("{}/complete_frames{}".format(output_path,file_name),'rb') as f:
                    all_complete_frames=pickle.load(f)
                with open("{}/scroll_sequences{}".format(output_path,file_name),'rb') as f:
                    all_sequences=pickle.load(f)
            all_scroll=[]
            for i,sequence in enumerate(all_sequences):
                all_scroll.append(processScroll(sequence, all_start_frames[i+1], all_complete_frames[i], i, save_temp_file, file_name))

            masterimage=[]
            count=len(all_complete_frames)
            cropped=[]
            not_found=False
            s =[ (i,scroll) for i,scroll in enumerate(reversed(all_scroll))]
            for i,scroll in s:
                print(scroll)
                if scroll:
                    x_change,y_change=scroll
                    if x_change==0 and y_change==0:
                        not_found=True

                    if x_change>0 or y_change>0:
                        x,y,_=all_complete_frames[count-1].shape
                        if len(masterimage)==0:
                            masterimage=all_complete_frames[count-i-1]

                        if save_temp_file:
                            cv2.imwrite('{}/output_first_{}.png'.format(output_path,file_name), masterimage)
                        if x_change>0 and y_change==0:
                            cropped=all_complete_frames[count-i-2][0:x_change,: ,:]

                        if len(cropped):
                            if save_temp_file:
                                cv2.imwrite('{}/output_cropped_{}_{}.png'.format(output_path,file_name,count-i), cropped)
                            masterimage=np.concatenate((cropped,masterimage), axis=0)
                        not_found=False
                else:
                    not_found=True
            if not_found:
                if len(masterimage)==0:
                    masterimage=all_complete_frames[count-i-1]

            cv2.imwrite('{}/output_final_{}.png'.format(output_path,file_name), masterimage)
        except:
            print("error")

if __name__ == '__main__':
    main()