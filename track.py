import sys
import cv2
import numpy as np

if not (cv2.__version__):
    print('Error: Install OpenCV')


def frame2img(path):
    count = 0
    success = True
    while success:
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if count > 10000:
            # save frame as JPEG file
            cv2.imwrite("frame%d.jpg" % count, image)
            count += 1
        cap.release()


class blob(object):
    def __init__(self):
        detector = cv2.SimpleBlobDetector_create()
        self.params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        self.params.minArea = 190
        self.params.maxArea = 950

        # Filter by Circularity
        # self.params.filterByCircularity = True
        # elf.params.minCircularity = 0.4

        # Filter by Convexity
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.2

        # Filter by Inertia
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.03

        # Distance Between Blobs
        self.params.minDistBetweenBlobs = 90

        detector = cv2.SimpleBlobDetector_create(self.params)
        self.detector = detector


class counters(object):
    def __init__(self):
        global leftcnt
        self.leftcnt = 0
        global rightcnt
        self.rightcnt = 0
        global framecnt
        self.framecnt = 0
        global leftmark
        self.leftmark = 0
        global rightmark
        self.rightmark = 0
        global cxleft
        self.cxleft = []
        global cxright
        self.cxright = []


if __name__ == '__main__':
    path = sys.argv[1]
    vidcap = cv2.VideoCapture(path)
    print("Press esc to quit stream")
        

    kernel = (7, 7)

    # Read first frame
    ret, frame = vidcap.read()
    if not ret:
        print("Cannot open video file!")
        sys.exit()

    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    
    outvid = cv2.VideoWriter('output2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    # Intialize
    fgbg = cv2.createBackgroundSubtractorMOG2()
    blobs = blob()


    # Create Kalman Tracker
    # tracker = cv2.TrackerKCF_create()
    # = tracker.init(frame, bbox)

    # Camshift termination criterion
    # either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # setup initial location of window

    counter = counters()

    while (1):
        ret, frame = vidcap.read()

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # GaussianBlur
        gauss = cv2.GaussianBlur(gray, kernel, 0)


        # CLAHE
        # clip_limit = 0.45
        # tile_size = (5,5)
        # tile_size = kernel
        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        # adjc = clahe.apply(gauss)
        # adjc = cv2.adaptiveThreshold(gauss,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
        histequ = cv2.equalizeHist(gauss)


        # MOG BS and Thresholding
        fgmask = fgbg.apply(histequ)
        ret, thresh = cv2.threshold(fgmask.copy(), 90, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh.copy(),100,200)

        # kernels
        kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))

        # Dilation
        # dkernel = np.ones((2,2), np.uint8)
        dkernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(edges, kernel1, iterations=1)
        # dilation = cv2.dilate(dilation, mkernel, iterations=1)

        # Erosion
        # ekernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(dilation, kernel0, iterations=1)
        ffg = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel1)
        # ffg = cv2.morphologyEx(ffg, cv2.MORPH_OPEN, kernel2)
        # ffg = cv2.morphologyEx(ffg, cv2.MORPH_CLOSE, kernel3)
        # ret, ffg = cv2.threshold(ffg, 120,255, cv2.THRESH_BINARY);
        # Find Contours
        _, countours, _ = cv2.findContours(
            ffg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thrshCnt = 650
        # savedCnt = []

        # hulls = []
        sortcnt = sorted(countours, key=cv2.contourArea, reverse=True)
        if len(sortcnt) in range(4):
            hulls = [cv2.convexHull(x) for x in sortcnt]
        else:
            hulls = [cv2.convexHull(x) for x in sortcnt[:3]]

        # fill contour
        bgmask = cv2.drawContours(ffg, hulls, -1, (255, 255, 255), -1)

        for obj in hulls:
            x, y, w, h = cv2.boundingRect(obj)
            track_window = (x, y, w, h)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mom = cv2.moments(obj)
            cX = int(mom["m10"] / mom["m00"])
            cY = int(mom["m01"] / mom["m00"])

            # Left carts
            if (120 < cX < 207) and (69 < cY < 105):
                counter.leftmark += 1
                counter.cxleft.append(cX)
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "Train ", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


            if (200 > counter.leftmark > 100) and (abs(max(counter.cxleft) -
                (sum(counter.cxleft)/len(counter.cxleft))) + 10) > 25:
                    counter.leftcnt += 1
                    counter.leftmark=0
                    counter.cxleft = []

            if counter.leftmark%210 == 0:
                counter.leftmark = 0

            # Right carts
            if (424 < cX < 560) and (100 < cY < 197):
                counter.rightmark += 1
                counter.cxright.append(cX)
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "Train ", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


            if (130 > counter.rightmark > 80) and (abs(max(counter.cxright) -
                (sum(counter.cxright)/len(counter.cxright))) + 10) > 30:
                    counter.rightmark=0
                    counter.rightcnt += 1
                    counter.cxright = []

            if counter.rightmark%160 == 0:
               counter.rightmark = 0


        cv2.putText(frame, "Left: " + str(counter.leftcnt), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Right: " + str(counter.rightcnt), (491, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ## Camshift
        #    ret, track_window = cv2.CamShift(bgmask, track_window, term_crit)
        #    x,y,w,h = track_window
        #    #cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        #    pts = cv2.boxPoints(ret)
        #    pts = np.int0(pts)
        #    cv2.polylines(frame, [pts], True, 255, 2)


        # since contours are gray or black --> invert image
        #retval, invertedMask = cv2.threshold(bgmask, 200, 255,cv2.THRESH_BINARY_INV)
        #Blobdetect
        #keypts = blobs.detector.detect(invertedMask)
        # frame_keypoints = cv2.drawKeypoints(frame, keypts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ##overlay = frame.copy()
        ##for k in keypts:
        #    cv2.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size/2), (0, 0, 255), -1)
        #    cv2.line(overlay, (int(k.pt[0])-20, int(k.pt[1])), (int(k.pt[0])+20, int(k.pt[1])), (0,0,0), 3)
        #    cv2.line(overlay, (int(k.pt[0]), int(k.pt[1])-20), (int(k.pt[0]), int(k.pt[1])+20), (0,0,0), 3)
        #    opacity = 0.6
        #    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        cv2.namedWindow('orig', cv2.WINDOW_NORMAL)
        cv2.namedWindow('filter', cv2.WINDOW_NORMAL)
        cv2.imshow('filter', frame)
        outvid.write(frame)

        # exit condition
        k=cv2.waitKey(30) & 0xff
        if k == 27:
            break
            vidcap.release()
            outvid.release()
            cv.destroyAllWindows()
