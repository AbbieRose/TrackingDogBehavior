from collections import deque, defaultdict
import numpy as np
import cv2
import sys

#why?
#tracker_type = 'KCF'
#https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
#https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture('samoa_tug.mov')
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)] #?
calc_timestamps = [0.0]

#green" 93E4BC = 147, 228, 188
#pink C8565D = 200, 86, 93
# also AE464A = 174, 70, 74 E47078 = 228, 112, 120 
sensitivity = 20
lower_pink = np.array([170 - sensitivity, 50, 100])  # lower bound
upper_pink = np.array([170 + sensitivity, 255, 255]) # upper bound

points = deque(maxlen=512)

MAX_POINTS = 2

# read first frame to select bounding box
res, frame = cap.read()
bounding_box = cv2.selectROI(frame, False)
tracker.init(frame, bounding_box)

# in rectangle: corner_1 and corner 2 are diagonals, target_x, target_y are pts
def in_rectangle(target_x, target_y, corner_1, corner_2) :
    x_in_range = (target_x >= corner_1[0]) and (target_x <= corner_2[0])
    y_in_range = (target_y >= corner_1[1]) and (target_x <= corner_2[1])

    return (x_in_range) and (y_in_range)

def pretty_print(points_deque) :
    print("Timestamp, Tracked Points")
    for frame_data in points_deque :
        points_list = frame_data["points"]
        points_string = ''
        for pt in points_list :
            points_string = points_string + ", " + str(pt)
        print(str(frame_data["timestamp"]) + points_string)

frame_count = 0
tracking_frames = defaultdict(int)

while(cap.isOpened()) :
    frame_count = frame_count + 1
    frame_data = {}
    tracked_points = []

    res, frame = cap.read()
    if not res :
        print("not res")
        break

    res, bounding_box = tracker.update(frame)
    
    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cur_timestamp = calc_timestamps[-1] + 1000/fps
    calc_timestamps.append(cur_timestamp)
    frame_data["timestamp"] = cur_timestamp

    p1 = 0
    p2 = 0
    
    if res :
        p1 = (int(bounding_box[0]), int(bounding_box[1]))
        p2 = (int(bounding_box[0] + bounding_box[2]),
              int(bounding_box[1] + bounding_box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else :
        print("Tracking failure detected")
        bounding_box = cv2.selectROI(frame)
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, bounding_box)

    # detecting colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0 :        
        for c in contours :
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
            if radius > 5 :
                # check out documentation for this
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                if in_rectangle(x, y, p1, p2) :
                    tracked_points.append((x,y))

    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    #cv2.imshow('res',res)
    if len(tracked_points) > 0 :
        frame_data["points"] = tracked_points
        points.append(frame_data)
        
        tracking_frames[str(len(tracked_points))] += 1 
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break


print("total " + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

pretty_print(points)

for key in tracking_frames :
    print("Number of points: " + key + ", Number of frames detected: " + str(tracking_frames[key]))

cap.release()
cv2.destroyAllWindows()
