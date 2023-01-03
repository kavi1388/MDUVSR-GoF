import os
import multiprocessing as mp
from functools import partial
import argparse
import cv2

# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("vid", type=str, help="Video Path")
parser.add_argument("save", type=str, help="Frame Path (to save)")
# Read arguments from command line
args = parser.parse_args()


video_path = args.vid
frame_path = args.save

def video_to_frames(video_path, frame_path, filename):
  # for dirname, _, filenames in os.walk(video_path):
  #   for filename in filenames:
      f=os.path.join(video_path, filename)
      vidcap = cv2.VideoCapture(f)
      success,image = vidcap.read()
      count = 0
      while success:
        # print(success)
        cv2.imwrite(f"{frame_path}/{filename.split('.')[0]}_frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

files = os.listdir(video_path)

with mp.Pool(processes=mp.cpu_count()) as pool:
  pool.map(partial(video_to_frames,video_path,frame_path),files)