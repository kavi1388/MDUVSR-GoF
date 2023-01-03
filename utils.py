import cv2
import glob
import os


def video_to_frames(video_path, frame_path):
  for dirname, _, filenames in os.walk(video_path):
    for filename in filenames:
        f=os.path.join(dirname, filename)
        vidcap = cv2.VideoCapture(f)
        success,image = vidcap.read()
        count = 0
        while success:
          # print(success)
          cv2.imwrite(f"{frame_path}/{filename}_frame%d.jpg" % count, image)     # save frame as JPEG file
          success,image = vidcap.read()
          print('Read a new frame: ', success)
          count += 1


def frames_to_video(img_path, vid_path, vidname, fps, frame_width, frame_height):
    out = cv2.VideoWriter(f'{vid_path}/{vidname}_HR_output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps,
                          (frame_width, frame_height))

    for filename in glob.glob(f'{img_path}/*.jpg'):
        img = cv2.imread(filename)
        out.write(img)

    out.release()
