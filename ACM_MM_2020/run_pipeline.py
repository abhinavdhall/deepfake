import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD


def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > min_track:
      
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(track,cropfile):

  flist = glob.glob(os.path.join(frames_dir,reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
    
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(tmp_dir,reference,'audio.wav')
  audiostart  = (track['frame'][0])/frame_rate
  audioend    = (track['frame'][-1]+1)/frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(avi_dir,reference,'audio.wav'),audiostart,audioend,audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=None)

  # print(output)

  # if output != 0:
  #   pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  # if output != 0:
  #   pdb.set_trace()

  print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video():

  DET = S3FD(device='cuda')

  flist = glob.glob(os.path.join(frames_dir,reference,'*.jpg'))
  flist.sort()

  dets = []
      
  for fidx, fname in enumerate(flist):

    start_time = time.time()
    
    image = cv2.imread(fname)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])

    dets.append([]);
    for bbox in bboxes:
      dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

    elapsed_time = time.time() - start_time

    print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(avi_dir,reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

  savepath = os.path.join(work_dir,reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)

  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect():

  video_manager = VideoManager([os.path.join(avi_dir,reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(work_dir,reference,'scene.pckl')

  
  scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  print('%s - scenes detected %d'%(os.path.join(avi_dir,reference,'video.avi'),len(scene_list)))

  return scene_list
    

def run_pipeline(arg_data_dir,arg_videofile,arg_reference,label):

  global data_dir; data_dir = arg_data_dir
  global videofile; videofile = arg_videofile
  global reference; reference = arg_reference

  global avi_dir; avi_dir = os.path.join(data_dir,'pyavi')
  global tmp_dir; tmp_dir = os.path.join(data_dir,'pytmp')
  global work_dir; work_dir = os.path.join(data_dir,'pywork')
  global crop_dir; crop_dir = os.path.join(data_dir,'pycrop',label)
  global frames_dir; frames_dir = os.path.join(data_dir,'pyframes')

  global facedet_scale; facedet_scale=0.25
  global crop_scale; crop_scale=0.40
  global min_track; min_track=50
  global frame_rate; frame_rate=30
  global num_failed_det; num_failed_det=40
  global min_face_size; min_face_size=50 

  if not os.path.exists(os.path.join(data_dir,'pycrop')):
    os.makedirs(os.path.join(data_dir,'pycrop'))

  if not os.path.exists(avi_dir):
    os.makedirs(avi_dir)

  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

  if not os.path.exists(work_dir):
    os.makedirs(work_dir)

  if not os.path.exists(crop_dir):
    os.makedirs(crop_dir)

  if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)


  # ========== DELETE EXISTING DIRECTORIES ==========

  if os.path.exists(os.path.join(work_dir,reference)):
    rmtree(os.path.join(work_dir,reference))

  if os.path.exists(os.path.join(crop_dir,reference)):
    rmtree(os.path.join(crop_dir,reference))

  if os.path.exists(os.path.join(avi_dir,reference)):
    rmtree(os.path.join(avi_dir,reference))

  if os.path.exists(os.path.join(frames_dir,reference)):
    rmtree(os.path.join(frames_dir,reference))

  if os.path.exists(os.path.join(tmp_dir,reference)):
    rmtree(os.path.join(tmp_dir,reference))

  # ========== MAKE NEW DIRECTORIES ==========

  os.makedirs(os.path.join(work_dir,reference))
  os.makedirs(os.path.join(crop_dir,reference))
  os.makedirs(os.path.join(avi_dir,reference))
  os.makedirs(os.path.join(frames_dir,reference))
  os.makedirs(os.path.join(tmp_dir,reference))

  # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
  # -r is for setting frame rate in fps
  # -y Overwrite output files without asking
  # -f image2 will extract frames from video
  # third command extracts audio from video

  command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 30 %s" % (videofile,os.path.join(avi_dir,reference,'video.avi')))
  output = subprocess.call(command, shell=True, stdout=None)

  command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(avi_dir,reference,'video.avi'),os.path.join(frames_dir,reference,'%06d.jpg'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 48000 %s" % (os.path.join(avi_dir,reference,'video.avi'),os.path.join(avi_dir,reference,'audio.wav'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  # ========== FACE DETECTION ==========

  faces = inference_video()

  # ========== SCENE DETECTION ==========

  scene = scene_detect()

  # ========== FACE TRACKING ==========

  alltracks = []
  vidtracks = []

  for shot in scene:

    if shot[1].frame_num - shot[0].frame_num >= min_track :
      alltracks.extend(track_shot(faces[shot[0].frame_num:shot[1].frame_num]))

  try:

    # ========== FACE TRACK CROP ==========

    for ii, track in enumerate(alltracks):
      vidtracks.append(crop_video(track,os.path.join(crop_dir,reference,'%05d'%ii)))

    # ========== SAVE RESULTS ==========

    savepath = os.path.join(work_dir,reference,'tracks.pckl')

    with open(savepath, 'wb') as fil:
      pickle.dump(vidtracks, fil)

    rmtree(os.path.join(tmp_dir,reference))

  except:
    rmtree(os.path.join(tmp_dir,reference))
    rmtree(os.path.join(crop_dir,reference))
    rmtree(os.path.join(avi_dir,reference))
    rmtree(os.path.join(work_dir,reference))
    rmtree(os.path.join(frames_dir,reference))

