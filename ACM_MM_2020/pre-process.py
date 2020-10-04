import argparse, imageio, subprocess, os, cv2
from shutil import rmtree
import shutil
from run_pipeline import *

parser = argparse.ArgumentParser(description = "PreProcess");
parser.add_argument('--out_dir',       type=str, default='', help='Output direcotry');
opt = parser.parse_args();

setattr(opt,'data_dir',os.path.join(opt.out_dir))
setattr(opt,'fake_dir',os.path.join(opt.data_dir,'fake'))
setattr(opt,'real_dir',os.path.join(opt.data_dir,'real'))

if not os.path.exists(os.path.join(opt.data_dir,'pycrop')):
    os.makedirs(os.path.join(opt.data_dir,'pycrop'))

if not os.path.exists(os.path.join(opt.data_dir,'pytmp')):
    os.makedirs(os.path.join(opt.data_dir,'pytmp'))

setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))

for video in os.listdir(opt.real_dir):
	if not os.path.exists(os.path.join(opt.data_dir,'pycrop','real',os.path.basename(video)[0:-4])):
		print(video)
		run_pipeline(opt.data_dir,os.path.join(opt.real_dir,video),os.path.basename(video)[0:-4],'real')

for video in os.listdir(opt.fake_dir):
	if not os.path.exists(os.path.join(opt.data_dir,'pycrop','fake',os.path.basename(video)[0:-4])):
		print(video)
		run_pipeline(opt.data_dir,os.path.join(opt.fake_dir,video),os.path.basename(video)[0:-4],'fake')


for directory in os.listdir(os.path.join(opt.crop_dir,'real')):
	if os.path.isdir(os.path.join(opt.crop_dir,'real',directory)):
		if not os.path.exists(os.path.join(opt.tmp_dir,'real',directory)):
			if len(os.listdir(os.path.join(opt.crop_dir,'real',directory)))==0:
				continue
			videoName = os.listdir(os.path.join(opt.crop_dir,'real',directory))[0]
			videopath = os.path.join(opt.crop_dir,'real',directory,videoName)
			framedir = os.path.join(opt.crop_dir,'real',directory,'frames')
			os.makedirs(framedir)

			audiodir = os.path.join(opt.crop_dir,'real',directory)

			command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (videopath,os.path.join(framedir,'%06d.jpg'))) 
			output = subprocess.call(command, shell=True, stdout=None)

			command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 48000 %s" % (videopath, os.path.join(audiodir,'audio.wav'))) 
			output = subprocess.call(command, shell=True, stdout=None)

			os.makedirs(os.path.join(opt.tmp_dir,'real',directory))

			total_frames = len(os.listdir(framedir))

			for frameNum in range(0,total_frames,30):
				if(frameNum+30>total_frames):
					continue
				if(frameNum==0):
					start_time = 0
				else:
					start_time = 30.0/frameNum
				videonum = '%05d'%(frameNum/30)
				os.makedirs(os.path.join(opt.tmp_dir,'real',directory,videonum))
				dst = os.path.join(opt.tmp_dir,'real',directory,videonum)

				for i in range(frameNum+1,frameNum+31):
					i_str = '%06d'%i
					i_jpg = i_str + '.jpg'
					shutil.copy(os.path.join(framedir,i_jpg),dst)

				output_audio = videonum + '.wav'
				audiotmp    = os.path.join(opt.tmp_dir,'real',directory,output_audio)
				audiostart  = frameNum/30
				audioend    = (frameNum+30)/30

				command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(audiodir,'audio.wav'),audiostart,audioend,audiotmp)) 
				output = subprocess.call(command, shell=True, stdout=None)


for directory in os.listdir(os.path.join(opt.crop_dir,'fake')):
	if os.path.isdir(os.path.join(opt.crop_dir,'fake',directory)):
		if not os.path.exists(os.path.join(opt.tmp_dir,'fake',directory)):
			if len(os.listdir(os.path.join(opt.crop_dir,'fake',directory)))==0:
				continue
			videoName = os.listdir(os.path.join(opt.crop_dir,'fake',directory))[0]
			videopath = os.path.join(opt.crop_dir,'fake',directory,videoName)
			framedir = os.path.join(opt.crop_dir,'fake',directory,'frames')
			os.makedirs(framedir)

			audiodir = os.path.join(opt.crop_dir,'fake',directory)

			command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (videopath,os.path.join(framedir,'%06d.jpg'))) 
			output = subprocess.call(command, shell=True, stdout=None)

			command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 48000 %s" % (videopath, os.path.join(audiodir,'audio.wav'))) 
			output = subprocess.call(command, shell=True, stdout=None)

			os.makedirs(os.path.join(opt.tmp_dir,'fake',directory))

			total_frames = len(os.listdir(framedir))

			for frameNum in range(0,total_frames,30):
				if(frameNum+30>total_frames):
					continue
				if(frameNum==0):
					start_time = 0
				else:
					start_time = 30.0/frameNum
				videonum = '%05d'%(frameNum/30)
				os.makedirs(os.path.join(opt.tmp_dir,'fake',directory,videonum))
				dst = os.path.join(opt.tmp_dir,'fake',directory,videonum)

				for i in range(frameNum+1,frameNum+31):
					i_str = '%06d'%i
					i_jpg = i_str + '.jpg'
					shutil.copy(os.path.join(framedir,i_jpg),dst)

				output_audio = videonum + '.wav'
				audiotmp    = os.path.join(opt.tmp_dir,'fake',directory,output_audio)
				audiostart  = frameNum/30
				audioend    = (frameNum+30)/30

				command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(audiodir,'audio.wav'),audiostart,audioend,audiotmp)) 
				output = subprocess.call(command, shell=True, stdout=None)



