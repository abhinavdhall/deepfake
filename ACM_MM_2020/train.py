import argparse, os, re, pickle
import numpy as np
from model import *
from dataset_3d import *
from resnet_2d3d import neq_load_customized
from utils import denorm, AverageMeter, ConfusionMeter, calc_loss, calc_accuracy, save_checkpoint, write_log

import torch
import torch.optim as optim
from torch.utils import data
import torch.utils.data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume training')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir',default='output',type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--print_freq', default=10, type=int, help='frequency of printing output during training')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation')
parser.add_argument('--threshold', default=0.3, type=float, help='threshold for testing')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)

def main():
	torch.manual_seed(0)
	np.random.seed(0)
	global args; args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	global cuda; cuda = torch.device('cuda')

	model = Audio_RNN(img_dim=args.img_dim, network=args.net, num_layers_in_fc_layers = args.final_dim, dropout=args.dropout)

	model = nn.DataParallel(model)
	model = model.to(cuda)
	global criterion; criterion = nn.CrossEntropyLoss()

	print('\n===========Check Grad============')
	for name, param in model.named_parameters():
		print(name, param.requires_grad)
	print('=================================\n')

	params = model.parameters()
	optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
	least_loss = 0
	global iteration; iteration = 0

	args.old_lr = None

	if args.test:
		if os.path.isfile(args.test):
			print("=> loading testing checkpoint '{}'".format(args.test))
			checkpoint = torch.load(args.test)
			try: model.load_state_dict(checkpoint['state_dict'])
			except:
				print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
				sys.exit()
			print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
			global num_epoch; num_epoch = checkpoint['epoch']
		elif args.test == 'random':
			print("=> [Warning] loaded random weights")
		else: 
			raise ValueError()

		transform = transforms.Compose([
		Scale(size=(args.img_dim,args.img_dim)),
		ToTensor(),
		Normalize()
		])
		test_loader = get_data(transform, 'test')
		global test_dissimilarity_score; test_dissimilarity_score = {}
		global test_target; test_target = {}
		global test_number_ofile_number_of_chunks; test_number_ofile_number_of_chunks = {}
		test_loss = test(test_loader, model)
		file_dissimilarity_score = open("file_dissimilarity_score.pkl","wb")
		pickle.dump(test_dissimilarity_score,file_dissimilarity_score)
		file_dissimilarity_score.close()
		file_target = open("file_target.pkl","wb")
		pickle.dump(test_target,file_target)
		file_target.close()
		file_number_of_chunks = open("file_number_of_chunks.pkl","wb")
		pickle.dump(test_number_ofile_number_of_chunks,file_number_of_chunks)
		file_number_of_chunks.close()
		sys.exit()
	else: # not test
		torch.backends.cudnn.benchmark = True

	if args.resume:
		if os.path.isfile(args.resume):
			args.old_lr = float(re.search('_lr(.+?)/', args.resume).group(1))
			print(args.old_lr)
			print("=> loading resumed checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
			args.start_epoch = checkpoint['epoch']
			iteration = checkpoint['iteration']
			least_loss = checkpoint['least_loss']
			try:
				model.load_state_dict(checkpoint['state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer'])

			except:
				print('=> [Warning]: weight structure is not equal to checkpoint; Use non-equal load ==')
				model = neq_load_customized(model, checkpoint['state_dict'])
			print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("[Warning] no checkpoint found at '{}'".format(args.resume))

	transform = transforms.Compose([
		Scale(size=(args.img_dim,args.img_dim)),
		ToTensor(),
		Normalize()
	])

	train_loader = get_data(transform, 'train')

	global total_sum_scores_real; total_sum_scores_real = 0
	global total_sum_scores_fake; total_sum_scores_fake = 0
	global count_real; count_real = 0
	global count_fake; count_fake = 0
	# setup tools
	global de_normalize; de_normalize = denorm()
	global img_path; img_path, model_path = set_path(args)
	global writer_train
	try: # old version
		writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
	except: # v1.7
		writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

	### main loop ###
	for epoch in range(args.start_epoch, args.epochs):
		train_loss, avg_score_real, avg_score_fake = train(train_loader, model, optimizer, epoch)

		writer_train.add_scalar('global/loss', train_loss, epoch)
		writer_train.add_scalar('global/avg_score_fake', avg_score_fake, epoch)
		writer_train.add_scalar('global/avg_score_real', avg_score_real, epoch)

		# save check_point
		if epoch==0:
			least_loss = train_loss
		is_best = train_loss <= least_loss
		least_loss = min(least_loss,train_loss)
		save_checkpoint({
			'epoch': epoch+1,
			'net': args.net,
			'state_dict': model.state_dict(),
			'least_loss': least_loss,
			'avg_score_real': avg_score_real,
			'avg_score_fake': avg_score_fake,
			'optimizer': optimizer.state_dict(),
			'iteration': iteration
		}, is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

	print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))



def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = deepfake_3d(out_dir=args.out_dir,mode=mode,
                         transform=transform)
    
    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      collate_fn=my_collate)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def my_collate(batch):
	batch = list(filter(lambda x: x is not None and x[1].size()[3] == 99, batch))
	return torch.utils.data.dataloader.default_collate(batch)

def train(data_loader, model, optimizer, epoch):
	losses = AverageMeter()
	model.train()
	global iteration
	dissimilarity_score_dict = {}
	target_dict = {}
	number_of_chunks_dict = {}
	for idx, (video_seq,audio_seq,target,audiopath) in enumerate(data_loader):
		tic = time.time()
		video_seq = video_seq.to(cuda)
		audio_seq = audio_seq.to(cuda)
		target = target.to(cuda)
		B = video_seq.size(0)

		vid_out  = model.module.forward_lip(video_seq)
		aud_out = model.module.forward_aud(audio_seq)

		vid_class = model.module.final_classification_lip(vid_out)
		aud_class = model.module.final_classification_aud(aud_out)

		del video_seq
		del audio_seq

		loss1 = calc_loss(vid_out, aud_out, target, args.hyper_param)
		loss2 = criterion(vid_class,target.view(-1))
		loss3 = criterion(aud_class,target.view(-1))
		acc = calc_accuracy(vid_out, aud_out, target, args.threshold)

		loss = loss1 + loss2 + loss3

		losses.update(loss.item(), B)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		for batch in range(B):
			vid_name = audiopath[batch].split('\\')[-2]
			dist = torch.dist(vid_out[batch,:].view(-1), aud_out[batch,:].view(-1), 2)
			tar = target[batch,:].view(-1).item()
			if(dissimilarity_score_dict.get(vid_name)):
				dissimilarity_score_dict[vid_name] += dist
				number_of_chunks_dict[vid_name] += 1
			else:
				dissimilarity_score_dict[vid_name] = dist
				number_of_chunks_dict[vid_name] = 1

			if(target_dict.get(vid_name)):
				pass
			else:
				target_dict[vid_name] = tar

		if idx % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
					'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'.format(
					epoch, idx, len(data_loader), time.time()-tic,
					loss=losses))

			total_weight = 0.0
			decay_weight = 0.0
			for m in model.parameters():
				if m.requires_grad: decay_weight += m.norm(2).data
				total_weight += m.norm(2).data
			print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))

			writer_train.add_scalar('local/loss', losses.val, iteration)

			iteration += 1

	avg_score_real, avg_score_fake = get_scores(dissimilarity_score_dict,number_of_chunks_dict,target_dict)
	return losses.local_avg, avg_score_real, avg_score_fake 

def test(data_loader, model):
	losses = AverageMeter()
	model.eval()
	with torch.no_grad():
		for idx, (video_seq,audio_seq,target,audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
			video_seq = video_seq.to(cuda)
			audio_seq = audio_seq.to(cuda)
			target = target.to(cuda)
			B = video_seq.size(0)

			vid_out  = model.module.forward_lip(video_seq)
			aud_out = model.module.forward_aud(audio_seq)

			vid_class = model.module.final_classification_lip(vid_out)
			aud_class = model.module.final_classification_aud(aud_out)

			del video_seq
			del audio_seq

			loss1 = calc_loss(vid_out, aud_out, target, args.hyper_param)
			loss2 = criterion(vid_class,target.view(-1))
			loss3 = criterion(aud_class,target.view(-1))

			loss = loss1 + loss2 + loss3
			losses.update(loss.item(), B)
			
			dist = torch.dist(vid_out[0,:].view(-1), aud_out[0,:].view(-1), 2)
			tar = target[0,:].view(-1).item()
			vid_name = audiopath[0].split('\\')[-2]
			print(vid_name)
			if(test_dissimilarity_score.get(vid_name)):
				test_dissimilarity_score[vid_name] += dist
				test_number_ofile_number_of_chunks[vid_name] += 1
			else:
				test_dissimilarity_score[vid_name] = dist
				test_number_ofile_number_of_chunks[vid_name] = 1

			if(test_target.get(vid_name)):
				pass
			else:
				test_target[vid_name] = tar

	print('Loss {loss.avg:.4f}\t'.format(loss=losses))
	write_log(content='Loss {loss.avg:.4f}\t'.format(loss=losses, args=args),
			epoch=num_epoch,
			filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
	return losses.avg

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_tmp/deepfake_audio-{args.img_dim}_{0}_bs{args.batch_size}_lr{1}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

def get_scores(dissimilarity_score_dict, number_of_chunks_dict, target_dict):
	sum_score_fake = 0
	sum_score_real = 0
	total_fake = 0
	total_real =0 
	for video,score in dissimilarity_score_dict.items():
		avg_score = score / number_of_chunks_dict[video]
		if target_dict[video] == 0:
			sum_score_fake += avg_score
			total_fake += 1
		else:
			sum_score_real += avg_score
			total_real += 1
	avg_score_real = sum_score_real / total_real
	avg_score_fake = sum_score_fake / total_fake
	return avg_score_real, avg_score_fake
if __name__ == '__main__':
    main()

