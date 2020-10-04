# Audio-Visual Dissonance-based Deepfake Detection and Localization
This repository contains the implementation of Audio Visual Deepfake Detection method proposed in the paper -  
  
Komal Chugh, Parul Gupta, Abhinav Dhall and Ramanathan Subramanian, Audio-Visual Dissonance-based Deepfake Detection and Localization, ACM Multimedia 2020.  
Links: [[PDF]](https://arxiv.org/pdf/2005.14405.pdf) 

![Network Architecture](https://github.com/abhinavdhall/deepfake/blob/main/ACM_MM_2020/network_architecture.PNG)
  
# Dependencies
1) Follow the steps given in `conda_requirements.txt`.  
2) Run the command: `pip install -r requirements.txt`.  
  
# Prepare data
1) Download DFDC dataset from [here](https://www.kaggle.com/c/deepfake-detection-challenge/data). 
  
2) Store the train and test videos as follows:  

   ```
   {your_path}/ACM_MM_2020/train/real/{videoname}.mp4  
   {your_path}/ACM_MM_2020/train/fake/{videoname}.mp4  
   {your_path}/ACM_MM_2020/test/real/{videoname}.mp4  
   {your_path}/ACM_MM_2020/test/fake/{videoname}.mp4
   ```
  
   If you wish to use the same videos as used by the paper authors, please refer to `train_list.txt` and `test_list.txt`. These files contain video name followed by '.' and then the part where that video is stored (here part refers to the smaller chunks into which DFDC dataset is split). `train_list.txt` contains fake videos used for training, please extract corresponding real videos using metadata files of DFDC.  
  
3) Once the videos have been placed at the above mentioned paths, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test` for pre-processing the videos.  
  
4) After the above step, you can delete `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test` folders. (Do not delete `pytmp` folder please!)  
  
5) Collect video paths in csv files by running `python write_csv.py --out_dir . ` command.  
  
# Training
```
python train.py --epochs 50 --out_dir .
```
There are other arguments that can be changed, for instance, learning rate, batch size etc. Have a look at arguments defined in `train.py`. You can use --resume argument for resuming training and give the path of model file.  
  
# Testing
```
python train.py --test log_tmp\deepfake_audio-224_r18_bs8_lr0.001\model\model_best_epoch48.pth.tar --out_dir .
```
Change the path of model file accordingly in the --test argument.  
  
For computing AUC score, run `python test.py` after executing the above command.  
  
# Reference
If you use the code, please cite the paper -
```
@InProceedings{chugh2020not,
  author       = "Chugh, Komal and Gupta, Parul and Dhall, Abhinav and Subramanian, Ramanathan",
  title        = "Not made for each other-Audio-Visual Dissonance-based Deepfake Detection and Localization",
  booktitle    = "Proceedings of the 28th ACM international conference on Multimedia",
  year         = "2020",
}
```
# Acknowledgements
Thanks to the code available at https://github.com/TengdaHan/DPC and https://github.com/joonson/syncnet_python.  
  



