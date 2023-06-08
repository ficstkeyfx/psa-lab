#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import os
import sys
operating_system = sys.platform

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if operating_system.find("win") == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from object_extraction import *
import ae.adversarial_training.trainer_appearance as appearance_trainer_adv
import ae.adversarial_training.trainer_motion as trainer_motion_adv
import compute_features
from dataset_reader_motion import MotionAeType

# do not delete this
RUNNING_ID = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
utils.set_vars(args.logs_folder, RUNNING_ID)
utils.create_dir(args.logs_folder)
args.log_parameters()

# train the appearance model
appearance_trainer_adv.train()
appearance_trainer_adv.test()

# train the motion models
trainer_motion_adv.train(MotionAeType.PREVIOUS)
trainer_motion_adv.test(MotionAeType.PREVIOUS)

trainer_motion_adv.train(MotionAeType.NEXT)
trainer_motion_adv.test(MotionAeType.NEXT)


import discriminator.fusion.trainer_discriminator as trainer_discriminator_fusion
trainer_discriminator_fusion.train('app')
trainer_discriminator_fusion.train('next')
trainer_discriminator_fusion.train('previous')

compute_features.compute_abnormality_scores(ProcessingType.TEST)

compute_features.compute_performance_indices(ProcessingType.TEST)

