from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from nets.crackformer import crackformer
import os


TRAIN_IMG  = os.path.join(args.data_dir, 'image')
TRAIN_MASK = os.path.join(args.data_dir, 'label')
train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.jpg')]
print(f'total train images = {len(train_img_names)}')

channel_means = [0.485, 0.456, 0.406]
channel_stds  = [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])
val_tfms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(channel_means, channel_stds)])
mask_tfms = transforms.Compose([transforms.ToTensor()])

dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
_dataset, test_dataset = random_split(dataset, [275, 40],torch.Generator().manual_seed(42))
test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)


# pretrain_dir="checkpoints/crack260.pth"
pretrain_dir="checkpoints/crack315.pth"
# pretrain_dir='checkpoints/crack537.pth'
valid_result_dir = "./datasets/" + datasetName + "/valid/Valid_result/"
def Test():
    crack=crackformer()
    crack.load_state_dict(torch.load(pretrain_dir))
    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack, image_format, lable_format)
    validator.validate('0')

if __name__ == '__main__':
    Test()