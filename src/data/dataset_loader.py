from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as tortrans

from src.data_io import transform as trans
from src.data_io.dataset_folder import Dataset

labels = open(
    "/home/ai/challenge/datasets/file_label.txt",
    "r",
)
data_label = labels.readlines()

train_label, val_label = train_test_split(data_label, test_size=0.2, random_state=111)

print(f"Train: {len(train_label)},Validation: {len(val_label)}")


def get_train_loader(conf):
    train_transform = trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size), scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
        ]
    )
    # root_path = "{}/{}".format(conf.train_root_path, conf.patch_info)

    trainset = Dataset(
        label_list=train_label,
        transforms=train_transform,
        ft_width=conf.ft_width,
        ft_height=conf.ft_height,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
    )
    return train_loader


def get_val_loader(conf):
    val_transform = trans.Compose(
        [
            trans.ToPILImage(),
            tortrans.Resize(size=tuple(conf.input_size)),
            trans.ToTensor(),
        ]
    )

    valset = Dataset(
        label_list=val_label,
        transforms=val_transform,
        ft_width=conf.ft_width,
        ft_height=conf.ft_height,
    )
    val_loader = DataLoader(
        valset,
        batch_size=conf.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )
    return val_loader
