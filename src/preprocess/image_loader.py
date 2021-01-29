from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class ImageTransform:
    """ Image preprocessing

    -convert tensor, normalize, padding
    -input image size : (3, 600, 800)

    """

    def __init__(self):
        """
        r_mean: 0.43032042947197113
        r_std: 0.21909041691171033
        g_mean: 0.4967263167264148
        g_std: 0.22394303049942132
        b_mean: 0.31342008204298993
        b_std: 0.20059191462725368
        """
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),  # range(0~1)
            transforms.Normalize(mean=[0.430, 0.496, 0.313], std=[0.219, 0.223, 0.200]),
            # transforms.Pad([0, 100])
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImgDataset(data.Dataset):

    def __init__(self, file_list, label_list, transform):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        label = self.label_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed, label
