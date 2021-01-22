from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class ImageTransform:
    """ Image preprocessing

    -convert tensor, normalize, padding
    -input image size : (3, 600, 800)

    """

    def __init__(self, mean, std):
        """

        Args:
            mean: average of learning data
            std: std of learning data
        """
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            transforms.Normalize((mean_1, mean_2, mean_3), (std_1, std_2, std_3))  # for 3 channel
            transforms.Pad([0, 100])
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
