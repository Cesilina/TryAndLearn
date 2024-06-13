'''
图片增强的方式
1.随机裁剪之后再resize成原来的大小【torchvision:transforms:RandomResizedCrop】
2.随机色彩失真
3.随机高斯模糊
'''
from PIL import Image
from torchvision import transforms


im = Image.open('./cat.png')
print(im)

im_aug = transforms.Compose([
    transforms.Resize(120), # 按照比例缩放
    transforms.RandomHorizontalFlip(), # 随机水平翻转 RandomVerticalFlip 随机竖直反转
    transforms.RandomCrop(96), # 随机裁剪出96*96
    transforms.RandomRotation(20), # 随机旋转
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5) # 对比度
])

im_new = im_aug(im)

print(im_new)


# MoCo中使用的增强方法
moco_aug1 = transforms.Compose(
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 如果是输入到模型中，则需要后两行的内容，不输入模型的话，就不需要totensor()
            normalize,
)

moco_aug2 = transforms.Compose(
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
)



