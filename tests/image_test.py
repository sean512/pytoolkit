
import pytest

import pytoolkit as tk


@pytest.fixture()
def save_dir(check_dir):
    d = check_dir / 'image'
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize('filename', ['cifar.png', 'Lenna.png'])
def test_data_augmentation(data_dir, save_dir, filename):
    """画像の変換のテスト。目視したいので結果を`../___check/image[12]/`に保存しちゃう。"""
    aug = tk.image.Compose([
        tk.image.RandomRotate(),
        tk.image.RandomTransform(256, 256),
        tk.image.Standardize(),
        tk.image.RandomColorAugmentors(),
    ])
    img_path = data_dir / filename
    img = tk.ndimage.load(img_path)
    for i in range(32):
        tk.ndimage.save(save_dir / f'{img_path.stem}.DA.{i}.png', aug(image=img)['image'])


def test_to_gray_scale(data_dir, save_dir):
    aug = tk.image.ToGrayScale(p=1)
    img = tk.ndimage.load(data_dir / 'Lenna.png')
    tk.ndimage.save(save_dir / f'Lenna.ToGrayScale.png', aug(image=img)['image'])


def test_to_random_binarize(data_dir, save_dir):
    aug = tk.image.RandomBinarize(p=1)
    img = tk.ndimage.load(data_dir / 'Lenna.png')
    for i in range(4):
        tk.ndimage.save(save_dir / f'Lenna.RandomBinarize.{i}.png', aug(image=img)['image'])
