from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('paddleocr', 'skimage')
hiddenimports += ['pyclipper', 'tools', 'skimage.morphology', 'imgaug', 'lmdb', 'requests']

