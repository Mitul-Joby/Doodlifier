import os
import progressbar
import urllib.request

pbar = None
current = None

def show_progress(block_num, block_size, total_size):
    global pbar, current
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size, widgets=['{:10}'.format(current), progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), '  ', progressbar.AdaptiveETA(), ' ', progressbar.Timer()])
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download(classListPath):
    global current
    if not os.path.exists('data/doodle'):
        os.mkdir('data/doodle')
    print('\nDownloading datasets...')
    with open(classListPath, 'r') as f:
        classes = f.readlines()
        classes = [c.replace('\n','').replace(' ','_') for c in classes]
        if classes is None or len(classes) == 0:
            print(f'No classes specified in the {classListPath} file. Please add one class per line. For a list of classes, see available.txt')
            return False
        for c in classes:
            if os.path.exists(f'data/doodle/{c}.npy'):
                print(f'Skipping {c}')
                continue
            cls_url = c.replace('_', '%20')
            current = c
            url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{cls_url}.npy'
            urllib.request.urlretrieve(url, f'data/doodle/{c}.npy', show_progress)
    return True

if __name__ == '__main__':
    download('doodle.txt')