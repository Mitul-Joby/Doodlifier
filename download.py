import urllib.request

with open("data/doodle/classes.txt", "r") as f:
    classes = f.readlines()
    classes = [c.replace('\n','').replace(' ','_') for c in classes]
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:        
        cls_url = c.replace('_', '%20')
        path = base+cls_url+'.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/doodle/' + c + '.npy')
