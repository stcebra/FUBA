def add_backdoor_all(img):
    img[:,:, :2, :5] = 1
    img[:,:, 4:6, 7:12] = 1
    img[:,:, :2, 7:12] = 1
    img[:,:, 4:6, :5] = 1
    return img  

def add_backdoor_1(img):
    img[:,:, :2, :5] = 1
    return img  

def add_backdoor_2(img):
    img[:,:, 4:6, 7:12] = 1
    return img  

def add_backdoor_4(img):
    img[:,:, :2, 7:12] = 1
    return img  

def add_backdoor_3(img):
    img[:,:, 4:6, :5] = 1
    return img  