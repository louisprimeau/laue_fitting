import scipy.ndimage

import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    ax.scatter([event.xdata], [event.ydata])
    points.append([event.xdata, event.ydata])
    fig.canvas.draw()
    fig.canvas.flush_events()

def sharpen_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # ensure image is normalized
    image = scipy.ndimage.gaussian_filter(image, sigma=3) # gausisan filter
    image[image < 0.1] = 0 # remove bottom 10%
    #image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image
    
path = 'data'

for filename in os.listdir(path):
    if (filename[-4:] == '.bmp' and (filename[:-4] + '.csv') not in os.listdir(path)):

        print(filename[:-4] + '.csv', os.listdir(path))
        points = []
        image = np.asarray(imageio.imread(os.path.join(path, filename))).astype('float')
        image = sharpen_image(image)

        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        points = np.array(points)
        points[:, 0] = (points[:, 0] - image.shape[0]/2) / image.shape[0]
        points[:, 1] = (points[:, 1] - image.shape[1]/2) / image.shape[1]

        
        np.savetxt(os.path.join(path, filename[:-4] + '.csv'), points, delimiter=",")
