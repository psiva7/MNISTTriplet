import numpy as np
import matplotlib.pyplot as plt

def GetClrImgs(gallery, gLabel):
    galleryImages = gallery.copy()
    galleryImages /= 2
    galleryImages += 0.5
    cm = plt.get_cmap('rainbow')
    numClass = np.unique(gLabel).shape[0]
    clrs = [cm(i/numClass) for i in range(numClass)]
    galleryImagesClr = np.zeros((galleryImages.shape[0],galleryImages.shape[1],galleryImages.shape[2],3),dtype=np.float)
    for i in range(0,galleryImages.shape[0]):
        x = galleryImages[i]
        y = gLabel[i]
        clr = np.array(clrs[y][0:3])
        galleryImagesClr[i,:,:,0] = x*clr[0]
        galleryImagesClr[i,:,:,1] = x*clr[1]
        galleryImagesClr[i,:,:,2] = x*clr[2]
    return galleryImagesClr

def GetPlotImage(images, xy, canvas_shape = (1024,1024,3)):
    h,w = images.shape[1:3]
    min_xy = np.amin(xy, 0)
    max_xy = np.amax(xy, 0)
    min_canvas = np.array((0, 0))
    max_canvas = np.array((canvas_shape[0] - h, canvas_shape[1] - w))
    canvas = np.zeros(canvas_shape, dtype=np.float)
    for I, pt in zip(images, xy):
        x_off, y_off = map_range(pt, min_xy, max_xy, min_canvas, max_canvas).astype(int)
        #should avoid the following loops it is slow
        for r in range(0,h):
            for c in range(0,w):
                if (I[r,c,0] > 0 or I[r,c,1] > 0 or I[r,c,2] > 0):
                    canvas[y_off+r,x_off+c,0] = I[r,c,0]
                    canvas[y_off+r,x_off+c,1] = I[r,c,1]
                    canvas[y_off+r,x_off+c,2] = I[r,c,2]
    return (canvas, min_xy, max_xy)

def map_range(x, in_min, in_max, out_min, out_max):
    return out_min + (out_max - out_min) * (x - in_min) / (in_max - in_min)