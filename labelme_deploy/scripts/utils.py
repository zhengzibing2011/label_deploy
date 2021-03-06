import base64
try:
    import io
except ImportError:
    import io as io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import scipy.misc
import skimage.color

#define labelcolormap depend on CamVid class-11
labelcolormap = np.array([[0,0,0],[128,0,0],[128,64,128]],dtype=np.float32) / 255

def img_b64_to_array(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

 
def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def draw_label(label, img, label_names, colormap=None):
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

#modify colormap
    if colormap is None:
        colormap = labelcolormap

    label_viz = skimage.color.label2rgb(
        label, colors=colormap[1:], bg_label=0, bg_color=colormap[0])
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):      
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    out = np.array(PIL.Image.open(f))[:, :, :3]
    out = scipy.misc.imresize(out, img.shape[:2])
    return out


def labelme_shapes_to_label(img_shape, shapes):
    label_name_to_val = {'Background': 0}
#change dtype to np.uint8
    lbl = np.zeros(img_shape[:2], dtype=np.uint8)
    for shape in sorted(shapes, key=lambda x: x['label']):
        polygons = shape['points']
        label_name = shape['label']

        if label_name in label_name_to_val:
            label_value = label_name_to_val[label_name]

        else:
            label_value = len(label_name_to_val)   
            label_name_to_val[label_name] = label_value
        mask = polygons_to_mask(img_shape[:2], polygons)
        lbl[mask] = label_value
    
    lbl_names = [None] * (max(label_name_to_val.values()) + 1)
    for label_name, label_value in label_name_to_val.items():
        lbl_names[label_value] = label_name
     
    return lbl, lbl_names 
