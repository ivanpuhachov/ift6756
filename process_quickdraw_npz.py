import numpy as np
from quickdrawing import Drawing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


dataname = "owl"
n_items = 10000
finalsize = 32
subset = 'train'

input_path = f"data/{dataname}.npz"
out_path = f"data/bitmap_{dataname}_{subset}_{finalsize}x{n_items}.npy"


def process_npz_item(item, img_size=32):
    drawing = Drawing.from_npz_data(item)
    npimg = drawing.draw_to_numpy(img_size=img_size, linewidth=6)
    return npimg


data = np.load(input_path, encoding='latin1', allow_pickle=True)
data_resized = data[subset][:n_items]
bitmaps = Parallel(n_jobs=4, verbose=10)(delayed(process_npz_item)(data_resized[i], finalsize) for i in range(n_items))
bitmaps = np.array(bitmaps)
plt.imshow(bitmaps[1], cmap='gray_r')
plt.colorbar()
plt.show()
print(bitmaps.shape)
np.save(out_path, bitmaps)
