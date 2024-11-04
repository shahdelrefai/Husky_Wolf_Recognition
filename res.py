# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import imagesize
import numpy as np

root = "data2/"

# Get the Image Resolutions
imgs = [img.name for img in Path(root).iterdir() if img.suffix == ".jpeg"]
img_meta = {}
for f in imgs:
    img_meta[f] = imagesize.get(Path(root) / f)

# Convert to DataFrame and compute aspect ratio
img_meta_df = pd.DataFrame.from_dict(img_meta, orient='index', columns=['Width', 'Height']).reset_index()
img_meta_df.rename(columns={'index': 'FileName'}, inplace=True)
img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)

# Display results
print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
img_meta_df.head()


# Visualize Image Resolutions

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha=0.5, s=img_meta_df["Aspect Ratio"]*100, picker=True)
ax.set_title("Image Resolution")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)


# Interactive Plotting

# Import libraries
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as mplPath

# Lasso Selection of data points
class SelectFromCollection:
	def __init__(self, ax, collection, alpha_other=0.3):
		self.canvas = ax.figure.canvas
		self.collection = collection

		self.xys = collection.get_offsets()
		self.lasso = LassoSelector(ax, onselect=self.onselect)
		self.ind = []

	def onselect(self, verts):
		path = mplPath(verts)
		self.ind = np.nonzero(path.contains_points(self.xys))[0]
		self.canvas.draw_idle()

	def disconnect(self):
		self.canvas.draw_idle()

# Show the original image upon picking the point
def on_pick(event):
	ind = event.ind[0]
	w, h = points.get_offsets().data[ind]
	img_file = Path(img_meta_df.iloc[ind, 0])
	if Path(root,img_file).is_file():
		print(f"Showing: {img_file}")
		img = Image.open(Path(root,img_file))
		figs = plt.figure(figsize=(5, 5))
		axs = figs.add_subplot(111)
		axs.set_title(Path(img_file).name, size=14)
		axs.set_xticks([])
		axs.set_yticks([])
		axs.set_xlabel(f'Dim: {round(w)} x {round(h)}', size=14)
		axs.imshow(img)
		figs.tight_layout()
		figs.show()

# Save selected image filenames 
def save_selected_imgs(df, fileName = Path("Images to discard.csv")):
	if fileName.is_file():
		orgData = pd.read_csv(fileName)
		df = pd.concat([orgData, df])
	df.set_axis(['FileName'], axis='columns').to_csv(fileName, index=False)

# Store selected points upon pressing "enter"
def accept(event):
	if event.key == "enter":
		selected_imgs = img_meta_df.iloc[selector.ind, 0].to_frame()
		save_selected_imgs(selected_imgs)
		print("Selected images:")
		print(selected_imgs)
		selector.disconnect()
		fig.canvas.draw()

# Plot the image resolutions
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha=0.5, s=img_meta_df["Aspect Ratio"]*100, picker=True)
ax.set_title("Press enter to after selecting the points.")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)

# Add interaction
selector = SelectFromCollection(ax, points)
fig.canvas.mpl_connect("key_press_event", accept)
fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()