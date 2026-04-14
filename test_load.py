from monai import transforms as T
import glob

files = glob.glob(r"D:\BrainTumorData\Task01_BrainTumour\imagesTr\*.nii.gz")
if not files:
    print("no files")
    exit()

img_path = files[0]
label_path = img_path.replace("imagesTr", "labelsTr")

data = {"image": img_path, "label": label_path}

t1 = T.LoadImaged(keys=["image", "label"], ensure_channel_first=True)
data1 = t1(data)
print("After LoadImaged image shape:", data1["image"].shape)
print("After LoadImaged label shape:", data1["label"].shape)

try:
    t2 = T.EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")
    data2 = t2(data1)
    print("After EnsureChannelFirst image shape:", data2["image"].shape)
    print("After EnsureChannelFirst label shape:", data2["label"].shape)
except Exception as e:
    print("EnsureChannelFirst Error:", e)

try:
    t3 = T.Orientationd(keys=["image", "label"], axcodes="RAS")
    data3 = t3(data2)
    print("After Orientation image shape:", data3["image"].shape)
except Exception as e:
    print("Orientation Error:", e)
