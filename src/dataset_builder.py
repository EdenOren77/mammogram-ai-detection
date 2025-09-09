import os
import pandas as pd

label_map={
    "normal":0,
    "benign":1,
    "malignant":2
}

def build_dataset(data_dir="data/Dataset_BUSI_with_GT"):
    entries=[]
    for label_name,lable_value in label_map.items():
        class_dir=os.path.join(data_dir,label_name)
        for filename in os.listdir(class_dir):
            # Only files with the .png extension and no "mask"
            if filename.endswith(".png") and "_mask" not in filename:
                filepath=os.path.join(class_dir,filename)
                entries.append({
                    "path":filepath,
                    "label":lable_value,
                    "label_name":label_name
                })
    df=pd.DataFrame(entries)
    return df

if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(f"Total samples: {len(df)}")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/image_data.csv", index=False)
