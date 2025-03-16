import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(input_csv, output_dir="data", 
              train_csv="train.csv", val_csv="val.csv", test_csv="test.csv",
              train_ratio=0.8, val_ratio=0.15, test_ratio=0.05, random_seed=42):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(input_csv)
    
    # Chia dữ liệu
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

    # Lưu vào thư mục `data/`
    train_df.to_csv(os.path.join(output_dir, train_csv), index=False)
    val_df.to_csv(os.path.join(output_dir, val_csv), index=False)
    test_df.to_csv(os.path.join(output_dir, test_csv), index=False)

    print("Dataset split completed:")
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

if __name__ == "__main__":
    split_csv("data/labels.csv")