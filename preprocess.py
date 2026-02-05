import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from literal import Folder, RawDataColumns


def main(args):
    train_df = pd.read_csv(args.train_csv_path, encoding='utf-8-sig')
    test_df = pd.read_csv(args.test_csv_path, encoding='utf-8-sig')

    # 데이터 경로 수정
    train_df[RawDataColumns.img_path] = train_df[RawDataColumns.img_path].apply(lambda x: x.replace("./", Folder.data, 1))
    test_df[RawDataColumns.img_path] = test_df[RawDataColumns.img_path].apply(lambda x: x.replace("./", Folder.data, 1))

    # 라벨 길이를 기준으로 데이터 분리 (k-fold용)
    train_df[RawDataColumns.length] = train_df[RawDataColumns.label].str.len()
    train_df_short = train_df[train_df[RawDataColumns.length] == 1].reset_index(drop=True)
    train_df_long = train_df[train_df[RawDataColumns.length] > 1].reset_index(drop=True)
    
    if not os.path.exists(Folder.data_preprocess):
        os.makedirs(Folder.data_preprocess)

    y_counts = train_df_long[RawDataColumns.length].value_counts()
    min_class_count = y_counts.min() if not y_counts.empty else 0

    if min_class_count < args.kfold_n_splits and len(train_df_long) > 1:
        print(f"Warning: The minimum number of samples for any class ({min_class_count}) is less than n_splits={args.kfold_n_splits}.")
        # 데이터가 너무 적을 경우, k-fold 대신 단일 분할만 수행
        
        # 길이를 기준으로 Stratified split 시도
        try:
            train_long_indices, valid_long_indices = train_test_split(
                range(len(train_df_long)),
                test_size=1/args.kfold_n_splits,
                stratify=train_df_long[RawDataColumns.length],
                random_state=42
            )
        except ValueError:
             print("Stratified split failed. Performing a non-stratified random split.")
             train_long_indices, valid_long_indices = train_test_split(
                range(len(train_df_long)),
                test_size=1/args.kfold_n_splits,
                random_state=42
            )

        fold_train_long = train_df_long.iloc[train_long_indices]
        fold_valid_long = train_df_long.iloc[valid_long_indices]

        # 1글자짜리 데이터는 항상 학습에만 포함
        fold_train_df = pd.concat([train_df_short, fold_train_long]).reset_index(drop=True)
        fold_valid_df = fold_valid_long

        train_csv_name = "fold0_train.csv"
        valid_csv_name = "fold0_valid.csv"
        fold_train_df.to_csv(os.path.join(Folder.data_preprocess, train_csv_name), index=False, encoding='utf-8-sig')
        fold_valid_df.to_csv(os.path.join(Folder.data_preprocess, valid_csv_name), index=False, encoding='utf-8-sig')
        print(f"Created a single train/validation split: {train_csv_name}, {valid_csv_name}")

    else:
        kfold = StratifiedKFold(n_splits=args.kfold_n_splits, shuffle=True, random_state=42)
        for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X=train_df_long, y=train_df_long[RawDataColumns.length])):
            fold_train_long = train_df_long.iloc[train_idx]
            fold_valid_long = train_df_long.iloc[test_idx]

            # 1글자짜리 데이터는 항상 학습에만 포함
            fold_train_df = pd.concat([train_df_short, fold_train_long]).reset_index(drop=True)
            fold_valid_df = fold_valid_long

            train_csv_name = f"fold{fold_num}_train.csv"
            valid_csv_name = f"fold{fold_num}_valid.csv"
            fold_train_df.to_csv(os.path.join(Folder.data_preprocess, train_csv_name), index=False, encoding='utf-8-sig')
            fold_valid_df.to_csv(os.path.join(Folder.data_preprocess, valid_csv_name), index=False, encoding='utf-8-sig')
        print(f"Successfully created {args.kfold_n_splits} folds.")

    # 전체 테스트 데이터를 저장 (이 부분은 원본 test_df를 그대로 사용)
    test_df.to_csv(os.path.join(Folder.data_preprocess, "test.csv"), index=False, encoding='utf-8-sig')
    print("Test data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", type=str, default="./data/train_data.csv")
    parser.add_argument("--test_csv_path", type=str, default="./data/test_data.csv")
    parser.add_argument("--kfold_n_splits", type=int, default=5)
    args = parser.parse_args()
    main(args)
