import os
import sys
import logging
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import models
import gc
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--model', type=str, default='fm', help='The model to run. deepfm, fm, ffm, or ffmv2.')
    parser.add_argument('--expid', type=str, default='default', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--dataset_id', type=str, default='criteo_x1_default', help='The dataset id to run.')

    args = vars(parser.parse_args())

    args['expid'] = (args['model'] + '_' + args['expid']).lower()
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id, dataset_id=args['dataset_id'])
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if not os.path.exists(feature_map_json):
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # 打印 feature_map 中每个特征的详细信息
    for feature_name in feature_map.features.keys():
        if feature_map.features[feature_name]['type'] == 'categorical':
            logging.info(f"Feature: {feature_name}, Vocab Size: {feature_map.features[feature_name]['vocab_size']}")
            logging.info(f"Details: {feature_map.features[feature_name]}")

    # 加载训练数据
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()

    # 初始化计数字典
    feature_value_counts = defaultdict(lambda: defaultdict(int))

    from tqdm import tqdm
    import sys
    # 遍历训练数据
    for batch_data in tqdm(train_gen):
        for feature_name in feature_map.features.keys():
            if feature_map.features[feature_name]['type'] == 'categorical':
                feature_values = batch_data[feature_name].cpu().numpy()
                for value in feature_values:
                    feature_value_counts[feature_name][value] += 1
    # 遍历训练数据
    for batch_data in tqdm(valid_gen):
        for feature_name in feature_map.features.keys():
            if feature_map.features[feature_name]['type'] == 'categorical':
                feature_values = batch_data[feature_name].cpu().numpy()
                for value in feature_values:
                    feature_value_counts[feature_name][value] += 1

    # # 打印特征值计数结果
    # for feature_name, value_counts in feature_value_counts.items():
    #     logging.info(f"Feature: {feature_name}")
    #     for value, count in value_counts.items():
    #         logging.info(f"  Value: {value}, Count: {count}")

    # 保存特征值计数结果到文件
    output_file = f'{params["dataset_id"]}_{args["model"].lower()}_feature_value_counts.json'
    with open(output_file, 'w') as f:
        json.dump(feature_value_counts, f)
    logging.info(f"Saved feature value counts to {output_file}")
