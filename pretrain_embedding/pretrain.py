import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--model', type=str, default='fm', help='The model to run. deepfm, fm or dcnv2.')
    parser.add_argument('--expid', type=str, default='default', help='The experiment id to run. Most of time no need to change.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--dataset_id', type=str, default='criteo_x1_default', help='The dataset id to run. criteo_x1_default or avazu_x1_default')
    
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
    
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    
    result_filename = Path(args['expid']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
    
    if not os.path.exists('../pretrain_result/' + params['dataset_id'] + '_' + args['model'].lower() + '_embeddings'):
        os.makedirs('../pretrain_result/' + params['dataset_id'] + '_' + args['model'].lower() + '_embeddings')
    
    for feature_name in feature_map.features.keys():
        if feature_map.features[feature_name]['type'] == 'categorical':
            if hasattr(model, 'embedding_layer'):
                embedding_layer_weight = model.embedding_layer.embedding_layer.embedding_layers[feature_name].weight.data.cpu().numpy()
            elif hasattr(model, 'embedding_layers'):    # specially designed for ffm
                embedding_layers = [field_emb.embedding_layer.embedding_layers[feature_name].weight.data.cpu().numpy() for field_emb in model.embedding_layers]
                embedding_layer_weight = np.hstack(embedding_layers)
            else:
                raise AttributeError("Neither 'embedding_layer' nor 'embedding_layers' found in model.embedding_layer")
            embedding_file = '../pretrain_result/' + params['dataset_id'] + '_' + args['model'].lower() + f'_embeddings/{feature_name}_embedding.npy'
            np.save(embedding_file, embedding_layer_weight)
            logging.info(f"Saved embedding file to {embedding_file}")