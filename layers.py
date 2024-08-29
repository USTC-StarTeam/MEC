import torch
from torch import nn
import os
import numpy as np
from collections import OrderedDict
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.utils import not_in_whitelist
from fuxictr.pytorch import layers
import io
import json
import logging
from utils import load_pretrain_emb


class FeatureEmbedding(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 pretrained_codes=None,
                 vq_num_embeddings=0):  
        super(FeatureEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing,
                                                    pretrained_codes=pretrained_codes,
                                                    vq_num_embeddings=vq_num_embeddings)  # 传递参数

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb

    def load_pretrained_embeddings(self, feature_name, embedding_files):
        """
        Load pretrained embeddings from files and set them to the corresponding embedding layers.
        :param embedding_files: List of file paths containing the pretrained embeddings.
        """
        for i, file in enumerate(embedding_files):
            embeddings = np.load(file)
            if isinstance(self.embedding_layer.embedding_layers[feature_name], nn.ModuleList):
                self.embedding_layer.embedding_layers[feature_name][i].weight.data.copy_(torch.from_numpy(embeddings).float())


class FeatureEmbeddingDict(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 pretrained_codes=None,
                 vq_num_embeddings=0): 
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.pretrained_codes = pretrained_codes 
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_dim = 1  # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_dim, bias=False)
                elif feature_spec["type"] in ["categorical", "sequence"]:
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        pretrain_path = os.path.join(feature_map.data_dir,
                                                        feature_spec["pretrained_emb"])
                        vocab_path = os.path.join(feature_map.data_dir, 
                                                    "feature_vocab.json")
                        pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
                        pretrain_usage = feature_spec.get("pretrain_usage", "init")
                        self.embedding_layers[feature] = PretrainedEmbedding(feature,
                                                                                feature_spec,
                                                                                pretrain_path,
                                                                                vocab_path,
                                                                                feat_dim,
                                                                                pretrain_dim,
                                                                                pretrain_usage)
                    else:
                        if self.pretrained_codes and feature in self.pretrained_codes:
                            code_num = self.pretrained_codes[feature].shape[1]
                            self.embedding_layers[feature] = nn.ModuleList([
                                nn.Embedding(vq_num_embeddings, embedding_dim//code_num) for _ in range(code_num)
                            ])
                        else:
                            padding_idx = feature_spec.get("padding_idx", None)
                            self.embedding_layers[feature] = nn.Embedding(feature_spec["vocab_size"], 
                                                                        feat_dim, 
                                                                        padding_idx=padding_idx)
        self.reset_parameters()

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))
        
    def reset_parameters(self):
        embedding_initializer = get_initializer(self.embedding_initializer)
        for k, v in self.embedding_layers.items():
            if k in self._feature_map.features and "share_embedding" in self._feature_map.features[k]:
                continue
            if type(v) == PretrainedEmbedding:  # skip pretrained
                v.reset_parameters(embedding_initializer)
            elif type(v) == nn.Embedding:
                if v.padding_idx is not None:
                    embedding_initializer(v.weight[1:, :])  # set padding_idx to zero
                else:
                    embedding_initializer(v.weight)
            elif isinstance(v, nn.ModuleList):  # handle ModuleList for pretrained codes
                for embed_layer in v:
                    embedding_initializer(embed_layer.weight)
                       
    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def dict2tensor(self, embedding_dict, flatten_emb=False, feature_list=[], feature_source=[], feature_type=[]):
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_list and not_in_whitelist(feature, feature_list):
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        feature_emb_dict = OrderedDict()
        for feature in inputs.keys():
            feature_spec = self._feature_map.features[feature]
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            if feature_type and not_in_whitelist(feature_spec["type"], feature_type):
                continue
            if feature_spec["type"] == "numeric":
                inp = inputs[feature].float().view(-1, 1)
                embeddings = self.embedding_layers[feature](inp)
            elif feature_spec["type"] == "categorical":
                if self.pretrained_codes and feature in self.pretrained_codes:
                    feature_codes = self.pretrained_codes[feature]  # tensor on GPU
                    code_embeddings = []
                    for i, embed_layer in enumerate(self.embedding_layers[feature]):
                        codes = feature_codes[inputs[feature].long(), i]
                        code_embedding = embed_layer(codes)
                        code_embeddings.append(code_embedding)
                    embeddings = torch.cat(code_embeddings, dim=-1)
                else:
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
            elif feature_spec["type"] == "sequence":
                inp = inputs[feature].long()
                embeddings = self.embedding_layers[feature](inp)
            else:
                raise NotImplementedError

            if feature in self.feature_encoders:
                embeddings = self.feature_encoders[feature](embeddings)
            feature_emb_dict[feature] = embeddings
        return feature_emb_dict


class PretrainedEmbedding(nn.Module):
    def __init__(self,
                 feature_name,
                 feature_spec,
                 pretrain_path,
                 vocab_path,
                 embedding_dim,
                 pretrain_dim,
                 pretrain_usage="init"):
        """
        Fusion pretrained embedding with ID embedding
        :param: fusion_type: init/sum/concat
        """
        super().__init__()
        assert pretrain_usage in ["init", "sum", "concat"]
        self.pretrain_usage = pretrain_usage
        padding_idx = feature_spec.get("padding_idx", None)
        self.oov_idx = feature_spec["oov_idx"]
        self.freeze_emb = feature_spec["freeze_emb"]
        self.pretrain_embedding = self.load_pretrained_embedding(feature_spec["vocab_size"],
                                                                 pretrain_dim,
                                                                 pretrain_path,
                                                                 vocab_path,
                                                                 feature_name,
                                                                 freeze=self.freeze_emb,
                                                                 padding_idx=padding_idx)
        if pretrain_usage != "init":
            self.id_embedding = nn.Embedding(feature_spec["vocab_size"],
                                             embedding_dim,
                                             padding_idx=padding_idx)
        self.proj = None
        if pretrain_usage in ["init", "sum"] and embedding_dim != pretrain_dim:
            self.proj = nn.Linear(pretrain_dim, embedding_dim, bias=False)
        if pretrain_usage == "concat":
            self.proj = nn.Linear(pretrain_dim + embedding_dim, embedding_dim, bias=False)

    def reset_parameters(self, embedding_initializer):
        if self.pretrain_usage in ["sum", "concat"]:
            nn.init.zeros_(self.id_embedding.weight) # set oov token embeddings to zeros
            embedding_initializer(self.id_embedding.weight[1:self.oov_idx, :])

    def load_feature_vocab(self, vocab_path, feature_name):
        with io.open(vocab_path, "r", encoding="utf-8") as fd:
            vocab = json.load(fd)
            vocab_type = type(list(vocab.items())[1][0]) # get key dtype
        return vocab[feature_name], vocab_type

    def load_pretrained_embedding(self, vocab_size, pretrain_dim, pretrain_path, vocab_path,
                                  feature_name, freeze=False, padding_idx=None):
        embedding_layer = nn.Embedding(vocab_size,
                                       pretrain_dim,
                                       padding_idx=padding_idx)
        if freeze:
            embedding_matrix = np.zeros((vocab_size, pretrain_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(vocab_size, pretrain_dim))
            if padding_idx:
                embedding_matrix[padding_idx, :] = np.zeros(pretrain_dim) # set as zero for PAD
        logging.info("Loading pretrained_emb: {}".format(pretrain_path))
        keys, embeddings = load_pretrain_emb(pretrain_path, keys=["key", "value"])
        assert embeddings.shape[-1] == pretrain_dim, f"pretrain_dim={pretrain_dim} not correct."
        vocab, vocab_type = self.load_feature_vocab(vocab_path, feature_name)
        keys = keys.astype(vocab_type) # ensure the same dtype between pretrained keys and vocab keys
        for idx, word in enumerate(keys):
            if word in vocab:
                embedding_matrix[vocab[word]] = embeddings[idx]
        embedding_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
        if freeze:
            embedding_layer.weight.requires_grad = False
        return embedding_layer

    def forward(self, inputs):
        mask = (inputs <= self.oov_idx).float()
        pretrain_emb = self.pretrain_embedding(inputs)
        if not self.freeze_emb:
            pretrain_emb = pretrain_emb * mask.unsqueeze(-1)
        if self.pretrain_usage == "init":
            if self.proj is not None:
                feature_emb = self.proj(pretrain_emb)
            else:
                feature_emb = pretrain_emb
        else:
            id_emb = self.id_embedding(inputs)
            id_emb = id_emb * mask.unsqueeze(-1)
            if self.pretrain_usage == "sum":
                if self.proj is not None:
                    feature_emb = self.proj(pretrain_emb) + id_emb
                else:
                    feature_emb = pretrain_emb + id_emb
            elif self.pretrain_usage == "concat":
                feature_emb = torch.cat([pretrain_emb, id_emb], dim=-1)
                feature_emb = self.proj(feature_emb)
        return feature_emb