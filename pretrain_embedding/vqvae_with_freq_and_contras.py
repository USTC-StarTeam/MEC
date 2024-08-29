import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (flat_x.pow(2).sum(1, keepdim=True)
                     + self.embedding.weight.pow(2).sum(1)
                     - 2 * flat_x @ self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = self.embedding(encoding_indices).view_as(x)
        
        # 计算正则化损失
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, encoding_indices, perplexity

    def get_embedding_from_codes(self, codes):
        # 输入codes，得到对应的embedding
        embeddings = self.embedding(codes)
        return embeddings

class MultiDimVQVAE(nn.Module):
    def __init__(self, input_dim, num_embeddings, embedding_dim, num_splits):
        super(MultiDimVQVAE, self).__init__()
        self.num_splits = num_splits
        self.embedding_dim = embedding_dim
        self.encoder = nn.Linear(input_dim, embedding_dim * num_splits)
        self.vq_layers = nn.ModuleList([VQEmbedding(num_embeddings, embedding_dim) for _ in range(num_splits)])
        self.decoder = nn.Linear(embedding_dim*num_splits, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        split_z = torch.chunk(z, self.num_splits, dim=1)
        quantized_list = []
        code_list = []
        perplexities = []
        for i, vq_layer in enumerate(self.vq_layers):
            quantized, code, perplexity = vq_layer(split_z[i])
            quantized_list.append(quantized)
            code_list.append(code.view(-1))
            perplexities.append(perplexity)
        
        # quantized = torch.mean(torch.stack(quantized_list, dim=0), dim=0)
        quantized = torch.cat(quantized_list, dim=1)
        x_recon = self.decoder(quantized)
        code_list = torch.stack(code_list).t()
        
        avg_perplexity = torch.mean(torch.tensor(perplexities))
        return x_recon, quantized, code_list, avg_perplexity

    def get_embeddings_from_codes(self, codes):
        batch_size, code_dim = codes.shape
        assert code_dim == self.num_splits, "code dim must be equal to num_splits"
        
        embeddings_list = []
        for i, vq_layer in enumerate(self.vq_layers):
            embeddings = vq_layer.get_embedding_from_codes(codes[:, i])
            embeddings_list.append(embeddings)

        embeddings = torch.cat(embeddings_list, dim=1)
        return embeddings

# 训练模型的函数
def train_model(args, feature_name, model, dataloader, epochs=10, lr=1e-4, reg=1e-4, contrastive_weight=2e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    best_model_path = './checkpoints/'+args['dataset_id']+'_'+args['model']+'_layers='+str(args['layers'])+'_codedim='+str(args['code_dim'])+'_embeddingdim='+str(int(40//float(args['layers'])))+'_'+str(args['cut_down'])+f"_best_model_{feature_name}.pth"
    
    if args['print_out'] == 1:
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for data in tqdm(dataloader, desc='Epoch='+str(epoch), leave=False):
                data = data[0].to(device) 
                optimizer.zero_grad()
                x_recon, quantized, codes, avg_perplexity = model(data)
                recon_loss = F.mse_loss(x_recon, data)
                
                reg_loss = (avg_perplexity - torch.log(torch.tensor(model.vq_layers[0].num_embeddings, device=device)))**2
                
                neg_codes = codes.clone()
                random_idx = torch.randint(0, codes.size(1), (codes.size(0),))
                neg_codes[:, random_idx] = torch.randint(0, model.vq_layers[0].num_embeddings, (codes.size(0),), device=device)
                neg_quantized = model.get_embeddings_from_codes(neg_codes)
                
                pos_sim = F.cosine_similarity(data, quantized, dim=-1)
                neg_sim = F.cosine_similarity(data, neg_quantized, dim=-1)
                contrastive_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
                contrastive_loss = contrastive_loss.mean()
                
                loss = recon_loss + reg_loss * reg + contrastive_loss * contrastive_weight
                
                loss.backward()
                optimizer.step()
                
                total_loss.append([loss.item(), reg_loss.item(), avg_perplexity.item(), contrastive_loss.item()])
                
            total_loss = np.array(total_loss).T
            print(np.mean(total_loss[0]), np.mean(total_loss[1]), np.mean(total_loss[2]), np.mean(total_loss[3]))
            if np.mean(total_loss[0]) < best_loss:
                best_loss = np.mean(total_loss[0])
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(torch.load(best_model_path))
                return model
    else:
        for epoch in range(epochs):
            model.train()
            total_loss = []
            for data in dataloader:
                data = data[0].to(device)  
                optimizer.zero_grad()
                x_recon, quantized, codes, avg_perplexity = model(data)
                recon_loss = F.mse_loss(x_recon, data)
                
                reg_loss = (avg_perplexity - torch.log(torch.tensor(model.vq_layers[0].num_embeddings, device=device)))**2
                
                neg_codes = codes.clone()
                random_idx = torch.randint(0, codes.size(1), (codes.size(0),))
                neg_codes[:, random_idx] = torch.randint(0, model.vq_layers[0].num_embeddings, (codes.size(0),), device=device)
                neg_quantized = model.get_embeddings_from_codes(neg_codes)
                
                pos_sim = F.cosine_similarity(quantized, quantized, dim=-1)
                neg_sim = F.cosine_similarity(quantized, neg_quantized, dim=-1)
                contrastive_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
                contrastive_loss = contrastive_loss.mean()
                
                loss = recon_loss + reg_loss * reg + contrastive_loss * contrastive_weight
                
                loss.backward()
                optimizer.step()
                
                total_loss.append([loss.item(), reg_loss.item(), avg_perplexity.item(), contrastive_loss.item()])
                
            total_loss = np.array(total_loss).T
            # print(np.mean(total_loss[0]), np.mean(total_loss[1]), np.mean(total_loss[2]), np.mean(total_loss[3]))
            if np.mean(total_loss[0]) < best_loss:
                best_loss = np.mean(total_loss[0])
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                model.load_state_dict(torch.load(best_model_path))
                return model
    
    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)
    return model

def save_codebook(model, output_dir, device):
    """
    保存模型的 code book，即每个 code 和对应的 embedding
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, vq_layer in enumerate(model.vq_layers):
        codes = torch.arange(vq_layer.num_embeddings, device=device)
        embeddings = vq_layer.embedding(codes).detach().cpu().numpy()
        
        np.save(os.path.join(output_dir, f"vq_layer_{i}_embeddings.npy"), embeddings)

def list_files(directory):
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            files.append(item)
    return files

# 将嵌入转换为codes并保存的函数
def convert_embeddings_to_codes(dataloader, model, batch_size):
    all_codes = []
    all_quantized = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].to(device)
            _, quantized, codes, _ = model(data)
            all_quantized.append(quantized.cpu())
            all_codes.append(codes.cpu())
    
    all_quantized = torch.cat(all_quantized, dim=0)
    all_codes = torch.cat(all_codes, dim=0)
    
    return all_quantized, all_codes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='fm', help='The model to run. deepfm, fm, ffm, or ffmv2.')
    parser.add_argument('--dataset_id', type=str, default='criteo_x1_default', help='The dataset id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--layers', type=int, default=4, help='The number of layers in the model.')
    parser.add_argument('--code_dim', type=int, default=4096, help='The dim of codebook.')
    parser.add_argument('--print_out', type=int, default=1, help='Whether to print the log of the model.')
    parser.add_argument('--cut_down', type=int, default=100000, help='When to use vqvae.')
    parser.add_argument('--batch_size', type=int, default=8000, help='Batch size for DataLoader.')
    parser.add_argument('--regularization_weight', type=float, default=1e-4, help='The weight of the regularization loss.')
    args = vars(parser.parse_args())
    
    # print("[Running] python {}".format(' '.join(sys.argv)))
    num_embeddings = args['code_dim']
    embedding_dim = int(40 / args['layers'])
    layers = args['layers']
    commitment_cost = 0.25
    epochs = 100
    learning_rate = 1e-4
    num_workers = 6

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args['gpu']))
    else:
        device = 'cpu'
    
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('../data/pretrain_result'):
        os.makedirs('../data/pretrain_result')
    
    cutdown_num = args['cut_down']
    model_name = args['model'].lower()
    tar_dir = f'../data/pretrain_result/' + args['dataset_id'] + '_' + f'{model_name}_embeddings/'
    npy_file_list = list_files(tar_dir)
    os.makedirs(f'../data/pretrain_result/' + args['dataset_id'] + '_' + f'{model_name}_layers={layers}_codedim={num_embeddings}_embeddingdim={embedding_dim}_{cutdown_num}_feature_logfreq_cons_2_codebook', exist_ok=True)
    
    import json
    with open(args['dataset_id'] + '_feature_value_counts.json', 'r') as f:
        feature_value_counts = json.load(f)
    
    import math
    for i, npy_file in enumerate(npy_file_list):
        data = np.load(os.path.join(tar_dir, npy_file))
        feature_name = npy_file.split('_embedding')[0]
        input_dim = data.shape[-1]
        feature_num = data.shape[0]
        if feature_num < cutdown_num:
            continue
        
        repeat_num = []
        for j in range(feature_num):
            if not f'{j+1}.0' in feature_value_counts[feature_name] or feature_value_counts[feature_name][f'{j+1}.0'] < 2:
                repeat_num.append(1)
            else:
                repeat_num.append(int(math.log2(feature_value_counts[feature_name][f'{j+1}.0'])))
        data_repeat = np.repeat(data, repeat_num, axis=0)
        input_data_repeat = torch.tensor(data_repeat, dtype=torch.float32).view(-1, input_dim)
        
        dataset_repeat = TensorDataset(input_data_repeat)
        dataloader = DataLoader(dataset_repeat, batch_size=args['batch_size'], shuffle=True, num_workers=num_workers)

        vqvae_model = MultiDimVQVAE(input_dim=input_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim, num_splits=layers).to(device)

        trained_model = train_model(args, feature_name, vqvae_model, dataloader, epochs=epochs, reg=args['regularization_weight'])

        save_codebook(trained_model, f'../data/pretrain_result/' + args['dataset_id'] + '_' + f'{model_name}_layers={layers}_codedim={num_embeddings}_embeddingdim={embedding_dim}_{cutdown_num}_feature_logfreq_cons_2_codebook/{feature_name}', device)
        if args['print_out'] == 1:
            print(f"Saved VQ layer {feature_name} code book. Now: {i+1}/{len(npy_file_list)}")
        
        input_data = torch.tensor(data, dtype=torch.float32).view(-1, input_dim)
        dataset = TensorDataset(input_data)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=num_workers)
        quantized, codes = convert_embeddings_to_codes(dataloader, trained_model, args['batch_size'])
        np.save(f"../data/pretrain_result/" + args['dataset_id'] + '_' + f"{model_name}_layers={layers}_codedim={num_embeddings}_embeddingdim={embedding_dim}_{cutdown_num}_feature_logfreq_cons_2_codebook/{feature_name}/codes.npy", codes.cpu().numpy())
        np.save(f"../data/pretrain_result/" + args['dataset_id'] + '_' + f"{model_name}_layers={layers}_codedim={num_embeddings}_embeddingdim={embedding_dim}_{cutdown_num}_feature_logfreq_cons_2_codebook/{feature_name}/embeddings.npy", quantized.cpu().detach().numpy())
        
        del input_data
        del vqvae_model
        del trained_model
        torch.cuda.empty_cache()