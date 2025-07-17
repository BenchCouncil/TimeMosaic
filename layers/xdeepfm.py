import torch
from torch import nn
import torch.nn.functional as F


class DNN(nn.Module):
    
    def __init__(self, input_dim, hidden_units, activation='relu', dropout_rate=0, use_bn=False):
        super(DNN, self).__init__()
        layers = []
        current_dim = input_dim
        
        if activation.lower() == 'relu':
            activation_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.Identity()

        for units in hidden_units:
            layers.append(nn.Linear(current_dim, units))
            
            if use_bn:
                layers.append(nn.BatchNorm1d(units))

            layers.append(activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = units
        
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)



class CIN(nn.Module):
    
    def __init__(self, configs, num_patches, d_model, cross_layer_sizes, activation='relu', dropout_rate=0.1, stride=8):
        super().__init__()
        self.d_model = d_model
        self.cross_layer_sizes = cross_layer_sizes
        self.all_patch = num_patches
        self.channel = configs.channel_xpatchfm
        self.num_patches = num_patches = int((configs.seq_len - configs.patch_len) / stride + 2) 

        self.activation = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'identity': nn.Identity()
        }[activation.lower()]
        self.weights = nn.ParameterList()
        last_H = self.all_patch 
        for i, H_k in enumerate(self.cross_layer_sizes):
            weight = nn.Parameter(torch.randn(1, last_H, self.all_patch, H_k))
            nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
            last_H = H_k

        total_feature_maps = sum(self.cross_layer_sizes)

        self.projection = nn.Conv1d(in_channels=total_feature_maps, out_channels=self.all_patch, kernel_size=1)

        # 正则化层
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        

        # softmask
        self.threshold = 0.5
        # self.mask_logits = nn.Parameter(torch.randn(self.all_patch))
        self.mask_linear = nn.Linear(d_model, 1)
        # self.mask_linear = nn.ModuleList([
        #     nn.Linear(d_model, 1) for _ in range(len(self.cross_layer_sizes))
        # ])


    def forward(self, x):
        """
        时序CIN块的前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 `[batch_size, num_patches, d_model]`。

        返回:
            torch.Tensor: 输出张量，形状与输入相同，为 `[batch_size, num_patches, d_model]`。
        """
        
        
    
        # print(x.shape)
      
            # mask 1
            # soft_mask = torch.sigmoid(self.mask_logits)
            # x = x * soft_mask.view(1, -1, 1)
            
            # mask 2 
            # mask_logits = self.mask_linear(x).squeeze(-1)      # shape: (B, L)
            # soft_mask = torch.sigmoid(mask_logits)             # range: (0,1)
            # x = x * soft_mask.unsqueeze(-1)
            
            # mask 3
            # mask_logits = self.mask_linear(x).squeeze(-1)      # shape: (B, L)
            # soft_mask = torch.sigmoid(mask_logits)             # range: (0,1)
            # hard_mask = (soft_mask > self.threshold).float()  # (B, L), binary
            # x = x * hard_mask.unsqueeze(-1)
        
        # cos_sim = F.cosine_similarity(x_i.unsqueeze(2), x_j.unsqueeze(1), dim=-1)  # [B, M, N]
        # outer = torch.einsum('bmd,bnd->bmnd', x_i, x_j)
        # output = outer * cos_sim.unsqueeze(-1)

        if self.channel == "CD":
            x = x.view(-1, self.all_patch, self.d_model)
            mask_logits = self.mask_linear(x).squeeze(-1)      # shape: (B, L)
            soft_mask = torch.sigmoid(mask_logits)             # range: (0,1)
            x = x * soft_mask.unsqueeze(-1)

        x_residual = x

        layer_outputs = []
        
        X_0 = x
        X_k = x
        
        for i, H_k in enumerate(self.cross_layer_sizes):

            # Method1_mask:
            # if self.channel == "CD":
            #     mask_logits = self.mask_linear(X_k).squeeze(-1)      # shape: (B, L)
            #     soft_mask = torch.sigmoid(mask_logits)             # range: (0,1)
            #     X_k = X_k * soft_mask.unsqueeze(-1)
            outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)
            
            # Method2_CosSimilarity
            # cos_sim = F.cosine_similarity(X_k.unsqueeze(2), X_0.unsqueeze(1), dim=-1)  # [B, M, N]
            # outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)
            # outer_product = outer_product * cos_sim.unsqueeze(-1)


            # cos_sim = F.cosine_similarity(X_k.unsqueeze(2), X_0.unsqueeze(1), dim=-1)  # [B, M, N]
            # cos_sim = (cos_sim+1)/2
            # aggregated_scores = cos_sim.sum(dim=2).unsqueeze(-1)
            # X_k = X_k * aggregated_scores
            # outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)
            # outer_product = outer_product * cos_sim.unsqueeze(-1)
             
            # Method3_Cos Similariry at X_k
            # cos_sim = F.cosine_similarity(X_k.unsqueeze(2), X_0.unsqueeze(1), dim=-1) 
            # aggregated_scores = cos_sim.sum(dim=2).unsqueeze(-1)
            # X_k = X_k * aggregated_scores
            # outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)

            # Method4 cos at outerproduct
            # cos_sim = F.cosine_similarity(X_k.unsqueeze(2), X_0.unsqueeze(1), dim=-1)
            # mean_cos_sim = cos_sim.mean() 
            # alpha = (mean_cos_sim + 1) / 2 
            # outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)
            # outer_product = outer_product * alpha

            # Method5 cos->alpha ,alpha at outerproduct
            # cos_sim = F.cosine_similarity(X_k.unsqueeze(2), X_0.unsqueeze(1), dim=-1)
            # print(cos_sim.min())
            # alpha = torch.sigmoid(cos_sim.mean(dim=[1,2]) ).view(-1,1,1,1)
            # outer_product = torch.einsum('bmd,bnd->bmnd', X_k, X_0)
            # outer_product = outer_product * alpha

            X_next = torch.einsum('bmnd,imnk->bikd', outer_product, self.weights[i])
          
            X_next = X_next.squeeze(1) 
            X_next = self.activation(X_next)
            
            layer_outputs.append(X_next)
            X_k = X_next

        cin_output = torch.cat(layer_outputs, dim=1) # concat or max or mean or ...
        projected_output = self.projection(cin_output)

        output = self.dropout(projected_output)
        output = self.norm(x_residual + output)
        
        if self.channel == "CD":
            output = output.view(-1, self.num_patches, self.d_model)
            
        return output


class xDeepFM_Interaction(nn.Module):

    def __init__(self, configs, num_patches, dropout_rate=0.1):
        super().__init__()
        
        # 通过configs对象安全地获取在run.py中定义的参数
        self.use_linear =configs.use_linear
        self.use_dnn =configs.use_dnn
        self.use_cin = configs.use_cin
        
        self.num_patch = num_patches
        self.d_model = configs.d_model
        self.gate_hidden_units = configs.gate_hidden_units

        if self.use_linear:
            self.linear_part = nn.Identity()
        '''
        在DNN中
        Input: (bs,N,d_model)
        '''
        if self.use_dnn:
            dnn_input_dim = self.num_patch * self.d_model #展平 >> [bs, N*d_model]
            self.dnn_part = DNN(
                input_dim = dnn_input_dim,
                hidden_units = configs.dnn_hidden_units, #隐藏层
                dropout_rate = configs.dropout
            )
            self.dnn_projection = nn.Linear(configs.dnn_hidden_units[-1], dnn_input_dim)  #映射 从dnn_hidden[-1] -> dnn_input_dim

        if self.use_cin:
            self.cin_part = CIN(
                configs = configs,
                num_patches=self.num_patch * configs.enc_in if configs.channel_xpatchfm == "CD" else self.num_patch,
                d_model=self.d_model,
                # cross_layer_sizes=[150,100,100,50]
                cross_layer_sizes=configs.cin_layer_size
            )
        


        # ==================== 新增门控网络 ====================
        # 我们只在CIN和DNN都使用时才需要门控
        if self.use_cin and self.use_dnn:
            self.gate_network = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid())
        # ======================================================

        # ==================== 补丁级门控网络  ====================
        if self.use_cin and self.use_dnn:
            self.gate_network_patchwise = nn.Sequential(
                nn.Linear(self.d_model, 32), 
                nn.ReLU(),
                nn.Linear(32, 1),           
                nn.Sigmoid()
            )
        # ==================================================================

        
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.d_model)
        self.alpha = nn.Parameter(torch.ones(1)) 


    def forward(self, x):

        # outputs = []
        # if self.use_linear:
        #     outputs.append(self.linear_part(x))
        # if self.use_dnn:
        #     batch_size, num_patch, d_model = x.shape
        #     dnn_input = x.flatten(start_dim=1)
        #     dnn_hidden_output = self.dnn_part(dnn_input)
        #     dnn_output = self.dnn_projection(dnn_hidden_output)
        #     dnn_output = dnn_output.view(batch_size, num_patch, d_model)
        #     outputs.append(dnn_output)
        # if self.use_cin:
        #     outputs.append(self.cin_part(x))
        # if not outputs:
        #     return torch.zeros_like(x)
        # final_output = torch.cat(outputs, dim=1)
        # return final_output

    
    
        '''    if self.use_dnn:
            batch_size, num_patch, d_model = x.shape
            dnn_input = x.flatten(start_dim=1)
            dnn_hidden_output = self.dnn_part(dnn_input)
            dnn_output = self.dnn_projection(dnn_hidden_output)
            dnn_output = dnn_output.view(batch_size, num_patch, d_model)
          
        if self.use_cin:
            cin_out = self.cin_part(x)
        print(alpha)
        alpha = torch.sigmoid(self.alpha)  
        output = alpha * cin_out + (1 - alpha) * dnn_output

        return output'''

        cin_out = self.cin_part(x) if self.use_cin else None

        if self.use_dnn:
            batch_size, num_patch, d_model = x.shape
            dnn_input = x.flatten(start_dim=1)
            dnn_hidden_output = self.dnn_part(dnn_input)
            dnn_output = self.dnn_projection(dnn_hidden_output)
            dnn_output = dnn_output.view(batch_size, num_patch, d_model)
        
        # gate = self.gate_network(x) 
        gate = self.gate_network_patchwise(x) 
        fused_output = gate * cin_out + (1 - gate) * dnn_output

        
        return fused_output