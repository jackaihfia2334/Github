'''
Author: jackaihfia2334 2598207826@qq.com
Date: 2022-10-21 09:59:36
LastEditors: jackaihfia2334 2598207826@qq.com
LastEditTime: 2022-10-24 19:37:14
FilePath: \DIN-pytorch1\layer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP( nn.Module):
    
    def __init__(self, input_dimension, hidden_size , target_dimension = 1, activation_layer = 'LeakyReLU'):
        super().__init__()

        Activation = nn.LeakyReLU

        # if activation_layer == 'DICE': pass
        # elif activation_layer == 'LeakyReLU': pass

        def _dense( in_dim, out_dim, bias = False):
            return nn.Sequential(
                nn.Linear( in_dim, out_dim, bias = bias),
                nn.BatchNorm1d( out_dim),
                Activation( 0.1 ))

        dimension_pair = [input_dimension] + hidden_size
        layers = [ _dense( dimension_pair[i], dimension_pair[i+1]) for i in range( len( hidden_size))]

        layers.append( nn.Linear( hidden_size[-1], target_dimension))
        layers.insert( 0, nn.BatchNorm1d( input_dimension) )

        self.model = nn.Sequential( *layers )
    
    def forward( self, X): return self.model( X)


class InputEmbedding( nn.Module):

    def __init__(self, n_uid, n_mid, n_cid, embedding_dim ):
        super().__init__()
        self.user_embedding_unit = nn.Embedding( n_uid, embedding_dim)
        self.material_embedding_unit = nn.Embedding( n_mid, embedding_dim)
        self.category_embedding_unit = nn.Embedding( n_cid, embedding_dim)

    def forward( self, user, material, category, material_historical, category_historical, 
                                       material_historical_neg, category_historical_neg, neg_smaple = False ):

        user_embedding = self.user_embedding_unit( user)

        material_embedding = self.material_embedding_unit( material)
        material_historical_embedding = self.material_embedding_unit( material_historical)

        category_embedding = self.category_embedding_unit( category)
        category_historical_embedding = self.category_embedding_unit( category_historical)

        material_historical_neg_embedding = self.material_embedding_unit( material_historical_neg) if neg_smaple else None  
        category_historical_neg_embedding = self.category_embedding_unit( category_historical_neg) if neg_smaple else None

        ans = [ user_embedding, material_historical_embedding, category_historical_embedding, 
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding ]
        return tuple( map( lambda x: x.squeeze() if x != None else None , ans) )



class AttentionLayer( nn.Module):

    def __init__(self, embedding_dim, hidden_size, activation_layer = 'sigmoid'):
        super().__init__()

        Activation = nn.Sigmoid
        if activation_layer == 'Dice': pass
        
        def _dense( in_dim, out_dim):
            return nn.Sequential( nn.Linear( in_dim, out_dim), Activation() )
        
        dimension_pair = [embedding_dim * 8] + hidden_size  #输入维度是8*embedding  D = 2*embedding
        #输入是 fact, query, fact * query, query - fact 
        #fact = item_historical_embedding
        #query = item_embedding
        layers = [ _dense( dimension_pair[i], dimension_pair[i+1]) for i in range(len( hidden_size))]
        layers.append( nn.Linear( hidden_size[-1], 1) )
        self.model = nn.Sequential(*layers)
    
    def forward( self, query, fact, mask, return_scores = False):
        B, T, D = fact.shape
        
        query = torch.ones((B, T, 1) ).type( query.type() ) * query.view( (B, 1, D)) 
        # query = query.view(-1).expand( T, -1).view( T, B, D).permute( 1, 0, 2)
        # 把item_embedding扩展为 (B, T, D)

        combination = torch.cat( [ fact, query, fact * query, query - fact ], dim = 2)

        scores = self.model( combination).squeeze()
        scores = torch.where( mask == 1, scores, torch.ones_like( scores) * ( -2 ** 31 ) )

        scores = ( scores.softmax( dim = -1) * mask ).view( (B , 1, T))

        if return_scores: return scores.squeeze()
        return torch.matmul( scores, fact).squeeze()