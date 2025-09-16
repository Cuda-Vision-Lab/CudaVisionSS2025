# from encoder import MultiModalVitEncoder
# from decoder import VitDecoder
#from predictor import VitPredictor

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self):
        
        encoder_embed_dim = None,
        decoder_embed_dim = None,
        num_heads = None,
        encoder_depth = None,
        decoder_depth = None
        patch_size=16, 
        in_chans=3
        
        # self.encoder = MultiModalVitEncoder()
        # self.decoder = VitDecoder()
        # self.predictor = VitPredictor(
        #                             embed_dim=embed_dim,
        #                             num_heads=4,
        #                             mlp_size=256
        #                             ) if use_predictor else nn.Identity()
        pass
    
    def make_encoder (self):
        pass
    
    def make_decoder(self):
        pass
    
    def make_redictor(self):
        pass