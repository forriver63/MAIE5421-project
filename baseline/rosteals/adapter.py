"""Official inference components with 56 useful bits and their native BCH code."""
from pathlib import Path
import ast,sys,types,json
import numpy as np
import os
ROOT=Path(os.environ['WM_ROOT']).resolve()

import torch
import torch.nn.functional as F
from PIL import Image

class NativeBCH:
    def __init__(self):
        import bchlib
        self.bch=bchlib.BCH(5,137)
        assert self.bch.ecc_bytes==5
    def encode(self,bits56):
        data=bytearray(np.packbits(bits56).tobytes())
        assert len(data)==7
        return np.concatenate([np.unpackbits(np.frombuffer(data+self.bch.encode(data),dtype=np.uint8)),np.zeros(4,dtype=np.uint8)])
    def decode(self,bits100):
        packet=np.packbits(np.asarray(bits100[:96],dtype=np.uint8)).tobytes()
        data,ecc=bytearray(packet[:7]),bytearray(packet[7:])
        errors=self.bch.decode(data,ecc)
        if errors<0:return None,errors
        self.bch.correct(data,ecc)
        return np.unpackbits(np.frombuffer(data,dtype=np.uint8)),errors

def resize_rgb(rgb,size):
    t=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
    return F.interpolate(t,size=(size,size),mode='bilinear',align_corners=False,antialias=True)

def extract_classes(path,names,namespace,disable_pretrained=False):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
    if disable_pretrained:
        for node in ast.walk(ast.Module(body=nodes,type_ignores=[])):
            if isinstance(node,ast.keyword) and node.arg=='pretrained':node.value=ast.Constant(False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)

class RoSteALS:
    size=256
    def __init__(self):
        import torchvision
        from contextlib import contextmanager
        folder=ROOT/'third_party/rosteals_author_demo';sys.path.insert(0,str(folder))
        from ldm.modules.diffusionmodules.model import Encoder,Decoder
        from ldm.modules.diffusionmodules.util import conv_nd,zero_module
        from ldm.util import instantiate_from_config
        from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
        ns=dict(torch=torch,nn=torch.nn,np=np,thf=F,F=F,torchvision=torchvision,
            pl=types.SimpleNamespace(LightningModule=torch.nn.Module),contextmanager=contextmanager,
            Encoder=Encoder,Decoder=Decoder,instantiate_from_config=instantiate_from_config,
            DiagonalGaussianDistribution=DiagonalGaussianDistribution,conv_nd=conv_nd,zero_module=zero_module)
        extract_classes(folder/'ldm/models/autoencoder.py',['VQModel','VQModelInterface'],ns)
        extract_classes(folder/'cldm/ae.py',['View','SecretEncoder3'],ns)
        extract_classes(folder/'cldm/diffsteg.py',['SecretDecoder'],ns,True)
        self.ae=ns['VQModelInterface'](embed_dim=3,n_embed=8192,lossconfig={'target':'torch.nn.Identity'},
            ddconfig=dict(double_z=False,z_channels=3,resolution=256,in_channels=3,out_ch=3,ch=128,ch_mult=[1,2,4],num_res_blocks=2,attn_resolutions=[],dropout=0.0))
        self.control=ns['SecretEncoder3'](secret_len=100,resolution=64,base_res=32)
        self.decoder=ns['SecretDecoder'](arch='resnet50',secret_len=100)
        state=torch.load(folder/'models/epoch=000017-step=000449999.ckpt',map_location='cpu',weights_only=False)
        state=state.get('state_dict',state)
        self.loaded={}
        for name in ['ae','control','decoder']:
            module=getattr(self,name);prefix=name+'.'
            sd={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            module.load_state_dict(sd,strict=True);module.eval();self.loaded[name]=len(sd)
        self.ignored_checkpoint_keys=[k for k in state if not k.startswith(('ae.','control.','decoder.'))]
    @torch.inference_mode()
    def native_marked(self,small,bits100):
        latent=self.ae.encode(small)
        residual,_=self.control(None,torch.tensor(bits100,dtype=torch.float32)[None])
        return self.ae.decode(latent+residual)
    @torch.inference_mode()
    def embed(self,rgb,bits100):
        small=resize_rgb(rgb,256)*2-1
        marked=self.native_marked(small,bits100).clamp(-1,1)
        residual=F.interpolate((marked-small)/2,size=rgb.shape[:2],mode='bilinear',align_corners=False,antialias=True)
        return np.rint(np.clip(rgb.astype(np.float32)/255+residual[0].permute(1,2,0).numpy(),0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def extract(self,rgb):
        return self.decoder(resize_rgb(rgb,256)*2-1)[0].numpy(),0.0
    @torch.inference_mode()
    def native_check(self,rgb,bits100):
        small=resize_rgb(rgb,256)*2-1
        return self.decoder(self.native_marked(small,bits100))[0].numpy()>0
