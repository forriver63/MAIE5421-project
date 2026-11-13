from pathlib import Path
import sys,os,types,importlib
import torch,numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
ROOT=Path(os.environ["WM_ROOT"]).resolve()
SOURCE=ROOT/"third_party/sepmark/network"
pkg=types.ModuleType('network');pkg.__path__=[str(SOURCE)]
pkg.torch=torch;pkg.nn=torch.nn;pkg.F=F;pkg.np=np;pkg.Image=Image;pkg.transforms=transforms
sys.modules['network']=pkg
for name in ['ResBlock','ConvBlock']:
 module=importlib.import_module('network.'+name)
 for k,v in vars(module).items():
  if not k.startswith('_'):setattr(pkg,k,v)
from network.Encoder_U import DW_Encoder
from network.Decoder_U import DW_Decoder
def small(rgb):
 x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255
 return F.interpolate(x,(256,256),mode='bilinear',align_corners=False,antialias=True)*2-1
class SepMark:
 capacity=128
 def __init__(self):
  state=torch.load(ROOT/'models/sepmark/EC_90.pth',map_location='cpu',weights_only=True)
  self.encoder=DW_Encoder(128,attention='se').eval();self.decoder=DW_Decoder(128,attention='se').eval()
  for prefix,m in [('encoder',self.encoder),('decoder_C',self.decoder)]:m.load_state_dict({k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')},strict=True)
 @torch.inference_mode()
 def embed(self,rgb,bits):
  x=small(rgb);marked=self.encoder(x,torch.from_numpy(bits.astype(np.float32))[None]*.2-.1)
  residual=F.interpolate((marked-x)/2,rgb.shape[:2],mode='bilinear',align_corners=False,antialias=True)
  return np.rint(np.clip(rgb.astype(np.float32)/255+residual[0].permute(1,2,0).numpy(),0,1)*255).astype(np.uint8)
 @torch.inference_mode()
 def extract(self,rgb):return self.decoder(small(rgb))[0].numpy(),0.
