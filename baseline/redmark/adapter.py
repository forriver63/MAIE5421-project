"""Model I/O only; evaluation conditions and payloads come from teammates."""
from pathlib import Path
import ast
import importlib.util
import os
import numpy as np

ROOT=Path(os.environ['WM_ROOT']).resolve()

class _ReDMarkBase:
    capacity=1024
    def __init__(self):
        import tensorflow as tf
        import scipy.io as sio
        import h5py
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        folder=ROOT/'third_party/redmark'
        spec=importlib.util.spec_from_file_location('redmark_circular',folder/'include/my_circular_layer.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tree=ast.parse((folder/'test.py').read_text(encoding='utf-8'))
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
                   ['multiply_255','divide_255','scalar_output_shape','multiply_scalar','buildModel']]
        # TF2.15 requires a shape matching the scalar broadcast used in the paper.
        # No learned layers or checkpoint tensors change.
        for node in ast.walk(functions[-1]):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='input_strenght_alpha' for t in node.targets):
                for kw in node.value.keywords:
                    if kw.arg=='shape': kw.value=ast.Tuple(elts=[ast.Constant(1)]*3,ctx=ast.Load())
        ns=dict(tf=tf,layers=tf.keras.layers,np=np,sio=sio,Conv2D_circular=module.Conv2D_circular)
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'redmark_build','exec'),ns)
        old=os.getcwd()
        try:
            os.chdir(folder)
            self.encoder,self.decoder=ns['buildModel']('logs/MT-Net/Weights/weights_final.h5')
        finally: os.chdir(old)
        with h5py.File(folder/'logs/MT-Net/Weights/weights_final.h5','r') as h:
            names={n.decode() if isinstance(n,bytes) else n for n in h.attrs['layer_names']}
            for model in [self.encoder,self.decoder]:
                for layer in model.layers:
                    if layer.trainable_weights:
                        assert layer.name in names,layer.name
                        group=h[layer.name]
                        saved=[np.array(group[n]) for n in group.attrs['weight_names']]
                        assert len(saved)==len(layer.get_weights())
                        for a,b in zip(saved,layer.get_weights()): np.testing.assert_array_equal(a,b)
        # Author's diagonal placement: 64 blocks of 16 bits, each repeated four times.
        mapping=np.empty((16,16),dtype=int)
        for position,(d,i) in enumerate((d,i) for d in range(16) for i in range(16)):
            mapping[i,(i+d)%16]=position//4
        self.mapping=mapping.reshape(-1)
    @staticmethod
    def patches(gray):
        return gray.reshape(16,32,16,32).transpose(0,2,1,3).reshape(256,32,32,1)
    @staticmethod
    def tile(patches):
        return patches.reshape(16,16,32,32).transpose(0,2,1,3).reshape(512,512)
    def embed_gray(self,gray,bits):
        normalized=(gray.astype(np.float32)-128)/255
        message=np.asarray(bits,dtype=np.float32).reshape(64,4,4,1)[self.mapping]
        output=self.encoder([self.patches(normalized),message,np.ones((256,1,1,1),np.float32)],training=False).numpy()
        return np.clip(self.tile(output)*255+128,0,255).astype(np.uint8)
    def extract_gray(self,gray):
        prediction=self.decoder(self.patches((gray.astype(np.float32)-128)/255),training=False).numpy()
        votes=np.zeros((64,4,4),dtype=np.float32)
        for i,k in enumerate(self.mapping): votes[k]+=prediction[i,:,:,0]>.5
        return votes.reshape(-1)/4,.5
    def embed(self,rgb,bits):
        import cv2
        # Standard luminance-only RGB adaptation; keep chroma, like transform baselines.
        ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
        ycc[:,:,0]=self.embed_gray(ycc[:,:,0],bits)
        return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB)
    def extract(self,rgb):
        import cv2
        # Follow MBRS: resize received whole image to fixed network geometry.
        if rgb.shape[:2]!=(512,512):
            import torch
            from torch.nn.functional import interpolate
            t=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
            t=interpolate(t,size=(512,512),mode='bilinear',align_corners=False,antialias=True)
            rgb=np.rint(t[0].permute(1,2,0).numpy()*255).astype(np.uint8)
        return self.extract_gray(cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0])

class ReDMark(_ReDMarkBase):
 def __init__(self):
  super().__init__()
  spec=importlib.util.spec_from_file_location('redmark_author_utils',ROOT/'third_party/redmark/include/utils.py')
  self.vf=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.vf)
 def embed_gray(self,gray,bits):
  normalized=(gray.astype(np.float64)-128.)/255.
  message=np.asarray(bits,np.float32).reshape(64,4,4,1)[self.mapping]
  output=self.encoder([self.vf.partitioning(normalized,32),message,np.ones((256,1,1,1),np.float32)],training=False).numpy()
  return np.clip(self.vf.tiling(output,512)*255.+128.,0,255).astype(np.uint8)
 def extract_gray(self,gray):
  normalized=(gray.astype(np.float64)-128.)/255.
  prediction=self.decoder(self.vf.partitioning(normalized,32),training=False).numpy()
  votes=np.zeros((64,4,4),np.float32)
  for i,k in enumerate(self.mapping):votes[k]+=prediction[i,:,:,0]>.5
  return votes.reshape(-1)/4,.5
