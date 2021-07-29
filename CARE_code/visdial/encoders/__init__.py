from visdial.encoders.mvan.mvan import MVANEncoder
from visdial.encoders.basemodel.basemodel import BaseModelEncoder
from visdial.encoders.transformermodel.transformermodel import TransformerModelEncoder
from visdial.encoders.rva.rvamodel import RvAEncoder

def Encoder(hparams, *args):
  name_enc_map = {
    "mvan": MVANEncoder,  # Ours
    "basemodel": BaseModelEncoder,
    'transformermodel': TransformerModelEncoder,
    "rvamodel": RvAEncoder
  }
  return name_enc_map[hparams.encoder](hparams, *args)