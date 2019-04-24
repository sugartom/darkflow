import tensorflow as tf
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
import json
import os

# # # Yitao-TLS-Begin
# # import os
# # import sys
# # from tensorflow.python.saved_model import builder as saved_model_builder
# # from tensorflow.python.saved_model import signature_constants
# # from tensorflow.python.saved_model import signature_def_utils
# # from tensorflow.python.saved_model import tag_constants
# # from tensorflow.python.saved_model import utils
# # from tensorflow.python.util import compat

# # tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# # FLAGS = tf.app.flags.FLAGS

# import grpc
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc

# from tensorflow.python.framework import tensor_util
# # # Yitao-TLS-End

class TFNet(object):

  _TRAINER = dict({
    'rmsprop': tf.train.RMSPropOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adagradDA': tf.train.AdagradDAOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'adam': tf.train.AdamOptimizer,
    'ftrl': tf.train.FtrlOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
  })

  # imported methods
  _get_fps = help._get_fps
  say = help.say
  train = flow.train
  camera = help.camera
  predict = flow.predict
  return_predict = flow.return_predict
  to_darknet = help.to_darknet
  build_train_op = help.build_train_op
  load_from_ckpt = help.load_from_ckpt

  def __init__(self, FLAGS, darknet = None):
    self.ntrain = 0

    if isinstance(FLAGS, dict):
      from ..defaults import argHandler
      newFLAGS = argHandler()
      newFLAGS.setDefaults()
      newFLAGS.update(FLAGS)
      FLAGS = newFLAGS

    self.FLAGS = FLAGS
    if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
      self.say('\nLoading from .pb and .meta')
      self.graph = tf.Graph()
      device_name = FLAGS.gpuName \
        if FLAGS.gpu > 0.0 else None
      with tf.device(device_name):
        with self.graph.as_default() as g:
          self.build_from_pb()
      return

    if darknet is None: 
      darknet = Darknet(FLAGS)
      self.ntrain = len(darknet.layers)

    self.darknet = darknet
    args = [darknet.meta, FLAGS]
    self.num_layer = len(darknet.layers)
    self.framework = create_framework(*args)
    
    self.meta = darknet.meta

    self.say('\nBuilding net ... Fake ...')
    # start = time.time()
    # self.graph = tf.Graph()
    # device_name = FLAGS.gpuName \
    #   if FLAGS.gpu > 0.0 else None
    # with tf.device(device_name):
    #   with self.graph.as_default() as g:
    #     self.build_forward()
    #     self.setup_meta_ops()
    # self.say('Finished in {}s\n'.format(
    #   time.time() - start))

    # ichannel = grpc.insecure_channel("localhost:8500")
    # self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)
  
  def build_from_pb(self):
    with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(
      graph_def,
      name=""
    )
    with open(self.FLAGS.metaLoad, 'r') as fp:
      self.meta = json.load(fp)
    self.framework = create_framework(self.meta, self.FLAGS)

    # Placeholders
    self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
    self.feed = dict() # other placeholders
    self.out = tf.get_default_graph().get_tensor_by_name('output:0')
    
    self.setup_meta_ops()
  
  def build_forward(self):
    pass
    # verbalise = self.FLAGS.verbalise

    # # Placeholders
    # inp_size = [None] + self.meta['inp_size']
    # self.inp = tf.placeholder(tf.float32, inp_size, 'input')
    # self.feed = dict() # other placeholders

    # # Build the forward pass
    # state = identity(self.inp)
    # roof = self.num_layer - self.ntrain
    # self.say(HEADER, LINE)
    # for i, layer in enumerate(self.darknet.layers):
    #   scope = '{}-{}'.format(str(i),layer.type)
    #   args = [layer, state, i, roof, self.feed]
    #   state = op_create(*args)
    #   mess = state.verbalise()
    #   self.say(mess)
    # self.say(LINE)

    # self.top = state
    # self.out = tf.identity(state.out, name='output')

  def setup_meta_ops(self):
    cfg = dict({
      'allow_soft_placement': False,
      'log_device_placement': False
    })

    utility = min(self.FLAGS.gpu, 1.)
    if utility > 0.0:
      self.say('GPU mode with {} usage'.format(utility))
      cfg['gpu_options'] = tf.GPUOptions(
        per_process_gpu_memory_fraction = utility)
      cfg['allow_soft_placement'] = True
    else: 
      self.say('Running entirely on CPU')
      cfg['device_count'] = {'GPU': 0}

    if self.FLAGS.train: self.build_train_op()
    
    if self.FLAGS.summary:
      self.summary_op = tf.summary.merge_all()
      self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')
    
    # self.sess = tf.Session(config = tf.ConfigProto(**cfg))
    # self.sess.run(tf.global_variables_initializer())

    if not self.ntrain: return
    self.saver = tf.train.Saver(tf.global_variables(), 
      max_to_keep = self.FLAGS.keep)
    if self.FLAGS.load != 0: self.load_from_ckpt()
    
    # if self.FLAGS.summary:
      # self.writer.add_graph(self.sess.graph)

    # if True:
    #   # Yitao-TLS-Begin
    #   # init_op = tf.initialize_all_variables()
    #   # self.session.run(init_op)

    #   export_path_base = "actdet_yolo"
    #   export_path = os.path.join(
    #     compat.as_bytes(export_path_base),
    #     compat.as_bytes(str(FLAGS.model_version)))
    #   print('Exporting trained model to ', export_path)
    #   builder = saved_model_builder.SavedModelBuilder(export_path)

    #   tensor_info_x = tf.saved_model.utils.build_tensor_info(self.inp)
    #   tensor_info_y = tf.saved_model.utils.build_tensor_info(self.out)

    #   prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #     inputs={'input': tensor_info_x},
    #     outputs={'output': tensor_info_y},
    #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #   legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    #   builder.add_meta_graph_and_variables(
    #     self.sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #       'predict_images':
    #         prediction_signature,
    #     },
    #     legacy_init_op=legacy_init_op)

    #   builder.save()

    #   print('Done exporting!')
    #   # Yitao-TLS-End

  def savepb(self):
    """
    Create a standalone const graph def that 
    C++ can load and run.
    """
    darknet_pb = self.to_darknet()
    flags_pb = self.FLAGS
    flags_pb.verbalise = False
    
    flags_pb.train = False
    # rebuild another tfnet. all const.
    tfnet_pb = TFNet(flags_pb, darknet_pb)    
    tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
    # tfnet_pb.predict() # uncomment for unit testing
    name = 'built_graph/{}.pb'.format(self.meta['name'])
    os.makedirs(os.path.dirname(name), exist_ok=True)
    #Save dump of everything in meta
    with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
      json.dump(self.meta, fp)
    self.say('Saving const graph def to {}'.format(name))
    graph_def = tfnet_pb.sess.graph_def
    tf.train.write_graph(graph_def,'./', name, False)