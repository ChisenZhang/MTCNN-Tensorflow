# -*-coding: utf-8 -*-
"""
    @Project: tensorflow_models_nets
    @info:
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import sys
sys.path.append('./')
# from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
# from PIL import Image
# import cv2
# import numpy as np

model_path = ['./data/MTCNN_model/PNet_landmark/PNet', './data/MTCNN_model/RNet_landmark/RNet', './data/MTCNN_model/ONet_landmark/ONet']

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 设置最小gpu使用量

from tensorflow.python.tools import freeze_graph
from tensorflow.python import pywrap_tensorflow

# 打印保存节点的内容、值
def printCPKV(checkPoint):
    reader = pywrap_tensorflow.NewCheckpointReader(checkPoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name:", key)
        # print(reader.get_tensor(key))

def freeze_graphOri(input_checkpoint, output_graph):
    slide_window = True
    test_mode = "ONet"
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "cls_fc/Softmax,bbox_fc/BiasAdd,landmark_fc/BiasAdd"# "Squeeze,Squeeze_1,Squeeze_2"
    imgSize = 12
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, 1, model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    # detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        imgSize = 24
        RNet = Detector(R_Net, 24, 1, model_path[1])
        # detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        imgSize = 48
        ONet = Detector(O_Net, 48, 1, model_path[2])
        # detectors[2] = ONet
    # x = tf.placeholder('float', shape=[None, imgSize, imgSize, 3], name='input_image')
    sess = ONet.sess

    # 写图
    # writer = tf.summary.FileWriter('./logs/', sess.graph)
    # writer.close()
    # exit(1)

    # saver = tf.train.Saver()
    # with tf.Session(config=config) as sess:
        # saver.restore(sess, tf.train.latest_checkpoint(input_checkpoint))  # 恢复图并得到数据

    model_dict = '/'.join(model_path[2].split('/')[:-1])
    ckpt = tf.train.latest_checkpoint(model_dict)
    tmpPB = './model_save/tmpGraph.pb'
    tf.train.write_graph(sess.graph_def, './model_save', 'tmpGraph.pb')
    freeze_graph.freeze_graph(input_graph=tmpPB, input_checkpoint=str(ckpt), output_node_names=output_node_names,
                              output_graph='./model_save/modelTest.pb', input_saver='', input_binary=False,
                              restore_op_name='', filename_tensor_name='', clear_devices=True, initializer_nodes='')
    exit(1)

    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=sess.graph_def,  # 等于:sess.graph_def
        output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = './model_save_NL01'#/mobileNet_adam_190513175358.ckpt-3363.meta'
    # 输出pb模型的路径
    pb_path = "./model_save/pb"
    if not os.path.exists(pb_path):
        os.makedirs(pb_path)
    out_pb_path = os.path.join(pb_path, 'frozen_model_ONet.pb')
    # 调用freeze_graph将ckpt转为pb
    freeze_graphOri(input_checkpoint, out_pb_path)