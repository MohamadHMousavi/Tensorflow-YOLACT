import tensorflow as tf
import numpy as np

class TFRecordCreator:
    def __init__(self):
        pass
    
    @staticmethod
    def _get_float_array_feature(data):
        if isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=data))
    
    @staticmethod
    def _get_int_feature(data):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))
    
    @staticmethod
    def _get_bytes_feature(data):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

    @classmethod
    def _get_tensor_feature(cls, tensor):
        serialized_tensor = tf.io.serialize_tensor(tensor)
        return cls.get_bytes_feature(serialized_tensor.numpy())
    
    @staticmethod
    def _decode_tensor_feature(tensor_feature, tensor_type=tf.float32):
        return tf.io.parse_tensor(tensor_feature, tensor_type)

    @staticmethod
    def _decode_image_feature(image_data, channels=3):
        return tf.io.decode_image(image_data, channels=channels)
    
    @staticmethod
    def _decode_jpeg_feature(jpeg_data):
        jpeg_data = tf.strings.join([jpeg_data, tf.convert_to_tensor(b'\xff\xd9')])
        # return cv2.imdecode(np.frombuffer(jpeg_data.numpy(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return tf.io.decode_jpeg(jpeg_data, channels=3)
    
    def _get_features(self, *args, **kwargs):
        raise NotImplementedError('features are not defined')
    
    def encode(self, *args, **kwargs):
        feature_dict = self._get_features(*args, **kwargs)
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    def _get_parsed_dict(self, tf_data, *args, **kwargs):
        raise NotImplementedError('parsing data into dict is not implemented')
    
    def _get_parsed_data(self, parsed_dict, *args, **kwargs):
        raise NotImplementedError('data parsing is not implemented')
    
    def decode(self, tf_data, *args, **kwargs):
        parsed_dict = self._get_parsed_dict(tf_data, *args, **kwargs)
        return self._get_parsed_data(parsed_dict, *args, **kwargs)
    
    

class DataTFRecorder(TFRecordCreator):
    # def __init__(self, tensor_type=tf.float16):
        # self.tensor_type = tf.float16
        
    def _get_parsed_dict(self, tf_data):
        return tf.io.parse_single_example(tf_data, {'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
                                                    'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
                                                    'image/filename': tf.io.FixedLenFeature([], dtype=tf.string),
                                                    'image/source_id': tf.io.FixedLenFeature([], dtype=tf.string),
                                                    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
                                                    'image/format': tf.io.FixedLenFeature([], dtype=tf.string),
                                                    'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                                                    'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                                                    'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                                                    'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                                                    'image/object/class/text': tf.io.VarLenFeature(dtype=tf.string),
                                                    'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
                                                    'image/object/mask': tf.io.VarLenFeature(tf.string)})
    
    def _decode_boxes(self, parsed_tensors):
        # pass the bbox in the order [xmin, ymin, xmax, ymax]
        xmin = parsed_tensors['image/object/bbox/xmin']
        ymin = parsed_tensors['image/object/bbox/ymin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        
        # return tf.cast(tf.stack([xmin, ymin, xmax, ymax], axis=-1), dtype=self.tensor_type)
    
    @staticmethod
    def _get_tensor(x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        return x

    def _decode_png_mask(self, image_buffer):
        image = tf.squeeze(
            self._decode_image_feature(image_buffer, channels=1), axis=2)
        image.set_shape([None, None])
        image = tf.cast(tf.greater(image, 0), dtype=tf.float32)
        return image

    def _get_mask_data(self, png_masks, height, width):
        return tf.cond(
            tf.greater(tf.size(png_masks), 0),
            lambda: tf.map_fn(self._decode_png_mask, png_masks, dtype=tf.float32),
            lambda: tf.zeros(tf.stack([0, height, width]), dtype=tf.float32))
    
    def _get_parsed_data(self, parsed_dict):
        parsed_data_dict = {k:self._get_tensor(v) for k, v in parsed_dict.items()}
        image = self._decode_image_feature(parsed_data_dict['image/encoded'], channels=3)
        image.set_shape([None, None, None])
        masks = self._get_mask_data(parsed_data_dict['image/object/mask'],
                                    parsed_data_dict['image/height'],
                                    parsed_data_dict['image/width'])
        bboxes = self._decode_boxes(parsed_data_dict)
        
        return {
            'image': image,
            'height': parsed_data_dict['image/height'],
            'width': parsed_data_dict['image/width'],
            'gt_classes': parsed_data_dict['image/object/class/label'],
            'gt_bboxes': bboxes,
            'gt_masks': masks}
        # return parsed_data_dict
        # return {"index": parsed_dict['index'],
        #         "pid": parsed_dict['pid'],
        #         "image_link": parsed_dict['image_link'], 
        #         "jpeg_encoded": self.decode_jpeg_feature(parsed_dict['jpeg_encoded'])}
