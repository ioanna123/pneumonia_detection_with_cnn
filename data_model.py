from enum import Enum

from pydantic import BaseModel


class PredictionClass(Enum):
    pneumonia = "PNEUMONIA"
    normal = "NORMAL"


class ImageModel(Enum):
    first_custom_cnn = "first_custome_cnn_model.h5"
    first_custom_cnn_class_weight = "first_custome_cnn_model_class_weights.h5"
    first_custom_cnn_smote = "first_custome_cnn_model_smote.h5"
    first_custom_cnn_undersampling = "first_custom_cnn_undersampling.h5"
    second_custom_cnn = "second_custom_cnn_model_smote.h5"
    second_custom_cnn_smote = "second_custom_cnn_model_smote.h5"
    densent121_cnn = "DenseNet121_cnn_model.h5"
    densent121_cnn_class_weight = "DenseNet121_cnn_model_class_weights.h5"
    resnet50_cnn = "ResNet50_cnn_model.h5"
    vagg16_cnn = "vgg16_cnn_model.h5"
    inception_cnn_model = "inception_cnn_model.h5"


class PredictionResult(BaseModel):
    """
    Pydantic dataclass representing the query result.
    """
    prediction_class: PredictionClass
    image_name: str
