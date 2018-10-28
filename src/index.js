import * as tf from '@tensorflow/tfjs';
import coco_classes from './coco_classes';
import voc_classes from './voc_classes';
import {
  v1_tiny_model,
  v2_tiny_model,
  v3_tiny_model,
  v3_model,
  v1_tiny_anchors,
  v2_tiny_anchors,
  v3_tiny_anchors,
  v3_anchors,
} from './config';
import postprocess from './postprocess';

const MAX_BOXES_PER_CLASS = 20;
const INPUT_SIZE = 416;
const SCORE_THRESHOLD = .5;
const IOU_THRESHOLD = .3;

async function _loadModel(
  pathOrIOHandler,
  modelUrl,
) {
  if (modelUrl) {
    return await tf.loadFrozenModel(modelUrl, pathOrIOHandler);
  } else {
    return await tf.loadModel(pathOrIOHandler);
  }
}

async function _predict(
  version,
  model,
  image,
  maxBoxesPerClass,
  scoreThreshold,
  iouThreshold,
  numClasses,
  anchors,
  classNames,
  inputSize,
) {
  let outputs = tf.tidy(() => {
    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, inputSize, inputSize);

    let imageTensor = tf.fromPixels(canvas, 3);
    imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

    const outputs = model.predict(imageTensor);
    return outputs;
  });

  const boxes = await postprocess(
    version,
    outputs,
    anchors,
    numClasses,
    classNames,
    [image.height, image.width],
    maxBoxesPerClass,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);

  return boxes;
}

async function v1tiny(
  pathOrIOHandler = v1_tiny_model,
  modelUrl = null,
) {
  const model = await _loadModel(pathOrIOHandler, modelUrl);

  return async function (
    image,
    {
      maxBoxesPerClass = MAX_BOXES_PER_CLASS,
      scoreThreshold = SCORE_THRESHOLD,
      iouThreshold = IOU_THRESHOLD,
      numClasses = voc_classes.length,
      anchors = v1_tiny_anchors,
      classNames = voc_classes,
      inputSize = INPUT_SIZE,
    } = {}
  ) {
    return await _predict(
      "v1tiny",
      model,
      image,
      maxBoxesPerClass,
      scoreThreshold,
      iouThreshold,
      numClasses,
      anchors,
      classNames,
      inputSize,
    );
  }
}

async function v2tiny(
  pathOrIOHandler = v2_tiny_model,
  modelUrl = null,
) {
  const model = await _loadModel(pathOrIOHandler, modelUrl);

  return async function (
    image,
    {
      maxBoxesPerClass = MAX_BOXES_PER_CLASS,
      scoreThreshold = SCORE_THRESHOLD,
      iouThreshold = IOU_THRESHOLD,
      numClasses = coco_classes.length,
      anchors = v2_tiny_anchors,
      classNames = coco_classes,
      inputSize = INPUT_SIZE,
    } = {}
  ) {
    return await _predict(
      "v2tiny",
      model,
      image,
      maxBoxesPerClass,
      scoreThreshold,
      iouThreshold,
      numClasses,
      anchors,
      classNames,
      inputSize,
    );
  }
}

async function v3tiny(
  pathOrIOHandler = v3_tiny_model,
  modelUrl = null,
) {
  const model = await _loadModel(pathOrIOHandler, modelUrl);

  return async function (
    image,
    {
      maxBoxesPerClass = MAX_BOXES_PER_CLASS,
      scoreThreshold = SCORE_THRESHOLD,
      iouThreshold = IOU_THRESHOLD,
      numClasses = coco_classes.length,
      anchors = v3_tiny_anchors,
      classNames = coco_classes,
      inputSize = INPUT_SIZE,
    } = {}
  ) {
    return await _predict(
      "v3tiny",
      model,
      image,
      maxBoxesPerClass,
      scoreThreshold,
      iouThreshold,
      numClasses,
      anchors,
      classNames,
      inputSize,
    );
  }
}

async function v3(
  pathOrIOHandler = v3_model,
  modelUrl = null,
) {
  const model = await _loadModel(pathOrIOHandler, modelUrl);

  return async function (
    image,
    {
      maxBoxesPerClass = MAX_BOXES_PER_CLASS,
      scoreThreshold = SCORE_THRESHOLD,
      iouThreshold = IOU_THRESHOLD,
      numClasses = coco_classes.length,
      anchors = v3_anchors,
      classNames = coco_classes,
      inputSize = INPUT_SIZE,
    } = {}
  ) {
    return await _predict(
      "v3",
      model,
      image,
      maxBoxesPerClass,
      scoreThreshold,
      iouThreshold,
      numClasses,
      anchors,
      classNames,
      inputSize,
    );
  }
}

const yolo = {
  v1tiny,
  v2tiny,
  v3tiny,
  v3
};

export default yolo;