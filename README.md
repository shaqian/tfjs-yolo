# tfjs-yolo

[`YOLO`](https://pjreddie.com/darknet/yolo/) object detection with [Tensorflow.js](https://js.tensorflow.org/). Supports YOLO v3 and Tiny YOLO v1, v2, v3.

## Demo

- Detect objects using your webcam:
https://shaqian.github.io/tfjs-yolo-demo/
![demo](https://github.com/shaqian/tfjs-yolo-demo/raw/master/demo.gif)

- Not hotdog PWA: https://shaqian.github.io/Not-Hotdog/

## Install
```
npm install tfjs-yolo
```

## Usage

### Import module

```javascript
import yolo from 'tfjs-yolo';
```

### Initialize and load model

```javascript
// Use default models (stored in my GitHub demo repo)
let myYolo = await yolo.v1tiny();
let myYolo = await yolo.v2tiny();
let myYolo = await yolo.v3tiny();
let myYolo = await yolo.v3();

// or specify path or handler, see https://js.tensorflow.org/api/0.13.3/#loadModel
let myYolo = await yolo.v3tiny("https://.../model.json");

// or use frozen model, see https://js.tensorflow.org/api/0.13.3/#loadFrozenModel
let myYolo = await yolo.v3tiny(
  "https://.../weights_manifest.json",
  "https://.../tensorflowjs_model.pb"
);

```

### Run model

Supported input html element:
- img
- canvas
- video

```javascript
const boxes = await myYolo.predict(canvas);

// Optional settings
const boxes = await myYolo.predict(
  canvas,
  {
    maxBoxes: 5,          // defaults to 20
    scoreThreshold: .2,   // defaults to .5
    iouThreshold: .5,     // defaults to .3
    numClasses: 80,       // defaults to 80 for yolo v3, tiny yolo v2, v3 and 20 for tiny yolo v1
    anchors: [...],       // See ./src/config.js for examples
    classNames: [...],    // defaults to coco classes for yolo v3, tiny yolo v2, v3 and voc classes for tiny yolo v1
    inputSize: 416,       // defaults to 416
  }
);
```

### Output box format

```javascript
{
  top,    // Float
  left,   // Float
  bottom, // Float
  right,  // Float
  height, // Float
  width,  // Float
  score,  // Float
  class   // String, e.g. person
}
```

### Dispose model

```javascript
myYolo.dispose();
```

## Credits

- https://github.com/qqwweee/keras-yolo3
- https://github.com/zqingr/tfjs-yolov3
- https://github.com/ModelDepot/tfjs-yolo-tiny
- https://github.com/allanzelener/YAD2K