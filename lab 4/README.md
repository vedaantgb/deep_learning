# Image Object Detection: EfficientDet-D7 and Region Proposals + VGG16

This notebook contains two separate image-recognition approaches. Both accept an uploaded image and display it with predicted labels and bounding boxes.

## Contents

1. **Approach 1 — EfficientDet-D7:** Uses a pre-trained object-detection model from TensorFlow Hub. It predicts object classes, confidence scores, and bounding boxes directly.
2. **Approach 2 — Region Proposals + VGG16:** Generates candidate image regions using OpenCV Selective Search (with a sliding-window fallback), classifies each region using pre-trained VGG16, and removes overlapping detections with Non-Maximum Suppression (NMS).

> These are alternative approaches, not a single combined pipeline. Run the relevant notebook cell for the method you want to try.

---

## Approach 1: EfficientDet-D7 Object Detection

### Overview

This implementation loads the pre-trained `tensorflow/efficientdet/d7/1` model from TensorFlow Hub. It reads the uploaded image, converts it to the format expected by TensorFlow, runs object detection, and draws a bounding box and class label for each detection whose confidence score is greater than `0.5`.

### Main components

- **Model:** EfficientDet-D7 loaded from TensorFlow Hub.
- **Labels:** A COCO class-ID mapping included in the notebook.
- **Image handling:** OpenCV reads the image in BGR format and converts it to RGB for inference.
- **Detection threshold:** `0.5`.
- **Output:** The original image with green bounding boxes and labels containing the class name and confidence score.

### Requirements

The notebook is written for **Google Colab** and uses its upload and image-display helpers.

- Python
- OpenCV (`opencv-python`)
- NumPy
- TensorFlow
- TensorFlow Hub

In a fresh Colab notebook, install any missing dependencies with:

```python
!pip install opencv-python numpy tensorflow tensorflow-hub
```

TensorFlow Hub must be able to download the model, so an internet connection is required for the first model load.

### How to run

1. Open the notebook in Google Colab.
2. Run the EfficientDet-D7 code cell.
3. Wait for the pre-trained model to load.
4. When prompted, upload an image.
5. Inspect the displayed image and its bounding boxes.

### Important function

```python
detect_objects_with_boxes(image_path, detector)
```

**Input:** Image file path and the loaded detector.

**Returns:** An OpenCV image array with detected objects drawn on it.

### Configuration

Change `confidence_threshold = 0.5` inside `detect_objects_with_boxes` to adjust how many detections are shown. A higher threshold usually displays fewer, more confident detections; a lower threshold may display more false positives.

---

## Approach 2: Region Proposals + VGG16 Classification

### Overview

This implementation uses a two-stage approach:

1. Propose candidate regions that may contain objects.
2. Classify each region with ImageNet-pretrained VGG16.

The code first resizes the image to a width of `800` pixels while preserving its aspect ratio. It then generates up to `300` candidate regions. Each crop is resized to `224 × 224`, preprocessed for VGG16, and classified. Predictions with a probability greater than `0.45` are retained, and Non-Maximum Suppression removes heavily overlapping boxes.

### Main components

- **Classifier:** VGG16 pretrained on ImageNet, with its classification head enabled.
- **Region proposals:** OpenCV Selective Search when `cv2.ximgproc.segmentation` is available.
- **Fallback:** Multi-scale sliding windows if Selective Search is unavailable or fails.
- **Classification threshold:** `0.45`.
- **Overlap filtering:** Non-Maximum Suppression with an IoU threshold of `0.3`.
- **Output:** The resized image with green bounding boxes, predicted ImageNet labels, and scores.

### Requirements

The code is designed for **Google Colab** and uses its upload and display helpers.

- Python
- OpenCV with the `ximgproc` module (`opencv-contrib-python`)
- NumPy
- TensorFlow / Keras

The notebook attempts to install `opencv-contrib-python` if Selective Search is unavailable and then restarts the runtime. If this setup cell restarts Colab, run the cell again after the runtime reconnects.

A typical installation command, if needed, is:

```python
!pip install opencv-contrib-python numpy tensorflow
```

### How to run

1. Open the notebook in Google Colab.
2. Run the setup/code cell. If Colab restarts, wait for it to reconnect and run the cell again.
3. When prompted, upload an image.
4. Wait for the region proposals and VGG16 predictions to finish.
5. Inspect the displayed image and labeled regions.

### Important functions

#### `get_region_proposals(img, max_regions=300)`

Creates candidate boxes in `(x, y, width, height)` format. It uses Selective Search when possible; otherwise, it uses multi-scale sliding windows. The returned list is capped at `max_regions`.

#### `non_max_suppression(dets, iou_thr=0.3)`

Filters overlapping detections by keeping higher-scoring boxes and suppressing boxes whose Intersection over Union (IoU) with a kept box is at least the configured threshold.

Each detection is represented as:

```python
((x, y, width, height), label, score)
```

### Configuration

- `max_regions=300`: Maximum number of candidate regions considered.
- `prob > 0.45`: Minimum VGG16 class probability for retaining a candidate.
- `iou_thr=0.3`: Overlap threshold used by NMS.
- `cv2.resize(img, (800, int(H0 * 800.0 / W0)))`: Sets the processing width to 800 pixels while preserving aspect ratio.

### Important limitation

VGG16 is an **image classifier**, not a detector trained to predict object locations. In this pipeline, the candidate region is assigned its top ImageNet class. Consequently, boxes can be inaccurate, the same object may initially produce many candidates, and some regions may be classified as an object even when they do not contain one. ImageNet labels also do not necessarily match the COCO label set used by EfficientDet.

---

## Comparison

| Feature | EfficientDet-D7 | Region Proposals + VGG16 |
|---|---|---|
| Core method | Object detector | Region proposals followed by image classification |
| Model source | TensorFlow Hub | Keras VGG16 pretrained on ImageNet |
| Box generation | Predicted directly by the detector | Selective Search or sliding-window fallback |
| Label set | COCO class mapping in the notebook | ImageNet categories |
| Confidence threshold | `0.5` | `0.45` |
| Overlap suppression | Uses the detector's returned predictions as provided by this code | Explicit NMS with IoU threshold `0.3` |
| Typical trade-off | More direct detection pipeline | More candidate-region processing; localization may be less reliable |

## Troubleshooting

- **Model download fails:** Check that the runtime has internet access and try loading the model again.
- **`ModuleNotFoundError`:** Install the missing dependency in Colab, then rerun the relevant cell.
- **OpenCV has no `ximgproc`:** Use `opencv-contrib-python`; the second approach also contains a sliding-window fallback.
- **The image cannot be read:** Upload a valid image file supported by OpenCV, such as JPG or PNG.
- **Processing is slow in Approach 2:** Reduce `max_regions`; each candidate region must be classified by VGG16.
- **No boxes appear:** Try a lower confidence threshold, while keeping in mind that this may increase false positives.

## Notes

- Both code cells use `google.colab.files.upload()` and `google.colab.patches.cv2_imshow()`. To run outside Colab, replace these with local file-path input and an appropriate image-display method.
- The first approach uses the COCO mapping included in the notebook. The second uses ImageNet class names returned by Keras `decode_predictions`.
- Neither implementation guarantees correct detections for every image. Results depend on image quality, object size, class coverage, and model confidence.
