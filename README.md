# CoreMLSegmentation

This is a replication of the PyTorch DeepLabV3 [demo](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) in Swift UI and Core ML.

<p align="center">
  <img width="320" alt="" src="https://user-images.githubusercontent.com/2617118/145712166-1c78846d-b364-46ed-9ccd-5dfcc8430779.png">
</p>

Download the `DeeplabV3` image segmentation model from [Apple Core ML models](https://developer.apple.com/machine-learning/models/) and copy to your project.

Download the following files from [CoreMLHelpers](https://github.com/hollance/CoreMLHelpers) and copy to your project.
- `MLMultiArray+Image.swift`
- `CGImage+RawBytes.swift`
- `UIImage+Extensions.swift`
- `Math.swift`

Create an instance of the DeeplabV3 model.
```swift
guard let model = try? VNCoreMLModel(for: DeepLabV3Int8LUT(configuration: .init()).model)
else { return }
```

Create a VNCoreMLRequest.
```swift
let request = VNCoreMLRequest(model: model, completionHandler: visionRequestDidComplete)
request.imageCropAndScaleOption = .scaleFill
DispatchQueue.global().async {
    let handler = VNImageRequestHandler(cgImage: inputImage.cgImage!, options: [:])
    do {
        try handler.perform([request])
    } catch {
        print(error)
    }
}
```

Create a color palette for all 21 classes of the DeeplabV3 model.
```swift
let palette: [Float] = [pow(2, 25) - 1, pow(2, 15) - 1, pow(2, 21) - 1]

let colors = try MLMultiArray(shape: [21, 3], dataType: .int32)
for i in 0..<21 {
    for j in 0..<3 {
        if j == 0 {
            colors[[i as NSNumber, j as NSNumber]] = i as NSNumber
        } else {
            colors[[i as NSNumber, j as NSNumber]] = ((i * Int(palette[j])) % 255) as NSNumber
        }
    }
}
```

Make a prediction using the DeeplabV3 model and fill the segmentation mask according to the class labels. The prediction output is a `MultiArray` of size 513x513 filled with class labels `0-20`
```swift
if let observations = request.results as? [VNCoreMLFeatureValueObservation],
   let labels = observations.first?.featureValue.multiArrayValue {
    let h = labels.shape[0]
    let w = labels.shape[1]

    // Set the RGB values of the segmentation mask according to the labels
    let mask = try MLMultiArray(shape: [3, h, w], dataType: .int32)
    for row in 0..<h.intValue {
        for col in 0..<w.intValue {
            for channel in 0..<3 {
                let label = labels[[row as NSNumber, col as NSNumber]]
                mask[[channel as NSNumber, row as NSNumber, col as NSNumber]] = colors[[label as NSNumber, channel as NSNumber]]
            }
        }
    }
}
```

Blend the segmentation mask with the input image
```swift
let maskImage: UIImage = mask.image(min: 0, max: 255)!.resized(to: self.inputImage.size)

UIGraphicsBeginImageContext(self.inputImage.size)
let areaSize = CGRect(x: 0, y: 0, width: self.inputImage.size.width, height: self.inputImage.size.height)

self.outputImage.draw(in: areaSize)
maskImage.draw(in: areaSize, blendMode: .normal, alpha: 0.5)

self.outputImage = UIGraphicsGetImageFromCurrentImageContext()!
UIGraphicsEndImageContext()
```
