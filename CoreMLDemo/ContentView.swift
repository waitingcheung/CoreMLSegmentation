//
//  ContentView.swift
//  CoreMLDemo
//
//  Created by Wai Ting Cheung on 9/12/2021.
//

import SwiftUI
import Vision

struct ContentView: View {
    @State var outputImage : UIImage = UIImage(named: "deeplab1")!
    @State var inputImage : UIImage = UIImage(named: "deeplab1")!
    
    var body: some View {
        ScrollView{
            VStack{
                Image(uiImage: inputImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Image(uiImage: outputImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Spacer()
                
                Button(action: {runVisionRequest()}, label: {
                    Text("Run Image Segmentation")
                })
                    .padding()
            }
        }
    }
    
    func runVisionRequest() {
        guard let model = try? VNCoreMLModel(for: DeepLabV3Int8LUT(configuration: .init()).model)
        else { return }
        
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
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            let palette: [Float] = [pow(2, 25) - 1, pow(2, 15) - 1, pow(2, 21) - 1]
            do {
                // Create a color palette for all 21 classes
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
                    
                    // Blend segmentation mask with input image
                    let maskImage: UIImage = mask.image(min: 0, max: 255)!.resized(to: self.inputImage.size)
                    
                    UIGraphicsBeginImageContext(self.inputImage.size)
                    let areaSize = CGRect(x: 0, y: 0, width: self.inputImage.size.width, height: self.inputImage.size.height)
                    
                    self.outputImage.draw(in: areaSize)
                    maskImage.draw(in: areaSize, blendMode: .normal, alpha: 0.5)
                    
                    self.outputImage = UIGraphicsGetImageFromCurrentImageContext()!
                    UIGraphicsEndImageContext()
                }
            } catch {
                print(error)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
