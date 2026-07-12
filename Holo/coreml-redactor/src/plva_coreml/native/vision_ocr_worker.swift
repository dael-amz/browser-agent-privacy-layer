import CoreGraphics
import Foundation
import ImageIO
import Vision

private func writeResponse(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value) else { return }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

private func number(_ value: Any?) -> CGFloat? {
    (value as? NSNumber).map { CGFloat(truncating: $0) }
}

private func recognize(_ request: [String: Any]) throws -> [[String: Any]] {
    guard
        let encoded = request["image"] as? String,
        let bytes = Data(base64Encoded: encoded),
        let source = CGImageSourceCreateWithData(bytes as CFData, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw NSError(domain: "PLVAVision", code: 1)
    }
    let mode = request["mode"] as? String ?? "fast"
    let requestedRegions = request["rois"] as? [[String: Any]]
    let regions: [[String: Any]] = requestedRegions?.isEmpty == false
        ? requestedRegions!
        : [["x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0]]
    var output: [[String: Any]] = []
    for region in regions {
        guard
            let x = number(region["x"]),
            let y = number(region["y"]),
            let width = number(region["width"]),
            let height = number(region["height"]),
            x >= 0, y >= 0, width > 0, height > 0,
            x + width <= 1.0001, y + height <= 1.0001
        else {
            throw NSError(domain: "PLVAVision", code: 2)
        }
        let pixelRect = CGRect(
            x: x * CGFloat(image.width),
            y: y * CGFloat(image.height),
            width: width * CGFloat(image.width),
            height: height * CGFloat(image.height)
        ).integral.intersection(
            CGRect(x: 0, y: 0, width: image.width, height: image.height)
        )
        guard !pixelRect.isEmpty, let crop = image.cropping(to: pixelRect) else { continue }
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = mode == "accurate" ? .accurate : .fast
        textRequest.usesLanguageCorrection = false
        textRequest.recognitionLanguages = ["en-US"]
        let handler = VNImageRequestHandler(cgImage: crop, options: [:])
        try handler.perform([textRequest])
        for observation in textRequest.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let box = observation.boundingBox
            // Input regions use a top-left origin; Vision observations use lower-left.
            let fullX = x + box.origin.x * width
            let fullY = 1.0 - (y + height) + box.origin.y * height
            output.append([
                "text": candidate.string,
                "confidence": candidate.confidence,
                "x": fullX,
                "y": fullY,
                "width": box.width * width,
                "height": box.height * height,
            ])
        }
    }
    return output
}

while let line = readLine() {
    autoreleasepool {
        do {
            guard
                let data = line.data(using: .utf8),
                let request = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let identifier = request["id"] as? String
            else {
                throw NSError(domain: "PLVAVision", code: 3)
            }
            let started = CFAbsoluteTimeGetCurrent()
            let observations = try recognize(request)
            writeResponse([
                "id": identifier,
                "ok": true,
                "observations": observations,
                "duration_ms": (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
            ])
        } catch {
            writeResponse([
                "ok": false,
                "error": String(describing: type(of: error)),
            ])
        }
    }
}
