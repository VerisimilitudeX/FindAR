import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var previewLayer: AVCaptureVideoPreviewLayer?
    let session = AVCaptureSession()
    var requests = [VNRequest]()
    var audioPlayer: AVAudioPlayer?
    
    var lastUpdateTime = Date()
    var lastDetectedObject: String?
    
    // UILabel to display the name and confidence of the detected object
    let detectionLabel: UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.backgroundColor = .black.withAlphaComponent(0.7)
        label.textColor = .white
        label.font = UIFont.boldSystemFont(ofSize: 18)
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // Work item for updating the label
    var updateLabelWorkItem: DispatchWorkItem?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCameraLiveView()
        setupVision()
        setupDetectionLabel()
        setupAudioPlayer()
    }
    
    func setupCameraLiveView() {
        session.sessionPreset = .high
        guard let captureDevice = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: captureDevice) else {
            fatalError("Unable to access camera.")
        }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        session.addOutput(output)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = view.bounds
        view.layer.addSublayer(previewLayer!)
        
        session.startRunning()
    }
    
    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: coreml().model) else {
            fatalError("Can't load Vision ML model")
        }
        let objectRecognitionRequest = VNCoreMLRequest(model: visionModel) { [weak self] (request, error) in
            DispatchQueue.main.async {
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    self?.handleObjectRecognition(results)
                }
            }
        }
        requests = [objectRecognitionRequest]
    }
    
    func setupDetectionLabel() {
        view.addSubview(detectionLabel)
        NSLayoutConstraint.activate([
            detectionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            detectionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            detectionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            detectionLabel.heightAnchor.constraint(equalToConstant: 50)
        ])
    }
    
    func setupAudioPlayer() {
        // Set up the audio session
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default, options: [])
            try audioSession.setActive(true)
        } catch {
            print("Failed to set up audio session: \(error)")
        }

        // Set up the audio player
        guard let soundURL = Bundle.main.url(forResource: "sound", withExtension: "mp3") else {
            print("Unable to find sound file.")
            return
        }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: soundURL)
            audioPlayer?.prepareToPlay()
            audioPlayer?.volume = 1.0  // Ensure the volume is up
        } catch {
            print("Unable to initialize AVAudioPlayer: \(error.localizedDescription)")
        }
    }

    
    func handleObjectRecognition(_ recognitions: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async { [weak self] in
            guard let strongSelf = self else { return }
            guard let previewLayer = strongSelf.previewLayer else {
                print("Preview layer is nil")
                return
            }

            // Draw bounding boxes
            previewLayer.sublayers?.removeSubrange(1...) // Clear previous boxes
            for observation in recognitions {
                strongSelf.drawBoundingBox(observation.boundingBox)
            }

            // Update detection label with a delay
            guard let mostConfidentRecognition = recognitions.max(by: { a, b in a.confidence < b.confidence }) else { return }
            let identifier = mostConfidentRecognition.labels.first?.identifier ?? "Unknown"
            let confidence = mostConfidentRecognition.labels.first?.confidence ?? 0
            
            // Update the label only if enough time has passed or the detected object has changed
            if Date().timeIntervalSince(strongSelf.lastUpdateTime) >= 1.0 || strongSelf.lastDetectedObject != identifier {
                strongSelf.detectionLabel.text = "\(identifier) \(Int(confidence * 100))%"
                if identifier == "Sharp object" {
                    strongSelf.audioPlayer?.play()
                }
                strongSelf.lastUpdateTime = Date()
                strongSelf.lastDetectedObject = identifier
            }
        }
    }
    
    func drawBoundingBox(_ rect: CGRect) {
        guard let previewLayer = self.previewLayer else {
            print("Preview layer is nil")
            return
        }
        let boundingBox = VNImageRectForNormalizedRect(rect, Int(previewLayer.bounds.width), Int(previewLayer.bounds.height))
        let boxLayer = CAShapeLayer()
        boxLayer.frame = boundingBox
        boxLayer.cornerRadius = 3
        boxLayer.borderWidth = 2
        boxLayer.borderColor = UIColor.red.cgColor
        boxLayer.backgroundColor = UIColor.clear.cgColor
        previewLayer.addSublayer(boxLayer)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try imageRequestHandler.perform(requests)
        } catch {
            print("Failed to perform image request: \(error)")
        }
    }
}

// Replace `coreml` with the actual class name of your Core ML model.
