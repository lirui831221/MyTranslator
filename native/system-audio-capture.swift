import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

private func writeStatus(_ message: String) {
    if let data = (message + "\n").data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

@available(macOS 13.0, *)
final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let sampleQueue = DispatchQueue(label: "com.rain.mytranslator.system-audio")
    private var stream: SCStream?

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: false
        )
        guard let display = content.displays.first else {
            throw NSError(
                domain: "MyTranslatorSystemAudio",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No capturable display is available"]
            )
        }

        let filter = SCContentFilter(
            display: display,
            excludingApplications: [],
            exceptingWindows: []
        )
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = 16_000
        configuration.channelCount = 1
        configuration.width = 2
        configuration.height = 2
        configuration.queueDepth = 1
        configuration.showsCursor = false
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(
            filter: filter,
            configuration: configuration,
            delegate: self
        )
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
        self.stream = stream
        try await stream.startCapture()
        writeStatus("READY sample_rate=16000 channels=1 format=f32le")
    }

    func stop() async {
        guard let stream else { return }
        try? await stream.stopCapture()
        self.stream = nil
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio,
              sampleBuffer.isValid,
              CMSampleBufferDataIsReady(sampleBuffer),
              let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(description),
              let format = AVAudioFormat(streamDescription: streamDescription)
        else {
            return
        }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0,
              let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
        else {
            return
        }
        pcm.frameLength = frameCount
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcm.mutableAudioBufferList
        )
        guard status == noErr else {
            writeStatus("ERROR failed to copy audio samples: \(status)")
            return
        }

        if let channels = pcm.floatChannelData {
            let byteCount = Int(frameCount) * MemoryLayout<Float>.size
            FileHandle.standardOutput.write(Data(bytes: channels[0], count: byteCount))
        } else if let channels = pcm.int16ChannelData {
            let samples = UnsafeBufferPointer(start: channels[0], count: Int(frameCount))
            var floats = samples.map { Float($0) / Float(Int16.max) }
            floats.withUnsafeMutableBytes { bytes in
                FileHandle.standardOutput.write(Data(bytes: bytes.baseAddress!, count: bytes.count))
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        writeStatus("ERROR capture stopped: \(error.localizedDescription)")
        exit(2)
    }
}

guard #available(macOS 13.0, *) else {
    writeStatus("ERROR macOS 13 or later is required")
    exit(1)
}

let capture = SystemAudioCapture()
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
let terminationSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
terminationSource.setEventHandler {
    Task {
        await capture.stop()
        exit(0)
    }
}
terminationSource.resume()

Task {
    do {
        try await capture.start()
    } catch {
        writeStatus("ERROR \(error.localizedDescription)")
        exit(1)
    }
}
dispatchMain()
