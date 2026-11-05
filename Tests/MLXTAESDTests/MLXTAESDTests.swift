import Foundation
import MLX
import XCTest

@testable import MLXTAESD

final class MLXTAESDTests: XCTestCase {

    /// The decoder must own exactly the 67 tensors in `taesd_decoder.safetensors`.
    func testParameterCount() {
        let model = TAESDDecoder()
        XCTAssertEqual(model.parameters().flattened().count, 67)
    }

    /// 8x upsampling, 3 channels out, values clamped to [0, 1]. Runs with random (unloaded) weights.
    func testDecodeShapeAndRange() {
        let model = TAESDDecoder()
        let latent = MLXRandom.normal([1, 8, 8, 4])
        let image = model.decodePixels(latent)
        XCTAssertEqual(image.shape, [1, 64, 64, 3])
        XCTAssertGreaterThanOrEqual(image.min().item(Float.self), 0)
        XCTAssertLessThanOrEqual(image.max().item(Float.self), 1)
    }

    /// Bit-level check against the original PyTorch TAESD. Skipped unless both are set:
    ///   TAESD_WEIGHTS        path to taesd_decoder.safetensors
    ///   TAESD_REFERENCE_DIR  a folder made by `Scripts/make_reference.py` (latent.npy, image.npy)
    /// With xcodebuild, pass them as TEST_RUNNER_TAESD_WEIGHTS / TEST_RUNNER_TAESD_REFERENCE_DIR.
    func testMatchesPyTorchReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["TAESD_WEIGHTS"], let dir = env["TAESD_REFERENCE_DIR"] else {
            throw XCTSkip("Set TAESD_WEIGHTS and TAESD_REFERENCE_DIR to compare against PyTorch.")
        }
        let model = try loadTAESDDecoder(url: URL(fileURLWithPath: weights))
        let latent = try loadArray(url: URL(fileURLWithPath: dir).appendingPathComponent("latent.npy"))
        let expected = try loadArray(url: URL(fileURLWithPath: dir).appendingPathComponent("image.npy"))
        let image = model.decodePixels(latent)
        XCTAssertEqual(image.shape, expected.shape)
        let maxDiff = abs(image - expected).max().item(Float.self)
        print("MLXTAESD vs PyTorch: max |difference| = \(maxDiff)")
        XCTAssertLessThan(maxDiff, 1e-3)
    }
}
