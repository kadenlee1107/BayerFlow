/*
 * Training Data Bridge — C-to-Swift
 *
 * @_cdecl bridge functions called from denoise_core.c to submit
 * extracted noisy/denoised patch pairs for model training.
 *
 * Follows the same pattern as MetalBridge.swift and MPSPostFilterBridge.swift.
 */

import Foundation

// MARK: - C Bridge Functions

/// Check if the user has opted in to training data collection.
@_cdecl("training_data_enabled")
func training_data_enabled() -> Int32 {
    return UserDefaults.standard.bool(forKey: "trainingDataConsent") ? 1 : 0
}

/// Submit a noisy/denoised patch pair from the C pipeline.
/// Called from denoise_core.c's extract_training_patches().
/// Copies pixel data immediately into a Swift-managed buffer.
@_cdecl("training_data_submit_patch")
func training_data_submit_patch(
    _ noisy: UnsafePointer<UInt16>,
    _ denoised: UnsafePointer<UInt16>,
    _ patchW: Int32, _ patchH: Int32,
    _ frameW: Int32, _ frameH: Int32,
    _ patchX: Int32, _ patchY: Int32,
    _ frameIdx: Int32,
    _ noiseSigma: Float, _ flowMag: Float,
    _ cameraModel: UnsafePointer<CChar>?,
    _ iso: Int32
) {
    let patchPixels = Int(patchW) * Int(patchH)

    // Copy pixel data into Swift-owned arrays
    let noisyData = Data(bytes: noisy, count: patchPixels * 2)
    let denoisedData = Data(bytes: denoised, count: patchPixels * 2)

    let camera = cameraModel.map { String(cString: $0) } ?? ""

    let patch = TrainingPatch(
        noisyData: noisyData,
        denoisedData: denoisedData,
        patchW: Int(patchW),
        patchH: Int(patchH),
        frameW: Int(frameW),
        frameH: Int(frameH),
        patchX: Int(patchX),
        patchY: Int(patchY),
        frameIdx: Int(frameIdx),
        noiseSigma: noiseSigma,
        flowMag: flowMag,
        cameraModel: camera,
        iso: Int(iso)
    )

    TrainingDataManager.shared.submitPatch(patch)
}

// MARK: - Patch Data Structure

struct TrainingPatch {
    let noisyData: Data
    let denoisedData: Data
    let patchW: Int
    let patchH: Int
    let frameW: Int
    let frameH: Int
    let patchX: Int
    let patchY: Int
    let frameIdx: Int
    let noiseSigma: Float
    let flowMag: Float
    let cameraModel: String
    let iso: Int
}
