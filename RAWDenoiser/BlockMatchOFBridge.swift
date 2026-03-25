import Foundation

/// C bridge for Metal block matching optical flow.

@_cdecl("compute_metal_flow")
func compute_metal_flow(
    _ center: UnsafePointer<UInt16>,
    _ neighbor: UnsafePointer<UInt16>,
    _ green_w: Int32, _ green_h: Int32,
    _ flow_x: UnsafeMutablePointer<Float>,
    _ flow_y: UnsafeMutablePointer<Float>
) -> Int32 {
    guard let of = MetalBlockMatchOF.shared else { return -1 }
    of.computeFlow(center: center, neighbor: neighbor,
                   width: Int(green_w), height: Int(green_h),
                   flowX: flow_x, flowY: flow_y)
    return 0
}

/// Batch version: ALL neighbors in a single GPU command buffer.
@_cdecl("compute_metal_flow_batch")
func compute_metal_flow_batch(
    _ center: UnsafePointer<UInt16>,
    _ neighbors: UnsafePointer<UnsafePointer<UInt16>?>,
    _ num_neighbors: Int32,
    _ green_w: Int32, _ green_h: Int32,
    _ fx_out: UnsafePointer<UnsafeMutablePointer<Float>?>,
    _ fy_out: UnsafePointer<UnsafeMutablePointer<Float>?>
) -> Int32 {
    guard let of = MetalBlockMatchOF.shared else { return -1 }

    let w = Int(green_w)
    let h = Int(green_h)
    let n = Int(num_neighbors)

    // Build center pyramid once
    of.buildCenterPyramid(center: center, width: w, height: h)

    // Collect valid neighbors
    var validNbrs: [UnsafePointer<UInt16>] = []
    var validFx: [UnsafeMutablePointer<Float>] = []
    var validFy: [UnsafeMutablePointer<Float>] = []
    for i in 0..<n {
        guard let nbr = neighbors[i],
              let fx = fx_out[i],
              let fy = fy_out[i] else { continue }
        validNbrs.append(nbr)
        validFx.append(fx)
        validFy.append(fy)
    }

    if validNbrs.isEmpty { return 0 }

    // Process all neighbors in a single GPU command buffer
    of.computeFlowBatch(neighbors: validNbrs, width: w, height: h,
                         flowXs: validFx, flowYs: validFy)
    return 0
}
