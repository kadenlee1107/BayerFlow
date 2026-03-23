import Cocoa
import SwiftUI

/// Principal class for the FCP Workflow Extension.
/// FCP instantiates this when the user opens Window → Extensions → BayerFlow.
class WorkflowExtensionViewController: NSViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        let rootView = ExtensionRootView()
        let hostingView = NSHostingView(rootView: rootView)
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hostingView)
        NSLayoutConstraint.activate([
            hostingView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingView.topAnchor.constraint(equalTo: view.topAnchor),
            hostingView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    override func loadView() {
        self.view = NSView(frame: NSRect(x: 0, y: 0, width: 480, height: 640))
    }
}
