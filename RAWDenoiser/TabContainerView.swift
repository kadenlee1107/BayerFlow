import SwiftUI

// MARK: - Session Model

struct DenoisingSession: Identifiable {
    let id = UUID()
    var label: String = "New Session"
    var formatColor: Color = .secondary
    var pendingRestore: SessionSnapshot? = nil
}

// MARK: - Tab Container

struct TabContainerView: View {
    @EnvironmentObject private var license: LicenseManager
    @State private var sessions: [DenoisingSession] = [DenoisingSession()]
    @State private var selectedSessionID: UUID?
    @State private var confirmClose: UUID? = nil
    @State private var recoveredSnapshots: [SessionSnapshot] = []
    @State private var showRecoveryAlert = false
    private let tabActionsObj = TabActions()

    var body: some View {
        VStack(spacing: 0) {
            tabBar
            Divider()

            // Show the selected session
            ZStack {
                ForEach(sessions) { session in
                    SessionView(
                        session: session,
                        onFileLoaded: { name, color in
                            if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
                                sessions[idx].label = name
                                sessions[idx].formatColor = color
                            }
                        }
                    )
                    .environmentObject(license)
                    .opacity(session.id == selectedSessionID ? 1 : 0)
                    .allowsHitTesting(session.id == selectedSessionID)
                }
            }
        }
        .onAppear {
            if selectedSessionID == nil {
                selectedSessionID = sessions.first?.id
            }
            tabActionsObj.newTab = { [self] in addSession() }
            tabActionsObj.closeTab = { [self] in closeCurrentTab() }

            // Check for crash-recovered sessions
            let saved = SessionPersistenceManager.shared.loadAll()
            if !saved.isEmpty {
                recoveredSnapshots = saved
                showRecoveryAlert = true
            }
        }
        .focusedValue(\.tabActions, tabActionsObj)
        .alert("Close Tab?", isPresented: Binding(
            get: { confirmClose != nil },
            set: { if !$0 { confirmClose = nil } }
        )) {
            Button("Close", role: .destructive) {
                if let id = confirmClose {
                    removeSession(id: id)
                }
                confirmClose = nil
            }
            Button("Cancel", role: .cancel) {
                confirmClose = nil
            }
        } message: {
            Text("This session may still be processing. Close it anyway?")
        }
        .alert("Recover Sessions?", isPresented: $showRecoveryAlert) {
            Button("Recover") {
                // Replace the default empty session with recovered ones
                var restoredSessions: [DenoisingSession] = []
                for snapshot in recoveredSnapshots {
                    var s = DenoisingSession()
                    s.label = snapshot.label
                    s.formatColor = Color.fromHex(snapshot.formatColorHex)
                    s.pendingRestore = snapshot
                    restoredSessions.append(s)
                }
                if !restoredSessions.isEmpty {
                    sessions = restoredSessions
                    selectedSessionID = sessions.first?.id
                }
                // Delete old snapshot files — SessionView will create fresh ones
                SessionPersistenceManager.shared.deleteAll()
            }
            Button("Discard", role: .destructive) {
                SessionPersistenceManager.shared.deleteAll()
            }
        } message: {
            Text("\(recoveredSnapshots.count) session(s) from a previous run were found. Would you like to restore them?")
        }
    }

    // MARK: - Tab Bar

    private var tabBar: some View {
        HStack(spacing: 2) {
            ForEach(sessions) { session in
                tabButton(for: session)
            }

            // Add tab button
            Button {
                addSession()
            } label: {
                Image(systemName: "plus")
                    .font(.caption)
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("New Tab (Cmd+T)")

            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.bar)
    }

    private func tabButton(for session: DenoisingSession) -> some View {
        let isSelected = session.id == selectedSessionID

        return Button {
            selectedSessionID = session.id
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(session.formatColor)
                    .frame(width: 8, height: 8)

                Text(session.label)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)

                if sessions.count > 1 {
                    Button {
                        closeSession(id: session.id)
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 8, weight: .bold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Close Tab")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(isSelected ? Color.accentColor.opacity(0.3) : Color.clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Session Management

    func addSession() {
        let session = DenoisingSession()
        sessions.append(session)
        selectedSessionID = session.id
    }

    private func closeSession(id: UUID) {
        // For now, just close directly (we could check engine state later)
        removeSession(id: id)
    }

    private func removeSession(id: UUID) {
        guard sessions.count > 1 else { return }
        SessionPersistenceManager.shared.delete(id: id.uuidString)
        sessions.removeAll { $0.id == id }
        if selectedSessionID == id {
            selectedSessionID = sessions.first?.id
        }
    }

    func closeCurrentTab() {
        guard let current = selectedSessionID, sessions.count > 1 else { return }
        closeSession(id: current)
    }
}
