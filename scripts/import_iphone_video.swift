#!/usr/bin/env swift
// Normal ImageCaptureCore API only. Enumerate metadata and optionally copy one
// exact known video; never requests thumbnails, deletes, PTP commands or UI.
import Foundation
import ImageCaptureCore
import Darwin

struct Options {
    var filename = "OQEM6774.MP4"
    var deviceName = "iPhone"
    var directory = URL(fileURLWithPath: "./video-script/import/source", isDirectory: true)
    var report = URL(fileURLWithPath: "./video-script/import/import-metadata.json")
    var download = false
    var timeout: TimeInterval = 300
    var discoveryTimeout: TimeInterval = 20
}
var options = Options()
var index = 1
while index < CommandLine.arguments.count {
    let arg = CommandLine.arguments[index]
    if arg == "--help" {
        print("Usage: imagecapture-import [--device NAME] [--filename OQEM6774.MP4] [--out DIR] [--report JSON] [--discovery-timeout SEC] [--timeout SEC] [--download]\nDevice name matches exactly (default iPhone). Discovery/session/catalog timeout defaults to 20 seconds; download timeout defaults to 300. Default is metadata only. --download copies exactly the validated target without deleting or overwriting anything.")
        exit(0)
    } else if arg == "--download" { options.download = true }
    else {
        index += 1
        guard index < CommandLine.arguments.count else { print("Missing value for \(arg)"); exit(2) }
        let value = CommandLine.arguments[index]
        switch arg {
        case "--device": options.deviceName = value
        case "--filename": options.filename = value
        case "--out": options.directory = URL(fileURLWithPath: value, isDirectory: true)
        case "--report": options.report = URL(fileURLWithPath: value)
        case "--timeout": options.timeout = Double(value) ?? 0
        case "--discovery-timeout": options.discoveryTimeout = Double(value) ?? 0
        default: print("Unknown argument: \(arg)"); exit(2)
        }
    }
    index += 1
}
guard options.timeout > 0, options.discoveryTimeout > 0, options.filename == URL(fileURLWithPath: options.filename).lastPathComponent else { print("Invalid filename/timeout"); exit(2) }

final class Importer: NSObject, ICDeviceBrowserDelegate, ICCameraDeviceDelegate {
    let options: Options
    let browser = ICDeviceBrowser()
    var camera: ICCameraDevice?
    var target: ICCameraFile?
    var progress: Progress?
    var done = false
    var exitCode: Int32 = 1
    var startedAt = Date()
    var deadline: Date
    var lastProgressAt = Date.distantPast
    var candidateCallbacks = 0
    var openPending = false
    var catalogReady = false
    var downloadRequested = false
    var output: [String: Any]

    init(_ options: Options) {
        self.options = options
        self.deadline = Date().addingTimeInterval(options.discoveryTimeout)
        self.output = ["status": "running", "target_name": options.filename, "expected_device_name": options.deviceName,
            "mode": options.download ? "copy_exact_target" : "target_metadata_only",
            "source_delete_enabled": false, "overwrite_enabled": false,
            "sidecar_download_enabled": false, "output_directory": options.directory.path,
            "discovery_timeout_seconds": options.discoveryTimeout, "download_timeout_seconds": options.timeout,
            "events": [[String: Any]](), "started_at": ISO8601DateFormatter().string(from: Date())]
        super.init()
        browser.delegate = self
        browser.browsedDeviceTypeMask = ICDeviceTypeMask(rawValue: ICDeviceTypeMask.camera.rawValue | ICDeviceLocationTypeMask.local.rawValue)!
    }

    func event(_ name: String, _ fields: [String: Any] = [:]) {
        var row = fields; row["event"] = name; row["elapsed_seconds"] = Date().timeIntervalSince(startedAt)
        var events = output["events"] as! [[String: Any]]; events.append(row); output["events"] = events
        if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), let line = String(data: data, encoding: .utf8) { print(line); fflush(stdout) }
    }

    func finish(_ status: String, _ message: String? = nil, code: Int32 = 1) {
        guard !done else { return }
        done = true; exitCode = code; output["status"] = status
        if let message = message { output["message"] = message }
        output["finished_at"] = ISO8601DateFormatter().string(from: Date())
        output["elapsed_seconds"] = Date().timeIntervalSince(startedAt)
        output["final_device_state"] = ["session_opened": camera?.hasOpenSession ?? false,
            "access_restricted": camera?.isAccessRestrictedAppleDevice ?? false,
            "catalog_ready": catalogReady, "open_request_pending": openPending]
        if let camera = camera, camera.hasOpenSession { camera.requestCloseSession() }
        browser.stop()
        do {
            try FileManager.default.createDirectory(at: options.report.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: options.report, options: .atomic)
            print("\(status.uppercased()): \(options.report.path)")
        } catch { print("Could not write report: \(error)") }
        fflush(stdout)
    }

    func checkTimeout() {
        if !done && Date() >= deadline {
            event("discovery_timeout_state", ["is_browsing": browser.isBrowsing,
                "mask": browser.browsedDeviceTypeMask.rawValue,
                "devices_count": browser.devices?.count ?? 0, "candidate_callbacks": candidateCallbacks,
                "session_opened": camera?.hasOpenSession ?? false,
                "access_restricted": camera?.isAccessRestrictedAppleDevice ?? false,
                "catalog_percent_completed": camera?.contentCatalogPercentCompleted ?? 0])
            progress?.cancel()
            if camera?.isAccessRestrictedAppleDevice == true {
                finish("access_restricted_timeout", "Device remained access-restricted for the bounded wait; no permission bypass or UI fallback attempted")
            } else {
                finish("timeout", "Normal ImageCaptureCore API timed out; no permission bypass or UI fallback attempted")
            }
        }
        if let progress = progress, !done, Date().timeIntervalSince(lastProgressAt) >= 5 {
            lastProgressAt = Date()
            event("download_progress", ["completed": progress.completedUnitCount, "total": progress.totalUnitCount, "fraction": progress.fractionCompleted])
        }
    }

    func deviceBrowser(_ browser: ICDeviceBrowser, didAdd device: ICDevice, moreComing: Bool) {
        candidateCallbacks += 1
        event("local_device_candidate", ["name": device.name ?? "unnamed", "is_camera": device is ICCameraDevice])
        guard camera == nil, let cameraDevice = device as? ICCameraDevice,
              device.name == options.deviceName else { return }
        camera = cameraDevice; cameraDevice.delegate = self
        output["device"] = ["name": device.name ?? "iPhone", "transport": device.transportType ?? "unknown"]
        event("iphone_discovered", ["name": device.name ?? "iPhone"])
        openPending = true; cameraDevice.requestOpenSession()
    }
    func deviceBrowser(_ browser: ICDeviceBrowser, didRemove device: ICDevice, moreGoing: Bool) { if device === camera { finish("device_removed", "iPhone disconnected") } }
    func didRemove(_ device: ICDevice) { if device === camera { finish("device_removed", "iPhone disconnected") } }
    func device(_ device: ICDevice, didOpenSessionWithError error: Error?) {
        openPending = false
        if let error = error {
            event("session_open_error", ["error": String(describing: error), "access_restricted": camera?.isAccessRestrictedAppleDevice ?? false])
            // An initially restricted device can become available after the
            // user's normal unlock/trust action. Wait for the bounded state
            // transition, rather than interpreting its notification as final.
            if camera?.isAccessRestrictedAppleDevice == true { return }
            finish("session_failed", String(describing: error)); return
        }
        event("session_opened")
    }
    func device(_ device: ICDevice, didCloseSessionWithError error: Error?) { if !done { finish("session_closed", error.map(String.init(describing:)) ?? "Device session closed unexpectedly") } }
    func device(_ device: ICDevice, didEncounterError error: Error?) { if let error = error { finish("device_error", String(describing: error)) } }
    func cameraDeviceDidEnableAccessRestriction(_ device: ICDevice) {
        event("access_restriction_enabled", ["property_access_restricted": camera?.isAccessRestrictedAppleDevice ?? false,
            "session_opened": camera?.hasOpenSession ?? false])
    }
    func cameraDeviceDidRemoveAccessRestriction(_ device: ICDevice) {
        event("access_restriction_removed", ["property_access_restricted": camera?.isAccessRestrictedAppleDevice ?? false,
            "session_opened": camera?.hasOpenSession ?? false])
        guard !done, let camera = camera, !camera.isAccessRestrictedAppleDevice else { return }
        if !camera.hasOpenSession && !openPending {
            // Retry only after the system reports normal access was granted.
            openPending = true; camera.requestOpenSession()
        } else if catalogReady && camera.hasOpenSession {
            deviceDidBecomeReady(withCompleteContentCatalog: camera)
        }
    }
    func cameraDevice(_ camera: ICCameraDevice, didAdd items: [ICCameraItem]) {
        // Inspect catalog names solely to select the requested file. Never read
        // or print any unrelated file contents, thumbnails or item metadata.
        for item in items where item.name == options.filename {
            if let file = item as? ICCameraFile { target = file }
        }
    }
    func cameraDevice(_ camera: ICCameraDevice, didRemove items: [ICCameraItem]) {}
    func cameraDevice(_ camera: ICCameraDevice, didRenameItems items: [ICCameraItem]) {}
    func cameraDeviceDidChangeCapability(_ camera: ICCameraDevice) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceivePTPEvent event: Data) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceiveThumbnail thumbnail: CGImage?, for item: ICCameraItem, error: Error?) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceiveMetadata metadata: [AnyHashable: Any]?, for item: ICCameraItem, error: Error?) {}

    func deviceDidBecomeReady(withCompleteContentCatalog device: ICCameraDevice) {
        guard !done, !downloadRequested else { return }
        catalogReady = true
        event("complete_catalog_ready", ["session_opened": device.hasOpenSession,
            "access_restricted": device.isAccessRestrictedAppleDevice])
        // A download is permitted only after a real open session and normal
        // unrestricted catalog readiness. Never force either property.
        guard device.hasOpenSession, !device.isAccessRestrictedAppleDevice else { return }
        let matches = (device.mediaFiles ?? []).compactMap { $0 as? ICCameraFile }.filter { $0.name == options.filename }
        guard matches.count == 1, let file = matches.first else { finish("target_not_unique", "Exact target appears \(matches.count) times in the available catalog"); return }
        target = file
        let formatter = ISO8601DateFormatter(); formatter.timeZone = TimeZone(identifier: "Asia/Shanghai")
        let metadata: [String: Any] = ["name": file.name ?? "", "original_filename": file.originalFilename ?? "",
            "bytes": file.fileSize, "width": file.width, "height": file.height, "duration_seconds": file.duration,
            "creation_date_Asia_Shanghai": file.creationDate.map { formatter.string(from: $0) } ?? NSNull(),
            "modification_date_Asia_Shanghai": file.modificationDate.map { formatter.string(from: $0) } ?? NSNull()]
        output["target_metadata"] = metadata; event("target_verified_metadata", metadata)
        guard options.download else { finish("metadata_verified", code: 0); return }
        guard file.fileSize > 3_000_000_000, file.fileSize < 4_500_000_000,
              file.width == 3840, file.height == 2160,
              file.creationDate.map({ formatter.string(from: $0).hasPrefix("2026-10-02T05:54:") }) == true else {
            finish("metadata_mismatch", "Target name matches but expected size/resolution/creation minute does not; nothing downloaded"); return
        }
        do {
            try FileManager.default.createDirectory(at: options.directory, withIntermediateDirectories: true)
            let destination = options.directory.appendingPathComponent(options.filename)
            guard !FileManager.default.fileExists(atPath: destination.path) else { finish("destination_exists", "Refusing to overwrite \(destination.path)"); return }
            output["destination"] = destination.path
            downloadRequested = true
            event("download_requested", ["destination": destination.path, "delete_after_download": false, "download_sidecars": false])
            deadline = Date().addingTimeInterval(options.timeout)
            progress = file.requestDownload(options: [.downloadsDirectoryURL: options.directory,
                .saveAsFilename: options.filename, .overwrite: false,
                .deleteAfterSuccessfulDownload: false, .sidecarFiles: false]) { filename, error in
                DispatchQueue.main.async {
                    if let error = error { self.finish("download_failed", String(describing: error)); return }
                    let actualURL = self.options.directory.appendingPathComponent(filename ?? self.options.filename)
                    do {
                        let attrs = try FileManager.default.attributesOfItem(atPath: actualURL.path)
                        let bytes = (attrs[.size] as? NSNumber)?.int64Value ?? -1
                        self.output["saved"] = ["path": actualURL.path, "bytes": bytes, "matches_device_size": bytes == file.fileSize]
                        self.finish(bytes == file.fileSize ? "downloaded" : "size_mismatch", code: bytes == file.fileSize ? 0 : 1)
                    } catch { self.finish("saved_file_unreadable", String(describing: error)) }
                }
            }
        } catch { finish("output_error", String(describing: error)) }
    }
}

let importer = Importer(options)
importer.event("normal_imagecapture_api_started", ["download": options.download])
importer.browser.start()
while !importer.done {
    RunLoop.current.run(until: Date().addingTimeInterval(0.1))
    importer.checkTimeout()
}
exit(importer.exitCode)
