// Generate the app's own vector-drawn controller artwork with AppKit.
// Usage: swift scripts/create_app_icon.swift resources/AppIcon.iconset
import AppKit
import Foundation
let output = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "resources/AppIcon.iconset"
try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let image = NSImage(size: NSSize(width: pixels, height: pixels))
        image.lockFocus()
        let context = NSGraphicsContext.current!.cgContext
        context.scaleBy(x: CGFloat(pixels)/1024, y: CGFloat(pixels)/1024)
        let rect = NSRect(x: 28, y: 28, width: 968, height: 968)
        let background = NSBezierPath(roundedRect: rect, xRadius: 216, yRadius: 216)
        let gradient = NSGradient(starting: NSColor(calibratedRed: 0.08, green: 0.22, blue: 0.43, alpha: 1),
                                  ending: NSColor(calibratedRed: 0.02, green: 0.06, blue: 0.16, alpha: 1))!
        gradient.draw(in: background, angle: 90)
        NSColor(calibratedRed: 0.18, green: 0.65, blue: 0.95, alpha: 0.35).setFill()
        NSBezierPath(ovalIn: NSRect(x: 215, y: 125, width: 594, height: 594)).fill()
        let body = NSBezierPath()
        body.move(to: NSPoint(x: 284, y: 654))
        body.curve(to: NSPoint(x: 175, y: 350), controlPoint1: NSPoint(x: 225, y: 640), controlPoint2: NSPoint(x: 140, y: 380))
        body.curve(to: NSPoint(x: 333, y: 365), controlPoint1: NSPoint(x: 196, y: 293), controlPoint2: NSPoint(x: 256, y: 292))
        body.line(to: NSPoint(x: 389, y: 426)); body.line(to: NSPoint(x: 635, y: 426)); body.line(to: NSPoint(x: 691, y: 365))
        body.curve(to: NSPoint(x: 849, y: 350), controlPoint1: NSPoint(x: 768, y: 292), controlPoint2: NSPoint(x: 828, y: 293))
        body.curve(to: NSPoint(x: 740, y: 654), controlPoint1: NSPoint(x: 884, y: 380), controlPoint2: NSPoint(x: 799, y: 640))
        body.curve(to: NSPoint(x: 284, y: 654), controlPoint1: NSPoint(x: 632, y: 693), controlPoint2: NSPoint(x: 392, y: 693)); body.close()
        NSColor(calibratedWhite: 0.96, alpha: 1).setFill(); body.fill()
        NSColor(calibratedRed: 0.08, green: 0.17, blue: 0.29, alpha: 1).setFill()
        NSBezierPath(roundedRect: NSRect(x: 282, y: 472, width: 128, height: 38), xRadius: 8, yRadius: 8).fill()
        NSBezierPath(roundedRect: NSRect(x: 327, y: 427, width: 38, height: 128), xRadius: 8, yRadius: 8).fill()
        for (x,y) in [(700,548),(746,500),(700,452),(654,500)] {
            NSBezierPath(ovalIn: NSRect(x: x-18,y: y-18,width: 36,height: 36)).fill()
        }
        NSColor(calibratedRed: 0.12, green: 0.53, blue: 0.83, alpha: 1).setFill()
        NSBezierPath(roundedRect: NSRect(x: 435, y: 557, width: 154, height: 63), xRadius: 15, yRadius: 15).fill()
        image.unlockFocus()
        let data = NSBitmapImageRep(data: image.tiffRepresentation!)!.representation(using: .png, properties: [:])!
        let name = "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png"
        try data.write(to: URL(fileURLWithPath: output).appendingPathComponent(name))
    }
}
