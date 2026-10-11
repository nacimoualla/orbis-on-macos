# GameController Framework for Input Abstraction

To implement `libScePad` (DualShock 4 emulation), we chose Apple's `GameController.framework` rather than raw IOKit HID driver access. `GameController.framework` natively supports DualShock 4 and DualSense gamepads over Bluetooth and USB (providing motion gyros, touchpad, and haptics without requiring private kernel entitlements) while providing fallback mapping for third-party Xbox and MFi controllers.
