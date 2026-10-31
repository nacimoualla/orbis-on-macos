// Game import / library GUI (AppKit).
//
// Test build for this branch: pick a folder, scan it for PS4 titles, run the
// selected one, and stream the runner's debug log.
//
// A "title" here is what a PS4 dump actually produces: a directory named
// <content-id>_<something> containing eboot.bin, which matches the layout of
// games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/.

#import <AppKit/AppKit.h>
#import <mach-o/dyld.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "ps4/game_import.hpp"

// The runner, resolved next to this binary so the GUI works from a build tree
// with no install step.
static std::string runner_path() {
    char buf[4096] = {0};
    uint32_t n = uint32_t(sizeof(buf) - 1);
    if (_NSGetExecutablePath(buf, &n) != 0) return "ps4-runner";
    std::string p(buf);
    const size_t slash = p.find_last_of('/');
    if (slash == std::string::npos) return "ps4-runner";
    return p.substr(0, slash) + "/ps4-runner";
}

// Where run logs go, and the ONLY place they go: <repo root>/Logs.
//
// Resolved by walking up from this executable until a CMakeLists.txt is found,
// so it resolves identically whether the app runs from the bundle
// (build/ps4-gui.app/Contents/MacOS/) or as a loose binary (build/).
//
// There is deliberately no fallback. An earlier version wrote inside the .app
// bundle, which is wrong twice over: it buries logs somewhere unintuitive, and
// a future code signature would be invalidated by files appearing under
// Contents/. Rather than silently scattering logs to a second location, an
// unresolved root is reported and the run is not logged.
static bool repo_root(std::string& out) {
    char buf[4096] = {0};
    uint32_t n = uint32_t(sizeof(buf) - 1);
    if (_NSGetExecutablePath(buf, &n) != 0) return false;
    std::string dir(buf);
    const size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos) dir = dir.substr(0, slash);

    for (int hops = 0; hops < 8; ++hops) {
        const size_t s2 = dir.find_last_of('/');
        if (s2 == std::string::npos) break;
        dir = dir.substr(0, s2);
        if (dir.empty()) break;
        if (::access((dir + "/CMakeLists.txt").c_str(), F_OK) == 0) {
            out = dir;
            return true;
        }
    }
    return false;
}

NSString* human_size(ps4::u64 bytes) {
    if (bytes >= (1ull << 30))
        return [NSString stringWithFormat:@"%.2f GB", double(bytes) / 1073741824.0];
    if (bytes >= (1ull << 20))
        return [NSString stringWithFormat:@"%.1f MB", double(bytes) / 1048576.0];
    return [NSString stringWithFormat:@"%llu B", (unsigned long long)bytes];
}

// ---------------------------------------------------------------------------
// Objective-C classes must live at global scope; they are not allowed inside a
// C++ namespace. A namespace alias keeps the unqualified C++ names below
// (GameEntry, scan_folder, runner_path, human_size) resolving correctly.
// ---------------------------------------------------------------------------
namespace gui = ps4::gui;
using ps4::gui::GameEntry;
using ps4::gui::scan_folder;

@interface GameLibraryController : NSWindowController <NSTableViewDataSource, NSTableViewDelegate>
@property (nonatomic, strong) NSTableView* table;
@property (nonatomic, strong) NSScrollView* tableScroll;
@property (nonatomic, strong) NSImageView* previewIconView;
@property (nonatomic, strong) NSTextField* previewTitleLabel;
@property (nonatomic, strong) NSTextField* previewMetaLabel;
@property (nonatomic, strong) NSButton* bigLaunchButton;
@property (nonatomic, strong) NSTextView* logView;
@property (nonatomic, strong) NSSearchField* logSearchField;
@property (nonatomic, strong) NSPopUpButton* logFilterPopup;
@property (nonatomic, strong) NSTextField* statusLabel;
@property (nonatomic, strong) NSTextField* folderLabel;
@property (nonatomic, strong) NSPopUpButton* budgetPopup;
@property (nonatomic, strong) NSButton* jitCheckbox;
@property (nonatomic, strong) NSButton* windowCheckbox;
@property (nonatomic, strong) NSButton* runButton;
@property (nonatomic, strong) NSButton* stopButton;
@property (nonatomic, strong) NSProgressIndicator* spinner;
@property (nonatomic, strong) NSMutableArray<NSString*>* logLines;
@property (nonatomic, strong) NSMutableArray<NSString*>* pendingLogLines;
@property (nonatomic, strong) NSLock* pendingLogLock;
@property (nonatomic, assign) BOOL logFlushScheduled;
@property (nonatomic, strong) dispatch_queue_t ioQueue;
@property (nonatomic, strong) NSTask* task;
@property (nonatomic, strong) NSFileHandle* logFile;
@property (nonatomic, copy) NSString* logPath;
@end

// C++ state cannot be an ObjC property; a class extension can still hold the
// ivar because the non-fragile ABI allows ivars there.
@interface GameLibraryController () {
    std::vector<gui::GameEntry> _found;
}
- (void)scanFolder:(NSString*)path;
- (void)setStatus:(NSString*)s;
- (void)appendLog:(NSString*)line;
- (void)appendLogLines:(NSArray<NSString*>*)lines;
- (void)flushPendingLogs;
- (void)updateDetailPaneForSelectedRow;
- (void)reapplyLogFilter;
@end

@implementation GameLibraryController

- (instancetype)init {
    self = [super init];
    if (self) {
        _logLines = [NSMutableArray array];
        _pendingLogLines = [NSMutableArray array];
        _pendingLogLock = [NSLock new];
        _logFlushScheduled = NO;
        _ioQueue = dispatch_queue_create("ps4.gui.io", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

#pragma mark - UI

- (void)buildWindow {
    const NSRect frame = NSMakeRect(0, 0, 1040, 700);
    NSWindow* win = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    win.title = @"PS4 Game Library";
    [win center];
    self.window = win;

    NSView* root = win.contentView;
    const CGFloat W = frame.size.width;

    // --- Header controls ---
    NSButton* pick = [NSButton buttonWithTitle:@"Import Folder…"
                                        target:self
                                        action:@selector(onPickFolder:)];
    pick.frame = NSMakeRect(16, frame.size.height - 46, 120, 28);
    [root addSubview:pick];

    NSButton* controlsBtn = [NSButton buttonWithTitle:@"Controls Guide…"
                                               target:self
                                               action:@selector(onShowControls:)];
    controlsBtn.frame = NSMakeRect(142, frame.size.height - 46, 126, 28);
    [root addSubview:controlsBtn];

    _folderLabel = [[NSTextField alloc] initWithFrame:
        NSMakeRect(276, frame.size.height - 44, W - 320, 20)];
    _folderLabel.stringValue = @"No folder selected";
    _folderLabel.editable = NO;
    _folderLabel.bordered = NO;
    _folderLabel.drawsBackground = NO;
    _folderLabel.font = [NSFont systemFontOfSize:11];
    _folderLabel.lineBreakMode = NSLineBreakByTruncatingHead;
    [root addSubview:_folderLabel];

    _spinner = [[NSProgressIndicator alloc] initWithFrame:
        NSMakeRect(W - 36, frame.size.height - 42, 16, 16)];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.hidden = YES;
    [root addSubview:_spinner];

    // --- Games list table ---
    _table = [[NSTableView alloc] initWithFrame:NSMakeRect(0, 0, 640, 350)];

    // Icon thumbnail column
    NSTableColumn* iconCol = [[NSTableColumn alloc] initWithIdentifier:@"icon"];
    iconCol.title = @"";
    iconCol.width = 44;
    NSImageCell* imgCell = [[NSImageCell alloc] init];
    imgCell.imageScaling = NSImageScaleProportionallyUpOrDown;
    iconCol.dataCell = imgCell;
    [_table addTableColumn:iconCol];

    NSDictionary* cols = @{
        @"title":    @230,
        @"title_id": @110,
        @"app_ver":  @70,
        @"size":     @80,
        @"status":   @90
    };
    NSDictionary* colTitles = @{
        @"title":    @"Title",
        @"title_id": @"Title ID",
        @"app_ver":  @"Version",
        @"size":     @"Size",
        @"status":   @"Status"
    };
    for (NSString* key in @[ @"title", @"title_id", @"app_ver", @"size", @"status" ]) {
        NSTableColumn* col = [[NSTableColumn alloc] initWithIdentifier:key];
        col.title = colTitles[key];
        col.width = [cols[key] doubleValue];
        [_table addTableColumn:col];
    }
    _table.dataSource = self;
    _table.delegate = self;
    _table.target = self;
    _table.doubleAction = @selector(onRun:);
    _table.rowHeight = 36;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.allowsMultipleSelection = NO;

    _tableScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(16, 295, 640, 350)];
    _tableScroll.documentView = _table;
    _tableScroll.hasVerticalScroller = YES;
    _tableScroll.borderType = NSBezelBorder;
    [root addSubview:_tableScroll];

    // --- Game Detail Preview Pane ---
    NSBox* detailBox = [[NSBox alloc] initWithFrame:NSMakeRect(666, 295, W - 666 - 16, 350)];
    detailBox.title = @"Game Details";
    detailBox.titleFont = [NSFont boldSystemFontOfSize:12];

    _previewIconView = [[NSImageView alloc] initWithFrame:NSMakeRect(16, 185, 128, 128)];
    _previewIconView.imageScaling = NSImageScaleProportionallyUpOrDown;
    _previewIconView.wantsLayer = YES;
    _previewIconView.layer.cornerRadius = 10.0;
    _previewIconView.layer.masksToBounds = YES;
    [detailBox.contentView addSubview:_previewIconView];

    _previewTitleLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(154, 185, 180, 128)];
    _previewTitleLabel.editable = NO;
    _previewTitleLabel.selectable = YES;
    _previewTitleLabel.bordered = NO;
    _previewTitleLabel.drawsBackground = NO;
    _previewTitleLabel.font = [NSFont boldSystemFontOfSize:14];
    _previewTitleLabel.cell.wraps = YES;
    _previewTitleLabel.cell.lineBreakMode = NSLineBreakByWordWrapping;
    _previewTitleLabel.stringValue = @"No game selected";
    [detailBox.contentView addSubview:_previewTitleLabel];

    _previewMetaLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(16, 56, 320, 115)];
    _previewMetaLabel.editable = NO;
    _previewMetaLabel.selectable = YES;
    _previewMetaLabel.bordered = NO;
    _previewMetaLabel.drawsBackground = NO;
    _previewMetaLabel.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
    _previewMetaLabel.textColor = [NSColor secondaryLabelColor];
    _previewMetaLabel.cell.wraps = YES;
    _previewMetaLabel.cell.lineBreakMode = NSLineBreakByWordWrapping;
    _previewMetaLabel.stringValue = @"Select a title from the list to view details and launch.";
    [detailBox.contentView addSubview:_previewMetaLabel];

    _bigLaunchButton = [NSButton buttonWithTitle:@"▶  Launch Title" target:self action:@selector(onRun:)];
    _bigLaunchButton.frame = NSMakeRect(16, 14, 320, 34);
    _bigLaunchButton.bezelStyle = NSBezelStyleRegularSquare;
    _bigLaunchButton.font = [NSFont boldSystemFontOfSize:13];
    _bigLaunchButton.enabled = NO;
    [detailBox.contentView addSubview:_bigLaunchButton];

    [root addSubview:detailBox];

    // --- Run controls ---
    _runButton = [NSButton buttonWithTitle:@"Run" target:self action:@selector(onRun:)];
    _runButton.frame = NSMakeRect(16, 257, 75, 28);
    [root addSubview:_runButton];

    _stopButton = [NSButton buttonWithTitle:@"Stop" target:self action:@selector(onStop:)];
    _stopButton.frame = NSMakeRect(96, 257, 75, 28);
    _stopButton.enabled = NO;
    [root addSubview:_stopButton];

    _budgetPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(178, 257, 185, 26)];
    [_budgetPopup addItemsWithTitles:@[ @"Unlimited (continuous)",
                                        @"Budget 100M (quick)",
                                        @"Budget 1G (long)",
                                        @"Budget 300G (soak)" ]];
    [_budgetPopup selectItemAtIndex:0];
    [root addSubview:_budgetPopup];

    _jitCheckbox = [NSButton checkboxWithTitle:@"JIT" target:nil action:nil];
    _jitCheckbox.frame = NSMakeRect(372, 261, 50, 18);
    _jitCheckbox.state = NSControlStateValueOn;
    [root addSubview:_jitCheckbox];

    _windowCheckbox = [NSButton checkboxWithTitle:@"Window" target:nil action:nil];
    _windowCheckbox.frame = NSMakeRect(430, 261, 75, 18);
    _windowCheckbox.state = NSControlStateValueOn;
    [root addSubview:_windowCheckbox];

    _statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(515, 261, W - 530, 18)];
    _statusLabel.stringValue = @"Idle";
    _statusLabel.editable = NO;
    _statusLabel.bordered = NO;
    _statusLabel.drawsBackground = NO;
    _statusLabel.font = [NSFont systemFontOfSize:11];
    [root addSubview:_statusLabel];

    // --- Log toolbar ---
    _logSearchField = [[NSSearchField alloc] initWithFrame:NSMakeRect(16, 225, 200, 24)];
    _logSearchField.placeholderString = @"Search logs…";
    _logSearchField.target = self;
    _logSearchField.action = @selector(onLogFilterChanged:);
    [root addSubview:_logSearchField];

    _logFilterPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(222, 224, 120, 26)];
    [_logFilterPopup addItemsWithTitles:@[ @"All Logs", @"GPU / PM4", @"JIT", @"HLE", @"Errors" ]];
    _logFilterPopup.target = self;
    _logFilterPopup.action = @selector(onLogFilterChanged:);
    [root addSubview:_logFilterPopup];

    NSButton* copyBtn = [NSButton buttonWithTitle:@"Copy Log" target:self action:@selector(onCopyLog:)];
    copyBtn.frame = NSMakeRect(W - 325, 224, 90, 26);
    [root addSubview:copyBtn];

    NSButton* clearBtn = [NSButton buttonWithTitle:@"Clear" target:self action:@selector(onClearLog:)];
    clearBtn.frame = NSMakeRect(W - 230, 224, 75, 26);
    [root addSubview:clearBtn];

    NSButton* revealBtn = [NSButton buttonWithTitle:@"Reveal Logs…" target:self action:@selector(onRevealLogs:)];
    revealBtn.frame = NSMakeRect(W - 150, 224, 134, 26);
    [root addSubview:revealBtn];

    // --- Log text view ---
    _logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 36, 196)];
    _logView.editable = NO;
    _logView.allowsUndo = NO;
    _logView.font = [NSFont monospacedDigitSystemFontOfSize:10
                                                     weight:NSFontWeightRegular];

    NSScrollView* logScroll = [[NSScrollView alloc] initWithFrame:
        NSMakeRect(16, 16, W - 32, 202)];
    logScroll.documentView = _logView;
    logScroll.hasVerticalScroller = YES;
    logScroll.borderType = NSBezelBorder;
    [root addSubview:logScroll];

    // Auto-scan <repo_root>/games if present on launch
    std::string rootDir;
    if (repo_root(rootDir)) {
        std::string gamesDir = rootDir + "/games";
        if (::access(gamesDir.c_str(), F_OK) == 0) {
            _folderLabel.stringValue = @(gamesDir.c_str());
            [self scanFolder:@(gamesDir.c_str())];
        }
    }
}

#pragma mark - Log

- (void)appendLog:(NSString*)line {
    if (line.length) {
        [self appendLogLines:@[line]];
    }
}

- (void)appendLogLines:(NSArray<NSString*>*)lines {
    if (lines.count == 0) return;
    [_pendingLogLock lock];
    [_pendingLogLines addObjectsFromArray:lines];
    if (!_logFlushScheduled) {
        _logFlushScheduled = YES;
        [_pendingLogLock unlock];
        __weak GameLibraryController* weakSelf = self;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(100 * NSEC_PER_MSEC)), dispatch_get_main_queue(), ^{
            [weakSelf flushPendingLogs];
        });
    } else {
        [_pendingLogLock unlock];
    }
}

- (void)reapplyLogFilter {
    NSString* search = [_logSearchField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]].lowercaseString;
    NSInteger cat = _logFilterPopup.indexOfSelectedItem; // 0=All, 1=GPU/PM4, 2=JIT, 3=HLE, 4=Errors

    if (search.length == 0 && cat == 0) {
        NSString* joined = [_logLines componentsJoinedByString:@"\n"];
        _logView.string = joined;
        [_logView scrollRangeToVisible:NSMakeRange(joined.length, 0)];
        return;
    }

    NSMutableArray<NSString*>* filtered = [NSMutableArray array];
    for (NSString* line in _logLines) {
        NSString* lower = line.lowercaseString;
        if (cat == 1) { // GPU/PM4
            if (!([lower containsString:@"[pm4]"] || [lower containsString:@"[gpu]"] ||
                  [lower containsString:@"[gcn]"] || [lower containsString:@"[metal]"] ||
                  [lower containsString:@"[shader]"] || [lower containsString:@"[draw]"])) {
                continue;
            }
        } else if (cat == 2) { // JIT
            if (!([lower containsString:@"[jit]"] || [lower containsString:@"[cpu]"] ||
                  [lower containsString:@"[x86]"])) {
                continue;
            }
        } else if (cat == 3) { // HLE
            if (!([lower containsString:@"[hle]"] || [lower containsString:@"[syscall]"] ||
                  [lower containsString:@"sce"])) {
                continue;
            }
        } else if (cat == 4) { // Errors
            if (!([lower containsString:@"error"] || [lower containsString:@"fail"] ||
                  [lower containsString:@"fatal"] || [lower containsString:@"assert"] ||
                  [lower containsString:@"invalid"] || [lower containsString:@"warn"])) {
                continue;
            }
        }

        if (search.length > 0 && ![lower containsString:search]) {
            continue;
        }

        [filtered addObject:line];
    }

    NSString* joined = [filtered componentsJoinedByString:@"\n"];
    _logView.string = joined;
    [_logView scrollRangeToVisible:NSMakeRange(joined.length, 0)];
}

- (void)onLogFilterChanged:(id)sender {
    [self reapplyLogFilter];
}

- (void)onCopyLog:(id)sender {
    NSPasteboard* pb = [NSPasteboard generalPasteboard];
    [pb clearContents];
    NSString* str = _logView.string;
    if (str.length) {
        [pb setString:str forType:NSPasteboardTypeString];
    }
}

- (void)onClearLog:(id)sender {
    [_pendingLogLock lock];
    [_pendingLogLines removeAllObjects];
    [_logLines removeAllObjects];
    [_pendingLogLock unlock];
    _logView.string = @"";
}

- (void)onRevealLogs:(id)sender {
    std::string root;
    if (repo_root(root)) {
        NSString* logDir = [@(root.c_str()) stringByAppendingPathComponent:@"Logs"];
        [[NSFileManager defaultManager] createDirectoryAtPath:logDir withIntermediateDirectories:YES attributes:nil error:nil];
        [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:logDir]];
    }
}

- (void)flushPendingLogs {
    NSArray* toAppend = nil;
    [_pendingLogLock lock];
    _logFlushScheduled = NO;
    if (_pendingLogLines.count > 0) {
        toAppend = [_pendingLogLines copy];
        [_pendingLogLines removeAllObjects];
    }
    [_pendingLogLock unlock];

    if (toAppend.count == 0) return;

    @autoreleasepool {
        for (NSString* line in toAppend) {
            if (line.length) [_logLines addObject:line];
        }
        if (_logLines.count > 10000) {
            [_logLines removeObjectsInRange:NSMakeRange(0, _logLines.count - 6000)];
        }
        NSString* search = [_logSearchField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        NSInteger cat = _logFilterPopup.indexOfSelectedItem;
        if (search.length > 0 || cat > 0) {
            [self reapplyLogFilter];
        } else {
            NSString* joined = [_logLines componentsJoinedByString:@"\n"];
            _logView.string = joined;
            [_logView scrollRangeToVisible:NSMakeRange(joined.length, 0)];
        }
    }
}

- (void)setStatus:(NSString*)s {
    dispatch_async(dispatch_get_main_queue(), ^{
        self->_statusLabel.textColor = [NSColor controlTextColor];
        self->_statusLabel.stringValue = s;
    });
}

#pragma mark - Actions

- (void)onShowControls:(id)sender {
    NSAlert* alert = [[NSAlert alloc] init];
    alert.messageText = @"Keyboard & Mouse Controls Guide";
    alert.informativeText =
        @"DualShock 4 Control           Keyboard & Mouse Mapping\n"
        @"─────────────────────────────────────────────────────────────\n"
        @"Left Analog Stick             W / A / S / D\n"
        @"Right Analog Stick            Mouse Movement (or I / J / K / L)\n"
        @"D-Pad (Up/Down/Left/Right)    Arrow Keys (↑ ↓ ← →)\n"
        @"Cross (✕)                     Spacebar / Enter / Left Click\n"
        @"Circle (○)                    C / Backspace / Right Click\n"
        @"Square (□)                    F / X\n"
        @"Triangle (△)                  E / R\n"
        @"L1 / R1 Bumpers               Q / Left Shift & Right Shift\n"
        @"L2 / R2 Triggers              Z / Left Ctrl & V\n"
        @"Touchpad Motion               Mouse Cursor inside Game Window\n"
        @"Touchpad Click                Tab Key / Mouse Click\n"
        @"Options / Start               Escape Key / P\n"
        @"Share                         Backquote (`) / F1\n"
        @"Quit / Close Window           Cmd + W or Cmd + Q\n";
    [alert addButtonWithTitle:@"Done"];
    [alert beginSheetModalForWindow:self.window completionHandler:nil];
}

- (void)onPickFolder:(id)sender {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseDirectories = YES;
    panel.canChooseFiles = NO;
    panel.allowsMultipleSelection = NO;
    panel.prompt = @"Import";
    panel.message = @"Choose a folder containing PS4 game directories.";

    [panel beginSheetModalForWindow:self.window
                  completionHandler:^(NSModalResponse resp) {
        if (resp != NSModalResponseOK) return;
        NSString* path = panel.URLs.firstObject.path;
        self->_folderLabel.stringValue = path;
        [self scanFolder:path];
    }];
}

- (void)updateDetailPaneForSelectedRow {
    NSInteger row = _table.selectedRow;
    if (row >= 0 && (size_t)row < _found.size()) {
        const GameEntry& g = _found[(size_t)row];
        if (!g.icon_path.empty()) {
            _previewIconView.image = [[NSImage alloc] initWithContentsOfFile:@(g.icon_path.c_str())];
        } else {
            if (@available(macOS 11.0, *)) {
                _previewIconView.image = [NSImage imageWithSystemSymbolName:@"gamecontroller" accessibilityDescription:@"Game"];
            } else {
                _previewIconView.image = nil;
            }
        }
        _previewTitleLabel.stringValue = @(g.name.c_str());
        _previewMetaLabel.stringValue = [NSString stringWithFormat:
            @"Title ID:  %s\n"
            @"Version:   %s\n"
            @"Size:      %@ (%u files)\n"
            @"Status:    %s\n"
            @"Directory: %s",
            g.title_id.c_str(),
            g.app_ver.c_str(),
            human_size(g.total_bytes),
            g.file_count,
            g.status.c_str(),
            g.id.c_str()];
        _bigLaunchButton.enabled = YES;
    } else {
        _previewIconView.image = nil;
        _previewTitleLabel.stringValue = @"No game selected";
        _previewMetaLabel.stringValue = @"Select a title from the list to view details and launch.";
        _bigLaunchButton.enabled = NO;
    }
}

- (void)scanFolder:(NSString*)path {
    [_spinner startAnimation:nil];
    [self setStatus:@"Scanning…"];

    const std::string native(path.UTF8String);
    dispatch_async(_ioQueue, ^{
        std::vector<GameEntry> games = scan_folder(native);
        dispatch_async(dispatch_get_main_queue(), ^{
            [self->_spinner stopAnimation:nil];
            self->_found = std::move(games);
            [self->_table reloadData];
            [self setStatus:[NSString stringWithFormat:@"%zu title(s) found",
                                                      self->_found.size()]];
            if (!self->_found.empty()) {
                [self->_table selectRowIndexes:[NSIndexSet indexSetWithIndex:0] byExtendingSelection:NO];
            }
            [self updateDetailPaneForSelectedRow];
        });
    });
}

- (void)onRun:(id)sender {
    NSInteger row = _table.selectedRow;
    if (row < 0 || (size_t)row >= _found.size()) {
        [self setStatus:@"Select a game first"];
        return;
    }
    if (_task.isRunning) return;

    const GameEntry& g = _found[(size_t)row];

    NSTask* t = [NSTask new];
    t.launchPath = @(runner_path().c_str());
    t.currentDirectoryURL = [NSURL fileURLWithPath:@(g.root.c_str())];

    NSMutableArray* args = [NSMutableArray array];
    if (_jitCheckbox.state != NSControlStateValueOn) {
        [args addObject:@"--no-jit"];
    }
    if (_windowCheckbox.state == NSControlStateValueOn) {
        [args addObject:@"--window"];
    }
    [args addObject:@"--budget"];
    switch (_budgetPopup.indexOfSelectedItem) {
        case 1:  [args addObject:@"100000000"];   break;
        case 2:  [args addObject:@"1000000000"];  break;
        case 3:  [args addObject:@"300000000000"]; break;
        default: [args addObject:@"0"];           break; // unlimited
    }
    [args addObject:@(g.eboot.c_str())];
    t.arguments = args;

    NSPipe* out = [NSPipe pipe];
    t.standardOutput = out;
    t.standardError = out;

    // Tee the run to <repo root>/Logs/run_<timestamp>.log so a session's
    // output survives the GUI closing.
    std::string root;
    if (!repo_root(root)) {
        // Single destination, no fallback: say so rather than write elsewhere.
        [self setStatus:@"Cannot locate repo root; logs disabled"];
        [self appendLog:@"[gui] could not locate the repo root (no CMakeLists.txt "
                        "above the executable); run logs are disabled"];
    } else {
        NSString* logDir =
            [[NSString stringWithUTF8String:root.c_str()]
                stringByAppendingPathComponent:@"Logs"];
        NSError* dirErr = nil;
        if (![[NSFileManager defaultManager] createDirectoryAtPath:logDir
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:&dirErr]) {
            [self setStatus:[NSString stringWithFormat:@"Log dir failed: %@",
                                                      dirErr.localizedDescription]];
        } else {
            NSDateFormatter* fmt = [NSDateFormatter new];
            fmt.dateFormat = @"yyyyMMdd-HHmmss";
            NSString* fname = [NSString stringWithFormat:@"run_%@.log",
                                                         [fmt stringFromDate:[NSDate date]]];
            NSString* path = [logDir stringByAppendingPathComponent:fname];
            // +fileHandleForWritingAtPath: only OPENS an existing file and returns
            // nil otherwise, so create it first -- otherwise no log is ever written
            // and nothing complains.
            NSFileManager* fm = [NSFileManager defaultManager];
            if (![fm fileExistsAtPath:path]) {
                [fm createFileAtPath:path contents:[NSData data] attributes:nil];
            }
            self.logFile = [NSFileHandle fileHandleForWritingAtPath:path];
            self.logPath = path;
            if (!self.logFile) {
                [self setStatus:[NSString stringWithFormat:@"Cannot open log %@", path]];
            } else {
                [self appendLog:[NSString stringWithFormat:@"=== logging to %@ ===", path]];
            }
        }
    }

    __weak GameLibraryController* weakSelf = self;
    out.fileHandleForReading.readabilityHandler = ^(NSFileHandle* fh) {
        @autoreleasepool {
            NSData* data = fh.availableData;
            if (data.length == 0) return;
            // Write the raw chunk first so the file keeps stdout/stderr interleaving
            // exactly as the runner produced it, rather than the line-split view.
            @try { [weakSelf.logFile writeData:data]; } @catch (NSException*) {}
            NSString* s = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (!s) return;
            NSArray* lines = [s componentsSeparatedByString:@"\n"];
            [weakSelf appendLogLines:lines];
        }
    };

    t.terminationHandler = ^(NSTask* finished) {
        GameLibraryController* strongSelf = weakSelf;
        if (!strongSelf) return;
        [strongSelf flushPendingLogs];
        dispatch_async(dispatch_get_main_queue(), ^{
            strongSelf.runButton.enabled = YES;
            strongSelf.bigLaunchButton.enabled = YES;
            strongSelf.stopButton.enabled = NO;
            int code = (int)finished.terminationStatus;
            NSString* lastComp = strongSelf.logPath.lastPathComponent;
            NSString* logName = lastComp ? lastComp : @"";
            if (code == 0) {
                strongSelf.statusLabel.textColor = [NSColor systemGreenColor];
                strongSelf.statusLabel.stringValue = [NSString stringWithFormat:@"✓ Finished cleanly (0)%@",
                    logName.length ? [NSString stringWithFormat:@" - %@", logName] : @""];
            } else {
                strongSelf.statusLabel.textColor = [NSColor systemRedColor];
                strongSelf.statusLabel.stringValue = [NSString stringWithFormat:@"✗ Exited with error (%d)%@",
                    code,
                    logName.length ? [NSString stringWithFormat:@" - %@", logName] : @""];
            }
            [strongSelf appendLog:@"\n=== runner exited ==="];
            [strongSelf flushPendingLogs];
            [strongSelf.logFile closeFile];
            strongSelf.logFile = nil;
        });
    };

    NSError* err = nil;
    if (![t launchAndReturnError:&err]) {
        [self setStatus:[NSString stringWithFormat:@"Failed to launch: %@",
                                                  err.localizedDescription]];
        return;
    }
    _task = t;
    _runButton.enabled = NO;
    _bigLaunchButton.enabled = NO;
    _stopButton.enabled = YES;
    _statusLabel.textColor = [NSColor controlTextColor];
    [self setStatus:[NSString stringWithFormat:@"Running %s…",
                                              g.name.empty() ? g.id.c_str()
                                                             : g.name.c_str()]];
}

- (void)onStop:(id)sender {
    [_task terminate];
    _task = nil;
    [self flushPendingLogs];
    [_logFile closeFile];
    _logFile = nil;
    _runButton.enabled = YES;
    _bigLaunchButton.enabled = YES;
    _stopButton.enabled = NO;
    _statusLabel.textColor = [NSColor controlTextColor];
    [self setStatus:_logPath.length ? [NSString stringWithFormat:@"Stopped - log: %@", _logPath]
                                    : @"Stopped"];
}

- (void)windowWillClose:(NSNotification*)n {
    (void)n;
    [_task terminate];
    [_logFile closeFile];
    _logFile = nil;
}

#pragma mark - Table

- (NSInteger)numberOfRowsInTableView:(NSTableView*)tv {
    (void)tv;
    return (NSInteger)_found.size();
}

- (id)tableView:(NSTableView*)tv
    objectValueForTableColumn:(NSTableColumn*)col
                          row:(NSInteger)row {
    (void)tv;
    if (row < 0 || (size_t)row >= _found.size()) return nil;
    const GameEntry& g = _found[(size_t)row];
    NSString* key = col.identifier;
    if ([key isEqualToString:@"icon"]) {
        if (!g.icon_path.empty()) {
            NSImage* img = [[NSImage alloc] initWithContentsOfFile:@(g.icon_path.c_str())];
            if (img) return img;
        }
        if (@available(macOS 11.0, *)) {
            return [NSImage imageWithSystemSymbolName:@"gamecontroller" accessibilityDescription:@"Game"];
        }
        return nil;
    }
    if ([key isEqualToString:@"title"])    return @(g.name.c_str());
    if ([key isEqualToString:@"title_id"]) return @(g.title_id.c_str());
    if ([key isEqualToString:@"app_ver"])  return @(g.app_ver.c_str());
    if ([key isEqualToString:@"size"])     return human_size(g.total_bytes);
    if ([key isEqualToString:@"status"])   return @(g.status.c_str());
    return @"";
}

- (void)tableViewSelectionDidChange:(NSNotification *)notification {
    (void)notification;
    [self updateDetailPaneForSelectedRow];
}

@end

namespace ps4 {
namespace gui {

int run_gui() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        GameLibraryController* ctl = [[GameLibraryController alloc] init];
        [ctl buildWindow];
        [ctl showWindow:nil];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    }
    return 0;
}

} // namespace gui
} // namespace ps4

// The GUI is its own binary, so it needs a plain main(). Kept here rather than
// in a separate translation unit so the whole AppKit layer stays in one file.
int main() { return ps4::gui::run_gui(); }
