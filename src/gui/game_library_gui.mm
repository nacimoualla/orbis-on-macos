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
@property (nonatomic, strong) NSTextView* logView;
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
    const NSRect frame = NSMakeRect(0, 0, 940, 660);
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

    NSButton* pick = [NSButton buttonWithTitle:@"Import Folder…"
                                        target:self
                                        action:@selector(onPickFolder:)];
    pick.frame = NSMakeRect(16, frame.size.height - 46, 132, 28);
    [root addSubview:pick];

    _folderLabel = [[NSTextField alloc] initWithFrame:
        NSMakeRect(156, frame.size.height - 44, W - 300, 20)];
    _folderLabel.stringValue = @"No folder selected";
    _folderLabel.editable = NO;
    _folderLabel.bordered = NO;
    _folderLabel.drawsBackground = NO;
    _folderLabel.font = [NSFont systemFontOfSize:11];
    _folderLabel.lineBreakMode = NSLineBreakByTruncatingHead;
    [root addSubview:_folderLabel];

    _spinner = [[NSProgressIndicator alloc] initWithFrame:
        NSMakeRect(W - 40, frame.size.height - 42, 16, 16)];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.hidden = YES;
    [root addSubview:_spinner];

    // --- table ---
    _table = [[NSTableView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, 250)];
    NSDictionary* cols = @{ @"id":     @220, @"name": @140,
                            @"eboot":  @200, @"size":  @90,
                            @"files":  @70,  @"status": @130 };
    for (NSString* key in @[ @"id", @"name", @"eboot", @"size", @"files", @"status" ]) {
        NSTableColumn* col = [[NSTableColumn alloc] initWithIdentifier:key];
        col.title = key;
        col.width = [cols[key] doubleValue];
        [_table addTableColumn:col];
    }
    _table.dataSource = self;
    _table.delegate = self;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.allowsMultipleSelection = NO;

    _tableScroll = [[NSScrollView alloc] initWithFrame:
        NSMakeRect(16, 300, W - 32, 250)];
    _tableScroll.documentView = _table;
    _tableScroll.hasVerticalScroller = YES;
    _tableScroll.borderType = NSBezelBorder;
    [root addSubview:_tableScroll];

    // --- run controls ---
    _runButton = [NSButton buttonWithTitle:@"Run" target:self action:@selector(onRun:)];
    _runButton.frame = NSMakeRect(16, 262, 80, 28);
    [root addSubview:_runButton];

    _stopButton = [NSButton buttonWithTitle:@"Stop" target:self action:@selector(onStop:)];
    _stopButton.frame = NSMakeRect(102, 262, 80, 28);
    _stopButton.enabled = NO;
    [root addSubview:_stopButton];

    _budgetPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(196, 262, 175, 26)];
    [_budgetPopup addItemsWithTitles:@[ @"Unlimited (continuous)",
                                        @"Budget 100M (quick)",
                                        @"Budget 1G (long)",
                                        @"Budget 300G (soak)" ]];
    [_budgetPopup selectItemAtIndex:0];
    [root addSubview:_budgetPopup];

    _jitCheckbox = [NSButton checkboxWithTitle:@"JIT" target:nil action:nil];
    _jitCheckbox.frame = NSMakeRect(381, 266, 50, 18);
    _jitCheckbox.state = NSControlStateValueOn;
    [root addSubview:_jitCheckbox];

    _windowCheckbox = [NSButton checkboxWithTitle:@"Window" target:nil action:nil];
    _windowCheckbox.frame = NSMakeRect(438, 266, 75, 18);
    _windowCheckbox.state = NSControlStateValueOn;
    [root addSubview:_windowCheckbox];

    _statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(525, 266, W - 540, 18)];
    _statusLabel.stringValue = @"Idle";
    _statusLabel.editable = NO;
    _statusLabel.bordered = NO;
    _statusLabel.drawsBackground = NO;
    _statusLabel.font = [NSFont systemFontOfSize:11];
    [root addSubview:_statusLabel];

    // --- log ---
    _logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 36, 214)];
    _logView.editable = NO;
    _logView.allowsUndo = NO;
    _logView.font = [NSFont monospacedDigitSystemFontOfSize:10
                                                     weight:NSFontWeightRegular];

    NSScrollView* logScroll = [[NSScrollView alloc] initWithFrame:
        NSMakeRect(16, 16, W - 32, 230)];
    logScroll.documentView = _logView;
    logScroll.hasVerticalScroller = YES;
    logScroll.borderType = NSBezelBorder;
    [root addSubview:logScroll];
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
        if (_logLines.count > 5000) {
            [_logLines removeObjectsInRange:NSMakeRange(0, _logLines.count - 3000)];
        }
        NSString* joined = [_logLines componentsJoinedByString:@"\n"];
        _logView.string = joined;
        [_logView scrollRangeToVisible:NSMakeRange(joined.length, 0)];
    }
}

- (void)setStatus:(NSString*)s {
    dispatch_async(dispatch_get_main_queue(), ^{ self->_statusLabel.stringValue = s; });
}

#pragma mark - Actions

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
        strongSelf.runButton.enabled = YES;
        strongSelf.stopButton.enabled = NO;
        [strongSelf setStatus:[NSString stringWithFormat:@"Exited (%d)%@",
            (int)finished.terminationStatus,
            strongSelf.logPath.length ? [NSString stringWithFormat:@" - log: %@",
                                                                strongSelf.logPath] : @""]];
        [strongSelf appendLog:@"\n=== runner exited ==="];
        [strongSelf flushPendingLogs];
        [strongSelf.logFile closeFile];
        strongSelf.logFile = nil;
    };

    NSError* err = nil;
    if (![t launchAndReturnError:&err]) {
        [self setStatus:[NSString stringWithFormat:@"Failed to launch: %@",
                                                  err.localizedDescription]];
        return;
    }
    _task = t;
    _runButton.enabled = NO;
    _stopButton.enabled = YES;
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
    _stopButton.enabled = NO;
    [self setStatus:_logPath.length ? [NSString stringWithFormat:@"Stopped - log: %@", _logPath]
                                    : @"Stopped"];
}

- (void)windowWillClose:(NSNotification*)n {
    [_task terminate];
    [_logFile closeFile];
    _logFile = nil;
}

#pragma mark - Table

- (NSInteger)numberOfRowsInTableView:(NSTableView*)tv { return (NSInteger)_found.size(); }

- (id)tableView:(NSTableView*)tv
    objectValueForTableColumn:(NSTableColumn*)col
                          row:(NSInteger)row {
    if (row < 0 || (size_t)row >= _found.size()) return @"";
    const GameEntry& g = _found[(size_t)row];
    NSString* key = col.identifier;
    if ([key isEqualToString:@"id"])     return @(g.id.c_str());
    if ([key isEqualToString:@"name"])   return @(g.name.c_str());
    if ([key isEqualToString:@"eboot"])  return @(g.eboot.c_str());
    if ([key isEqualToString:@"size"])   return human_size(g.total_bytes);
    if ([key isEqualToString:@"files"])  return @(g.file_count);
    if ([key isEqualToString:@"status"]) return @(g.status.c_str());
    return @"";
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
