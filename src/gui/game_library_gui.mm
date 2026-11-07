// Native library shell. Python owns import receipts and persisted library data;
// AppKit owns presentation and the running child process.
#import <AppKit/AppKit.h>
#import <mach-o/dyld.h>
#include <string>
#include <signal.h>
#include <unistd.h>

static NSString* executableDirectory() {
    char path[4096]; uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) return nil;
    return [@(path) stringByDeletingLastPathComponent];
}
static NSString* repositoryRoot() {
    NSString* path = executableDirectory();
    for (int i = 0; i < 10 && path.length > 1; ++i) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:[path stringByAppendingPathComponent:@"scripts/run_workflow.py"]]) return path;
        path = path.stringByDeletingLastPathComponent;
    }
    return nil;
}
static NSMutableDictionary* toolEnvironment() {
    NSMutableDictionary* environment = [NSProcessInfo.processInfo.environment mutableCopy];
    // Finder-launched apps do not inherit the terminal's Homebrew PATH.
    NSString* path = environment[@"PATH"] ?: @"";
    environment[@"PATH"] = [path stringByAppendingString:@":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"];
    environment[@"PYTHONUNBUFFERED"] = @"1";
    return environment;
}
static NSTextField* label(NSString* text, CGFloat size, BOOL bold) {
    NSTextField* view = [NSTextField wrappingLabelWithString:text];
    view.font = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    return view;
}
static NSStackView* stack(NSArray<NSView*>* views, NSUserInterfaceLayoutOrientation orientation, CGFloat spacing) {
    NSStackView* view = [NSStackView stackViewWithViews:views];
    view.orientation = orientation; view.spacing = spacing;
    view.alignment = orientation == NSUserInterfaceLayoutOrientationVertical ? NSLayoutAttributeLeading : NSLayoutAttributeCenterY;
    return view;
}
static void pin(NSView* child, NSView* parent, CGFloat inset) {
    child.translatesAutoresizingMaskIntoConstraints = NO;
    [parent addSubview:child];
    [NSLayoutConstraint activateConstraints:@[
        [child.leadingAnchor constraintEqualToAnchor:parent.leadingAnchor constant:inset],
        [child.trailingAnchor constraintEqualToAnchor:parent.trailingAnchor constant:-inset],
        [child.topAnchor constraintEqualToAnchor:parent.topAnchor constant:inset],
        [child.bottomAnchor constraintEqualToAnchor:parent.bottomAnchor constant:-inset]]];
}
static NSImage* gameImage(NSDictionary* game) {
    NSImage* image = [[NSImage alloc] initWithContentsOfFile:game[@"icon_path"] ?: @""];
    return image ?: [NSImage imageWithSystemSymbolName:@"gamecontroller.fill" accessibilityDescription:@"Game"];
}

@interface GameCard : NSCollectionViewItem
@property (nonatomic, strong) NSTextField* titleLabel;
@property (nonatomic, strong) NSImageView* art;
@end
@implementation GameCard
- (void)loadView {
    self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 156, 190)];
    self.view.wantsLayer = YES; self.view.layer.cornerRadius = 12;
    _art = [NSImageView new]; _art.imageScaling = NSImageScaleProportionallyUpOrDown;
    _titleLabel = label(@"", 12, YES); _titleLabel.maximumNumberOfLines = 2;
    NSStackView* content = stack(@[_art, _titleLabel], NSUserInterfaceLayoutOrientationVertical, 8);
    pin(content, self.view, 10);
    [_art.heightAnchor constraintEqualToConstant:130].active = YES;
    [_art.widthAnchor constraintEqualToAnchor:content.widthAnchor].active = YES;
    [_titleLabel.widthAnchor constraintEqualToAnchor:content.widthAnchor].active = YES;
}
- (void)setSelected:(BOOL)selected {
    [super setSelected:selected];
    self.view.layer.backgroundColor = (selected ? NSColor.selectedContentBackgroundColor : NSColor.controlBackgroundColor).CGColor;
    _titleLabel.textColor = selected ? NSColor.alternateSelectedControlTextColor : NSColor.labelColor;
}
@end

@interface GameLibraryController : NSWindowController <NSTableViewDataSource, NSTableViewDelegate, NSCollectionViewDataSource, NSCollectionViewDelegate, NSSearchFieldDelegate, NSWindowDelegate>
@property (nonatomic, strong) NSString* root;
@property (nonatomic, strong) NSArray<NSDictionary*>* games;
@property (nonatomic, strong) NSArray<NSDictionary*>* visibleGames;
@property (nonatomic, strong) NSArray<NSDictionary*>* jobs;
@property (nonatomic, strong) NSArray<NSDictionary*>* diagnostics;
@property (nonatomic, strong) NSDictionary* selectedGame;
@property (nonatomic, strong) NSMutableDictionary* pendingProfiles;
@property (nonatomic, strong) NSTabView* tabs;
@property (nonatomic, strong) NSSegmentedControl* navigation;
@property (nonatomic, strong) NSSearchField* search;
@property (nonatomic, strong) NSTextField* librarySummary;
@property (nonatomic, strong) NSTableView* table;
@property (nonatomic, strong) NSCollectionView* grid;
@property (nonatomic, strong) NSScrollView* gridScroll;
@property (nonatomic, strong) NSScrollView* tableScroll;
@property (nonatomic, strong) NSSegmentedControl* viewMode;
@property (nonatomic, strong) NSImageView* artwork;
@property (nonatomic, strong) NSTextField* titleLabel;
@property (nonatomic, strong) NSTextField* metaLabel;
@property (nonatomic, strong) NSButton* launchButton;
@property (nonatomic, strong) NSButton* stopButton;
@property (nonatomic, strong) NSButton* jit;
@property (nonatomic, strong) NSButton* windowMode;
@property (nonatomic, strong) NSButton* hleMono;
@property (nonatomic, strong) NSTextField* budget;
@property (nonatomic, strong) NSTextField* profileStatus;
@property (nonatomic, strong) NSTextField* status;
@property (nonatomic, strong) NSProgressIndicator* spinner;
@property (nonatomic, strong) NSTableView* importsTable;
@property (nonatomic, strong) NSTableView* diagnosticsTable;
@property (nonatomic, strong) NSPopUpButton* buildAction;
@property (nonatomic, strong) NSTextField* testFilter;
@property (nonatomic, strong) NSPopUpButton* logFilter;
@property (nonatomic, strong) NSSearchField* logSearch;
@property (nonatomic, strong) NSTextView* logView;
@property (nonatomic, strong) NSView* console;
@property (nonatomic, strong) NSSplitView* mainSplit;
@property (nonatomic, strong) NSMutableString* logBuffer;
@property (nonatomic, strong) NSMutableData* pendingOutput;
@property (nonatomic, strong) NSLock* outputLock;
@property (nonatomic, strong) NSFileHandle* logFile;
@property (nonatomic, strong) NSString* logPath;
@property (nonatomic, strong) NSTask* process;
@property (nonatomic, strong) NSTask* serviceTask;
@property (nonatomic, strong) dispatch_queue_t serviceQueue;
@property (nonatomic, assign) BOOL serviceBusy;
@property (nonatomic, assign) BOOL closing;
@property (nonatomic, assign) BOOL updatingSelection;
@property (nonatomic, strong) NSTimer* scanTimer;
@property (nonatomic, strong) NSTimer* pollTimer;
@property (nonatomic, strong) NSString* workerLogPath;
@property (nonatomic, assign) unsigned long long workerLogOffset;
@property (nonatomic, assign) BOOL workerRunning;
@property (nonatomic, strong) NSTimer* flushTimer;
@end

@implementation GameLibraryController
- (NSButton*)button:(NSString*)title action:(SEL)action {
    return [NSButton buttonWithTitle:title target:self action:action];
}
- (NSTableView*)tableWithColumns:(NSArray<NSArray*>*)columns {
    NSTableView* table = [NSTableView new]; table.delegate = self; table.dataSource = self;
    table.rowHeight = 44; table.usesAlternatingRowBackgroundColors = YES;
    table.columnAutoresizingStyle = NSTableViewLastColumnOnlyAutoresizingStyle;
    for (NSArray* spec in columns) {
        NSTableColumn* column = [[NSTableColumn alloc] initWithIdentifier:spec[0]];
        column.title = spec[1]; column.width = [spec[2] doubleValue]; column.minWidth = 80;
        [table addTableColumn:column];
    }
    return table;
}
- (NSScrollView*)scrollFor:(NSView*)view {
    NSScrollView* scroll = [NSScrollView new]; scroll.documentView = view;
    scroll.hasVerticalScroller = YES; scroll.autohidesScrollers = YES;
    scroll.borderType = NSNoBorder; return scroll;
}
- (void)addPage:(NSString*)name view:(NSView*)view {
    NSView* container = [NSView new]; pin(view, container, 0);
    NSTabViewItem* tab = [[NSTabViewItem alloc] initWithIdentifier:name]; tab.label = name; tab.view = container;
    [_tabs addTabViewItem:tab];
}
- (void)buildWindow {
    _root = repositoryRoot();
    _pendingProfiles = [[[NSUserDefaults standardUserDefaults] dictionaryForKey:@"pendingLaunchProfiles"] mutableCopy] ?: [NSMutableDictionary new];
    _games = @[]; _visibleGames = @[]; _jobs = @[]; _diagnostics = @[];
    _logBuffer = [NSMutableString new]; _pendingOutput = [NSMutableData new]; _outputLock = [NSLock new];
    _serviceQueue = dispatch_queue_create("dev.ps4.library.service", DISPATCH_QUEUE_SERIAL);
    NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1180, 820)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"PS4 to macOS"; win.subtitle = @"Game Library"; win.minSize = NSMakeSize(940, 660);
    win.delegate = self; self.window = win; [win center];
    NSView* rootView = win.contentView;
    _navigation = [NSSegmentedControl segmentedControlWithLabels:@[@"Library", @"Imports", @"Developer", @"Diagnostics"]
        trackingMode:NSSegmentSwitchTrackingSelectOne target:self action:@selector(onNavigate:)];
    _navigation.selectedSegment = 0;
    NSStackView* header = stack(@[label(@"PS4 to macOS", 20, YES), _navigation,
        [self button:@"Import…" action:@selector(onImport:)], [self button:@"Refresh" action:@selector(onRefresh:)],
        [self button:@"Controls" action:@selector(onShowControls:)], [self button:@"Console" action:@selector(onToggleConsole:)]],
        NSUserInterfaceLayoutOrientationHorizontal, 14);
    _tabs = [NSTabView new]; _tabs.tabViewType = NSNoTabsNoBorder;
    [self buildLibraryPage]; [self buildImportsPage]; [self buildDeveloperPage]; [self buildDiagnosticsPage];
    [self buildConsole];
    _mainSplit = [NSSplitView new]; _mainSplit.vertical = NO; _mainSplit.dividerStyle = NSSplitViewDividerStyleThin;
    [_mainSplit addArrangedSubview:_tabs]; [_mainSplit addArrangedSubview:_console];
    _tabs.translatesAutoresizingMaskIntoConstraints = NO; _console.translatesAutoresizingMaskIntoConstraints = NO;
    [_tabs.heightAnchor constraintGreaterThanOrEqualToConstant:380].active = YES;
    [_console.heightAnchor constraintGreaterThanOrEqualToConstant:100].active = YES;
    _status = label(@"Opening library…", 12, NO); _status.lineBreakMode = NSLineBreakByTruncatingMiddle;
    _spinner = [NSProgressIndicator new]; _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.controlSize = NSControlSizeSmall; _spinner.displayedWhenStopped = NO;
    NSStackView* footer = stack(@[_spinner, _status], NSUserInterfaceLayoutOrientationHorizontal, 8);
    NSStackView* outer = stack(@[header, _mainSplit, footer], NSUserInterfaceLayoutOrientationVertical, 12);
    pin(outer, rootView, 16);
    for (NSView* view in @[header, _mainSplit, footer]) [view.widthAnchor constraintEqualToAnchor:outer.widthAnchor].active = YES;
    [header.heightAnchor constraintEqualToConstant:34].active = YES;
    [footer.heightAnchor constraintEqualToConstant:22].active = YES;
    [_mainSplit setPosition:530 ofDividerAtIndex:0];
    _pollTimer = [NSTimer scheduledTimerWithTimeInterval:2 target:self selector:@selector(pollLibrary:) userInfo:nil repeats:YES];
    _flushTimer = [NSTimer scheduledTimerWithTimeInterval:0.15 target:self selector:@selector(flushOutput:) userInfo:nil repeats:YES];
    [self updateDetails];
    [self refreshLibrary:NO];
    if (![[NSProcessInfo processInfo].environment[@"PS4_LIBRARY_AUTOSCAN"] isEqualToString:@"0"]) {
        _scanTimer = [NSTimer scheduledTimerWithTimeInterval:15 target:self selector:@selector(autoScan:) userInfo:nil repeats:YES];
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2*NSEC_PER_SEC), dispatch_get_main_queue(), ^{ [self autoScan:nil]; });
    }
}
- (void)buildLibraryPage {
    _search = [NSSearchField new]; _search.placeholderString = @"Search title or title ID"; _search.delegate = self;
    [_search.widthAnchor constraintGreaterThanOrEqualToConstant:240].active = YES;
    _viewMode = [NSSegmentedControl segmentedControlWithLabels:@[@"Grid", @"List"] trackingMode:NSSegmentSwitchTrackingSelectOne target:self action:@selector(onViewMode:)];
    _viewMode.selectedSegment = [[NSUserDefaults standardUserDefaults] integerForKey:@"libraryViewMode"];
    _librarySummary = label(@"No titles yet — import a PKG or game folder", 12, NO);
    _librarySummary.textColor = NSColor.secondaryLabelColor;
    NSStackView* toolbar = stack(@[_search, _viewMode], NSUserInterfaceLayoutOrientationHorizontal, 10);
    _table = [self tableWithColumns:@[@[@"title", @"Title", @240], @[@"title_id", @"Title ID", @110], @[@"status", @"Status", @120]]];
    _table.target = self; _table.doubleAction = @selector(onLaunch:); _tableScroll = [self scrollFor:_table];
    NSCollectionViewFlowLayout* layout = [NSCollectionViewFlowLayout new]; layout.itemSize = NSMakeSize(156, 194);
    layout.minimumInteritemSpacing = 12; layout.minimumLineSpacing = 12; layout.sectionInset = NSEdgeInsetsMake(8, 8, 8, 8);
    _grid = [NSCollectionView new]; _grid.collectionViewLayout = layout; _grid.selectable = YES;
    _grid.allowsMultipleSelection = NO; _grid.dataSource = self; _grid.delegate = self;
    [_grid registerClass:GameCard.class forItemWithIdentifier:@"game"];
    _grid.backgroundColors = @[NSColor.windowBackgroundColor]; _gridScroll = [self scrollFor:_grid];
    NSView* browser = [NSView new]; pin(_gridScroll, browser, 0); pin(_tableScroll, browser, 0);
    NSStackView* library = stack(@[toolbar, _librarySummary, browser], NSUserInterfaceLayoutOrientationVertical, 8);
    for (NSView* view in @[toolbar, browser]) [view.widthAnchor constraintEqualToAnchor:library.widthAnchor].active = YES;
    [library.widthAnchor constraintGreaterThanOrEqualToConstant:400].active = YES;
    _artwork = [NSImageView new]; _artwork.imageScaling = NSImageScaleProportionallyUpOrDown;
    [_artwork.widthAnchor constraintEqualToConstant:140].active = YES; [_artwork.heightAnchor constraintEqualToConstant:140].active = YES;
    _titleLabel = label(@"Your game library", 22, YES); _titleLabel.maximumNumberOfLines = 2;
    _metaLabel = label(@"Import a PKG, archive, or extracted game folder. New PKGs in games/ are picked up automatically.", 12, NO);
    _metaLabel.textColor = NSColor.secondaryLabelColor; _metaLabel.maximumNumberOfLines = 5;
    _jit = [NSButton checkboxWithTitle:@"JIT translation" target:self action:@selector(onSaveProfile:)];
    _windowMode = [NSButton checkboxWithTitle:@"Game window" target:self action:@selector(onSaveProfile:)];
    _hleMono = [NSButton checkboxWithTitle:@"Use HLE Mono shim" target:self action:@selector(onSaveProfile:)];
    _budget = [NSTextField new]; _budget.stringValue = @"0"; _budget.placeholderString = @"0 = unlimited";
    _budget.target = self; _budget.action = @selector(onSaveProfile:);
    NSStackView* budgetRow = stack(@[label(@"Instruction budget", 12, NO), _budget], NSUserInterfaceLayoutOrientationHorizontal, 8);
    [_budget.widthAnchor constraintGreaterThanOrEqualToConstant:100].active = YES;
    _launchButton = [self button:@"Launch Game" action:@selector(onLaunch:)]; _launchButton.bezelStyle = NSBezelStyleRounded;
    _launchButton.keyEquivalent = @"\r"; _launchButton.enabled = NO;
    _stopButton = [self button:@"Stop" action:@selector(onStop:)]; _stopButton.enabled = NO;
    _profileStatus = label(@"Settings are saved per game", 11, NO); _profileStatus.textColor = NSColor.secondaryLabelColor;
    NSStackView* detail = stack(@[_artwork, _titleLabel, _metaLabel, label(@"Launch settings", 13, YES), _jit, _windowMode,
        _hleMono, budgetRow, _profileStatus, stack(@[_launchButton, _stopButton], NSUserInterfaceLayoutOrientationHorizontal, 8),
        [self button:@"Reveal Game in Finder" action:@selector(onRevealGame:)]], NSUserInterfaceLayoutOrientationVertical, 10);
    NSView* detailPanel = [NSView new]; pin(detail, detailPanel, 16);
    [detailPanel.widthAnchor constraintGreaterThanOrEqualToConstant:300].active = YES;
    [detailPanel.widthAnchor constraintLessThanOrEqualToConstant:390].active = YES;
    // A flexible bottom spacer keeps launch settings grouped at the top.
    NSView* spacer = [NSView new]; [detail addArrangedSubview:spacer];
    for (NSView* view in @[_titleLabel, _metaLabel, budgetRow, _profileStatus]) [view.widthAnchor constraintEqualToAnchor:detail.widthAnchor].active = YES;
    NSSplitView* split = [NSSplitView new]; split.vertical = YES; split.dividerStyle = NSSplitViewDividerStyleThin;
    [split addArrangedSubview:library]; [split addArrangedSubview:detailPanel];
    [self addPage:@"Library" view:split]; [self onViewMode:nil];
}
- (void)buildImportsPage {
    _importsTable = [self tableWithColumns:@[@[@"source", @"Package / archive", @340], @[@"status", @"State", @130], @[@"message", @"Details", @480]]];
    NSStackView* page = stack(@[label(@"Imports", 24, YES),
        label(@"PKGs in games/ are extracted automatically. Validated imports are added to the library, then their source PKG is deleted. Failed imports keep the source. Original archives and external sources are retained.", 13, NO),
        stack(@[[self button:@"Import File or Folder…" action:@selector(onImport:)], [self button:@"Scan games/" action:@selector(onRefresh:)],
            [self button:@"Retry Failed Imports" action:@selector(onRetryImports:)]], NSUserInterfaceLayoutOrientationHorizontal, 10),
        [self scrollFor:_importsTable]], NSUserInterfaceLayoutOrientationVertical, 14);
    for (NSView* view in page.arrangedSubviews) [view.widthAnchor constraintEqualToAnchor:page.widthAnchor].active = YES;
    [self addPage:@"Imports" view:page];
}
- (void)buildDeveloperPage {
    _buildAction = [NSPopUpButton new]; [_buildAction addItemsWithTitles:@[@"App + runner", @"Runner", @"All targets"]];
    _testFilter = [NSTextField new]; _testFilter.placeholderString = @"Optional phase or regex (flappy, x87, 5…)";
    [_testFilter.widthAnchor constraintGreaterThanOrEqualToConstant:330].active = YES;
    NSView* spacer = [NSView new];
    NSStackView* page = stack(@[label(@"Developer", 24, YES), label(@"Build and test the compatibility layer. Output streams to the console; complete logs are saved in Logs/.", 13, NO),
        label(@"Build", 16, YES), stack(@[_buildAction, [self button:@"Build" action:@selector(onBuild:)]], NSUserInterfaceLayoutOrientationHorizontal, 12),
        label(@"Tests", 16, YES), stack(@[_testFilter, [self button:@"Run Tests" action:@selector(onTest:)], [self button:@"Rerun Failures" action:@selector(onRerunTests:)]], NSUserInterfaceLayoutOrientationHorizontal, 12),
        [self button:@"Open Repository in Finder" action:@selector(onRevealRepo:)], spacer], NSUserInterfaceLayoutOrientationVertical, 20);
    [self addPage:@"Developer" view:page];
}
- (void)buildDiagnosticsPage {
    _diagnosticsTable = [self tableWithColumns:@[@[@"name", @"Log", @300], @[@"modified", @"Updated", @180], @[@"size", @"Size", @100], @[@"summary", @"Result / crash signature", @380]]];
    _diagnosticsTable.target = self; _diagnosticsTable.doubleAction = @selector(onInspectLog:);
    NSStackView* page = stack(@[label(@"Diagnostics", 24, YES), label(@"Recent run, build, test, and import logs. Double-click a log to inspect its tail. Signatures are hints, not a root-cause diagnosis.", 13, NO),
        stack(@[[self button:@"Refresh Logs" action:@selector(onRefreshLogs:)], [self button:@"Inspect Selected" action:@selector(onInspectLog:)],
            [self button:@"Collect Diagnostics" action:@selector(onCollect:)], [self button:@"Reveal Logs" action:@selector(onRevealLogs:)]], NSUserInterfaceLayoutOrientationHorizontal, 10),
        [self scrollFor:_diagnosticsTable]], NSUserInterfaceLayoutOrientationVertical, 14);
    for (NSView* view in page.arrangedSubviews) [view.widthAnchor constraintEqualToAnchor:page.widthAnchor].active = YES;
    [self addPage:@"Diagnostics" view:page];
}
- (void)buildConsole {
    _logSearch = [NSSearchField new]; _logSearch.placeholderString = @"Search console"; _logSearch.target = self; _logSearch.action = @selector(onFilterLog:);
    [_logSearch.widthAnchor constraintGreaterThanOrEqualToConstant:200].active = YES;
    _logFilter = [NSPopUpButton new]; [_logFilter addItemsWithTitles:@[@"All output", @"GPU / Metal", @"JIT / CPU", @"HLE", @"Errors"]];
    _logFilter.target = self; _logFilter.action = @selector(onFilterLog:);
    NSStackView* toolbar = stack(@[label(@"Console", 13, YES), _logSearch, _logFilter,
        [self button:@"Copy" action:@selector(onCopyLog:)], [self button:@"Clear" action:@selector(onClearLog:)],
        [self button:@"Reveal Logs" action:@selector(onRevealLogs:)]], NSUserInterfaceLayoutOrientationHorizontal, 10);
    _logView = [NSTextView new]; _logView.editable = NO; _logView.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    _logView.textContainerInset = NSMakeSize(10, 10); _logView.autoresizingMask = NSViewWidthSizable;
    _logView.textContainer.widthTracksTextView = YES; _logView.horizontallyResizable = NO;
    NSScrollView* scroll = [self scrollFor:_logView];
    _console = stack(@[toolbar, scroll], NSUserInterfaceLayoutOrientationVertical, 8);
    for (NSView* view in @[toolbar, scroll]) [view.widthAnchor constraintEqualToAnchor:_console.widthAnchor].active = YES;
}
#pragma mark - Library service
- (void)showStatus:(NSString*)text { _status.stringValue = text ?: @""; }
- (void)service:(NSArray<NSString*>*)arguments completion:(void (^)(NSDictionary*, NSString*))completion {
    if (!_root) { completion(nil, @"Cannot find the repository. Keep the app in its build directory."); return; }
    if (_serviceBusy) { completion(nil, @"A library operation is already running."); return; }
    _serviceBusy = YES; [_spinner startAnimation:nil];
    NSString* root = _root;
    dispatch_async(_serviceQueue, ^{
        @autoreleasepool {
            NSTask* task = [NSTask new]; task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/env"];
            task.currentDirectoryURL = [NSURL fileURLWithPath:root]; task.environment = toolEnvironment();
            task.arguments = [@[@"python3", [root stringByAppendingPathComponent:@"scripts/run_workflow.py"]] arrayByAddingObjectsFromArray:arguments];
            NSPipe* output = [NSPipe pipe]; NSPipe* errors = [NSPipe pipe]; task.standardOutput = output; task.standardError = errors;
            // Read stderr independently: extraction output must never fill a pipe
            // and block the JSON response. Output chunks are batched for AppKit.
            errors.fileHandleForReading.readabilityHandler = ^(NSFileHandle* handle) {
                NSData* data = handle.availableData;
                if (data.length) { [self.outputLock lock]; [self.pendingOutput appendData:data]; [self.outputLock unlock]; }
            };
            NSError* error = nil;
            [self.outputLock lock]; self.serviceTask = task; BOOL closing = self.closing; [self.outputLock unlock];
            BOOL launched = !closing && [task launchAndReturnError:&error];
            NSData* data = launched ? [output.fileHandleForReading readDataToEndOfFile] : nil;
            if (launched) [task waitUntilExit];
            errors.fileHandleForReading.readabilityHandler = nil;
            // Drain any bytes not delivered to the handler before EOF.
            NSData* remainder = launched ? [errors.fileHandleForReading readDataToEndOfFile] : nil;
            if (remainder.length) { [self.outputLock lock]; [self.pendingOutput appendData:remainder]; [self.outputLock unlock]; }
            NSDictionary* result = data.length ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
            NSString* failure = nil;
            if (!launched) failure = error.localizedDescription ?: @"Library operation cancelled.";
            else if (![result isKindOfClass:NSDictionary.class] || task.terminationStatus != 0)
                failure = error.localizedDescription ?: [NSString stringWithFormat:@"Library operation failed (%d). See console.", task.terminationStatus];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.serviceBusy = NO; self.serviceTask = nil; [self.spinner stopAnimation:nil];
                if (!self.closing) completion(failure ? nil : result, failure);
            });
        }
    });
}
- (void)applyLibrary:(NSDictionary*)library {
    NSDictionary* worker = library[@"worker"];
    _workerRunning = [worker[@"running"] boolValue];
    NSString* workerLog = worker[@"log_path"];
    if (workerLog.length && ![_workerLogPath isEqualToString:workerLog]) {
        _workerLogPath = workerLog;
        NSDictionary* attributes = [NSFileManager.defaultManager attributesOfItemAtPath:workerLog error:nil];
        unsigned long long bytes = [attributes fileSize];
        _workerLogOffset = bytes > 65536 ? bytes - 65536 : 0;
        [self appendText:[NSString stringWithFormat:@"\n=== Import worker log: %@ ===\n", workerLog]];
    }
    [self readWorkerLog];
    NSString* selected = _selectedGame[@"root"];
    NSMutableArray* incoming = [([library[@"games"] isKindOfClass:NSArray.class] ? library[@"games"] : @[]) mutableCopy];
    for (NSUInteger i = 0; i < incoming.count; ++i) {
        NSDictionary* game = incoming[i]; NSDictionary* pending = _pendingProfiles[game[@"root"]];
        if (pending) { NSMutableDictionary* copy = [game mutableCopy]; copy[@"profile"] = pending; incoming[i] = copy; }
    }
    BOOL gamesChanged = ![_games isEqualToArray:incoming];
    _games = incoming;
    _jobs = [library[@"jobs"] isKindOfClass:NSArray.class] ? library[@"jobs"] : @[];
    [_importsTable reloadData];
    // Rebuild grid and profile controls only when library metadata changes;
    // polling must not erase settings the user is currently editing.
    if (gamesChanged) [self filterGamesPreserving:selected];
    NSUInteger failures = 0;
    for (NSDictionary* job in _jobs) if ([job[@"status"] isEqualToString:@"failed"]) ++failures;
    if (!_process.running) [self showStatus:[NSString stringWithFormat:@"%lu titles · %lu import jobs%@%@", (unsigned long)_games.count,
        (unsigned long)_jobs.count, failures ? [NSString stringWithFormat:@" · %lu failed (see Imports)", (unsigned long)failures] : @"",
        _workerRunning ? @" · Extracting in background — safe to close this window" : @""]];
}
- (void)readWorkerLog {
    if (!_workerLogPath.length) return;
    NSFileHandle* handle = [NSFileHandle fileHandleForReadingAtPath:_workerLogPath];
    if (!handle) return;
    @try {
        unsigned long long size = [handle seekToEndOfFile];
        if (size < _workerLogOffset) _workerLogOffset = 0;
        [handle seekToFileOffset:_workerLogOffset];
        NSData* data = [handle readDataOfLength:65536]; _workerLogOffset += data.length;
        [handle closeFile];
        if (data.length) [self appendText:[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]
            ?: [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding]];
    } @catch (NSException* exception) { (void)exception; [handle closeFile]; }
}
- (void)pollLibrary:(id)sender {
    (void)sender;
    if (_serviceBusy || _closing) return;
    if (!_workerRunning && _pendingProfiles.count) {
        NSString* root = _pendingProfiles.allKeys.firstObject;
        [self saveProfile:_pendingProfiles[root] game:@{@"root":root} completion:^(BOOL saved) { (void)saved; }];
        return;
    }
    // The worker publishes atomic snapshots; this request never waits for
    // extraction to finish and reconnects after an application restart.
    [self service:@[@"library", @"--json"] completion:^(NSDictionary* result, NSString* error) {
        if (result) [self applyLibrary:result]; else [self showStatus:error];
    }];
}
- (void)refreshLibrary:(BOOL)import {
    if (_serviceBusy || _closing) return;
    [self showStatus:import ? @"Scanning games/ and importing new packages…" : @"Loading library…"];
    [self service:import ? @[@"import", @"--background", @"--json"] : @[@"library", @"--json"] completion:^(NSDictionary* result, NSString* error) {
        if (result) [self applyLibrary:result]; else [self showStatus:error];
    }];
}
- (void)autoScan:(id)sender {
    (void)sender;
    if (!_process.running && !_serviceBusy) [self refreshLibrary:YES];
}
- (void)onRefresh:(id)sender { (void)sender; [self refreshLibrary:YES]; }
- (void)onRetryImports:(id)sender {
    (void)sender;
    if (_serviceBusy) { [self showStatus:@"Wait for the current import to finish."]; return; }
    [self showStatus:@"Retrying failed imports…"];
    [self service:@[@"import", @"--background", @"--retry", @"--json"] completion:^(NSDictionary* result, NSString* error) {
        if (result) [self applyLibrary:result]; else [self showStatus:error];
    }];
}
- (void)onImport:(id)sender {
    (void)sender;
    if (_serviceBusy) { [self showStatus:@"Wait for the current import to finish."]; return; }
    NSOpenPanel* panel = [NSOpenPanel openPanel]; panel.canChooseDirectories = YES; panel.canChooseFiles = YES;
    panel.allowsMultipleSelection = NO; panel.prompt = @"Import";
    panel.message = @"Choose a PKG, ZIP/7z/RAR archive, or extracted game folder. External source files are retained.";
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse response) {
        if (response != NSModalResponseOK) return;
        self.navigation.selectedSegment = 1; [self onNavigate:nil];
        [self showStatus:@"Importing…"];
        [self service:@[@"import", panel.URL.path, @"--background", @"--json"] completion:^(NSDictionary* result, NSString* error) {
            if (result) [self applyLibrary:result]; else [self showStatus:error];
        }];
    }];
}
- (void)filterGamesPreserving:(NSString*)root {
    NSString* search = _search.stringValue.lowercaseString;
    NSMutableArray* visible = [NSMutableArray new];
    for (NSDictionary* game in _games) {
        NSString* text = [NSString stringWithFormat:@"%@ %@", game[@"title"] ?: @"", game[@"title_id"] ?: @""];
        if (!search.length || [text.lowercaseString containsString:search]) [visible addObject:game];
    }
    _visibleGames = visible;
    _librarySummary.stringValue = visible.count ? [NSString stringWithFormat:@"%lu of %lu titles", (unsigned long)visible.count, (unsigned long)_games.count]
        : (_games.count ? @"No matching titles" : @"No titles yet — import a PKG or game folder");
    _updatingSelection = YES; [_table reloadData]; [_grid reloadData];
    NSUInteger index = [visible indexOfObjectPassingTest:^BOOL(NSDictionary* game, NSUInteger idx, BOOL* stop) {
        (void)idx; (void)stop; return [game[@"root"] isEqualToString:root];
    }];
    if (index == NSNotFound && visible.count) index = 0;
    if (index != NSNotFound) {
        [_table selectRowIndexes:[NSIndexSet indexSetWithIndex:index] byExtendingSelection:NO];
        _grid.selectionIndexPaths = [NSSet setWithObject:[NSIndexPath indexPathForItem:index inSection:0]];
        _selectedGame = visible[index];
    } else { [_table deselectAll:nil]; _grid.selectionIndexPaths = [NSSet set]; _selectedGame = nil; }
    _updatingSelection = NO; [self updateDetails];
}
- (void)controlTextDidChange:(NSNotification*)notification {
    if (notification.object == _search) [self filterGamesPreserving:_selectedGame[@"root"]];
}
- (void)onViewMode:(id)sender {
    (void)sender; BOOL list = _viewMode.selectedSegment == 1;
    _gridScroll.hidden = list; _tableScroll.hidden = !list;
    [[NSUserDefaults standardUserDefaults] setInteger:_viewMode.selectedSegment forKey:@"libraryViewMode"];
}
- (void)onNavigate:(id)sender {
    (void)sender; [_tabs selectTabViewItemAtIndex:_navigation.selectedSegment];
    if (_navigation.selectedSegment == 3) [self onRefreshLogs:nil];
}
- (void)onToggleConsole:(id)sender { (void)sender; _console.hidden = !_console.hidden; [_mainSplit adjustSubviews]; }
- (void)updateDetails {
    NSDictionary* game = _selectedGame; BOOL selected = game != nil;
    _artwork.image = gameImage(game ?: @{});
    _titleLabel.stringValue = game[@"title"] ?: @"Your game library";
    _metaLabel.stringValue = selected ? [NSString stringWithFormat:@"%@ · Version %@\n%@\n%@", game[@"title_id"] ?: @"No title ID",
        game[@"app_ver"] ?: @"—", game[@"status"] ?: @"Unknown", game[@"root"] ?: @""]
        : @"Import a PKG, archive, or extracted game folder to get started.";
    NSDictionary* profile = _pendingProfiles[game[@"root"] ?: @""] ?: game[@"profile"] ?: @{};
    _jit.state = (!profile[@"jit"] || [profile[@"jit"] boolValue]) ? NSControlStateValueOn : NSControlStateValueOff;
    _windowMode.state = (!profile[@"window"] || [profile[@"window"] boolValue]) ? NSControlStateValueOn : NSControlStateValueOff;
    _hleMono.state = [profile[@"hle_mono"] boolValue] ? NSControlStateValueOn : NSControlStateValueOff;
    _budget.stringValue = [profile[@"budget"] description] ?: @"0";
    for (NSControl* control in @[_jit, _windowMode, _hleMono, _budget]) control.enabled = selected;
    _profileStatus.stringValue = selected ? @"Settings are saved per game" : @"Select a game to configure it";
    _launchButton.enabled = selected && !_process.running && ([game[@"status"] isEqualToString:@"ready"] || [game[@"status"] isEqualToString:@"ok"]);
}
- (NSDictionary*)currentProfile {
    NSString* value = [_budget.stringValue stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
    NSCharacterSet* invalid = NSCharacterSet.decimalDigitCharacterSet.invertedSet;
    if (!value.length || [value rangeOfCharacterFromSet:invalid].location != NSNotFound || value.length > 18) {
        _profileStatus.stringValue = @"Budget must be a nonnegative integer (0 = unlimited)."; return nil;
    }
    return @{@"jit": @(_jit.state == NSControlStateValueOn), @"window": @(_windowMode.state == NSControlStateValueOn),
        @"hle_mono": @(_hleMono.state == NSControlStateValueOn), @"budget": @([value longLongValue])};
}
- (void)saveProfile:(NSDictionary*)profile game:(NSDictionary*)game completion:(void (^)(BOOL))completion {
    _pendingProfiles[game[@"root"]] = profile;
    [[NSUserDefaults standardUserDefaults] setObject:_pendingProfiles forKey:@"pendingLaunchProfiles"];
    if (_serviceBusy || _workerRunning) {
        _pendingProfiles[game[@"root"]] = profile;
        [[NSUserDefaults standardUserDefaults] setObject:_pendingProfiles forKey:@"pendingLaunchProfiles"];
        _profileStatus.stringValue = @"Saved locally · library sync pending";
        completion(YES); return;
    }
    NSArray* args = @[@"profile", game[@"root"], @"--json", @"--jit", [profile[@"jit"] boolValue] ? @"on" : @"off",
        @"--window", [profile[@"window"] boolValue] ? @"on" : @"off", @"--budget", [profile[@"budget"] description],
        @"--hle-mono", [profile[@"hle_mono"] boolValue] ? @"on" : @"off"];
    [self service:args completion:^(NSDictionary* result, NSString* error) {
        if (result) {
            if ([self.pendingProfiles[game[@"root"]] isEqual:profile]) [self.pendingProfiles removeObjectForKey:game[@"root"]];
            [[NSUserDefaults standardUserDefaults] setObject:self.pendingProfiles forKey:@"pendingLaunchProfiles"];
            NSMutableArray* games = [self.games mutableCopy];
            for (NSUInteger i = 0; i < games.count; ++i) if ([games[i][@"root"] isEqual:game[@"root"]]) {
                NSMutableDictionary* updated = [games[i] mutableCopy]; updated[@"profile"] = profile; games[i] = updated;
                if ([self.selectedGame[@"root"] isEqual:game[@"root"]]) self.selectedGame = updated;
            }
            self.games = games;
            NSMutableArray* visible = [self.visibleGames mutableCopy];
            for (NSUInteger i = 0; i < visible.count; ++i) if ([visible[i][@"root"] isEqual:game[@"root"]]) {
                NSMutableDictionary* updated = [visible[i] mutableCopy]; updated[@"profile"] = profile; visible[i] = updated;
            }
            self.visibleGames = visible; self.profileStatus.stringValue = @"Launch settings saved";
        } else self.profileStatus.stringValue = @"Saved locally · library sync pending";
        completion(YES);
    }];
}
- (void)onSaveProfile:(id)sender {
    (void)sender; if (!_selectedGame) return; NSDictionary* profile = [self currentProfile];
    if (profile) [self saveProfile:profile game:_selectedGame completion:^(BOOL saved) { (void)saved; }];
}
#pragma mark - Running processes
- (void)onLaunch:(id)sender {
    (void)sender; if (!_selectedGame || _process.running) return;
    NSDictionary* game = _selectedGame; NSDictionary* profile = [self currentProfile]; if (!profile) return;
    if (![game[@"status"] isEqualToString:@"ready"] && ![game[@"status"] isEqualToString:@"ok"]) { [self showStatus:@"This title has no validated boot executable."]; return; }
    [self saveProfile:profile game:game completion:^(BOOL saved) {
        if (!saved || self.process.running) return;
        NSMutableArray* args = [NSMutableArray array];
        if (![profile[@"jit"] boolValue]) [args addObject:@"--no-jit"];
        [args addObject:[profile[@"window"] boolValue] ? @"--window" : @"--headless"];
        [args addObjectsFromArray:@[@"--budget", [profile[@"budget"] description], game[@"eboot"]]];
        NSMutableDictionary* env = toolEnvironment();
        if ([profile[@"hle_mono"] boolValue]) [env removeObjectForKey:@"PS4_LOAD_PRX"]; else env[@"PS4_LOAD_PRX"] = @"1";
        [self startExecutable:[executableDirectory() stringByAppendingPathComponent:@"ps4-runner"] arguments:args
            directory:game[@"root"] environment:env title:[NSString stringWithFormat:@"Run %@", game[@"title"]]];
    }];
}
- (void)startExecutable:(NSString*)executable arguments:(NSArray*)args directory:(NSString*)directory environment:(NSDictionary*)environment title:(NSString*)title {
    if (_process.running) { [self showStatus:@"Wait for the current run, build, or test to finish."]; return; }
    NSFileManager* fm = NSFileManager.defaultManager;
    NSString* logs = [_root stringByAppendingPathComponent:@"Logs"];
    NSError* error = nil;
    if (!logs || ![fm createDirectoryAtPath:logs withIntermediateDirectories:YES attributes:nil error:&error]) {
        [self showStatus:error.localizedDescription ?: @"Cannot create log directory."]; return;
    }
    NSDateFormatter* format = [NSDateFormatter new]; format.dateFormat = @"yyyyMMdd-HHmmss-SSS";
    _logPath = [logs stringByAppendingPathComponent:[NSString stringWithFormat:@"gui_%@.log", [format stringFromDate:NSDate.date]]];
    [fm createFileAtPath:_logPath contents:nil attributes:nil];
    _logFile = [NSFileHandle fileHandleForWritingAtPath:_logPath];
    if (!_logFile) { [self showStatus:@"Cannot create session log."]; return; }
    [self appendText:[NSString stringWithFormat:@"\n=== %@ ===\nLog: %@\n", title, _logPath]];
    [_logFile writeData:[[NSString stringWithFormat:@"=== %@ ===\n%@ %@\n", title, executable, [args componentsJoinedByString:@" "]] dataUsingEncoding:NSUTF8StringEncoding]];
    NSTask* task = [NSTask new]; task.executableURL = [NSURL fileURLWithPath:executable]; task.arguments = args;
    task.currentDirectoryURL = [NSURL fileURLWithPath:directory ?: _root]; task.environment = environment ?: toolEnvironment();
    NSPipe* pipe = [NSPipe pipe]; task.standardOutput = pipe; task.standardError = pipe;
    NSFileHandle* file = _logFile; __weak GameLibraryController* weakSelf = self;
    void (^readOutput)(void) = ^{
        @autoreleasepool {
            for (;;) {
                NSData* data = [pipe.fileHandleForReading readDataOfLength:32768];
                if (!data.length) break;
                GameLibraryController* owner = weakSelf;
                [owner.outputLock lock];
                @try { [file writeData:data]; } @catch (NSException* exception) { (void)exception; }
                [owner.pendingOutput appendData:data]; [owner.outputLock unlock];
            }
            [task waitUntilExit];
            GameLibraryController* owner = weakSelf;
            NSString* result = [NSString stringWithFormat:@"\n=== %@ finished: exit %d%@ ===\n", title, task.terminationStatus,
                task.terminationReason == NSTaskTerminationReasonUncaughtSignal ? @" (signal)" : @""];
            [owner.outputLock lock];
            @try { [file writeData:[result dataUsingEncoding:NSUTF8StringEncoding]]; [file closeFile]; } @catch (NSException* exception) { (void)exception; }
            [owner.pendingOutput appendData:[result dataUsingEncoding:NSUTF8StringEncoding]]; [owner.outputLock unlock];
            dispatch_async(dispatch_get_main_queue(), ^{
                if (owner.process == task) { owner.process = nil; owner.logFile = nil; }
                owner.stopButton.enabled = NO; [owner updateDetails]; [owner flushOutput:nil];
                [owner showStatus:[NSString stringWithFormat:@"%@ ended · exit %d (see log for runtime result) · %@", title, task.terminationStatus, owner.logPath.lastPathComponent]];
                [owner onRefreshLogs:nil];
            });
        }
    };
    if (![task launchAndReturnError:&error]) {
        [_logFile closeFile]; _logFile = nil;
        [self showStatus:error.localizedDescription]; return;
    }
    _process = task; dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), readOutput);
    _launchButton.enabled = NO; _stopButton.enabled = YES;
    [self showStatus:[NSString stringWithFormat:@"%@ · running", title]];
    if (_console.hidden) [self onToggleConsole:nil];
}
- (void)workflow:(NSArray*)arguments title:(NSString*)title {
    if (!_root) { [self showStatus:@"Cannot find repository."]; return; }
    [self startExecutable:@"/usr/bin/env" arguments:[@[@"python3", [_root stringByAppendingPathComponent:@"scripts/run_workflow.py"]] arrayByAddingObjectsFromArray:arguments]
        directory:_root environment:nil title:title];
}
- (void)onStop:(id)sender {
    (void)sender; NSTask* task = _process; if (!task.running) return;
    if ([task.arguments containsObject:@"python3"]) kill(task.processIdentifier, SIGINT); else [task terminate];
    [self showStatus:@"Stopping…"];
    // Keep the task and log alive until EOF. Escalate only for this same child.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 3*NSEC_PER_SEC), dispatch_get_main_queue(), ^{
        if (task.running) kill(task.processIdentifier, SIGKILL);
    });
}
- (void)onBuild:(id)sender {
    (void)sender; NSArray* targets = @[@"ps4-gui-app", @"ps4-runner", @"all"];
    [self workflow:@[@"build", @"--target", targets[_buildAction.indexOfSelectedItem]] title:@"Build"];
}
- (void)onTest:(id)sender {
    (void)sender; NSMutableArray* args = [NSMutableArray arrayWithObject:@"test"];
    if (_testFilter.stringValue.length) [args addObjectsFromArray:@[@"--phase", _testFilter.stringValue]];
    [self workflow:args title:@"Tests"];
}
- (void)onRerunTests:(id)sender { (void)sender; [self workflow:@[@"test", @"--rerun-failed"] title:@"Rerun failed tests"]; }
- (void)onCollect:(id)sender { (void)sender; [self workflow:@[@"logs", @"--collect"] title:@"Collect diagnostics"]; }
#pragma mark - Console and diagnostics
- (void)appendText:(NSString*)text {
    if (!text.length) return;
    [_logBuffer appendString:text];
    if (_logBuffer.length > 1000000) [_logBuffer deleteCharactersInRange:NSMakeRange(0, _logBuffer.length - 700000)];
    [self onFilterLog:nil];
}
- (void)flushOutput:(id)sender {
    (void)sender; [_outputLock lock]; NSData* data = [_pendingOutput copy]; [_pendingOutput setLength:0]; [_outputLock unlock];
    if (!data.length) return;
    // Use lossy conversion when arbitrary guest bytes aren't UTF-8. Disk logs
    // keep exact bytes; the visible console must keep making progress.
    NSString* text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    if (!text) text = [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding];
    [self appendText:text];
}
- (void)onFilterLog:(id)sender {
    (void)sender; NSString* search = _logSearch.stringValue.lowercaseString;
    NSInteger category = _logFilter.indexOfSelectedItem;
    if (!category && !search.length) _logView.string = _logBuffer;
    else {
        NSArray* needles = category == 1 ? @[@"gpu", @"pm4", @"gcn", @"metal", @"shader"] :
            category == 2 ? @[@"jit", @"cpu", @"x86"] : category == 3 ? @[@"hle", @"syscall", @"sce"] : @[@"error", @"fail", @"fatal", @"assert", @"invalid", @"warn", @"signal"];
        NSMutableArray* lines = [NSMutableArray new];
        for (NSString* line in [_logBuffer componentsSeparatedByString:@"\n"]) {
            NSString* lower = line.lowercaseString;
            if (search.length && ![lower containsString:search]) continue;
            BOOL match = category == 0;
            for (NSString* needle in needles) if ([lower containsString:needle]) { match = YES; break; }
            if (match) [lines addObject:line];
        }
        _logView.string = [lines componentsJoinedByString:@"\n"];
    }
    [_logView scrollRangeToVisible:NSMakeRange(_logView.string.length, 0)];
}
- (void)onCopyLog:(id)sender {
    (void)sender; [NSPasteboard.generalPasteboard clearContents];
    [NSPasteboard.generalPasteboard setString:_logView.string forType:NSPasteboardTypeString];
}
- (void)onClearLog:(id)sender { (void)sender; [_logBuffer setString:@""]; [self onFilterLog:nil]; }
- (void)onRevealLogs:(id)sender {
    (void)sender; if (!_root) return; NSString* logs = [_root stringByAppendingPathComponent:@"Logs"];
    [NSFileManager.defaultManager createDirectoryAtPath:logs withIntermediateDirectories:YES attributes:nil error:nil];
    [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:logs]];
}
- (void)onRevealRepo:(id)sender { (void)sender; if (_root) [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:_root]]; }
- (void)onRevealGame:(id)sender {
    (void)sender; if (_selectedGame[@"root"]) [NSWorkspace.sharedWorkspace activateFileViewerSelectingURLs:@[[NSURL fileURLWithPath:_selectedGame[@"root"]]]];
}
- (NSString*)tailOfLog:(NSString*)path {
    NSFileHandle* handle = [NSFileHandle fileHandleForReadingAtPath:path]; if (!handle) return @"Cannot read log.";
    @try {
        unsigned long long size = [handle seekToEndOfFile]; [handle seekToFileOffset:size > 65536 ? size - 65536 : 0];
        NSData* data = [handle readDataToEndOfFile]; [handle closeFile];
        return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding] ?: @"";
    } @catch (NSException* exception) { (void)exception; [handle closeFile]; return @"Cannot read log."; }
}
- (void)onRefreshLogs:(id)sender {
    (void)sender; if (!_root) return;
    NSString* logs = [_root stringByAppendingPathComponent:@"Logs"];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSArray<NSURL*>* urls = [NSFileManager.defaultManager contentsOfDirectoryAtURL:[NSURL fileURLWithPath:logs]
            includingPropertiesForKeys:@[NSURLContentModificationDateKey, NSURLFileSizeKey, NSURLIsRegularFileKey] options:NSDirectoryEnumerationSkipsHiddenFiles error:nil];
        NSMutableArray* entries = [NSMutableArray new];
        for (NSURL* url in urls) {
            NSDictionary* values = [url resourceValuesForKeys:@[NSURLContentModificationDateKey, NSURLFileSizeKey, NSURLIsRegularFileKey] error:nil];
            if (![values[NSURLIsRegularFileKey] boolValue]) continue;
            [entries addObject:@{@"path": url.path, @"date": values[NSURLContentModificationDateKey] ?: NSDate.distantPast,
                @"bytes": values[NSURLFileSizeKey] ?: @0}];
        }
        [entries sortUsingComparator:^NSComparisonResult(NSDictionary* a, NSDictionary* b) { return [b[@"date"] compare:a[@"date"]]; }];
        NSMutableArray* rows = [NSMutableArray new]; NSDateFormatter* format = [NSDateFormatter new]; format.dateFormat = @"MMM d, HH:mm:ss";
        for (NSDictionary* entry in [entries subarrayWithRange:NSMakeRange(0, MIN(40, entries.count))]) {
            NSString* tail = [self tailOfLog:entry[@"path"]]; NSString* summary = @"Log available";
            for (NSString* line in [[tail componentsSeparatedByString:@"\n"] reverseObjectEnumerator]) {
                NSString* lower = line.lowercaseString;
                if ([lower containsString:@"finished:"] || [lower containsString:@"fatal"] || [lower containsString:@"sigsegv"] ||
                    [lower containsString:@"assertion"] || [lower containsString:@"error"] || [lower containsString:@"signal"]) { summary = line; break; }
            }
            [rows addObject:@{@"path":entry[@"path"], @"name":[entry[@"path"] lastPathComponent], @"modified":[format stringFromDate:entry[@"date"]],
                @"size":[NSByteCountFormatter stringFromByteCount:[entry[@"bytes"] longLongValue] countStyle:NSByteCountFormatterCountStyleFile], @"summary":summary}];
        }
        dispatch_async(dispatch_get_main_queue(), ^{ if (!self.closing) { self.diagnostics = rows; [self.diagnosticsTable reloadData]; } });
    });
}
- (void)onInspectLog:(id)sender {
    (void)sender; NSInteger row = _diagnosticsTable.selectedRow;
    if (row < 0 || (NSUInteger)row >= _diagnostics.count) { [self showStatus:@"Select a log first."]; return; }
    NSDictionary* entry = _diagnostics[row];
    NSString* tail = [self tailOfLog:entry[@"path"]];
    if (_console.hidden) [self onToggleConsole:nil];
    [self appendText:[NSString stringWithFormat:@"\n=== %@ (last 64 KB) ===\n%@\n", entry[@"name"], tail]];
}
#pragma mark - Library view delegates
- (NSInteger)numberOfRowsInTableView:(NSTableView*)table {
    return table == _table ? _visibleGames.count : table == _importsTable ? _jobs.count : _diagnostics.count;
}
- (NSView*)tableView:(NSTableView*)table viewForTableColumn:(NSTableColumn*)column row:(NSInteger)row {
    NSArray* source = table == _table ? _visibleGames : table == _importsTable ? _jobs : _diagnostics;
    if (row < 0 || (NSUInteger)row >= source.count) return nil;
    NSDictionary* item = source[row]; NSString* text = [item[column.identifier] description] ?: @"";
    if ([column.identifier isEqualToString:@"source"]) text = text.lastPathComponent;
    NSTextField* view = [table makeViewWithIdentifier:column.identifier owner:self];
    if (!view) { view = [NSTextField labelWithString:@""]; view.identifier = column.identifier; view.lineBreakMode = NSLineBreakByTruncatingMiddle; }
    view.stringValue = text; view.toolTip = [item[column.identifier] description];
    view.font = [NSFont systemFontOfSize:12 weight:[column.identifier isEqualToString:@"title"] ? NSFontWeightMedium : NSFontWeightRegular];
    return view;
}
- (void)tableViewSelectionDidChange:(NSNotification*)notification {
    if (_updatingSelection || notification.object != _table) return;
    NSInteger row = _table.selectedRow;
    _selectedGame = row >= 0 && (NSUInteger)row < _visibleGames.count ? _visibleGames[row] : nil;
    _updatingSelection = YES;
    _grid.selectionIndexPaths = row >= 0 ? [NSSet setWithObject:[NSIndexPath indexPathForItem:row inSection:0]] : [NSSet set];
    _updatingSelection = NO; [self updateDetails];
}
- (NSInteger)collectionView:(NSCollectionView*)view numberOfItemsInSection:(NSInteger)section { (void)view; (void)section; return _visibleGames.count; }
- (NSCollectionViewItem*)collectionView:(NSCollectionView*)view itemForRepresentedObjectAtIndexPath:(NSIndexPath*)path {
    GameCard* card = (GameCard*)[view makeItemWithIdentifier:@"game" forIndexPath:path];
    NSDictionary* game = _visibleGames[path.item]; card.titleLabel.stringValue = game[@"title"] ?: @"Untitled";
    card.art.image = gameImage(game); card.view.toolTip = [NSString stringWithFormat:@"%@ · %@", game[@"title_id"] ?: @"", game[@"status"] ?: @""];
    return card;
}
- (void)collectionView:(NSCollectionView*)view didSelectItemsAtIndexPaths:(NSSet<NSIndexPath*>*)paths {
    (void)view; if (_updatingSelection) return; NSIndexPath* path = paths.anyObject;
    if (!path || (NSUInteger)path.item >= _visibleGames.count) return;
    _selectedGame = _visibleGames[path.item]; _updatingSelection = YES;
    [_table selectRowIndexes:[NSIndexSet indexSetWithIndex:path.item] byExtendingSelection:NO]; _updatingSelection = NO; [self updateDetails];
}
- (void)windowWillClose:(NSNotification*)notification {
    (void)notification; [_scanTimer invalidate]; [_pollTimer invalidate]; [_flushTimer invalidate];
    [_outputLock lock]; _closing = YES; NSTask* service = _serviceTask; [_outputLock unlock];
    // Import extraction belongs to a detached worker, never to this window.
    // A short JSON request may be left to finish rather than interrupt queue writes.
    (void)service;
    if (_process.running) {
        if ([_process.arguments containsObject:@"python3"]) kill(_process.processIdentifier, SIGINT); else [_process terminate];
    }
    [NSApp terminate:nil];
}
- (void)onShowControls:(id)sender {
    NSAlert* alert = [[NSAlert alloc] init];
    alert.messageText = @"Keyboard & Mouse Controls Guide";
    alert.informativeText =
        @"DualShock 4 Control           Keyboard & Mouse Mapping\n"
        @"─────────────────────────────────────────────────────────────\n"
        @"Left Analog Stick             W / A / S / D\n"
        @"Right Analog Stick            Mouse Movement (or I / J / K / L)\n"
        @"D-Pad (Up/Down/Left/Right)    Arrow Keys (↑ ↓ ← →)\n"
        @"Cross (✕)                     Spacebar / Enter / Z / Left Click\n"
        @"Circle (○)                    X / C / Backspace / Right Click\n"
        @"Square (□)                    C / F / X\n"
        @"Triangle (△)                  E / R / C\n"
        @"L1 / R1 Bumpers               Q / Left Shift & Right Shift\n"
        @"L2 / R2 Triggers              Left Ctrl & V\n"
        @"Touchpad Motion               Mouse Cursor inside Game Window\n"
        @"Touchpad Click                Tab Key / Mouse Click\n"
        @"Options / Start               Escape Key / P\n"
        @"Share                         Backquote (`) / F1\n"
        @"Quit / Close Window           Cmd + W or Cmd + Q\n";
    [alert addButtonWithTitle:@"Done"];
    [alert beginSheetModalForWindow:self.window completionHandler:nil];
}

@end

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication]; [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSMenu* menu = [NSMenu new]; NSMenuItem* appItem = [NSMenuItem new]; [menu addItem:appItem];
        NSMenu* appMenu = [NSMenu new];
        [appMenu addItemWithTitle:@"Quit PS4 to macOS" action:@selector(terminate:) keyEquivalent:@"q"]; appItem.submenu = appMenu;
        NSMenuItem* editItem = [NSMenuItem new]; editItem.title = @"Edit"; [menu addItem:editItem];
        NSMenu* editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
        [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
        [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
        [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
        [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"]; editItem.submenu = editMenu; NSApp.mainMenu = menu;
        GameLibraryController* controller = [GameLibraryController new]; [controller buildWindow];
        [controller showWindow:nil]; [NSApp activateIgnoringOtherApps:YES]; [NSApp run];
    }
    return 0;
}
