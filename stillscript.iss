; StillScript Confidential Transcripts — Windows installer (Inno Setup)
;
; Masterplan 4.1 renamed this from DanScribe. Two things deliberately did NOT
; change, and must not:
;
;   * AppId. Windows identifies an installed product by AppId, not by name.
;     Changing it would make an existing DanScribe installation invisible to
;     this installer — the old one would stay on disk, un-upgraded and
;     separately listed in Add/Remove Programs, instead of being upgraded in
;     place. Same reasoning as the ~/.danscribe* -> ~/.stillscript* migration
;     in stillscript.py: renaming a key that identifies existing user state
;     orphans that state.
;   * MyAppVersion. This rename is not a release.
;
; BUILD PATHS: all source/output paths below are relative to this script's
; own location ({#SourcePath}, set via SourceDir below), not to any specific
; user account or machine. Check out the repo, run `pyinstaller` (or
; equivalent) so dist\StillScript.exe exists alongside this .iss file, and
; compile — no path edits needed regardless of Windows username or drive.
; The CI workflow (.github/workflows/build-windows-release.yml) builds the
; .exe on a hosted runner and does not use this file, so CI is unaffected
; either way.
;
; The two bundled PDFs still carry their DanScribe-era filenames and content.
; They are stale v2 documents that Golf 5.1/5.2 (README / User Manual) will
; regenerate; renaming the files now would leave a StillScript-named file
; whose every page still says DanScribe, which is worse than an obviously
; old name. Left for those waves, deliberately, not overlooked.

#define MyAppName "StillScript Confidential Transcripts"
#define MyAppShortName "StillScript"
#define MyAppVersion "3.1.0"
#define MyAppPublisher "Acutus Consulting"
#define MyAppExeName "StillScript.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\StillScript
DefaultGroupName={#MyAppShortName}
AllowNoIcons=yes
SourceDir={#SourcePath}
OutputDir={#SourcePath}installer_output
OutputBaseFilename=StillScript_Setup_v3.1.0
SetupIconFile={#SourcePath}stillscript.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\StillScript.exe";           DestDir: "{app}"; Flags: ignoreversion
Source: "DanScribe_UserGuide_v2.pdf";     DestDir: "{app}"; Flags: ignoreversion
Source: "DanScribe_ReleaseNotes_v2.pdf";  DestDir: "{app}"; Flags: ignoreversion
Source: "README.md";                      DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppShortName}";            Filename: "{app}\{#MyAppExeName}"
Name: "{group}\User Guide";                   Filename: "{app}\DanScribe_UserGuide_v2.pdf"
Name: "{group}\Release Notes";                Filename: "{app}\DanScribe_ReleaseNotes_v2.pdf"
Name: "{group}\Uninstall {#MyAppShortName}";  Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppShortName}";      Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppShortName}"; Flags: nowait postinstall skipifsilent
Filename: "winget"; Parameters: "install --id Gyan.FFmpeg -e --silent"; Flags: runhidden nowait; StatusMsg: "Installing ffmpeg (required for audio processing)..."
