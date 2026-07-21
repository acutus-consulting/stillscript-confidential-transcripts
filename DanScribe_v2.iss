#define MyAppName "DanScribe AI"
#define MyAppVersion "3.0.1"
#define MyAppPublisher "DanScribe"
#define MyAppExeName "DanScribe_v3.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\DanScribe AI
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=C:\Users\danie\Danscribe\installer_output
OutputBaseFilename=DanScribe_AI_Setup_v3.0.1
SetupIconFile=C:\Users\danie\Danscribe\danscribe.ico
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
Source: "C:\Users\danie\Danscribe\dist\DanScribe_v3.exe";         DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\danie\Danscribe\DanScribe_UserGuide_v2.pdf";    DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\danie\Danscribe\DanScribe_ReleaseNotes_v2.pdf"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\danie\Danscribe\README.md";                      DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";            Filename: "{app}\{#MyAppExeName}"
Name: "{group}\User Guide";              Filename: "{app}\DanScribe_UserGuide_v2.pdf"
Name: "{group}\Release Notes";           Filename: "{app}\DanScribe_ReleaseNotes_v2.pdf"
Name: "{group}\Uninstall {#MyAppName}";  Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";      Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
Filename: "winget"; Parameters: "install --id Gyan.FFmpeg -e --silent"; Flags: runhidden nowait; StatusMsg: "Installing ffmpeg (required for audio processing)..."
