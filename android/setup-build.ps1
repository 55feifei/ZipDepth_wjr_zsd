param([switch]$Build, [switch]$SkipGradle)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$toolRoot = Join-Path $repoRoot 'output/tools/android'
$downloads = Join-Path $toolRoot 'downloads'
New-Item -ItemType Directory -Force -Path $downloads | Out-Null
$env:ANDROID_HOME = Join-Path $toolRoot 'sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:ANDROID_USER_HOME = Join-Path $toolRoot 'android-user'
$env:GRADLE_USER_HOME = Join-Path $toolRoot 'gradle-user'
New-Item -ItemType Directory -Force -Path $env:ANDROID_HOME,$env:ANDROID_USER_HOME,$env:GRADLE_USER_HOME | Out-Null

function Get-VerifiedDownload($Url, $Path, $Hash, $Algorithm = 'SHA256') {
    $alreadyVerified = (Test-Path -LiteralPath $Path) -and ((Get-FileHash -LiteralPath $Path -Algorithm $Algorithm).Hash.ToLowerInvariant() -eq $Hash.Trim().ToLowerInvariant())
    if (!$alreadyVerified) {
        Write-Host "Downloading $Url"
        & curl.exe --fail --location --retry 3 --continue-at - --connect-timeout 45 --max-time 3600 --output $Path $Url
        if ($LASTEXITCODE -ne 0) { throw "Download incomplete; run setup again to resume: $Path" }
    }
    if ((Get-FileHash -LiteralPath $Path -Algorithm $Algorithm).Hash.ToLowerInvariant() -ne $Hash.Trim().ToLowerInvariant()) {
        throw "Checksum verification failed: $Path"
    }
}

$jdkFolder = Get-ChildItem -LiteralPath $toolRoot -Directory -Filter 'jdk-17*' -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$jdkFolder) {
    $jdkAssets = Invoke-RestMethod -Uri 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jdk&os=windows'
    $jdkPackage = $jdkAssets[0].binary.package
    $jdkZip = Join-Path $downloads $jdkPackage.name
    Get-VerifiedDownload $jdkPackage.link $jdkZip $jdkPackage.checksum
    Write-Host 'Extracting project-local JDK 17'
    Expand-Archive -LiteralPath $jdkZip -DestinationPath $toolRoot -Force
    $jdkFolder = Get-ChildItem -LiteralPath $toolRoot -Directory -Filter 'jdk-17*' | Select-Object -First 1
}
$env:JAVA_HOME = $jdkFolder.FullName
$env:PATH = "$env:JAVA_HOME/bin;$env:PATH"

$gradleRoot = Join-Path $toolRoot 'gradle-8.9'
if (!$SkipGradle -and !(Test-Path -LiteralPath (Join-Path $gradleRoot 'bin/gradle.bat'))) {
    $gradleZip = Join-Path $downloads 'gradle-8.9-bin.zip'
    $gradleHash = Invoke-RestMethod -Uri 'https://services.gradle.org/distributions/gradle-8.9-bin.zip.sha256'
    if ($gradleHash -is [byte[]]) { $gradleHash = [Text.Encoding]::UTF8.GetString($gradleHash) }
    Get-VerifiedDownload 'https://services.gradle.org/distributions/gradle-8.9-bin.zip' $gradleZip $gradleHash
    Write-Host 'Extracting project-local Gradle 8.9'
    Expand-Archive -LiteralPath $gradleZip -DestinationPath $toolRoot -Force
}

$sdkTaskLock = $null
while ($null -eq $sdkTaskLock) {
    try { $sdkTaskLock = [IO.File]::Open((Join-Path $toolRoot 'sdk-setup.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
    catch [IO.IOException] { Start-Sleep -Seconds 1 }
}
$sdkManager = Join-Path $env:ANDROID_HOME 'cmdline-tools/latest/bin/sdkmanager.bat'
if (!(Test-Path -LiteralPath $sdkManager)) {
    [xml]$repository = Invoke-RestMethod -Uri 'https://dl.google.com/android/repository/repository2-1.xml'
    $toolsPackage = @($repository.SelectNodes('//*[local-name()="remotePackage"]')) | Where-Object { $_.path -eq 'cmdline-tools;latest' } | Select-Object -First 1
    $archive = @($toolsPackage.archives.archive) | Where-Object { $_.'host-os' -eq 'windows' } | Select-Object -First 1
    $toolsName = [string]$archive.complete.url
    $toolsZip = Join-Path $downloads $toolsName
    Get-VerifiedDownload ('https://dl.google.com/android/repository/' + $toolsName) $toolsZip ([string]$archive.complete.checksum) 'SHA1'
    $extractRoot = Join-Path $toolRoot 'cmdline-tools-extract'
    Expand-Archive -LiteralPath $toolsZip -DestinationPath $extractRoot -Force
    New-Item -ItemType Directory -Force -Path (Join-Path $env:ANDROID_HOME 'cmdline-tools') | Out-Null
    $sourceTools = [IO.Path]::GetFullPath((Join-Path $extractRoot 'cmdline-tools'))
    $destinationTools = [IO.Path]::GetFullPath((Join-Path $env:ANDROID_HOME 'cmdline-tools/latest'))
    $verifiedRoot = [IO.Path]::GetFullPath($toolRoot).TrimEnd('\') + '\'
    if (!$sourceTools.StartsWith($verifiedRoot,[StringComparison]::OrdinalIgnoreCase) -or !$destinationTools.StartsWith($verifiedRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Android tools move escaped the workspace tools directory' }
    Move-Item -LiteralPath $sourceTools -Destination $destinationTools
}
Write-Host 'Installing Android SDK platform 35, build tools 35.0.0 and adb'
$modernAndroidCli = Join-Path $env:ANDROID_HOME 'cmdline-tools/latest/bin/android.exe'
if (Test-Path -LiteralPath $modernAndroidCli) {
    & $modernAndroidCli --no-metrics "--sdk=$env:ANDROID_HOME" sdk install 'platform-tools' 'platforms/android-35' 'build-tools/35.0.0'
} else {
    1..100 | ForEach-Object { 'y' } | & $sdkManager "--sdk_root=$env:ANDROID_HOME" --licenses
    if ($LASTEXITCODE -ne 0) { throw 'Android SDK license setup failed' }
    & $sdkManager "--sdk_root=$env:ANDROID_HOME" 'platform-tools' 'platforms;android-35' 'build-tools;35.0.0'
}
if ($LASTEXITCODE -ne 0) { throw 'Android SDK package setup failed' }
foreach ($requiredSdkFile in @('platforms/android-35/android.jar','build-tools/35.0.0/aapt2.exe','platform-tools/adb.exe')) {
    if (!(Test-Path -LiteralPath (Join-Path $env:ANDROID_HOME $requiredSdkFile))) { throw "SDK package incomplete: $requiredSdkFile" }
}
$sdkProperty = $env:ANDROID_HOME.Replace('\','/').Replace(':','\:')
$sdkProperty = -join ($sdkProperty.ToCharArray() | ForEach-Object { if ([int]$_ -gt 127) { '\u{0:x4}' -f [int]$_ } else { [string]$_ } })
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'local.properties'), "sdk.dir=$sdkProperty`n", [Text.UTF8Encoding]::new($false))
$sdkTaskLock.Dispose()
& (Join-Path $env:JAVA_HOME 'bin/java.exe') -version
& (Join-Path $env:ANDROID_HOME 'platform-tools/adb.exe') devices -l
if ($Build -and !$SkipGradle) {
    & (Join-Path $gradleRoot 'bin/gradle.bat') --project-dir $PSScriptRoot --no-daemon testDebugUnitTest assembleDebug
    if ($LASTEXITCODE -ne 0) { throw 'Android build failed' }
}
Write-Host "Build environment ready: $toolRoot"
