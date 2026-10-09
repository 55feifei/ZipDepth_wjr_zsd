param([ValidateSet('baseline','final')][string]$Model='baseline', [switch]$Offline)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
# Gradle's Windows test worker cannot load its main class when the project or Gradle path contains non-ASCII characters.
if ($repoRoot -match '[^\x00-\x7F]') {
    $link=Join-Path $env:LOCALAPPDATA 'ZipDepthRepo'
    $resolvedRepo=(Resolve-Path -LiteralPath $repoRoot).Path.TrimEnd('\')
    if (Test-Path -LiteralPath $link) {
        $target=@((Get-Item -LiteralPath $link).Target)[0]
        $resolvedLink=[IO.Path]::GetFullPath([string]$target).TrimEnd('\')
        if ($resolvedLink -ne $resolvedRepo) { throw "ASCII junction $link points at $resolvedLink, not this repository ($resolvedRepo)." }
    } else {
        & cmd.exe /c mklink /J "$link" "$resolvedRepo"
        if ($LASTEXITCODE -ne 0) { throw "Could not create ASCII junction $link. Gradle unit tests fail on this non-ASCII project path." }
    }
    $child=Join-Path $link 'android\build.ps1'
    $arguments=@('-ExecutionPolicy','Bypass','-File',$child,'-Model',$Model)
    if ($Offline) { $arguments += '-Offline' }
    & powershell @arguments
    exit $LASTEXITCODE
}
$toolRoot=Join-Path $repoRoot 'output/tools/android'
$jdk=Get-ChildItem -LiteralPath $toolRoot -Directory -Filter 'jdk-17*' -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$jdk) { throw 'Run android/setup-build.ps1 first to prepare project-local JDK 17 and Android SDK.' }
$env:JAVA_HOME=$jdk.FullName
$env:ANDROID_HOME=Join-Path $toolRoot 'sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:ANDROID_USER_HOME=Join-Path $toolRoot 'android-user'
$env:GRADLE_USER_HOME=Join-Path $toolRoot 'gradle-user'
$env:PYTHONPATH=$repoRoot
$gradle=Join-Path $toolRoot 'gradle-8.9/bin/gradle.bat'
if (!(Test-Path -LiteralPath $gradle)) { throw 'Project-local Gradle 8.9 missing. Run android/setup-build.ps1. This project has no Gradle Wrapper.' }
foreach ($requiredSdkFile in @('platforms/android-35/android.jar','build-tools/35.0.0/aapt.exe','build-tools/35.0.0/apksigner.bat','platform-tools/adb.exe')) {
    if (!(Test-Path -LiteralPath (Join-Path $env:ANDROID_HOME $requiredSdkFile))) { throw "SDK package incomplete: $requiredSdkFile. Run android/setup-build.ps1." }
}
$pythonCandidates=@($env:ZIPDEPTH_PYTHON,'F:\ANACONDA\envs\ML1\python.exe')
$python=$pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (!$python) {
    $command=Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $python=$command.Source }
}
if (!$python) { throw 'Python is required to verify ONNX metadata hashes. Set ZIPDEPTH_PYTHON or install the ML1 environment.' }
$inputSizeSource=Join-Path $PSScriptRoot 'app/src/main/java/edu/course/zipdepth/InputSize.kt'
$inputSizeText=[IO.File]::ReadAllText($inputSizeSource)
if ($inputSizeText -notmatch 'const val DEFAULT = 384') { throw 'First-launch input size must stay 384. Refusing to package another default.' }
Push-Location $repoRoot
try {
    & $python -B -m zipdepth.course.model_assets check-sources --root $repoRoot --prefix "zipdepth_$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Model metadata hash check failed; APK was not built.' }
} finally { Pop-Location }
$arguments=@('--project-dir',$PSScriptRoot,'--no-daemon','--no-watch-fs','--max-workers=1',"-PmodelPrefix=zipdepth_$Model",'testDebugUnitTest','assembleDebug')
if ($Offline) { $arguments += '--offline' }
& $gradle @arguments
if ($LASTEXITCODE -ne 0) { throw 'Tests or APK build failed; no successful build is claimed.' }
$apk=Join-Path $PSScriptRoot 'app/build/outputs/apk/debug/app-debug.apk'
$destination=Join-Path $repoRoot "output/apk/zipdepth-$Model-debug.apk"
New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
Copy-Item -LiteralPath $apk -Destination $destination -Force
$apkHash=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
$testReports=Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'app/build/test-results/testDebugUnitTest') -Filter 'TEST-*.xml'
if (!$testReports) { throw 'Kotlin unit-test XML reports are missing.' }
$testSuites=@($testReports | ForEach-Object {
    [xml]$report=Get-Content -LiteralPath $_.FullName -Raw
    if ([int]$report.testsuite.failures -gt 0 -or [int]$report.testsuite.errors -gt 0) { throw "Unit-test report contains failures: $($_.Name)" }
    [ordered]@{name=[string]$report.testsuite.name; tests=[int]$report.testsuite.tests; failures=[int]$report.testsuite.failures; errors=[int]$report.testsuite.errors; skipped=[int]$report.testsuite.skipped; report=$_.FullName}
})
$aapt=Join-Path $env:ANDROID_HOME 'build-tools/35.0.0/aapt.exe'
$apksigner=Join-Path $env:ANDROID_HOME 'build-tools/35.0.0/apksigner.bat'
& $apksigner verify --min-sdk-version 28 $destination
if ($LASTEXITCODE -ne 0) { throw 'APK signature verification failed.' }
$badging=& $aapt dump badging $destination
if ($LASTEXITCODE -ne 0) { throw 'aapt dump badging failed.' }
$badgingText=$badging -join "`n"
if ($badgingText -notmatch "package: name='edu\.course\.zipdepth'") { throw "APK package is not edu.course.zipdepth.`n$badgingText" }
if ($badgingText -notmatch "sdkVersion:'28'") { throw "APK minSdk is not 28.`n$badgingText" }
if ($badgingText -notmatch "native-code: 'arm64-v8a'") { throw "APK ABI is not arm64-v8a.`n$badgingText" }
$assetReport=Join-Path $repoRoot "output/apk/apk_asset_check_$Model.json"
Push-Location $repoRoot
try {
    & $python -B -m zipdepth.course.model_assets check-apk --root $repoRoot --prefix "zipdepth_$Model" --apk $destination --report $assetReport
    if ($LASTEXITCODE -ne 0) { throw 'APK asset hash check failed.' }
} finally { Pop-Location }
$assetObject=Get-Content -LiteralPath $assetReport -Raw -Encoding UTF8 | ConvertFrom-Json
$models=@(256,384 | ForEach-Object {
    $path=Join-Path $repoRoot "output/models/zipdepth_${Model}_${_}.onnx"
    $metadataPath=Join-Path $repoRoot "output/models/zipdepth_${Model}_${_}.json"
    $metadata=Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $sha=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha -ne [string]$metadata.onnx_sha256) { throw "Source hash for size $_ does not match metadata." }
    [ordered]@{size=$_; path=$path; sha256=$sha; metadata_sha256=[string]$metadata.onnx_sha256; checkpoint_sha256=[string]$metadata.checkpoint_sha256; matches_metadata=$true; bytes=(Get-Item -LiteralPath $path).Length}
})
$evidence=[ordered]@{
    generated_at_utc=[DateTime]::UtcNow.ToString('o')
    model_prefix="zipdepth_$Model"
    default_input_size=384
    gradle_wrapper_present=$false
    build_command="powershell -ExecutionPolicy Bypass -File android/build.ps1 -Model $Model"
    build_result='success'
    apk=[ordered]@{path=$destination; sha256=$apkHash; bytes=(Get-Item -LiteralPath $destination).Length; signature_verified=$true; package_name='edu.course.zipdepth'; min_sdk=28; target_sdk=35; abi='arm64-v8a'}
    tooling=[ordered]@{jdk='17'; gradle='8.9'; agp='8.7.3'; kotlin='2.0.21'; cameraX='1.4.2'; onnxruntime='1.20.0'; compileSdk=35; buildTools='35.0.0'; minSdk=28; abi='arm64-v8a'; gradle_wrapper='absent; project-local gradle-8.9 is invoked by android/build.ps1'}
    models=$models
    apk_assets=$assetObject
    unit_test_suites=$testSuites
    shared_smoothing_fixture_sha256=(Get-FileHash -LiteralPath (Join-Path $repoRoot 'tests/fixtures/smoothing.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    device_validation=[ordered]@{status='not_run'; note='Build and JVM unit tests do not establish real camera behavior or device FPS. Honor 100 Pro validation must be recorded separately. 15 FPS is a project performance target, not a course requirement.'}
}
$json=$evidence | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $repoRoot "output/apk/build_evidence_${Model}.json"),$json,[Text.UTF8Encoding]::new($false))
if ($Model -eq 'baseline') { [IO.File]::WriteAllText((Join-Path $repoRoot 'output/apk/build_evidence.json'),$json,[Text.UTF8Encoding]::new($false)) }
Write-Host "APK SHA256: $apkHash"
Write-Host "Verified APK: $destination"
