param([ValidateSet('baseline','final')][string]$Model='baseline', [switch]$Offline)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
$toolRoot=Join-Path $repoRoot 'output/tools/android'
$jdk=Get-ChildItem -LiteralPath $toolRoot -Directory -Filter 'jdk-17*' -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$jdk) { throw 'Run android/setup-build.ps1 first to prepare project-local JDK 17 and Android SDK.' }
$env:JAVA_HOME=$jdk.FullName
$env:ANDROID_HOME=Join-Path $toolRoot 'sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:ANDROID_USER_HOME=Join-Path $toolRoot 'android-user'
$env:GRADLE_USER_HOME=Join-Path $toolRoot 'gradle-user'
$gradle=Join-Path $toolRoot 'gradle-8.9/bin/gradle.bat'
if (!(Test-Path -LiteralPath $gradle)) { throw 'Project-local Gradle 8.9 missing. Run android/setup-build.ps1.' }
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
    [ordered]@{name=[string]$report.testsuite.name; tests=[int]$report.testsuite.tests; failures=[int]$report.testsuite.failures; errors=[int]$report.testsuite.errors; skipped=[int]$report.testsuite.skipped; report=$_.FullName}
})
if (($testSuites | Measure-Object -Property failures -Sum).Sum -gt 0 -or ($testSuites | Measure-Object -Property errors -Sum).Sum -gt 0) { throw 'Unit-test report contains failures.' }
$models=@(256,384 | ForEach-Object {
    $path=Join-Path $repoRoot "output/models/zipdepth_${Model}_${_}.onnx"
    [ordered]@{size=$_; path=$path; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant(); bytes=(Get-Item -LiteralPath $path).Length}
})
$evidence=[ordered]@{
    generated_at_utc=[DateTime]::UtcNow.ToString('o')
    model_prefix="zipdepth_$Model"
    build_command="powershell -ExecutionPolicy Bypass -File android/build.ps1 -Model $Model"
    build_result='success'
    apk=[ordered]@{path=$destination; sha256=$apkHash; bytes=(Get-Item -LiteralPath $destination).Length}
    tooling=[ordered]@{jdk='17'; gradle='8.9'; agp='8.7.3'; kotlin='2.0.21'; cameraX='1.4.2'; onnxruntime='1.20.0'; compileSdk=35; buildTools='35.0.0'; minSdk=28; abi='arm64-v8a'}
    models=$models
    unit_test_suites=$testSuites
    shared_smoothing_fixture_sha256=(Get-FileHash -LiteralPath (Join-Path $repoRoot 'tests/fixtures/smoothing.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    device_validation=[ordered]@{status='not_run'; note='Build and JVM unit tests do not establish real camera behavior or device FPS. Honor 100 Pro validation must be recorded separately.'}
}
$json=$evidence | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $repoRoot "output/apk/build_evidence_${Model}.json"),$json,[Text.UTF8Encoding]::new($false))
if ($Model -eq 'baseline') { [IO.File]::WriteAllText((Join-Path $repoRoot 'output/apk/build_evidence.json'),$json,[Text.UTF8Encoding]::new($false)) }
Write-Host "APK SHA256: $apkHash"
Write-Host "Verified APK: $destination"
