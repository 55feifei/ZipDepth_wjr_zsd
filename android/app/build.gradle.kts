import java.io.File
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun sha256Hex(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

fun metadataField(file: File, name: String): String {
    val match = Regex(""""$name"\s*:\s*"([0-9a-fA-F]{64})"""").find(file.readText())
        ?: error("$name missing in ${file.name}")
    return match.groupValues[1].lowercase()
}

val modelPrefix = providers.gradleProperty("modelPrefix").orElse("zipdepth_baseline")
val generatedModelAssets = layout.buildDirectory.dir("generated/modelAssets")
val debugKeystore = rootProject.file("../output/tools/android/debug.keystore")
val ensureDebugKeystore by tasks.registering(Exec::class) {
    outputs.file(debugKeystore)
    onlyIf { !debugKeystore.isFile }
    doFirst { debugKeystore.parentFile.mkdirs() }
    commandLine(
        File(System.getProperty("java.home"), "bin/keytool.exe").absolutePath,
        "-genkeypair", "-noprompt", "-keystore", debugKeystore.absolutePath,
        "-alias", "androiddebugkey", "-storepass", "android", "-keypass", "android",
        "-dname", "CN=Android Debug,O=Android,C=US", "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000"
    )
}
val copyModels by tasks.registering {
    val sourceModels = listOf(256, 384).associateWith { rootProject.file("../output/models/${modelPrefix.get()}_${it}.onnx") }
    val metadataFiles = sourceModels.values.map { File(it.parentFile, "${it.nameWithoutExtension}.json") }
    inputs.files(sourceModels.values)
    inputs.files(metadataFiles)
    inputs.file(rootProject.file("../tests/fixtures/turbo_lut.json"))
    inputs.property("modelPrefix", modelPrefix)
    outputs.dir(generatedModelAssets)
    doLast {
        val missing = sourceModels.values.filter { !it.isFile || it.length() < 1024 }
        check(missing.isEmpty()) { "Real validated ONNX models required before APK packaging: $missing" }
        val checkpoints = mutableSetOf<String>()
        sourceModels.forEach { (_, source) ->
            val metadata = File(source.parentFile, "${source.nameWithoutExtension}.json")
            check(metadata.isFile) { "ONNX metadata required before APK packaging: $metadata" }
            val actual = sha256Hex(source)
            val expected = metadataField(metadata, "onnx_sha256")
            check(actual == expected) { "Refusing to package ${source.name}: SHA256 $actual does not match metadata $expected" }
            checkpoints += metadataField(metadata, "checkpoint_sha256")
        }
        check(checkpoints.size == 1) { "256/384 models do not come from the same checkpoint: $checkpoints" }
        val destination = generatedModelAssets.get().asFile.apply { mkdirs() }
        sourceModels.forEach { (size, source) -> source.copyTo(destination.resolve("zipdepth_${size}.onnx"), overwrite = true) }
        rootProject.file("../tests/fixtures/turbo_lut.json").copyTo(destination.resolve("turbo_lut.json"), overwrite = true)
    }
}

android {
    namespace = "edu.course.zipdepth"
    compileSdk = 35
    buildToolsVersion = "35.0.0"
    defaultConfig {
        applicationId = "edu.course.zipdepth"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0-${modelPrefix.get().substringAfter("zipdepth_")}"
        ndk { abiFilters += "arm64-v8a" }
        buildConfigField("String", "MODEL_FLAVOR", "\"${modelPrefix.get()}\"")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures { viewBinding = true; buildConfig = true }
    signingConfigs.getByName("debug") {
        storeFile = debugKeystore
        storePassword = "android"
        keyAlias = "androiddebugkey"
        keyPassword = "android"
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    sourceSets["main"].assets.srcDir(generatedModelAssets)
    sourceSets["test"].resources.srcDir(rootProject.file("../tests/fixtures"))
    androidResources { noCompress += "onnx" }
    packaging { resources.excludes += setOf("META-INF/LICENSE.md", "META-INF/LICENSE-notice.md") }
}

tasks.named("preBuild") { dependsOn(copyModels) }
tasks.matching { it.name == "validateSigningDebug" }.configureEach { dependsOn(ensureDebugKeystore) }

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.camera:camera-core:1.4.2")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.google.code.gson:gson:2.11.0")
}
