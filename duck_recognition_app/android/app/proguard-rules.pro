# Please add these rules to your existing keep rules in order to suppress warnings.
# This is generated automatically by the Android Gradle plugin.
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options

# TensorFlow
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory$Options { *; }

# Flutter
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.plugins.** { *; }

# Native methods
-keepclassmembers class * {
    native <methods>;
}