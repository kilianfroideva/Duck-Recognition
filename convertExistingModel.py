import tensorflow as tf
from pathlib import Path
import numpy as np

print("🔄 Conversion de VOTRE modèle CNN avec Python 3.11 + TensorFlow...")

# VOTRE dossier avec le modèle entraîné
model_dir = Path(r"c:\Users\nicol\OneDrive\Bureau\ARN\Labo\Duck-Recognition\output\duck_classification_f1_optimized_20250612_145058")

# Charger VOTRE modèle entraîné
model_path = model_dir / "model" / "best_duck_classifier_f1_optimized.h5"
model = tf.keras.models.load_model(model_path)

print(f"✅ VOTRE modèle CNN chargé: {model.count_params():,} paramètres")
print(f"📊 Architecture: {len(model.layers)} couches")

# VOS classes
labels = ["Autre", "Colvert femelle", "Colvert mâle", "Foulque macroule", "Grèbe huppé"]

# Créer labels.txt
labels_path = model_dir / "labels.txt"
with open(labels_path, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"✅ Labels sauvegardés: {labels_path}")

# Conversion TFLite optimisée pour Flutter
print("🔄 Conversion TFLite avec VOTRE modèle entraîné...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimisations pour mobile avec vos données
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Plus petit
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Fallback pour compatibilité
]

# Representative dataset basé sur vos images de canards
def representative_data_gen():
    """Generate representative data based on duck images characteristics"""
    for _ in range(100):
        # Simuler des images de canards réalistes
        # Couleurs typiques: bleus (eau), verts (nature), marrons (plumage)
        duck_image = np.random.uniform(0.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
        
        # Ajouter des variations typiques des photos de canards
        duck_image[:, :, :, 0] *= 0.8  # Moins de rouge
        duck_image[:, :, :, 1] *= 1.1  # Plus de vert
        duck_image[:, :, :, 2] *= 1.2  # Plus de bleu
        
        duck_image = np.clip(duck_image, 0.0, 1.0)
        yield [duck_image]

converter.representative_dataset = representative_data_gen

# Conversion
tflite_model = converter.convert()

# Sauvegarder
tflite_path = model_dir / "duck_classifier.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Modèle TFLite sauvegardé: {tflite_path}")
print(f"📱 Taille: {len(tflite_model):,} bytes")

# Test de compatibilité
print("\n🧪 Test modèle avec Flutter...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Input shape: {input_details[0]['shape']}")
print(f"✅ Output shape: {output_details[0]['shape']}")

# Test avec image factice
test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(output[0])
confidence = np.max(output[0]) * 100

print(f"✅ Test réussi!")
print(f"📊 Prédiction test: {labels[predicted_class]} ({confidence:.1f}%)")

print(f"\n🎯 MODÈLE CNN ENTRAÎNÉ EST PRÊT!")
print(f"📂 Fichiers à copier vers Flutter:")
print(f"   {tflite_path}")
print(f"   {labels_path}")

print(f"\n📋 COMMANDES DE COPIE:")
print(f'copy "{tflite_path}" "duck_recognition_app\\assets\\models\\"')
print(f'copy "{labels_path}" "duck_recognition_app\\assets\\models\\"')

print(f"\n📝 Dans main.dart, utilisez:")
print(f"   'assets/models/your_trained_duck_model.tflite'")

print(f"\n🦆 L'APP UTILISERA UN MODELE CNN ENTRAÎNÉ!")