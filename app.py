import streamlit as st
import google.generativeai as genai
import os
import io
import uuid
import time
from duckduckgo_search import DDGS
import requests
import re
import datetime
from PIL import Image
import numpy as np
import logging
import json

# --- İsteğe Bağlı Kütüphaneler (Platforma özel kurulum gerektirebilir) ---
# Not: speech_recognition kaldırıldı çünkü sesli sohbet modu kaldırıldı.
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 modülü bulunamadı. Metin okuma özelliği (TTS) devre dışı bırakıldı.")

# --- Global Değişkenler ve Ayarlar ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Anahtarı Kontrolü
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets else os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'ı veya ortam değişkenlerini kontrol edin.")
    logger.error("GOOGLE_API_KEY bulunamadı. Uygulama durduruluyor.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google API Anahtarı başarıyla yapılandırıldı.")
except Exception as e:
    logger.error(f"Genel API Yapılandırma Hatası: {e}")
    st.error(f"API anahtarı yapılandırılamadı: {e}. Lütfen anahtarınızı kontrol edin.")
    st.stop()

# Gemini Model Parametreleri
GLOBAL_MODEL_NAME = 'gemini-1.5-flash-latest'
GLOBAL_TEMPERATURE = 0.7
GLOBAL_TOP_P = 0.95
GLOBAL_TOP_K = 40
GLOBAL_MAX_OUTPUT_TOKENS = 4096

# --- Dil Ayarları ---
LANGUAGES = {
    "TR": {"name": "Türkçe", "emoji": "🇹🇷"},
    "EN": {"name": "English", "emoji": "🇬🇧"},
    "FR": {"name": "Français", "emoji": "🇫🇷"},
    "ES": {"name": "Español", "emoji": "🇪🇸"},
    "DE": {"name": "Deutsch", "emoji": "🇩🇪"},
    "RU": {"name": "Русский", "emoji": "🇷🇺"},
    "SA": {"name": "العربية", "emoji": "🇸🇦"},
    "AZ": {"name": "Azərbaycan dili", "emoji": "🇦🇿"},
    "JP": {"name": "日本語", "emoji": "🇯🇵"},
    "KR": {"name": "한국어", "emoji": "🇰🇷"},
    "PT": {"name": "Português", "emoji": "🇵🇹"},
    "BR": {"name": "Português (Brasil)", "emoji": "🇧🇷"},
    "CA": {"name": "Français (Canada)", "emoji": "🇨🇦"},
    "MX": {"name": "Español (México)", "emoji": "🇲🇽"},
    "AR": {"name": "Español (Argentina)", "emoji": "🇦🇷"},
    "CN": {"name": "中文", "emoji": "🇨🇳"},
    "IN": {"name": "हिन्दी", "emoji": "🇮🇳"},
    "PK": {"name": "اردو", "emoji": "🇵🇰"},
    "UZ": {"name": "O'zbekcha", "emoji": "🇺🇿"},
    "KZ": {"name": "Қазақша", "emoji": "🇰🇿"},
}

# --- Yardımcı Fonksiyonlar ---

def get_text(key):
    """Seçili dile göre metin döndürür."""
    # Tüm diller için çevirileri içeren ana sözlük
    texts = {
        # --- MEVCUT (ESKİ) DİLLERİN ÇEVİRİLERİ ---
        "TR": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Yeni Kişisel Yapay Zeka Asistanınız!",
            "profile_title": "Size Nasıl Hitap Etmeliyim?",
            "profile_name_label": "Adınız:",
            "profile_upload_label": "Profil Resmi Yükle (isteğe bağlı)",
            "profile_save_button": "Kaydet",
            "profile_greeting": "Merhaba, {name}!",
            "profile_edit_info": "Ayarlar & Kişiselleştirme bölümünden profilinizi düzenleyebilirsiniz.",
            "ai_features_title": "Hanogt AI Özellikleri:",
            "feature_general_chat": "Genel sohbet",
            "feature_web_search": "Web araması (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Bilgi tabanı yanıtları",
            "feature_creative_text": "Yaratıcı metin üretimi",
            "feature_image_generation": "Basit görsel oluşturma (örnek)",
            "feature_text_to_speech": "Metin okuma (TTS)",
            "feature_feedback": "Geri bildirim mekanizması",
            "settings_button": "⚙️ Ayarlar & Kişiselleştirme",
            "about_button": "ℹ️ Hakkımızda",
            "app_mode_title": "Uygulama Modu",
            "chat_mode_text": "💬 Yazılı Sohbet",
            "chat_mode_image": "🖼️ Görsel Oluşturucu",
            "chat_mode_voice": "🎤 Sesli Sohbet (Dosya Yükle)", # Bu metin artık kullanılmıyor ama uyumluluk için kalabilir
            "chat_mode_creative": "✨ Yaratıcı Stüdyo",
            "chat_input_placeholder": "Mesajınızı yazın veya bir komut girin: Örn: 'Merhaba', 'web ara: Streamlit', 'yaratıcı metin: uzaylılar'...",
            "generating_response": "Yanıt oluşturuluyor...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Geri bildirim için teşekkürler!",
            "image_gen_title": "Görsel Oluşturucu",
            "image_gen_input_label": "Oluşturmak istediğiniz görseli tanımlayın:",
            "image_gen_button": "Görsel Oluştur",
            "image_gen_warning_placeholder": "Görsel oluşturma özelliği şu anda bir placeholder'dır ve gerçek bir API'ye bağlı değildir.",
            "image_gen_warning_prompt_missing": "Lütfen bir görsel açıklaması girin.",
            "voice_chat_title": "Sesli Sohbet", # Bu metin artık kullanılmıyor
            "creative_studio_title": "Yaratıcı Stüdyo",
            "creative_studio_info": "Bu bölüm, yaratıcı metin üretimi gibi gelişmiş özellikler için tasarlanmıştır.",
            "creative_studio_input_label": "Yaratıcı metin isteğinizi girin:",
            "creative_studio_button": "Metin Oluştur",
            "creative_studio_warning_prompt_missing": "Lütfen bir yaratıcı metin isteği girin.",
            "settings_personalization_title": "Ayarlar & Kişiselleştirme",
            "settings_name_change_label": "Adınızı Değiştir:",
            "settings_avatar_change_label": "Profil Resmini Değiştir (isteğe bağlı)",
            "settings_update_profile_button": "Profil Bilgilerini Güncelle",
            "settings_profile_updated_toast": "Profil güncellendi!",
            "settings_chat_management_title": "Sohbet Yönetimi",
            "settings_clear_chat_button": "🧹 Aktif Sohbet Geçmişini Temizle",
            "about_us_title": "ℹ️ Hakkımızda",
            "about_us_text": "Hanogt AI HanStudios'un Sahibi Oğuz Han Guluzade Tarafından 2025 Yılında Yapılmıştır, Açık Kaynak Kodludur, Gemini Tarafından Eğitilmiştir Ve Bütün Telif Hakları Saklıdır.",
            "footer_user": "Kullanıcı: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Aktif ({model_name}) | Log: Aktif",
            "model_init_success": "Gemini Modeli başarıyla başlatıldı!",
            "model_init_error": "Gemini modelini başlatırken bir hata oluştu: {error}. Lütfen API anahtarınızın doğru ve aktif olduğundan emin olun.",
            "gemini_model_not_initialized": "Gemini modeli başlatılmamış. Lütfen API anahtarınızı kontrol edin.",
            "image_load_error": "Görsel yüklenemedi: {error}",
            "image_not_convertible": "Bu içerik konuşmaya çevrilemez (metin değil).",
            "duckduckgo_error": "DuckDuckGo araması yapılırken hata oluştu: {error}",
            "wikipedia_network_error": "Wikipedia araması yapılırken ağ hatası oluştu: {error}",
            "wikipedia_json_error": "Wikipedia yanıtı çözümlenirken hata oluştu: {error}",
            "wikipedia_general_error": "Wikipedia araması yapılırken genel bir hata oluştu: {error}",
            "unexpected_response_error": "Yanıt alınırken beklenmeyen bir hata oluştu: {error}",
            "chat_cleared_toast": "Aktif sohbet temizlendi!",
            "profile_image_load_error": "Profil resmi yüklenemedi: {error}",
            "web_search_results": "Web Arama Sonuçları:",
            "web_search_no_results": "Aradığınız terimle ilgili sonuç bulunamadı.",
            "wikipedia_search_results": "Wikipedia Arama Sonuçları:",
            "wikipedia_search_no_results": "Aradığınız terimle ilgili sonuç bulunamadı.",
            "image_generated_example": "'{prompt}' için bir görsel oluşturuldu (örnek).",
            "image_upload_caption": "Yüklenen Görsel",
            "image_processing_error": "Görsel işlenirken bir hata oluştu: {error}",
            "image_vision_query": "Bu görselde ne görüyorsun?",
            "tts_sr_not_available": "Metin okuma özelliği (TTS) şu anda kullanılamıyor. Gerekli kütüphane (pyttsx3) yüklenmemiş veya uyumlu değil.",
            "gemini_response_error": "Yanıt alınırken beklenmeyen bir hata oluştu: {error}",
            "creative_text_generated": "Yaratıcı Metin Oluşturuldu:",
            "turkish_voice_not_found": "Türkçe ses bulunamadı, varsayılan ses kullanılacak. İşletim sisteminizin ses ayarlarını kontrol ediniz."
        },
        "EN": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Your New Personal AI Assistant!",
            "profile_title": "How Should I Address You?",
            "profile_name_label": "Your Name:",
            "profile_upload_label": "Upload Profile Picture (optional)",
            "profile_save_button": "Save",
            "profile_greeting": "Hello, {name}!",
            "profile_edit_info": "You can edit your profile in the Settings & Personalization section.",
            "ai_features_title": "Hanogt AI Features:",
            "feature_general_chat": "General chat",
            "feature_web_search": "Web search (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Knowledge base responses",
            "feature_creative_text": "Creative text generation",
            "feature_image_generation": "Simple image generation (placeholder)",
            "feature_text_to_speech": "Text-to-speech (TTS)",
            "feature_feedback": "Feedback mechanism",
            "settings_button": "⚙️ Settings & Personalization",
            "about_button": "ℹ️ About Us",
            "app_mode_title": "Application Mode",
            "chat_mode_text": "💬 Text Chat",
            "chat_mode_image": "🖼️ Image Generator",
            "chat_mode_creative": "✨ Creative Studio",
            "chat_input_placeholder": "Type your message or enter a command: E.g., 'Hello', 'web search: Streamlit', 'creative text: aliens'...",
            "generating_response": "Generating response...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Thanks for your feedback!",
            "image_gen_title": "Image Generator",
            "image_gen_input_label": "Describe the image you want to create:",
            "image_gen_button": "Generate Image",
            "image_gen_warning_placeholder": "Image generation feature is currently a placeholder and not connected to a real API.",
            "image_gen_warning_prompt_missing": "Please enter an image description.",
            "creative_studio_title": "Creative Studio",
            "creative_studio_info": "This section is designed for advanced features like creative text generation.",
            "creative_studio_input_label": "Enter your creative text request:",
            "creative_studio_button": "Generate Text",
            "creative_studio_warning_prompt_missing": "Please enter a creative text request.",
            "settings_personalization_title": "Settings & Personalization",
            "settings_name_change_label": "Change Your Name:",
            "settings_avatar_change_label": "Change Profile Picture (optional)",
            "settings_update_profile_button": "Update Profile Info",
            "settings_profile_updated_toast": "Profile updated!",
            "settings_chat_management_title": "Chat Management",
            "settings_clear_chat_button": "🧹 Clear Active Chat History",
            "about_us_title": "ℹ️ About Us",
            "about_us_text": "Hanogt AI was created by Oğuz Han Guluzade, owner of HanStudios, in 2025. It is open-source, trained by Gemini, and all copyrights are reserved.",
            "footer_user": "User: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Active ({model_name}) | Log: Active",
            "model_init_success": "Gemini Model successfully initialized!",
            "model_init_error": "An error occurred while initializing the Gemini model: {error}. Please ensure your API key is correct and active.",
            "gemini_model_not_initialized": "Gemini model not initialized. Please check your API key.",
            "image_load_error": "Could not load image: {error}",
            "image_not_convertible": "This content cannot be converted to speech (not text).",
            "duckduckgo_error": "An error occurred while performing DuckDuckGo search: {error}",
            "wikipedia_network_error": "Network error occurred while performing Wikipedia search: {error}",
            "wikipedia_json_error": "Error occurred while parsing Wikipedia response: {error}",
            "wikipedia_general_error": "A general error occurred while performing Wikipedia search: {error}",
            "unexpected_response_error": "An unexpected error occurred while getting a response: {error}",
            "chat_cleared_toast": "Active chat cleared!",
            "profile_image_load_error": "Could not load profile image: {error}",
            "web_search_results": "Web Search Results:",
            "web_search_no_results": "No results found for your search term.",
            "wikipedia_search_results": "Wikipedia Search Results:",
            "wikipedia_search_no_results": "No results found for your search term.",
            "image_generated_example": "An image for '{prompt}' was generated (example).",
            "image_upload_caption": "Uploaded Image",
            "image_processing_error": "An error occurred while processing the image: {error}",
            "image_vision_query": "What do you see in this image?",
            "tts_sr_not_available": "Text-to-speech (TTS) feature is currently unavailable. Make sure the required library (pyttsx3) is installed and compatible.",
            "gemini_response_error": "An unexpected error occurred while getting a response: {error}",
            "creative_text_generated": "Creative Text Generated:",
            "turkish_voice_not_found": "Turkish voice not found, default voice will be used. Please check your operating system's sound settings."
        },
        "FR": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Votre Nouvel Assistant IA Personnel !",
            "profile_title": "Comment dois-je vous appeler ?",
            "profile_name_label": "Votre nom :",
            "profile_upload_label": "Télécharger une photo de profil (facultatif)",
            "profile_save_button": "Enregistrer",
            "profile_greeting": "Bonjour, {name} !",
            "profile_edit_info": "Vous pouvez modifier votre profil dans la section Paramètres et Personnalisation.",
            "ai_features_title": "Fonctionnalités de Hanogt AI :",
            "feature_general_chat": "Chat général",
            "feature_web_search": "Recherche Web (DuckDuckGo, Wikipédia)",
            "feature_knowledge_base": "Réponses basées sur la connaissance",
            "feature_creative_text": "Génération de texte créatif",
            "feature_image_generation": "Génération d'images simple (aperçu)",
            "feature_text_to_speech": "Synthèse vocale (TTS)",
            "feature_feedback": "Mécanisme de feedback",
            "settings_button": "⚙️ Paramètres & Personnalisation",
            "about_button": "ℹ️ À Propos",
            "app_mode_title": "Mode de l'application",
            "chat_mode_text": "💬 Chat Textuel",
            "chat_mode_image": "🖼️ Générateur d'Images",
            "chat_mode_creative": "✨ Studio Créatif",
            "chat_input_placeholder": "Tapez votre message ou une commande : Ex: 'Bonjour', 'recherche web: Streamlit', 'texte créatif: aliens'...",
            "generating_response": "Génération de la réponse...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Merci pour votre feedback !",
            "image_gen_title": "Générateur d'Images",
            "image_gen_input_label": "Décrivez l'image que vous voulez créer :",
            "image_gen_button": "Générer l'Image",
            "image_gen_warning_placeholder": "La fonction de génération d'images est actuellement un aperçu et n'est pas connectée à une véritable API.",
            "image_gen_warning_prompt_missing": "Veuillez entrer une description d'image.",
            "creative_studio_title": "Studio Créatif",
            "creative_studio_info": "Cette section est conçue pour des fonctionnalités avancées comme la génération de texte créatif.",
            "creative_studio_input_label": "Entrez votre demande de texte créatif :",
            "creative_studio_button": "Générer du Texte",
            "creative_studio_warning_prompt_missing": "Veuillez entrer une demande de texte créatif.",
            "settings_personalization_title": "Paramètres & Personnalisation",
            "settings_name_change_label": "Changer votre nom :",
            "settings_avatar_change_label": "Changer la photo de profil (facultatif)",
            "settings_update_profile_button": "Mettre à jour les informations du profil",
            "settings_profile_updated_toast": "Profil mis à jour !",
            "settings_chat_management_title": "Gestion du Chat",
            "settings_clear_chat_button": "🧹 Effacer l'historique du chat actif",
            "about_us_title": "ℹ️ À Propos de Nous",
            "about_us_text": "Hanogt AI a été créé par Oğuz Han Guluzade, propriétaire de HanStudios, en 2025. Il est open-source, entraîné par Gemini, et tous les droits d'auteur sont réservés.",
            "footer_user": "Utilisateur : {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA : Actif ({model_name}) | Journal : Actif",
            "model_init_success": "Modèle Gemini initialisé avec succès !",
            "model_init_error": "Une erreur s'est produite lors de l'initialisation du modèle Gemini : {error}. Veuillez vous assurer que votre clé API est correcte et active.",
            "gemini_model_not_initialized": "Modèle Gemini non initialisé. Veuillez vérifier votre clé API.",
            "image_load_error": "Impossible de charger l'image : {error}",
            "image_not_convertible": "Ce contenu ne peut pas être converti en parole (pas du texte).",
            "duckduckgo_error": "Une erreur s'est produite lors de la recherche DuckDuckGo : {error}",
            "wikipedia_network_error": "Erreur réseau lors de la recherche Wikipédia : {error}",
            "wikipedia_json_error": "Erreur lors de l'analyse de la réponse Wikipédia : {error}",
            "wikipedia_general_error": "Une erreur générale s'est produite lors de la recherche Wikipédia : {error}",
            "unexpected_response_error": "Une erreur inattendue s'est produite lors de l'obtention d'une réponse : {error}",
            "chat_cleared_toast": "Chat actif effacé !",
            "profile_image_load_error": "Impossible de charger l'image de profil : {error}",
            "web_search_results": "Résultats de la recherche Web :",
            "web_search_no_results": "Aucun résultat trouvé pour votre terme de recherche.",
            "wikipedia_search_results": "Résultats de la recherche Wikipédia :",
            "wikipedia_search_no_results": "Aucun résultat trouvé pour votre terme de recherche.",
            "image_generated_example": "Une image pour '{prompt}' a été générée (exemple).",
            "image_upload_caption": "Image Téléchargée",
            "image_processing_error": "Une erreur s'est produite lors du traitement de l'image : {error}",
            "image_vision_query": "Que voyez-vous dans cette image ?",
            "tts_sr_not_available": "La fonction de synthèse vocale (TTS) est actuellement indisponible. Assurez-vous que la bibliothèque requise (pyttsx3) est installée.",
            "gemini_response_error": "Une erreur inattendue s'est produite lors de l'obtention d'une réponse : {error}",
            "creative_text_generated": "Texte Créatif Généré :",
            "turkish_voice_not_found": "Voix turque non trouvée, la voix par défaut sera utilisée. Veuillez vérifier les paramètres sonores de votre système d'exploitation."
        },
        # ... Diğer eski diller (ES, DE, RU, SA, AZ, JP, KR) buraya eklenecek...
        "ES": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "¡Tu Nuevo Asistente Personal de IA!",
            "profile_title": "¿Cómo debo llamarte?",
            "profile_name_label": "Tu nombre:",
            "profile_upload_label": "Subir foto de perfil (opcional)",
            "profile_save_button": "Guardar",
            "profile_greeting": "¡Hola, {name}!",
            "profile_edit_info": "Puedes editar tu perfil en la sección de Configuración y Personalización.",
            "ai_features_title": "Características de Hanogt AI:",
            "feature_general_chat": "Chat general",
            "feature_web_search": "Búsqueda web (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Respuestas de la base de conocimientos",
            "feature_creative_text": "Generación de texto creativo",
            "feature_image_generation": "Generación simple de imágenes (ejemplo)",
            "feature_text_to_speech": "Texto a voz (TTS)",
            "feature_feedback": "Mecanismo de retroalimentación",
            "settings_button": "⚙️ Configuración & Personalización",
            "about_button": "ℹ️ Acerca de Nosotros",
            "app_mode_title": "Modo de Aplicación",
            "chat_mode_text": "💬 Chat de Texto",
            "chat_mode_image": "🖼️ Generador de Imágenes",
            "chat_mode_creative": "✨ Estudio Creativo",
            "chat_input_placeholder": "Escribe tu mensaje o un comando: Ej.: 'Hola', 'búsqueda web: Streamlit', 'texto creativo: alienígenas'...",
            "generating_response": "Generando respuesta...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "¡Gracias por tu comentario!",
            "image_gen_title": "Generador de Imágenes",
            "image_gen_input_label": "Describe la imagen que quieres crear:",
            "image_gen_button": "Generar Imagen",
            "image_gen_warning_placeholder": "La función de generación de imágenes es actualmente un marcador de posición y no está conectada a una API real.",
            "image_gen_warning_prompt_missing": "Por favor, introduce una descripción de la imagen.",
            "creative_studio_title": "Estudio Creativo",
            "creative_studio_info": "Esta sección está diseñada para funciones avanzadas como la generación de texto creativo.",
            "creative_studio_input_label": "Introduce tu solicitud de texto creativo:",
            "creative_studio_button": "Generar Texto",
            "creative_studio_warning_prompt_missing": "Por favor, introduce una solicitud de texto creativo.",
            "settings_personalization_title": "Configuración & Personalización",
            "settings_name_change_label": "Cambiar tu nombre:",
            "settings_avatar_change_label": "Cambiar foto de perfil (opcional)",
            "settings_update_profile_button": "Actualizar información de perfil",
            "settings_profile_updated_toast": "¡Perfil actualizado!",
            "settings_chat_management_title": "Gestión de Chat",
            "settings_clear_chat_button": "🧹 Borrar Historial de Chat Activo",
            "about_us_title": "ℹ️ Acerca de Nosotros",
            "about_us_text": "Hanogt AI fue creado por Oğuz Han Guluzade, propietario de HanStudios, en 2025. Es de código abierto, entrenado por Gemini y todos los derechos de autor están reservados.",
            "footer_user": "Usuario: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA: Activa ({model_name}) | Registro: Activo",
            "model_init_success": "¡Modelo Gemini inicializado con éxito!",
            "model_init_error": "Se produjo un error al inicializar el modelo Gemini: {error}. Asegúrate de que tu clave API sea correcta y esté activa.",
            "gemini_model_not_initialized": "Modelo Gemini no inicializado. Por favor, verifica tu clave API.",
            "image_load_error": "No se pudo cargar la imagen: {error}",
            "image_not_convertible": "Este contenido no se puede convertir a voz (no es texto).",
            "duckduckgo_error": "Se produjo un error al realizar la búsqueda en DuckDuckGo: {error}",
            "wikipedia_network_error": "Se produjo un error de red al realizar la búsqueda en Wikipedia: {error}",
            "wikipedia_json_error": "Error al analizar la respuesta de Wikipedia: {error}",
            "wikipedia_general_error": "Se produjo un error general al realizar la búsqueda en Wikipedia: {error}",
            "unexpected_response_error": "Se produjo un error inesperado al obtener una respuesta: {error}",
            "chat_cleared_toast": "¡Chat activo borrado!",
            "profile_image_load_error": "No se pudo cargar la imagen de perfil: {error}",
            "web_search_results": "Resultados de la Búsqueda Web:",
            "web_search_no_results": "No se encontraron resultados para su término de búsqueda.",
            "wikipedia_search_results": "Resultados de la Búsqueda de Wikipedia:",
            "wikipedia_search_no_results": "No se encontraron resultados para su término de búsqueda.",
            "image_generated_example": "Se generó una imagen para '{prompt}' (ejemplo).",
            "image_upload_caption": "Imagen Subida",
            "image_processing_error": "Se produjo un error al procesar la imagen: {error}",
            "image_vision_query": "¿Qué ves en esta imagen?",
            "tts_sr_not_available": "La función de texto a voz (TTS) no está disponible actualmente. Asegúrate de que la biblioteca requerida (pyttsx3) esté instalada.",
            "gemini_response_error": "Se produjo un error inesperado al obtener una respuesta: {error}",
            "creative_text_generated": "Texto Creativo Generado:",
            "turkish_voice_not_found": "No se encontró voz turca, se utilizará la voz predeterminada. Por favor, verifica la configuración de sonido de tu sistema operativo."
        },

        # --- YENİ EKLENEN DİLLERİN ÇEVİRİLERİ ---
        "BR": { # Portekizce (Brezilya)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Seu Novo Assistente Pessoal de IA!",
            "profile_title": "Como devo me dirigir a você?",
            "profile_name_label": "Seu Nome:",
            "profile_upload_label": "Carregar Foto de Perfil (opcional)",
            "profile_save_button": "Salvar",
            "profile_greeting": "Olá, {name}!",
            "profile_edit_info": "Você pode editar seu perfil na seção Configurações e Personalização.",
            "ai_features_title": "Recursos do Hanogt AI:",
            "feature_general_chat": "Chat geral",
            "feature_web_search": "Pesquisa na web (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Respostas da base de conhecimento",
            "feature_creative_text": "Geração de texto criativo",
            "feature_image_generation": "Geração de imagem simples (exemplo)",
            "feature_text_to_speech": "Conversão de texto em fala (TTS)",
            "feature_feedback": "Mecanismo de feedback",
            "settings_button": "⚙️ Configurações e Personalização",
            "about_button": "ℹ️ Sobre Nós",
            "app_mode_title": "Modo do Aplicativo",
            "chat_mode_text": "💬 Chat de Texto",
            "chat_mode_image": "🖼️ Gerador de Imagens",
            "chat_mode_creative": "✨ Estúdio Criativo",
            "chat_input_placeholder": "Digite sua mensagem ou um comando: Ex: 'Olá', 'pesquisa web: Streamlit', 'texto criativo: alienígenas'...",
            "generating_response": "Gerando resposta...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Obrigado pelo seu feedback!",
            "image_gen_title": "Gerador de Imagens",
            "image_gen_input_label": "Descreva a imagem que você quer criar:",
            "image_gen_button": "Gerar Imagem",
            "image_gen_warning_placeholder": "O recurso de geração de imagens é atualmente um placeholder e não está conectado a uma API real.",
            "image_gen_warning_prompt_missing": "Por favor, insira uma descrição da imagem.",
            "creative_studio_title": "Estúdio Criativo",
            "creative_studio_info": "Esta seção é para recursos avançados como geração de texto criativo.",
            "creative_studio_input_label": "Insira seu pedido de texto criativo:",
            "creative_studio_button": "Gerar Texto",
            "creative_studio_warning_prompt_missing": "Por favor, insira um pedido de texto criativo.",
            "settings_personalization_title": "Configurações e Personalização",
            "settings_name_change_label": "Mudar Seu Nome:",
            "settings_avatar_change_label": "Mudar Foto de Perfil (opcional)",
            "settings_update_profile_button": "Atualizar Informações do Perfil",
            "settings_profile_updated_toast": "Perfil atualizado!",
            "settings_chat_management_title": "Gerenciamento de Chat",
            "settings_clear_chat_button": "🧹 Limpar Histórico do Chat Ativo",
            "about_us_title": "ℹ️ Sobre Nós",
            "about_us_text": "O Hanogt AI foi criado por Oğuz Han Guluzade, proprietário da HanStudios, em 2025. É de código aberto, treinado pelo Gemini, e todos os direitos autorais são reservados.",
            "footer_user": "Usuário: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA: Ativa ({model_name}) | Log: Ativo",
            "model_init_success": "Modelo Gemini iniciado com sucesso!",
            "model_init_error": "Ocorreu um erro ao iniciar o modelo Gemini: {error}. Por favor, verifique se sua chave de API está correta e ativa.",
            "gemini_model_not_initialized": "Modelo Gemini não iniciado. Por favor, verifique sua chave de API.",
            "image_load_error": "Não foi possível carregar a imagem: {error}",
            "image_not_convertible": "Este conteúdo não pode ser convertido em fala (não é texto).",
            "duckduckgo_error": "Ocorreu um erro ao realizar a pesquisa no DuckDuckGo: {error}",
            "wikipedia_network_error": "Ocorreu um erro de rede ao pesquisar na Wikipedia: {error}",
            "wikipedia_json_error": "Ocorreu um erro ao analisar a resposta da Wikipedia: {error}",
            "wikipedia_general_error": "Ocorreu um erro geral ao pesquisar na Wikipedia: {error}",
            "unexpected_response_error": "Ocorreu um erro inesperado ao obter uma resposta: {error}",
            "chat_cleared_toast": "Chat ativo limpo!",
            "profile_image_load_error": "Não foi possível carregar a imagem de perfil: {error}",
            "web_search_results": "Resultados da Pesquisa na Web:",
            "web_search_no_results": "Nenhum resultado encontrado para o seu termo de pesquisa.",
            "wikipedia_search_results": "Resultados da Pesquisa na Wikipedia:",
            "wikipedia_search_no_results": "Nenhum resultado encontrado para o seu termo de pesquisa.",
            "image_generated_example": "Uma imagem para '{prompt}' foi gerada (exemplo).",
            "image_upload_caption": "Imagem Carregada",
            "image_processing_error": "Ocorreu um erro ao processar a imagem: {error}",
            "image_vision_query": "O que você vê nesta imagem?",
            "tts_sr_not_available": "O recurso de texto para fala (TTS) não está disponível. Certifique-se de que a biblioteca necessária (pyttsx3) está instalada.",
            "gemini_response_error": "Ocorreu um erro inesperado ao obter uma resposta: {error}",
            "creative_text_generated": "Texto Criativo Gerado:",
            "turkish_voice_not_found": "Voz em turco não encontrada, será usada a voz padrão. Verifique as configurações de som do seu sistema operacional."
        },
        # ... Diğer yeni dillerin çevirileri (CA, MX, AR, PT, CN, IN, PK, UZ, KZ) buraya eklenecek...
    }
    # Tüm dilleri kapsayacak şekilde genişletilmiş sözlük...
    # (Yukarıdaki kodda olduğu gibi tüm dillerin tam çevirilerini ekleyin)

    # Seçilen dile ait metin sözlüğünü al, yoksa varsayılan olarak Türkçe (TR) kullan
    return texts.get(st.session_state.current_language, texts["TR"]).get(key, f"TEXT_MISSING: {key}")


def initialize_session_state():
    """Uygulama oturum durumunu başlatır."""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_avatar" not in st.session_state:
        st.session_state.user_avatar = None
    if "models_initialized" not in st.session_state:
        st.session_state.models_initialized = False
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat_0"
        if "chat_0" not in st.session_state.all_chats:
            st.session_state.all_chats["chat_0"] = []
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "💬 Yazılı Sohbet"
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    if "show_about" not in st.session_state:
        st.session_state.show_about = False
    if "current_language" not in st.session_state:
        st.session_state.current_language = "TR"
    
    if "gemini_model" not in st.session_state or not st.session_state.models_initialized:
        initialize_gemini_model()

    load_chat_history()

def initialize_gemini_model():
    """Gemini modelini başlatır ve oturum durumuna kaydeder."""
    if st.session_state.get("gemini_model") is None or not st.session_state.get("models_initialized", False):
        try:
            st.session_state.gemini_model = genai.GenerativeModel(
                model_name=GLOBAL_MODEL_NAME,
                generation_config=genai.GenerationConfig( 
                    temperature=GLOBAL_TEMPERATURE,
                    top_p=GLOBAL_TOP_P,
                    top_k=GLOBAL_TOP_K,
                    max_output_tokens=GLOBAL_MAX_OUTPUT_TOKENS,
                )
            )
            st.session_state.models_initialized = True
            st.toast(get_text("model_init_success"), icon="✅")
            logger.info(f"Gemini Modeli başlatıldı: {GLOBAL_MODEL_NAME}")
        except Exception as e:
            st.error(get_text("model_init_error").format(error=e))
            st.session_state.models_initialized = False
            logger.error(f"Gemini modeli başlatma hatası: {e}")

def add_to_chat_history(chat_id, role, content):
    """Sohbet geçmişine mesaj ekler."""
    if chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[chat_id] = []
    
    # İçerik türüne göre farklı işlem yap
    if isinstance(content, Image.Image):
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='PNG')
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [img_byte_arr.getvalue()]})
    elif isinstance(content, bytes): # Zaten byte ise doğrudan ekle
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    else: # Metin ise
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [str(content)]})
    
    logger.info(f"Sohbet geçmişine eklendi: Chat ID: {chat_id}, Rol: {role}, İçerik Türü: {type(content)}")

def load_chat_history():
    """Sohbet geçmişini yükler."""
    if st.session_state.active_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []

def clear_active_chat():
    """Aktif sohbetin içeriğini temizler."""
    if st.session_state.active_chat_id in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session # Gemini oturumunu da temizle
        st.toast(get_text("chat_cleared_toast"), icon="🧹")
        logger.info(f"Aktif sohbet ({st.session_state.active_chat_id}) temizlendi.")
    st.rerun()

def text_to_speech(text):
    """Metni konuşmaya çevirir ve sesi oynatır."""
    if not TTS_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return
    try:
        engine = pyttsx3.init()
        # Dil'e özel ses arama (isteğe bağlı, geliştirilebilir)
        # Şimdilik varsayılan sesi kullanıyoruz.
        engine.say(text)
        engine.runAndWait()
        logger.info("Metinden sese çevirme başarılı.")
    except Exception as e:
        st.error(get_text("unexpected_response_error").format(error=e))
        logger.error(f"Metinden sese çevirme hatası: {e}")


@st.cache_data(ttl=3600)
def duckduckgo_search(query):
    """DuckDuckGo kullanarak web araması yapar."""
    try:
        with DDGS() as ddgs:
            # max_results'ı artırarak daha fazla sonuç alabilirsiniz
            results = list(ddgs.text(query, max_results=5))
            return results
    except Exception as e:
        st.error(get_text("duckduckgo_error").format(error=e))
        return []

@st.cache_data(ttl=3600)
def wikipedia_search(query):
    """Wikipedia'da arama yapar."""
    lang_code = st.session_state.current_language.lower().split('-')[0]
    api_url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("query", {}).get("search", [])
    except requests.exceptions.RequestException as e:
        st.error(get_text("wikipedia_network_error").format(error=e))
    except json.JSONDecodeError as e:
        st.error(get_text("wikipedia_json_error").format(error=e))
    except Exception as e:
        st.error(get_text("wikipedia_general_error").format(error=e))
    return []

def generate_image(prompt):
    """Görsel oluşturma (örnek - placeholder)."""
    st.warning(get_text("image_gen_warning_placeholder"))
    # Metni URL uyumlu hale getir
    safe_prompt = requests.utils.quote(prompt)
    placeholder_image_url = f"https://via.placeholder.com/512x512.png?text={safe_prompt}"
    
    with st.chat_message("assistant"):
        st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", get_text("image_generated_example").format(prompt=prompt))


# --- UI Bileşenleri ---

def display_welcome_and_profile_setup():
    """Hoş geldiniz mesajı ve profil oluşturma ekranı."""
    st.markdown(f"<h1 style='text-align: center;'>{get_text('welcome_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>{get_text('welcome_subtitle')}</h4>", unsafe_allow_html=True)
    st.write("---")

    with st.container():
        st.subheader(get_text("profile_title"))
        
        new_name = st.text_input(get_text("profile_name_label"), key="initial_name_input")
        uploaded_avatar = st.file_uploader(get_text("profile_upload_label"), type=["png", "jpg", "jpeg"], key="initial_avatar_upload")

        if st.button(get_text("profile_save_button"), key="initial_save_button"):
            if new_name:
                st.session_state.user_name = new_name
                if uploaded_avatar:
                    st.session_state.user_avatar = uploaded_avatar.read()
                st.rerun()
            else:
                st.warning("Lütfen bir ad girin.")

def display_settings_and_personalization():
    """Ayarlar ve Kişiselleştirme bölümünü gösterir."""
    with st.expander(get_text("settings_personalization_title"), expanded=True):
        new_name = st.text_input(get_text("settings_name_change_label"), value=st.session_state.user_name, key="settings_name_input")
        uploaded_avatar = st.file_uploader(get_text("settings_avatar_change_label"), type=["png", "jpg", "jpeg"], key="settings_avatar_upload")

        if st.button(get_text("settings_update_profile_button"), key="update_profile_button"):
            st.session_state.user_name = new_name
            if uploaded_avatar:
                st.session_state.user_avatar = uploaded_avatar.read()
            st.toast(get_text("settings_profile_updated_toast"), icon="✅")
            st.rerun()

        st.markdown("---")
        st.markdown(f"### {get_text('settings_chat_management_title')}")
        if st.button(get_text("settings_clear_chat_button"), key="clear_active_chat_button", type="primary"):
            clear_active_chat()

def display_about_section():
    """'Hakkımızda' bölümünü gösterir."""
    with st.expander(get_text("about_us_title"), expanded=True):
        st.markdown(get_text("about_us_text"))

def display_main_chat_interface():
    """Ana sohbet arayüzünü oluşturur ve yönetir."""
    
    # Ayarlar ve Hakkımızda butonları
    col1, col2 = st.columns(2)
    with col1:
        if st.button(get_text("settings_button"), use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_about = False 
    with col2:
        if st.button(get_text("about_button"), use_container_width=True):
            st.session_state.show_about = not st.session_state.show_about
            st.session_state.show_settings = False

    if st.session_state.show_settings:
        display_settings_and_personalization()
    if st.session_state.show_about:
        display_about_section()

    st.markdown("---")
    st.markdown(f"### {get_text('app_mode_title')}")

    # Sesli sohbet modu kaldırıldı
    mode_options = [
        get_text("chat_mode_text"),
        get_text("chat_mode_image"),
        get_text("chat_mode_creative")
    ]
    
    # Dil değişikliğinde index hatasını önlemek için kontrol
    try:
        current_mode_index = mode_options.index(st.session_state.chat_mode)
    except ValueError:
        st.session_state.chat_mode = mode_options[0] # Hata olursa varsayılana dön
        current_mode_index = 0

    selected_mode = st.radio(
        "Mod Seçimi",
        mode_options,
        index=current_mode_index,
        key="main_mode_radio",
        horizontal=True,
        label_visibility="collapsed"
    )

    if selected_mode != st.session_state.chat_mode:
        st.session_state.chat_mode = selected_mode
        st.rerun()

    # Modlara göre ilgili fonksiyonu çağır
    if st.session_state.chat_mode == get_text("chat_mode_text"):
        handle_text_chat()
    elif st.session_state.chat_mode == get_text("chat_mode_image"):
        handle_image_generation()
    elif st.session_state.chat_mode == get_text("chat_mode_creative"):
        handle_creative_studio()

def handle_text_chat():
    """Yazılı sohbet modunu yönetir."""
    chat_history = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    # Sohbet geçmişini göster
    for i, message in enumerate(chat_history):
        role = "assistant" if message["role"] == "model" else message["role"]
        avatar = st.session_state.user_avatar if role == "user" else None
        
        with st.chat_message(role, avatar=avatar):
            content = message["parts"][0]
            if isinstance(content, str):
                st.markdown(content)
                # Sadece asistan mesajları için butonları göster
                if role == "assistant":
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        if st.button("▶️", key=f"tts_{i}", help="Oku"):
                           text_to_speech(content)
                    with col2:
                       if st.button("👍", key=f"fb_{i}", help="Beğen"):
                           st.toast(get_text("feedback_toast"))

            elif isinstance(content, bytes):
                try:
                    img = Image.open(io.BytesIO(content))
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(get_text("image_load_error").format(error=str(e)))

    # Sohbet girişi
    prompt = st.chat_input(get_text("chat_input_placeholder"))
    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        st.rerun()

    # Eğer son mesaj kullanıcıdan ise ve cevap bekleniyorsa
    if chat_history and chat_history[-1]["role"] == "user":
        last_prompt = chat_history[-1]["parts"][0]
        
        with st.chat_message("assistant"):
            with st.spinner(get_text("generating_response")):
                response_text = ""
                try:
                    if last_prompt.lower().startswith("web ara:"):
                        query = last_prompt[len("web ara:"):].strip()
                        results = duckduckgo_search(query)
                        if results:
                            response_text = get_text("web_search_results") + "\n"
                            for r in results:
                                response_text += f"- **{r['title']}**: {r['body']} [link]({r['href']})\n"
                        else:
                            response_text = get_text("web_search_no_results")
                    elif last_prompt.lower().startswith("wiki ara:"):
                        query = last_prompt[len("wiki ara:"):].strip()
                        results = wikipedia_search(query)
                        if results:
                            lang_code = st.session_state.current_language.lower().split('-')[0]
                            response_text = get_text("wikipedia_search_results") + "\n"
                            for r in results:
                                response_text += f"- **{r['title']}**: [https://{lang_code}.wikipedia.org/?curid={r['pageid']}]\n"
                        else:
                            response_text = get_text("wikipedia_search_no_results")
                    elif last_prompt.lower().startswith("görsel oluştur:"):
                        image_prompt = last_prompt[len("görsel oluştur:"):].strip()
                        generate_image(image_prompt)
                        return # generate_image zaten mesajı ekliyor, bu yüzden burada dur
                    else:
                        # Normal Gemini sohbeti
                        gemini_history = [
                            {"role": ("assistant" if msg["role"] == "model" else msg["role"]), "parts": msg["parts"]}
                            for msg in chat_history[:-1]
                        ]
                        chat_session = st.session_state.gemini_model.start_chat(history=gemini_history)
                        response = chat_session.send_message(last_prompt, stream=True)
                        
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            response_placeholder.markdown(response_text + "▌")
                        response_placeholder.markdown(response_text) # Son halini imleçsiz yaz
                    
                    add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                    st.rerun()

                except Exception as e:
                    error_message = get_text("gemini_response_error").format(error=str(e))
                    st.error(error_message)
                    add_to_chat_history(st.session_state.active_chat_id, "model", f"Hata: {error_message}")
                    st.rerun()


def handle_image_generation():
    """Görsel oluşturma modunu yönetir."""
    st.subheader(get_text("image_gen_title"))
    image_prompt = st.text_input(get_text("image_gen_input_label"), key="image_prompt_input")
    if st.button(get_text("image_gen_button"), key="generate_image_button"):
        if image_prompt:
            generate_image(image_prompt)
        else:
            st.warning(get_text("image_gen_warning_prompt_missing"))

def handle_creative_studio():
    """Yaratıcı stüdyo modunu yönetir."""
    st.subheader(get_text("creative_studio_title"))
    st.info(get_text("creative_studio_info"))
    
    creative_prompt = st.text_area(get_text("creative_studio_input_label"), height=150, key="creative_prompt_input")
    if st.button(get_text("creative_studio_button"), key="generate_creative_text_button"):
        if creative_prompt:
            with st.spinner(get_text("generating_response")):
                try:
                    model = genai.GenerativeModel(GLOBAL_MODEL_NAME)
                    response = model.generate_content(f"Yaratıcı bir metin oluştur: {creative_prompt}", stream=True)
                    
                    st.success(get_text("creative_text_generated"))
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(get_text("unexpected_response_error").format(error=e))
        else:
            st.warning(get_text("creative_studio_warning_prompt_missing"))


# --- Ana Uygulama Mantığı ---

def main():
    """Ana Streamlit uygulamasını çalıştırır."""
    st.set_page_config(
        page_title="Hanogt AI Asistan",
        page_icon="✨",
        layout="centered", # Daha iyi bir görünüm için 'centered' kullanılabilir
        initial_sidebar_state="collapsed"
    )

    initialize_session_state()

    # Sol Üst Köşeye Dil Seçimini Koy
    col_lang, _ = st.columns([0.3, 0.7])
    with col_lang:
        lang_options = list(LANGUAGES.keys())
        try:
            current_lang_index = lang_options.index(st.session_state.current_language)
        except ValueError:
            current_lang_index = 0

        # format_func güncellendi: emoji + kısaltma
        selected_lang_code = st.selectbox(
            label="Dil Seçimi",
            options=lang_options,
            index=current_lang_index,
            key="language_selector",
            format_func=lambda code: f"{LANGUAGES[code]['emoji']} {code}",
            label_visibility="collapsed"
        )
        
        if selected_lang_code != st.session_state.current_language:
            st.session_state.current_language = selected_lang_code
            # Dil değiştiğinde mod isimlerini de güncellemek için chat_mode'u sıfırla
            st.session_state.chat_mode = get_text("chat_mode_text")
            st.rerun()

    if not st.session_state.get("user_name"):
        display_welcome_and_profile_setup()
    else:
        st.markdown(f"<h1 style='text-align: center;'>{get_text('welcome_title')}</h1>", unsafe_allow_html=True)
        display_main_chat_interface()

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center; font-size: 12px; color: gray;">
            {get_text('footer_user').format(user_name=st.session_state.user_name if st.session_state.user_name else "Misafir")} <br>
            {get_text('footer_version').format(year=datetime.datetime.now().year)} <br>
            {get_text('footer_ai_status').format(model_name=GLOBAL_MODEL_NAME)}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
