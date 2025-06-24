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

# --- Optional Libraries (May require platform-specific installation) ---
try:
    import pyttsx3
    import speech_recognition as sr
    TTS_SR_AVAILABLE = True
except ImportError:
    TTS_SR_AVAILABLE = False
    logging.warning("pyttsx3 or speech_recognition modules not found. Voice features disabled.")

# --- Global Variables and Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Key Check
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if st.secrets else os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check Streamlit Secrets or environment variables.")
    logger.error("GOOGLE_API_KEY not found. Application stopping.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google API Key successfully configured.")
except Exception as e:
    logger.error(f"General API Configuration Error: {e}")
    st.error(f"API key could not be configured: {e}. Please check your key.")
    st.stop()

# Gemini Model Parameters
GLOBAL_MODEL_NAME = 'gemini-1.5-flash-latest'
GLOBAL_TEMPERATURE = 0.7
GLOBAL_TOP_P = 0.95
GLOBAL_TOP_K = 40
GLOBAL_MAX_OUTPUT_TOKENS = 4096

# --- Language Settings ---
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
}

# --- Helper Functions ---

def get_text(key):
    """Returns text based on the selected language."""
    texts = {
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
            "chat_mode_voice": "🎤 Sesli Sohbet (Dosya Yükle)",
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
            "voice_chat_title": "Sesli Sohbet",
            "voice_upload_label": "Ses dosyası yükle (MP3, WAV)",
            "voice_upload_warning": "Ses dosyasından metin transkripsiyonu özelliği şu anda bir placeholder'dır.",
            "voice_live_input_title": "Canlı Ses Girişi",
            "voice_mic_button": "Mikrofonu Başlat",
            "voice_not_available": "Sesli sohbet özellikleri kullanılamıyor. Gerekli kütüphanelerin (pyttsx3, SpeechRecognition) kurulu olduğundan emin olun.",
            "voice_listening": "Dinleniyor...",
            "voice_heard": "Sen dedin: {text}",
            "voice_no_audio": "Ses algılanamadı, lütfen tekrar deneyin.",
            "voice_unknown": "Ne dediğini anlayamadım.",
            "voice_api_error": "Ses tanıma servisine ulaşılamıyor; {error}",
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
            "source_error": "Kaynak: Hata ({error})",
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
            "loading_audio_file": "Ses dosyası yükleniyor...",
            "tts_sr_not_available": "Sesli sohbet ve metin okuma özellikleri şu anda kullanılamıyor. Gerekli kütüphaneler yüklenmemiş veya uyumlu değil.",
            "mic_listen_timeout": "Ses algılama zaman aşımına uğradı.",
            "unexpected_audio_record_error": "Ses kaydı sırasında beklenmeyen bir hata oluştu: {error}",
            "gemini_response_error": "Yanıt alınırken beklenmeyen bir hata oluştu: {error}",
            "creative_text_generated": "Yaratıcı Metin Oluşturuldu: {text}",
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
            "chat_mode_voice": "🎤 Voice Chat (Upload File)",
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
            "voice_chat_title": "Voice Chat",
            "voice_upload_label": "Upload audio file (MP3, WAV)",
            "voice_upload_warning": "Audio file transcription feature is currently a placeholder.",
            "voice_live_input_title": "Live Voice Input",
            "voice_mic_button": "Start Microphone",
            "voice_not_available": "Voice chat features are currently unavailable. Make sure required libraries (pyttsx3, SpeechRecognition) are installed and compatible.",
            "voice_listening": "Listening...",
            "voice_heard": "You said: {text}",
            "voice_no_audio": "No audio detected, please try again.",
            "voice_unknown": "Could not understand what you said.",
            "voice_api_error": "Could not reach speech recognition service; {error}",
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
            "source_error": "Source: Error ({error})",
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
            "loading_audio_file": "Loading audio file...",
            "tts_sr_not_available": "Voice chat and text-to-speech features are currently unavailable. Make sure required libraries are installed and compatible.",
            "mic_listen_timeout": "Audio detection timed out.",
            "unexpected_audio_record_error": "An unexpected error occurred during audio recording: {error}",
            "gemini_response_error": "An unexpected error occurred while getting a response: {error}",
            "creative_text_generated": "Creative Text Generated: {text}",
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
            "chat_mode_voice": "🎤 Chat Vocal (Télécharger fichier)",
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
            "voice_chat_title": "Chat Vocal",
            "voice_upload_label": "Télécharger un fichier audio (MP3, WAV)",
            "voice_upload_warning": "La fonction de transcription de fichier audio est actuellement un aperçu.",
            "voice_live_input_title": "Entrée Vocale en Direct",
            "voice_mic_button": "Démarrer le Microphone",
            "voice_not_available": "Les fonctions de chat vocal sont actuellement indisponibles. Assurez-vous que les bibliothèques requises (pyttsx3, SpeechRecognition) sont installées et compatibles.",
            "voice_listening": "Écoute...",
            "voice_heard": "Vous avez dit : {text}",
            "voice_no_audio": "Aucun audio détecté, veuillez réessayer.",
            "voice_unknown": "Je n'ai pas compris ce que vous avez dit.",
            "voice_api_error": "Impossible d'atteindre le service de reconnaissance vocale ; {error}",
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
            "source_error": "Source : Erreur ({error})",
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
            "loading_audio_file": "Chargement du fichier audio...",
            "tts_sr_not_available": "Les fonctions de chat vocal et de synthèse vocale sont actuellement indisponibles. Assurez-vous que les bibliothèques requises sont installées et compatibles.",
            "mic_listen_timeout": "Détection audio expirée.",
            "unexpected_audio_record_error": "Une erreur inattendue s'est produite lors de l'enregistrement audio : {error}",
            "gemini_response_error": "Une erreur inattendue s'est produite lors de l'obtention d'une réponse : {error}",
            "creative_text_generated": "Texte Créatif Généré : {text}",
            "turkish_voice_not_found": "Voix turque non trouvée, la voix par défaut sera utilisée. Veuillez vérifier les paramètres sonores de votre système d'exploitation."
        },
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
            "chat_mode_voice": "🎤 Chat de Voz (Subir archivo)",
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
            "voice_chat_title": "Chat de Voz",
            "voice_upload_label": "Subir archivo de audio (MP3, WAV)",
            "voice_upload_warning": "La función de transcripción de archivos de audio es actualmente un marcador de posición.",
            "voice_live_input_title": "Entrada de Voz en Vivo",
            "voice_mic_button": "Iniciar Micrófono",
            "voice_not_available": "Las funciones de chat de voz no están disponibles actualmente. Asegúrate de que las bibliotecas requeridas (pyttsx3, SpeechRecognition) estén instaladas y sean compatibles.",
            "voice_listening": "Escuchando...",
            "voice_heard": "Dijiste: {text}",
            "voice_no_audio": "No se detectó audio, por favor, inténtalo de nuevo.",
            "voice_unknown": "No pude entender lo que dijiste.",
            "voice_api_error": "No se puede acceder al servicio de reconocimiento de voz; {error}",
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
            "source_error": "Fuente: Error ({error})",
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
            "loading_audio_file": "Cargando archivo de audio...",
            "tts_sr_not_available": "Las funciones de chat de voz y texto a voz no están disponibles actualmente. Asegúrate de que las bibliotecas requeridas estén instaladas y sean compatibles.",
            "mic_listen_timeout": "Tiempo de espera de detección de audio agotado.",
            "unexpected_audio_record_error": "Se produjo un error inesperado durante la grabación de audio: {error}",
            "gemini_response_error": "Se produjo un error inesperado al obtener una respuesta: {error}",
            "creative_text_generated": "Texto Creativo Generado: {text}",
            "turkish_voice_not_found": "No se encontró voz turca, se utilizará la voz predeterminada. Por favor, verifica la configuración de sonido de tu sistema operativo."
        },
        "DE": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Ihr Neuer Persönlicher KI-Assistent!",
            "profile_title": "Wie soll ich Sie ansprechen?",
            "profile_name_label": "Ihr Name:",
            "profile_upload_label": "Profilbild hochladen (optional)",
            "profile_save_button": "Speichern",
            "profile_greeting": "Hallo, {name}!",
            "profile_edit_info": "Sie können Ihr Profil im Bereich Einstellungen & Personalisierung bearbeiten.",
            "ai_features_title": "Hanogt AI Funktionen:",
            "feature_general_chat": "Allgemeiner Chat",
            "feature_web_search": "Websuche (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Wissensdatenbank-Antworten",
            "feature_creative_text": "Kreative Texterstellung",
            "feature_image_generation": "Einfache Bilderzeugung (Beispiel)",
            "feature_text_to_speech": "Text-to-Speech (TTS)",
            "feature_feedback": "Feedback-Mechanismus",
            "settings_button": "⚙️ Einstellungen & Personalisierung",
            "about_button": "ℹ️ Über Uns",
            "app_mode_title": "Anwendungsmodus",
            "chat_mode_text": "💬 Text-Chat",
            "chat_mode_image": "🖼️ Bilderzeuger",
            "chat_mode_voice": "🎤 Sprach-Chat (Datei hochladen)",
            "chat_mode_creative": "✨ Kreativ-Studio",
            "chat_input_placeholder": "Geben Sie Ihre Nachricht oder einen Befehl ein: Z.B. 'Hallo', 'websuche: Streamlit', 'kreativer Text: Aliens'...",
            "generating_response": "Antwort wird generiert...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Vielen Dank für Ihr Feedback!",
            "image_gen_title": "Bilderzeuger",
            "image_gen_input_label": "Beschreiben Sie das Bild, das Sie erstellen möchten:",
            "image_gen_button": "Bild erzeugen",
            "image_gen_warning_placeholder": "Die Bilderzeugungsfunktion ist derzeit ein Platzhalter und nicht mit einer echten API verbunden.",
            "image_gen_warning_prompt_missing": "Bitte geben Sie eine Bildbeschreibung ein.",
            "voice_chat_title": "Sprach-Chat",
            "voice_upload_label": "Audiodatei hochladen (MP3, WAV)",
            "voice_upload_warning": "Die Audiodatei-Transkriptionsfunktion ist derzeit ein Platzhalter.",
            "voice_live_input_title": "Live-Spracheingabe",
            "voice_mic_button": "Mikrofon starten",
            "voice_not_available": "Sprach-Chat-Funktionen sind derzeit nicht verfügbar. Stellen Sie sicher, dass die erforderlichen Bibliotheken (pyttsx3, SpeechRecognition) installiert und kompatibel sind.",
            "voice_listening": "Hören...",
            "voice_heard": "Sie sagten: {text}",
            "voice_no_audio": "Kein Audio erkannt, bitte versuchen Sie es erneut.",
            "voice_unknown": "Ich konnte nicht verstehen, was Sie gesagt haben.",
            "voice_api_error": "Spracherkennungsdienst nicht erreichbar; {error}",
            "creative_studio_title": "Kreativ-Studio",
            "creative_studio_info": "Dieser Bereich ist für erweiterte Funktionen wie die Erstellung kreativer Texte konzipiert.",
            "creative_studio_input_label": "Geben Sie Ihre kreative Textanfrage ein:",
            "creative_studio_button": "Text erzeugen",
            "creative_studio_warning_prompt_missing": "Bitte geben Sie eine kreative Textanfrage ein.",
            "settings_personalization_title": "Einstellungen & Personalisierung",
            "settings_name_change_label": "Namen ändern:",
            "settings_avatar_change_label": "Profilbild ändern (optional)",
            "settings_update_profile_button": "Profilinformationen aktualisieren",
            "settings_profile_updated_toast": "Profil aktualisiert!",
            "settings_chat_management_title": "Chat-Verwaltung",
            "settings_clear_chat_button": "🧹 Aktuellen Chatverlauf löschen",
            "about_us_title": "ℹ️ Über Uns",
            "about_us_text": "Hanogt AI wurde 2025 von Oğuz Han Guluzade, dem Eigentümer von HanStudios, entwickelt. Es ist quelloffen, von Gemini trainiert und alle Urheberrechte sind vorbehalten.",
            "footer_user": "Benutzer: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "KI: Aktiv ({model_name}) | Protokoll: Aktiv",
            "model_init_success": "Gemini-Modell erfolgreich initialisiert!",
            "model_init_error": "Beim Initialisieren des Gemini-Modells ist ein Fehler aufgetreten: {error}. Stellen Sie sicher, dass Ihr API-Schlüssel korrekt und aktiv ist.",
            "gemini_model_not_initialized": "Gemini-Modell nicht initialisiert. Bitte überprüfen Sie Ihren API-Schlüssel.",
            "image_load_error": "Bild konnte nicht geladen werden: {error}",
            "image_not_convertible": "Dieser Inhalt kann nicht in Sprache umgewandelt werden (kein Text).",
            "duckduckgo_error": "Beim Durchführen der DuckDuckGo-Suche ist ein Fehler aufgetreten: {error}",
            "wikipedia_network_error": "Netzwerkfehler bei der Wikipedia-Suche: {error}",
            "wikipedia_json_error": "Fehler beim Parsen der Wikipedia-Antwort: {error}",
            "wikipedia_general_error": "Ein allgemeiner Fehler bei der Wikipedia-Suche: {error}",
            "unexpected_response_error": "Beim Abrufen einer Antwort ist ein unerwarteter Fehler aufgetreten: {error}",
            "source_error": "Quelle: Fehler ({error})",
            "chat_cleared_toast": "Aktueller Chat gelöscht!",
            "profile_image_load_error": "Profilbild konnte nicht geladen werden: {error}",
            "web_search_results": "Websuchergebnisse:",
            "web_search_no_results": "Keine Ergebnisse für Ihren Suchbegriff gefunden.",
            "wikipedia_search_results": "Wikipedia-Suchergebnisse:",
            "wikipedia_search_no_results": "Keine Ergebnisse für Ihren Suchbegriff gefunden.",
            "image_generated_example": "Ein Bild für '{prompt}' wurde generiert (Beispiel).",
            "image_upload_caption": "Hochgeladenes Bild",
            "image_processing_error": "Beim Verarbeiten des Bildes ist ein Fehler aufgetreten: {error}",
            "image_vision_query": "Was sehen Sie auf diesem Bild?",
            "loading_audio_file": "Audiodatei wird geladen...",
            "tts_sr_not_available": "Sprach-Chat- und Text-to-Speech-Funktionen sind derzeit nicht verfügbar. Stellen Sie sicher, dass die erforderlichen Bibliotheken installiert und kompatibel sind.",
            "mic_listen_timeout": "Audioerkennung Zeitüberschreitung.",
            "unexpected_audio_record_error": "Ein unerwarteter Fehler bei der Audioaufnahme: {error}",
            "gemini_response_error": "Ein unerwarteter Fehler beim Abrufen einer Antwort: {error}",
            "creative_text_generated": "Kreativer Text generiert: {text}",
            "turkish_voice_not_found": "Türkische Stimme nicht gefunden, Standardstimme wird verwendet. Bitte überprüfen Sie die Soundeinstellungen Ihres Betriebssystems."
        },
        "RU": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Ваш новый персональный ИИ-ассистент!",
            "profile_title": "Как мне к вам обращаться?",
            "profile_name_label": "Ваше имя:",
            "profile_upload_label": "Загрузить фото профиля (необязательно)",
            "profile_save_button": "Сохранить",
            "profile_greeting": "Привет, {name}!",
            "profile_edit_info": "Вы можете редактировать свой профиль в разделе «Настройки и персонализация».",
            "ai_features_title": "Функции Hanogt AI:",
            "feature_general_chat": "Общий чат",
            "feature_web_search": "Веб-поиск (DuckDuckGo, Википедия)",
            "feature_knowledge_base": "Ответы из базы знаний",
            "feature_creative_text": "Генерация креативного текста",
            "feature_image_generation": "Простая генерация изображений (пример)",
            "feature_text_to_speech": "Преобразование текста в речь (TTS)",
            "feature_feedback": "Механизм обратной связи",
            "settings_button": "⚙️ Настройки и персонализация",
            "about_button": "ℹ️ О нас",
            "app_mode_title": "Режим приложения",
            "chat_mode_text": "💬 Текстовый чат",
            "chat_mode_image": "🖼️ Генератор изображений",
            "chat_mode_voice": "🎤 Голосовой чат (загрузить файл)",
            "chat_mode_creative": "✨ Креативная студия",
            "chat_input_placeholder": "Введите сообщение или команду: Например, 'Привет', 'веб-поиск: Streamlit', 'креативный текст: инопланетяне'...",
            "generating_response": "Генерация ответа...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Спасибо за ваш отзыв!",
            "image_gen_title": "Генератор изображений",
            "image_gen_input_label": "Опишите изображение, которое вы хотите создать:",
            "image_gen_button": "Сгенерировать изображение",
            "image_gen_warning_placeholder": "Функция генерации изображений в настоящее время является заглушкой и не подключена к реальному API.",
            "image_gen_warning_prompt_missing": "Пожалуйста, введите описание изображения.",
            "voice_chat_title": "Голосовой чат",
            "voice_upload_label": "Загрузить аудиофайл (MP3, WAV)",
            "voice_upload_warning": "Функция транскрипции аудиофайлов в настоящее время является заглушкой.",
            "voice_live_input_title": "Ввод голоса в реальном времени",
            "voice_mic_button": "Запустить микрофон",
            "voice_not_available": "Функции голосового чата в настоящее время недоступны. Убедитесь, что необходимые библиотеки (pyttsx3, SpeechRecognition) установлены и совместимы.",
            "voice_listening": "Слушаю...",
            "voice_heard": "Вы сказали: {text}",
            "voice_no_audio": "Аудио не обнаружено, пожалуйста, попробуйте еще раз.",
            "voice_unknown": "Не удалось понять, что вы сказали.",
            "voice_api_error": "Служба распознавания речи недоступна; {error}",
            "creative_studio_title": "Креативная студия",
            "creative_studio_info": "Этот раздел предназначен для расширенных функций, таких как генерация креативного текста.",
            "creative_studio_input_label": "Введите свой запрос на креативный текст:",
            "creative_studio_button": "Сгенерировать текст",
            "creative_studio_warning_prompt_missing": "Пожалуйста, введите запрос на креативный текст.",
            "settings_personalization_title": "Настройки и персонализация",
            "settings_name_change_label": "Изменить ваше имя:",
            "settings_avatar_change_label": "Изменить фото профиля (необязательно)",
            "settings_update_profile_button": "Обновить информацию профиля",
            "settings_profile_updated_toast": "Профиль обновлен!",
            "settings_chat_management_title": "Управление чатом",
            "settings_clear_chat_button": "🧹 Очистить историю активного чата",
            "about_us_title": "ℹ️ О нас",
            "about_us_text": "Hanogt AI был создан Огузом Ханом Гулузаде, владельцем HanStudios, в 2025 году. Он имеет открытый исходный код, обучен Gemini, и все авторские права защищены.",
            "footer_user": "Пользователь: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "ИИ: Активен ({model_name}) | Журнал: Активен",
            "model_init_success": "Модель Gemini успешно инициализирована!",
            "model_init_error": "Произошла ошибка при инициализации модели Gemini: {error}. Убедитесь, что ваш ключ API верен и активен.",
            "gemini_model_not_initialized": "Модель Gemini не инициализирована. Пожалуйста, проверьте свой ключ API.",
            "image_load_error": "Не удалось загрузить изображение: {error}",
            "image_not_convertible": "Этот контент не может быть преобразован в речь (не текст).",
            "duckduckgo_error": "Произошла ошибка при выполнении поиска DuckDuckGo: {error}",
            "wikipedia_network_error": "Произошла сетевая ошибка при выполнении поиска в Википедии: {error}",
            "wikipedia_json_error": "Ошибка при разборе ответа Википедии: {error}",
            "wikipedia_general_error": "Произошла общая ошибка при выполнении поиска в Википедии: {error}",
            "unexpected_response_error": "Произошла непредвиденная ошибка при получении ответа: {error}",
            "source_error": "Источник: Ошибка ({error})",
            "chat_cleared_toast": "Активный чат очищен!",
            "profile_image_load_error": "Не удалось загрузить изображение профиля: {error}",
            "web_search_results": "Результаты веб-поиска:",
            "web_search_no_results": "Результаты по вашему запросу не найдены.",
            "wikipedia_search_results": "Результаты поиска Википедии:",
            "wikipedia_search_no_results": "Результаты по вашему запросу не найдены.",
            "image_generated_example": "Изображение для '{prompt}' сгенерировано (пример).",
            "image_upload_caption": "Загруженное изображение",
            "image_processing_error": "Произошла ошибка при обработке изображения: {error}",
            "image_vision_query": "Что вы видите на этом изображении?",
            "loading_audio_file": "Загрузка аудиофайла...",
            "tts_sr_not_available": "Функции голосового чата и преобразования текста в речь в настоящее время недоступны. Убедитесь, что необходимые библиотеки установлены и совместимы.",
            "mic_listen_timeout": "Время ожидания обнаружения аудио истекло.",
            "unexpected_audio_record_error": "Произошла непредвиденная ошибка во время записи аудио: {error}",
            "gemini_response_error": "Произошла непредвиденная ошибка при получении ответа: {error}",
            "creative_text_generated": "Креативный текст сгенерирован: {text}",
            "turkish_voice_not_found": "Турецкий голос не найден, будет использоваться голос по умолчанию. Пожалуйста, проверьте настройки звука вашей операционной системы."
        },
        "SA": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "مساعدك الشخصي الجديد للذكاء الاصطناعي!",
            "profile_title": "كيف أجب أن أناديك؟",
            "profile_name_label": "اسمك:",
            "profile_upload_label": "تحميل صورة ملف شخصي (اختياري)",
            "profile_save_button": "حفظ",
            "profile_greeting": "مرحبًا، {name}!",
            "profile_edit_info": "يمكنك تعديل ملفك الشخصي في قسم الإعدادات والتخصيص.",
            "ai_features_title": "ميزات Hanogt AI:",
            "feature_general_chat": "دردشة عامة",
            "feature_web_search": "بحث الويب (DuckDuckGo, ويكيبيديا)",
            "feature_knowledge_base": "استجابات قاعدة المعرفة",
            "feature_creative_text": "إنشاء نص إبداعي",
            "feature_image_generation": "إنشاء صور بسيطة (مثال)",
            "feature_text_to_speech": "تحويل النص إلى كلام (TTS)",
            "feature_feedback": "آلية التغذية الراجعة",
            "settings_button": "⚙️ الإعدادات والتخصيص",
            "about_button": "ℹ️ حولنا",
            "app_mode_title": "وضع التطبيق",
            "chat_mode_text": "💬 الدردشة النصية",
            "chat_mode_image": "🖼️ منشئ الصور",
            "chat_mode_voice": "🎤 الدردشة الصوتية (تحميل ملف)",
            "chat_mode_creative": "✨ استوديو إبداعي",
            "chat_input_placeholder": "اكتب رسالتك أو أدخل أمرًا: مثال: 'مرحبًا', 'بحث ويب: Streamlit', 'نص إبداعي: كائنات فضائية'...",
            "generating_response": "جاري إنشاء الرد...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "شكرًا لملاحظاتك!",
            "image_gen_title": "منشئ الصور",
            "image_gen_input_label": "صف الصورة التي تريد إنشاءها:",
            "image_gen_button": "إنشاء صورة",
            "image_gen_warning_placeholder": "ميزة إنشاء الصور هي حاليًا مكان مؤقت وغير متصلة بواجهة برمجة تطبيقات حقيقية.",
            "image_gen_warning_prompt_missing": "الرجاء إدخال وصف للصورة.",
            "voice_chat_title": "الدردشة الصوتية",
            "voice_upload_label": "تحميل ملف صوتي (MP3, WAV)",
            "voice_upload_warning": "ميزة تحويل الملف الصوتي إلى نص هي حاليًا مكان مؤقت.",
            "voice_live_input_title": "إدخال صوت مباشر",
            "voice_mic_button": "تشغيل الميكروفون",
            "voice_not_available": "ميزات الدردشة الصوتية غير متاحة حاليًا. تأكد من تثبيت المكتبات المطلوبة (pyttsx3, SpeechRecognition) وتوافقها.",
            "voice_listening": "جاري الاستماع...",
            "voice_heard": "قلت: {text}",
            "voice_no_audio": "لم يتم اكتشاف صوت، يرجى المحاولة مرة أخرى.",
            "voice_unknown": "لم أتمكن من فهم ما قلته.",
            "voice_api_error": "لا يمكن الوصول إلى خدمة التعرف على الكلام؛ {error}",
            "creative_studio_title": "استوديو إبداعي",
            "creative_studio_info": "تم تصميم هذا القسم للميزات المتقدمة مثل إنشاء النص الإبداعي.",
            "creative_studio_input_label": "أدخل طلب النص الإبداعي الخاص بك:",
            "creative_studio_button": "إنشاء نص",
            "creative_studio_warning_prompt_missing": "الرجاء إدخال طلب نص إبداعي.",
            "settings_personalization_title": "الإعدادات والتخصيص",
            "settings_name_change_label": "تغيير اسمك:",
            "settings_avatar_change_label": "تغيير صورة الملف الشخصي (اختياري)",
            "settings_update_profile_button": "تحديث معلومات الملف الشخصي",
            "settings_profile_updated_toast": "تم تحديث الملف الشخصي!",
            "settings_chat_management_title": "إدارة الدردشة",
            "settings_clear_chat_button": "🧹 مسح سجل الدردشة النشط",
            "about_us_title": "ℹ️ حولنا",
            "about_us_text": "تم إنشاء Hanogt AI بواسطة أوغوز هان جولوزاده، مالك HanStudios، في عام 2025. إنه مفتوح المصدر، تم تدريبه بواسطة Gemini، وجميع حقوق النشر محفوظة.",
            "footer_user": "المستخدم: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "الذكاء الاصطناعي: نشط ({model_name}) | السجل: نشط",
            "model_init_success": "تم تهيئة نموذج Gemini بنجاح!",
            "model_init_error": "حدث خطأ أثناء تهيئة نموذج Gemini: {error}. يرجى التأكد من أن مفتاح API الخاص بك صحيح ونشط.",
            "gemini_model_not_initialized": "نموذج Gemini غير مهيأ. يرجى التحقق من مفتاح API الخاص بك.",
            "image_load_error": "تعذر تحميل الصورة: {error}",
            "image_not_convertible": "لا يمكن تحويل هذا المحتوى إلى كلام (ليس نصًا).",
            "duckduckgo_error": "حدث خطأ أثناء إجراء بحث DuckDuckGo: {error}",
            "wikipedia_network_error": "حدث خطأ في الشبكة أثناء إجراء بحث ويكيبيديا: {error}",
            "wikipedia_json_error": "خطأ أثناء تحليل استجابة ويكيبيديا: {error}",
            "wikipedia_general_error": "حدث خطأ عام أثناء إجراء بحث ويكيبيديا: {error}",
            "unexpected_response_error": "حدث خطأ غير متوقع أثناء تلقي رد: {error}",
            "source_error": "المصدر: خطأ ({error})",
            "chat_cleared_toast": "تم مسح الدردشة النشطة!",
            "profile_image_load_error": "تعذر تحميل صورة الملف الشخصي: {error}",
            "web_search_results": "نتائج بحث الويب:",
            "web_search_no_results": "لم يتم العثور على نتائج لمصطلح البحث الخاص بك.",
            "wikipedia_search_results": "نتائج بحث ويكيبيديا:",
            "wikipedia_search_no_results": "لم يتم العثور على نتائج لمصطلح البحث الخاص بك.",
            "image_generated_example": "تم إنشاء صورة لـ '{prompt}' (مثال).",
            "image_upload_caption": "الصورة المحملة",
            "image_processing_error": "حدث خطأ أثناء معالجة الصورة: {error}",
            "image_vision_query": "ماذا ترى في هذه الصورة؟",
            "loading_audio_file": "جاري تحميل الملف الصوتي...",
            "tts_sr_not_available": "ميزات الدردشة الصوتية وتحويل النص إلى كلام غير متاحة حاليًا. تأكد من تثبيت المكتبات المطلوبة وتوافقها.",
            "mic_listen_timeout": "انتهت مهلة اكتشاف الصوت.",
            "unexpected_audio_record_error": "حدث خطأ غير متوقع أثناء تسجيل الصوت: {error}",
            "gemini_response_error": "حدث خطأ غير متوقع أثناء تلقي رد: {error}",
            "creative_text_generated": "تم إنشاء النص الإبداعي: {text}",
            "turkish_voice_not_found": "لم يتم العثور على صوت تركي، سيتم استخدام الصوت الافتراضي. يرجى التحقق من إعدادات الصوت في نظام التشغيل الخاص بك."
        },
        "AZ": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Yeni Şəxsi Süni İntellekt Köməkçiniz!",
            "profile_title": "Sizə necə müraciət edim?",
            "profile_name_label": "Adınız:",
            "profile_upload_label": "Profil şəkli yükləyin (isteğe bağlı)",
            "profile_save_button": "Yadda saxla",
            "profile_greeting": "Salam, {name}!",
            "profile_edit_info": "Profilinizi Ayarlar və Fərdiləşdirmə bölməsində redaktə edə bilərsiniz.",
            "ai_features_title": "Hanogt AI Xüsusiyyətləri:",
            "feature_general_chat": "Ümumi söhbət",
            "feature_web_search": "Veb axtarış (DuckDuckGo, Vikipediya)",
            "feature_knowledge_base": "Bilik bazası cavabları",
            "feature_creative_text": "Yaradıcı mətn yaratma",
            "feature_image_generation": "Sadə şəkil yaratma (nümunə)",
            "feature_text_to_speech": "Mətnin səsə çevrilməsi (TTS)",
            "feature_feedback": "Rəy mexanizmi",
            "settings_button": "⚙️ Ayarlar & Fərdiləşdirmə",
            "about_button": "ℹ️ Haqqımızda",
            "app_mode_title": "Tətbiq Rejimi",
            "chat_mode_text": "💬 Yazılı Söhbət",
            "chat_mode_image": "🖼️ Şəkil Yaradıcı",
            "chat_mode_voice": "🎤 Səsli Söhbət (Fayl Yüklə)",
            "chat_mode_creative": "✨ Yaradıcı Studiya",
            "chat_input_placeholder": "Mesajınızı yazın və ya əmr daxil edin: Məsələn: 'Salam', 'veb axtar: Streamlit', 'yaradıcı mətn: yadplanetlilər'...",
            "generating_response": "Cavab yaradılır...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Rəyiniz üçün təşəkkür edirik!",
            "image_gen_title": "Şəkil Yaradıcı",
            "image_gen_input_label": "Yaratmaq istədiyiniz şəkli təsvir edin:",
            "image_gen_button": "Şəkil Yarat",
            "image_gen_warning_placeholder": "Şəkil yaratma xüsusiyyəti hazırda bir yer tutucudur və real API-yə qoşulmayıb.",
            "image_gen_warning_prompt_missing": "Zəhmət olmasa, bir şəkil təsviri daxil edin.",
            "voice_chat_title": "Səsli Söhbət",
            "voice_upload_label": "Səs faylı yükləyin (MP3, WAV)",
            "voice_upload_warning": "Səs faylından mətn transkripsiyası xüsusiyyəti hazırda bir yer tutucudur.",
            "voice_live_input_title": "Canlı Səs Girişi",
            "voice_mic_button": "Mikrofonu Başlat",
            "voice_not_available": "Səsli söhbət xüsusiyyətləri hazırda mövcud deyil. Lazımi kitabxanaların (pyttsx3, SpeechRecognition) quraşdırıldığından və uyğun olduğundan əmin olun.",
            "voice_listening": "Dinlənilir...",
            "voice_heard": "Sən dedin: {text}",
            "voice_no_audio": "Səs aşkarlanmadı, zəhmət olmasa yenidən cəhd edin.",
            "voice_unknown": "Nə dediyinizi başa düşmədim.",
            "voice_api_error": "Səs tanıma xidmətinə çatmaq mümkün deyil; {error}",
            "creative_studio_title": "Yaradıcı Studiya",
            "creative_studio_info": "Bu bölmə yaradıcı mətn yaratma kimi qabaqcıl xüsusiyyətlər üçün nəzərdə tutulub.",
            "creative_studio_input_label": "Yaradıcı mətn istəyinizi daxil edin:",
            "creative_studio_button": "Mətn Yarat",
            "creative_studio_warning_prompt_missing": "Zəhmət olmasa, bir yaradıcı mətn istəyi daxil edin.",
            "settings_personalization_title": "Ayarlar & Fərdiləşdirmə",
            "settings_name_change_label": "Adınızı Dəyişdirin:",
            "settings_avatar_change_label": "Profil Şəklini Dəyişdirin (isteğe bağlı)",
            "settings_update_profile_button": "Profil Məlumatlarını Yeniləyin",
            "settings_profile_updated_toast": "Profil yeniləndi!",
            "settings_chat_management_title": "Söhbət İdarəetməsi",
            "settings_clear_chat_button": "🧹 Aktiv Söhbət Keçmişini Təmizlə",
            "about_us_title": "ℹ️ Haqqımızda",
            "about_us_text": "Hanogt AI 2025-ci ildə HanStudios-un Sahibi Oğuz Xan Quluzadə tərəfindən hazırlanmışdır. Açıq Mənbə Kodludur, Gemini tərəfindən öyrədilmişdir və Bütün Müəllif Hüquqları Qorunur.",
            "footer_user": "İstifadəçi: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Aktiv ({model_name}) | Log: Aktiv",
            "model_init_success": "Gemini Modeli uğurla başladıldı!",
            "model_init_error": "Gemini modelini başladarkən bir səhv baş verdi: {error}. Zəhmət olmasa, API açarınızın doğru və aktiv olduğundan əmin olun.",
            "gemini_model_not_initialized": "Gemini modeli başladılmayıb. Zəhmət olmasa, API açarınızı yoxlayın.",
            "image_load_error": "Şəkil yüklənmədi: {error}",
            "image_not_convertible": "Bu məzmun səsə çevrilə bilməz (mətn deyil).",
            "duckduckgo_error": "DuckDuckGo axtarışı zamanı səhv baş verdi: {error}",
            "wikipedia_network_error": "Vikipediya axtarışı zamanı şəbəkə səhvi baş verdi: {error}",
            "wikipedia_json_error": "Vikipediya cavabı ayrıştırılarkən səhv baş verdi: {error}",
            "wikipedia_general_error": "Vikipediya axtarışı zamanı ümumi bir səhv baş verdi: {error}",
            "unexpected_response_error": "Cavab alınarkən gözlənilməz bir səhv baş verdi: {error}",
            "source_error": "Mənbə: Səhv ({error})",
            "chat_cleared_toast": "Aktiv söhbət təmizləndi!",
            "profile_image_load_error": "Profil şəkli yüklənmədi: {error}",
            "web_search_results": "Veb Axtarış Nəticələri:",
            "web_search_no_results": "Axtarış termininizlə əlaqəli nəticə tapılmadı.",
            "wikipedia_search_results": "Vikipediya Axtarış Nəticələri:",
            "wikipedia_search_no_results": "Axtarış termininizlə əlaqəli nəticə tapılmadı.",
            "image_generated_example": "'{prompt}' üçün bir şəkil yaradıldı (nümunə).",
            "image_upload_caption": "Yüklənən Şəkil",
            "image_processing_error": "Şəkil işlənərkən bir səhv baş verdi: {error}",
            "image_vision_query": "Bu şəkildə nə görürsən?",
            "loading_audio_file": "Səs faylı yüklənir...",
            "tts_sr_not_available": "Səsli söhbət və mətnin səsə çevrilməsi xüsusiyyətləri hazırda mövcud deyil. Lazımi kitabxanaların quraşdırıldığından və uyğun olduğundan əmin olun.",
            "mic_listen_timeout": "Səs aşkarlama vaxt aşımına uğradı.",
            "unexpected_audio_record_error": "Səs yazma zamanı gözlənilməz bir səhv baş verdi: {error}",
            "gemini_response_error": "Cavab alınarkən gözlənilməz bir səhv baş verdi: {error}",
            "creative_text_generated": "Yaradıcı Mətn Yaradıldı: {text}",
            "turkish_voice_not_found": "Türk səsi tapılmadı, standart səs istifadə olunacaq. Əməliyyat sisteminizin səs parametrlərini yoxlayın."
        },
        "JP": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "あなたの新しいパーソナルAIアシスタント！",
            "profile_title": "何とお呼びしましょうか？",
            "profile_name_label": "あなたの名前：",
            "profile_upload_label": "プロフィール画像をアップロード (オプション)",
            "profile_save_button": "保存",
            "profile_greeting": "こんにちは、{name}！",
            "profile_edit_info": "プロフィールは「設定とパーソナライズ」セクションで編集できます。",
            "ai_features_title": "Hanogt AI の機能：",
            "feature_general_chat": "一般チャット",
            "feature_web_search": "ウェブ検索 (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "ナレッジベースの回答",
            "feature_creative_text": "クリエイティブテキスト生成",
            "feature_image_generation": "簡易画像生成 (例)",
            "feature_text_to_speech": "テキスト読み上げ (TTS)",
            "feature_feedback": "フィードバックメカニズム",
            "settings_button": "⚙️ 設定とパーソナライズ",
            "about_button": "ℹ️ 会社概要",
            "app_mode_title": "アプリケーションモード",
            "chat_mode_text": "💬 テキストチャット",
            "chat_mode_image": "🖼️ 画像生成",
            "chat_mode_voice": "🎤 音声チャット (ファイルアップロード)",
            "chat_mode_creative": "✨ クリエイティブスタジオ",
            "chat_input_placeholder": "メッセージまたはコマンドを入力してください: 例: 'こんにちは', 'ウェブ検索: Streamlit', 'クリエイティブテキスト: エイリアン'...",
            "generating_response": "応答を生成中...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "フィードバックありがとうございます！",
            "image_gen_title": "画像生成",
            "image_gen_input_label": "作成したい画像を説明してください：",
            "image_gen_button": "画像を生成",
            "image_gen_warning_placeholder": "画像生成機能は現在プレースホルダーであり、実際のAPIには接続されていません。",
            "image_gen_warning_prompt_missing": "画像の説明を入力してください。",
            "voice_chat_title": "音声チャット",
            "voice_upload_label": "音声ファイルをアップロード (MP3, WAV)",
            "voice_upload_warning": "音声ファイルからのテキスト書き起こし機能は現在プレースホルダーです。",
            "voice_live_input_title": "ライブ音声入力",
            "voice_mic_button": "マイクを起動",
            "voice_not_available": "音声チャット機能は現在利用できません。必要なライブラリ (pyttsx3, SpeechRecognition) がインストールされ、互換性があることを確認してください。",
            "voice_listening": "聴いています...",
            "voice_heard": "あなたは言いました：{text}",
            "voice_no_audio": "音声が検出されませんでした。もう一度お試しください。",
            "voice_unknown": "何を言ったか理解できませんでした。",
            "voice_api_error": "音声認識サービスに到達できません; {error}",
            "creative_studio_title": "クリエイティブスタジオ",
            "creative_studio_info": "このセクションは、クリエイティブなテキスト生成などの高度な機能向けに設計されています。",
            "creative_studio_input_label": "クリエイティブなテキストリクエストを入力してください：",
            "creative_studio_button": "テキストを生成",
            "creative_studio_warning_prompt_missing": "クリエイティブなテキストリクエストを入力してください。",
            "settings_personalization_title": "設定とパーソナライズ",
            "settings_name_change_label": "名前を変更：",
            "settings_avatar_change_label": "プロフィール画像を変更 (オプション)",
            "settings_update_profile_button": "プロフィール情報を更新",
            "settings_profile_updated_toast": "プロフィールが更新されました！",
            "settings_chat_management_title": "チャット管理",
            "settings_clear_chat_button": "🧹 アクティブなチャット履歴をクリア",
            "about_us_title": "ℹ️ 会社概要",
            "about_us_text": "Hanogt AI は、HanStudios のオーナーである Oğuz Han Guluzade によって2025年に作成されました。オープンソースであり、Gemini によって訓練されており、すべての著作権は留保されています。",
            "footer_user": "ユーザー: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: アクティブ ({model_name}) | ログ: アクティブ",
            "model_init_success": "Geminiモデルが正常に初期化されました！",
            "model_init_error": "Geminiモデルの初期化中にエラーが発生しました：{error}。APIキーが正しいことを確認してください。",
            "gemini_model_not_initialized": "Geminiモデルが初期化されていません。APIキーを確認してください。",
            "image_load_error": "画像を読み込めませんでした：{error}",
            "image_not_convertible": "このコンテンツは音声に変換できません (テキストではありません)。",
            "duckduckgo_error": "DuckDuckGo検索の実行中にエラーが発生しました：{error}",
            "wikipedia_network_error": "Wikipedia検索の実行中にネットワークエラーが発生しました：{error}",
            "wikipedia_json_error": "Wikipediaの応答を解析中にエラーが発生しました：{error}",
            "wikipedia_general_error": "Wikipedia検索の実行中に一般的なエラーが発生しました：{error}",
            "unexpected_response_error": "応答の取得中に予期しないエラーが発生しました：{error}",
            "source_error": "ソース: エラー ({error})",
            "chat_cleared_toast": "アクティブなチャットがクリアされました！",
            "profile_image_load_error": "プロフィール画像を読み込めませんでした：{error}",
            "web_search_results": "ウェブ検索結果：",
            "web_search_no_results": "検索語句に一致する結果は見つかりませんでした。",
            "wikipedia_search_results": "Wikipedia検索結果：",
            "wikipedia_search_no_results": "検索語句に一致する結果は見つかりませんでした。",
            "image_generated_example": "'{prompt}'の画像が生成されました (例)。",
            "image_upload_caption": "アップロードされた画像",
            "image_processing_error": "画像の処理中にエラーが発生しました：{error}",
            "image_vision_query": "この画像に何が見えますか？",
            "loading_audio_file": "音声ファイルを読み込み中...",
            "tts_sr_not_available": "音声チャットおよびテキスト読み上げ機能は現在利用できません。必要なライブラリがインストールされ、互換性があることを確認してください。",
            "mic_listen_timeout": "音声検出がタイムアウトしました。",
            "unexpected_audio_record_error": "音声録音中に予期しないエラーが発生しました：{error}",
            "gemini_response_error": "応答の取得中に予期しないエラーが発生しました：{error}",
            "creative_text_generated": "クリエイティブテキスト生成済み：{text}",
            "turkish_voice_not_found": "トルコ語の音声が見つかりませんでした。デフォルトの音声が使用されます。オペレーティングシステムのサウンド設定を確認してください。"
        },
        "KR": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "新しい 개인 AI 어시스턴트!",
            "profile_title": "어떻게 불러드릴까요?",
            "profile_name_label": "이름:",
            "profile_upload_label": "프로필 사진 업로드 (선택 사항)",
            "profile_save_button": "저장",
            "profile_greeting": "안녕하세요, {name}님!",
            "profile_edit_info": "설정 및 개인화 섹션에서 프로필을 편집할 수 있습니다.",
            "ai_features_title": "Hanogt AI 기능:",
            "feature_general_chat": "일반 채팅",
            "feature_web_search": "웹 검색 (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "지식 기반 응답",
            "feature_creative_text": "창의적인 텍스트 생성",
            "feature_image_generation": "간단한 이미지 생성 (예시)",
            "feature_text_to_speech": "텍스트 음성 변환 (TTS)",
            "feature_feedback": "피드백 메커니즘",
            "settings_button": "⚙️ 설정 및 개인화",
            "about_button": "ℹ️ 회사 소개",
            "app_mode_title": "애플리케이션 모드",
            "chat_mode_text": "💬 텍스트 채팅",
            "chat_mode_image": "🖼️ 이미지 생성기",
            "chat_mode_voice": "🎤 음성 채팅 (파일 업로드)",
            "chat_mode_creative": "✨ 크리에이티브 스튜디오",
            "chat_input_placeholder": "메시지를 입력하거나 명령을 입력하세요: 예: '안녕하세요', '웹 검색: Streamlit', '창의적인 텍스트: 외계인'...",
            "generating_response": "응답 생성 중...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "피드백 감사합니다!",
            "image_gen_title": "이미지 생성기",
            "image_gen_input_label": "생성하려는 이미지를 설명하세요:",
            "image_gen_button": "이미지 생성",
            "image_gen_warning_placeholder": "이미지 생성 기능은 현재 플레이스홀더이며 실제 API에 연결되어 있지 않습니다.",
            "image_gen_warning_prompt_missing": "이미지 설명을 입력하세요.",
            "voice_chat_title": "음성 채팅",
            "voice_upload_label": "오디오 파일 업로드 (MP3, WAV)",
            "voice_upload_warning": "오디오 파일 전사 기능은 현재 플레이스홀더입니다.",
            "voice_live_input_title": "실시간 음성 입력",
            "voice_mic_button": "마이크 시작",
            "voice_not_available": "음성 채팅 기능은 현재 사용할 수 없습니다. 필요한 라이브러리(pyttsx3, SpeechRecognition)가 설치되어 있고 호환되는지 확인하세요.",
            "voice_listening": "듣는 중...",
            "voice_heard": "당신이 말했습니다: {text}",
            "voice_no_audio": "오디오가 감지되지 않았습니다. 다시 시도하세요.",
            "voice_unknown": "무슨 말을 했는지 이해할 수 없었습니다.",
            "voice_api_error": "음성 인식 서비스에 연결할 수 없습니다. {error}",
            "creative_studio_title": "크리에이티브 스튜디오",
            "creative_studio_info": "이 섹션은 창의적인 텍스트 생성과 같은 고급 기능을 위해 설계되었습니다.",
            "creative_studio_input_label": "창의적인 텍스트 요청을 입력하세요:",
            "creative_studio_button": "텍스트 생성",
            "creative_studio_warning_prompt_missing": "창의적인 텍스트 요청을 입력하세요.",
            "settings_personalization_title": "설정 및 개인화",
            "settings_name_change_label": "이름 변경:",
            "settings_avatar_change_label": "프로필 사진 변경 (선택 사항)",
            "settings_update_profile_button": "프로필 정보 업데이트",
            "settings_profile_updated_toast": "프로필이 업데이트되었습니다!",
            "settings_chat_management_title": "채팅 관리",
            "settings_clear_chat_button": "🧹 활성 채팅 기록 지우기",
            "about_us_title": "ℹ️ 회사 소개",
            "about_us_text": "Hanogt AI는 HanStudios의 소유자인 Oğuz Han Guluzade에 의해 2025년에 만들어졌습니다. 오픈 소스이며 Gemini에 의해 훈련되었으며 모든 저작권은 보호됩니다.",
            "footer_user": "사용자: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: 활성 ({model_name}) | 로그: 활성",
            "model_init_success": "Gemini 모델이 성공적으로 초기화되었습니다!",
            "model_init_error": "Gemini 모델 초기화 중 오류가 발생했습니다: {error}. API 키가 올바르고 활성 상태인지 확인하세요.",
            "gemini_model_not_initialized": "Gemini 모델이 초기화되지 않았습니다. API 키를 확인하세요.",
            "image_load_error": "이미지를 로드할 수 없습니다: {error}",
            "image_not_convertible": "이 콘텐츠는 음성으로 변환할 수 없습니다(텍스트가 아님).",
            "duckduckgo_error": "DuckDuckGo 검색 수행 중 오류가 발생했습니다: {error}",
            "wikipedia_network_error": "Wikipedia 검색 수행 중 네트워크 오류가 발생했습니다: {error}",
            "wikipedia_json_error": "Wikipedia 응답을 파싱하는 중 오류가 발생했습니다: {error}",
            "wikipedia_general_error": "Wikipedia 검색 수행 중 일반적인 오류가 발생했습니다: {error}",
            "unexpected_response_error": "응답을 가져오는 중 예기치 않은 오류가 발생했습니다: {error}",
            "source_error": "출처: 오류 ({error})",
            "chat_cleared_toast": "활성 채팅이 지워졌습니다!",
            "profile_image_load_error": "프로필 이미지를 로드할 수 없습니다: {error}",
            "web_search_results": "웹 검색 결과:",
            "web_search_no_results": "검색어에 대한 결과가 없습니다.",
            "wikipedia_search_results": "위키백과 검색 결과:",
            "wikipedia_search_no_results": "검색어에 대한 결과가 없습니다.",
            "image_generated_example": "'{prompt}'에 대한 이미지가 생성되었습니다(예시).",
            "image_upload_caption": "업로드된 이미지",
            "image_processing_error": "이미지 처리 중 오류가 발생했습니다: {error}",
            "image_vision_query": "이 이미지에서 무엇을 보시나요?",
            "loading_audio_file": "오디오 파일 로드 중...",
            "tts_sr_not_available": "음성 채팅 및 텍스트 음성 변환 기능은 현재 사용할 수 없습니다. 필요한 라이브러리가 설치되어 있고 호환되는지 확인하세요.",
            "mic_listen_timeout": "오디오 감지 시간 초과.",
            "unexpected_audio_record_error": "오디오 녹음 중 예기치 않은 오류가 발생했습니다: {error}",
            "gemini_response_error": "응답을 가져오는 중 예기치 않은 오류가 발생했습니다: {error}",
            "creative_text_generated": "창의적인 텍스트 생성됨: {text}",
            "turkish_voice_not_found": "터키어 음성을 찾을 수 없습니다. 기본 음성이 사용됩니다. 운영 체제의 사운드 설정을 확인하십시오."
        },
    }
    return texts.get(st.session_state.current_language, texts["TR"]).get(key, "TEXT_MISSING")

def initialize_session_state():
    """Initializes the application session state."""
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
        st.session_state.chat_mode = "💬 Yazılı Sohbet" # Updated to include emoji
    if "current_mode_index" not in st.session_state:
        st.session_state.current_mode_index = 0
    if "show_settings" not in st.session_state: # Control to show settings section
        st.session_state.show_settings = False
    if "show_about" not in st.session_state: # Control to show about section
        st.session_state.show_about = False
    if "current_language" not in st.session_state:
        st.session_state.current_language = "TR" # Default language Turkish
    
    # ADDED PART: Check and initialize gemini_model here
    # This check avoids re-initializing the model on every app reload
    if "gemini_model" not in st.session_state or not st.session_state.models_initialized:
        initialize_gemini_model() # Call the model initialization function

    load_chat_history()

def initialize_gemini_model():
    """Initializes the Gemini model and saves it to session state."""
    # Only initialize if 'gemini_model' is None or models_initialized is False
    if st.session_state.get("gemini_model") is None or not st.session_state.get("models_initialized", False):
        try:
            st.session_state.gemini_model = genai.GenerativeModel(
                model_name=GLOBAL_MODEL_NAME,
                # Correction: Used 'GenerationConfig' instead of 'Generation_config'.
                generation_config=genai.GenerationConfig( 
                    temperature=GLOBAL_TEMPERATURE,
                    top_p=GLOBAL_TOP_P,
                    top_k=GLOBAL_TOP_K,
                    max_output_tokens=GLOBAL_MAX_OUTPUT_TOKENS,
                )
            )
            st.session_state.models_initialized = True
            st.toast(get_text("model_init_success"), icon="✅")
            logger.info(f"Gemini Model initialized: {GLOBAL_MODEL_NAME}")
        except Exception as e:
            st.error(get_text("model_init_error").format(error=e))
            st.session_state.models_initialized = False
            logger.error(f"Gemini model initialization error: {e}")

def add_to_chat_history(chat_id, role, content):
    """Adds a message to the chat history."""
    if chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[chat_id] = []
    
    if isinstance(content, Image.Image):
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='PNG')
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [img_byte_arr.getvalue()]})
    elif isinstance(content, bytes):
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    else:
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    
    logger.info(f"Added to chat history: Chat ID: {chat_id}, Role: {role}, Content Type: {type(content)}")

def load_chat_history():
    """Loads chat history."""
    if st.session_state.active_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []

def clear_active_chat():
    """Clears the content of the active chat."""
    if st.session_state.active_chat_id in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.toast(get_text("chat_cleared_toast"), icon="🧹")
        logger.info(f"Active chat ({st.session_state.active_chat_id}) cleared.")
    st.rerun()

def text_to_speech(text):
    """Converts text to speech and plays the audio."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return False
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        found_turkish_voice = False
        for voice in voices:
            if "turkish" in voice.name.lower() or "tr-tr" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                found_turkish_voice = True
                break
        if not found_turkish_voice:
            st.warning(get_text("turkish_voice_not_found"))

        engine.say(text)
        engine.runAndWait()
        logger.info("Text-to-speech successful.")
        return True
    except Exception as e:
        st.error(get_text("unexpected_response_error").format(error=e))
        logger.error(f"Text-to-speech error: {e}")
        return False

def record_audio():
    """Captures audio input from the user."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(get_text("voice_listening"))
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning(get_text("voice_no_audio"))
            return ""
        except Exception as e:
            st.error(get_text("unexpected_audio_record_error").format(error=e))
            return ""
            
    try:
        text = r.recognize_google(audio, language="tr-TR") # Always use TR for recognition
        st.write(get_text("voice_heard").format(text=text))
        logger.info(f"Recognized audio: {text}")
        return text
    except sr.UnknownValueError:
        st.warning(get_text("voice_unknown"))
        return ""
    except sr.RequestError as e:
        st.error(get_text("voice_api_error").format(error=e))
        return ""
    except Exception as e:
        st.error(get_text("unexpected_audio_record_error").format(error=e))
        return ""

@st.cache_data(ttl=3600)
def duckduckgo_search(query):
    """Performs a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results
    except Exception as e:
        st.error(get_text("duckduckgo_error").format(error=e))
        return []

@st.cache_data(ttl=3600)
def wikipedia_search(query):
    """Searches Wikipedia."""
    try:
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
        response.raise_for_status()
        data = response.json()
        if data and "query" in data and "search" in data["query"]:
            return data["query"]["search"]
        return []
    except requests.exceptions.RequestException as e:
        st.error(get_text("wikipedia_network_error").format(error=e))
        return []
    except json.JSONDecodeError as e:
        st.error(get_text("wikipedia_json_error").format(error=e))
        return []
    except Exception as e:
        st.error(get_text("wikipedia_general_error").format(error=e))
        return []

def generate_image(prompt):
    """Image generation (example - placeholder)."""
    st.warning(get_text("image_gen_warning_placeholder"))
    placeholder_image_url = "https://via.placeholder.com/400x300.png?text=Görsel+Oluşturuldu"
    st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", get_text("image_generated_example").format(prompt=prompt))

def process_image_input(uploaded_file):
    """Processes the uploaded image and converts it to text."""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=get_text("image_upload_caption"), use_column_width=True)
            add_to_chat_history(st.session_state.active_chat_id, "user", image)
            
            if st.session_state.gemini_model:
                vision_chat_session = st.session_state.gemini_model.start_chat(history=[])
                response = vision_chat_session.send_message([image, get_text("image_vision_query")])
                response_text = response.text
                st.markdown(response_text)
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
            else:
                st.error(get_text("gemini_model_not_initialized"))
        except Exception as e:
            st.error(get_text("image_processing_error").format(error=e))

# --- UI Components ---

def display_welcome_and_profile_setup():
    """Displays welcome message and profile creation/editing."""
    st.markdown("<h1 style='text-align: center;'>Hanogt AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>{get_text('welcome_subtitle')}</h4>", unsafe_allow_html=True)
    st.write("---")

    col_features, col_profile = st.columns([1, 1])

    with col_features:
        st.subheader(get_text("ai_features_title"))
        st.markdown(f"""
            * {get_text('feature_general_chat')}
            * {get_text('feature_web_search')}
            * {get_text('feature_knowledge_base')}
            * {get_text('feature_creative_text')}
            * {get_text('feature_image_generation')}
            * {get_text('feature_text_to_speech')}
            * {get_text('feature_feedback')}
        """)

    with col_profile:
        st.markdown(f"<h3 style='text-align: center;'>{get_text('welcome_title')}</h3>", unsafe_allow_html=True)
        st.subheader(get_text("profile_title"))
        
        # Display profile picture
        if st.session_state.user_avatar:
            try:
                profile_image = Image.open(io.BytesIO(st.session_state.user_avatar))
                st.image(profile_image, caption=st.session_state.user_name if st.session_state.user_name else "User", width=150)
            except Exception as e:
                st.warning(get_text("profile_image_load_error").format(error=e))
                st.image("https://via.placeholder.com/150?text=Profile", width=150)
        else:
            st.image("https://via.placeholder.com/150?text=Profile", width=150)
        
        new_name = st.text_input(get_text("profile_name_label"), key="initial_name_input")
        uploaded_avatar = st.file_uploader(get_text("profile_upload_label"), type=["png", "jpg", "jpeg"], key="initial_avatar_upload")

        if st.button(get_text("profile_save_button"), key="initial_save_button"):
            if new_name:
                st.session_state.user_name = new_name
            if uploaded_avatar:
                st.session_state.user_avatar = uploaded_avatar.read()
            st.rerun()

    st.write("---")

def display_settings_and_personalization():
    """Displays the Settings and Personalization section."""
    st.markdown(f"## {get_text('settings_personalization_title')}")

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
    if st.button(get_text("settings_clear_chat_button"), key="clear_active_chat_button"):
        clear_active_chat()

    st.write("---")

def display_about_section():
    """Displays the 'About Us' section."""
    st.markdown(f"## {get_text('about_us_title')}")
    st.markdown(get_text("about_us_text"))
    st.write("---")

def display_main_chat_interface():
    """Displays the main chat interface."""
    
    # Settings and About buttons
    col_settings, col_about = st.columns(2)
    with col_settings:
        if st.button(get_text("settings_button"), key="toggle_settings"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_about = False # Close the other
    with col_about:
        if st.button(get_text("about_button"), key="toggle_about"):
            st.session_state.show_about = not st.session_state.show_about
            st.session_state.show_settings = False # Close the other

    if st.session_state.show_settings:
        display_settings_and_personalization()
    if st.session_state.show_about:
        display_about_section()

    st.markdown("---")
    st.markdown(f"## {get_text('app_mode_title')}")

    mode_options = [
        get_text("chat_mode_text"),
        get_text("chat_mode_image"),
        get_text("chat_mode_voice"),
        get_text("chat_mode_creative")
    ]
    st.session_state.chat_mode = st.radio(
        "Mode Selection", # Label should be filled for accessibility, even if hidden
        mode_options,
        horizontal=True,
        index=mode_options.index(st.session_state.chat_mode) if st.session_state.chat_mode in mode_options else 0,
        key="main_mode_radio"
    )
    
    current_mode_string = st.session_state.chat_mode 

    if current_mode_string == get_text("chat_mode_text"):
        handle_text_chat()
    elif current_mode_string == get_text("chat_mode_image"):
        handle_image_generation()
    elif current_mode_string == get_text("chat_mode_voice"):
        handle_voice_chat()
    elif current_mode_string == get_text("chat_mode_creative"):
        handle_creative_studio()

def handle_text_chat():
    """Manages the text chat mode."""
    chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    for message_index, message in enumerate(chat_messages):
        avatar_src = None
        if message["role"] == "user" and st.session_state.user_avatar:
            try:
                avatar_src = Image.open(io.BytesIO(st.session_state.user_avatar))
            except Exception as e:
                logger.warning(f"Failed to load user avatar for chat message: {e}")
                avatar_src = None

        with st.chat_message(message["role"], avatar=avatar_src):
            content_part = message["parts"][0]
            if isinstance(content_part, str):
                st.markdown(content_part)
            elif isinstance(content_part, bytes):
                try:
                    image = Image.open(io.BytesIO(content_part))
                    st.image(image, caption=get_text("image_upload_caption"), use_column_width=True)
                except Exception as e:
                    st.warning(get_text("image_load_error").format(error=e))

            col_btn1, col_btn2 = st.columns([0.05, 1])
            with col_btn1:
                if st.button(get_text("tts_button"), key=f"tts_btn_{st.session_state.active_chat_id}_{message_index}"):
                    if isinstance(content_part, str):
                        text_to_speech(content_part)
                    else:
                        st.warning(get_text("image_not_convertible"))
            with col_btn2:
                if st.button(get_text("feedback_button"), key=f"fb_btn_{st.session_state.active_chat_id}_{message_index}"):
                    st.toast(get_text("feedback_toast"), icon="🙏")

    prompt = st.chat_input(get_text("chat_input_placeholder"))

    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        
        if prompt.lower().startswith("web ara:") or prompt.lower().startswith("web search:"):
            query = prompt[prompt.find(":")+1:].strip()
            results = duckduckgo_search(query)
            if results:
                response_text = get_text("web_search_results") + "\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n{r['body']}\n{r['href']}\n\n"
            else:
                response_text = get_text("web_search_no_results")
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("wiki ara:") or prompt.lower().startswith("wiki search:"):
            query = prompt[prompt.find(":")+1:].strip()
            results = wikipedia_search(query)
            if results:
                response_text = get_text("wikipedia_search_results") + "\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n"
            else:
                response_text = get_text("wikipedia_search_no_results")
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("görsel oluştur:") or prompt.lower().startswith("generate image:"):
            image_prompt = prompt[prompt.find(":")+1:].strip()
            generate_image(image_prompt)
        else:
            # You can directly use this check here because it's already initialized in initialize_session_state().
            if st.session_state.gemini_model: 
                with st.spinner(get_text("generating_response")):
                    try:
                        processed_history = []
                        for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                            if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                try:
                                    processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                except Exception:
                                    continue
                            else:
                                processed_history.append(msg)

                        # Initialize or reset chat_session only the first time or if history changes
                        if "chat_session" not in st.session_state or st.session_state.chat_session.history != processed_history:
                             st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=processed_history)
                        
                        response = st.session_state.chat_session.send_message(prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty() 
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        
        st.rerun()

def handle_image_generation():
    """Manages the image generation mode."""
    st.subheader(get_text("image_gen_title"))
    image_prompt = st.text_input(get_text("image_gen_input_label"), key="image_prompt_input")
    if st.button(get_text("image_gen_button"), key="generate_image_button"):
        if image_prompt:
            generate_image(image_prompt)
        else:
            st.warning(get_text("image_gen_warning_prompt_missing"))

def handle_voice_chat():
    """Manages the voice chat mode."""
    st.subheader(get_text("voice_chat_title"))
    
    if not TTS_SR_AVAILABLE:
        st.info(get_text("voice_not_available"))
    else:
        uploaded_audio_file = st.file_uploader(get_text("voice_upload_label"), type=["mp3", "wav"], key="audio_uploader")
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
            st.warning(get_text("voice_upload_warning"))

        st.markdown("---")
        st.subheader(get_text("voice_live_input_title"))
        if st.button(get_text("voice_mic_button"), key="start_mic_button"):
            recognized_text = record_audio()
            if recognized_text:
                add_to_chat_history(st.session_state.active_chat_id, "user", recognized_text)

                if st.session_state.gemini_model:
                    with st.spinner(get_text("generating_response")):
                        try:
                            processed_history = []
                            for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                                if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                    try:
                                        processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                    except Exception:
                                        continue
                                else:
                                    processed_history.append(msg)

                            # Initialize or reset chat_session only the first time or if history changes
                            if "chat_session" not in st.session_state or st.session_state.chat_session.history != processed_history:
                                st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=processed_history)
                            
                            response = st.session_state.chat_session.send_message(recognized_text, stream=True)
                            response_text = ""
                            response_placeholder = st.empty()
                            for chunk in response:
                                response_text += chunk.text
                                with response_placeholder.container():
                                    st.markdown(response_text)
                            
                            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                            text_to_speech(response_text)
                            st.rerun()

                        except Exception as e:
                            st.error(get_text("gemini_response_error").format(error=e))
                else:
                    st.warning(get_text("gemini_model_not_initialized"))

def handle_creative_studio():
    """Manages the creative studio mode."""
    st.subheader(get_text("creative_studio_title"))
    st.write(get_text("creative_studio_info"))
    
    creative_prompt = st.text_area(get_text("creative_studio_input_label"), height=150, key="creative_prompt_input")
    if st.button(get_text("creative_studio_button"), key="generate_creative_text_button"):
        if creative_prompt:
            if st.session_state.gemini_model:
                with st.spinner(get_text("generating_response")):
                    try:
                        # For creative studio, a new session can always be started or the previous one can be used.
                        # If you don't want to keep history, starting with 'history=[]' makes sense.
                        creative_chat_session = st.session_state.gemini_model.start_chat(history=[])
                        response = creative_chat_session.send_message(f"Creative text generation: {creative_prompt}", stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", get_text("creative_text_generated").format(text=response_text))
                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        else:
            st.warning(get_text("creative_studio_warning_prompt_missing"))


# --- Main Application Logic ---

def main():
    """Runs the main Streamlit application."""
    st.set_page_config(
        page_title="Hanogt AI Assistant",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    initialize_session_state()

    # CSS injection (limited effect on Streamlit)
    st.markdown("""
        <style>
            /* Hide Streamlit header - includes top-right menus */
            header.st-emotion-cache-zq5bqg.ezrtsby0 {
                display: none;
            }
            /* Hide top-left menu open button */
            .st-emotion-cache-1avcm0k.e1tzin5v2 {
                display: none;
            }
            /* Center the app title */
            h1 {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)


    # Language Selector Button (Top-left corner)
    col_lang, _ = st.columns([0.1, 0.9])
    with col_lang:
        current_lang_display = f"{LANGUAGES[st.session_state.current_language]['emoji']} {st.session_state.current_language}"
        lang_options = [f"{v['emoji']} {k}" for k, v in LANGUAGES.items()]
        
        # Find the index of the selected language, default to the first option if not found
        selected_lang_index = 0 
        if current_lang_display in lang_options:
            selected_lang_index = lang_options.index(current_lang_display)

        # Correction: A meaningful value was given to the 'label' parameter and visually hidden.
        selected_lang_display = st.selectbox(
            label="Select application language", # Non-empty label
            options=lang_options,
            index=selected_lang_index,
            key="language_selector",
            help="Select application language",
            label_visibility="hidden" # Visually hide the label
        )
        
        new_lang_code = selected_lang_display.split(" ")[1] 
        if new_lang_code != st.session_state.current_language:
            st.session_state.current_language = new_lang_code
            st.rerun()

    # If profile information is not entered, show the initial screen
    if st.session_state.user_name == "":
        display_welcome_and_profile_setup()
    else:
        # Main title of the application
        st.markdown("<h1 style='text-align: center;'>Hanogt AI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: gray;'>{get_text('welcome_subtitle')}</h4>", unsafe_allow_html=True)
        st.write("---")

        display_main_chat_interface()

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; font-size: 12px; color: gray;">
            {get_text('footer_user').format(user_name=st.session_state.user_name if st.session_state.user_name else "Guest")} <br>
            {get_text('footer_version').format(year=datetime.datetime.now().year)} <br>
            {get_text('footer_ai_status').format(model_name=GLOBAL_MODEL_NAME)}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
