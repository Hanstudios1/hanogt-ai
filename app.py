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
import json # json modülünü ekledik

# --- İsteğe Bağlı Kütüphaneler (Platforma özel kurulum gerektirebilir) ---
try:
    import pyttsx3
    import speech_recognition as sr
    TTS_SR_AVAILABLE = True
except ImportError:
    TTS_SR_AVAILABLE = False
    logging.warning("pyttsx3 veya speech_recognition modülleri bulunamadı. Sesli özellikler devre dışı bırakıldı.")

# --- Global Değişkenler ve Ayarlar ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Anahtarı Kontrolü
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if st.secrets else os.environ.get("GOOGLE_API_KEY")

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
    # Tüm diller için metinleri içeren ana sözlük
    texts = {
        # ... (Mevcut diller: TR, EN, FR, ES, DE, RU, SA, AZ, JP, KR)
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
        # ... Diğer mevcut diller buraya gelecek
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
            "chat_mode_voice": "🎤 Chat de Voz (Carregar Arquivo)",
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
            "voice_chat_title": "Chat de Voz",
            "voice_upload_label": "Carregar arquivo de áudio (MP3, WAV)",
            "voice_upload_warning": "O recurso de transcrição de arquivo de áudio é atualmente um placeholder.",
            "voice_live_input_title": "Entrada de Voz ao Vivo",
            "voice_mic_button": "Iniciar Microfone",
            "voice_not_available": "Recursos de chat de voz indisponíveis. Certifique-se de que as bibliotecas necessárias (pyttsx3, SpeechRecognition) estão instaladas.",
            "voice_listening": "Ouvindo...",
            "voice_heard": "Você disse: {text}",
            "voice_no_audio": "Nenhum áudio detectado, por favor, tente novamente.",
            "voice_api_error": "Não foi possível acessar o serviço de reconhecimento de fala; {error}",
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
            "source_error": "Fonte: Erro ({error})",
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
            "loading_audio_file": "Carregando arquivo de áudio...",
            "tts_sr_not_available": "Os recursos de chat de voz e conversão de texto em fala não estão disponíveis. Certifique-se de que as bibliotecas necessárias estão instaladas.",
            "mic_listen_timeout": "Tempo de detecção de áudio esgotado.",
            "unexpected_audio_record_error": "Ocorreu um erro inesperado durante a gravação de áudio: {error}",
            "gemini_response_error": "Ocorreu um erro inesperado ao obter uma resposta: {error}",
            "creative_text_generated": "Texto Criativo Gerado: {text}",
            "turkish_voice_not_found": "Voz em turco não encontrada, será usada a voz padrão. Verifique as configurações de som do seu sistema operacional."
        },
        "CA": { # Fransızca (Kanada)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Votre Nouvel Assistant IA Personnel!",
            "profile_title": "Comment devrais-je vous appeler?",
            "profile_name_label": "Votre nom :",
            "profile_upload_label": "Téléverser une photo de profil (optionnel)",
            "profile_save_button": "Sauvegarder",
            "profile_greeting": "Bonjour, {name}!",
            "profile_edit_info": "Vous pouvez modifier votre profil dans la section Paramètres et Personnalisation.",
            "ai_features_title": "Fonctionnalités de Hanogt AI :",
            "feature_general_chat": "Clavardage général",
            "feature_web_search": "Recherche Web (DuckDuckGo, Wikipédia)",
            "feature_knowledge_base": "Réponses de la base de connaissances",
            "feature_creative_text": "Génération de texte créatif",
            "feature_image_generation": "Génération d'image simple (exemple)",
            "feature_text_to_speech": "Synthèse vocale (TTS)",
            "feature_feedback": "Mécanisme de rétroaction",
            "settings_button": "⚙️ Paramètres & Personnalisation",
            "about_button": "ℹ️ À Propos",
            "app_mode_title": "Mode de l'Application",
            "chat_mode_text": "💬 Clavardage Écrit",
            "chat_mode_image": "🖼️ Générateur d'Images",
            "chat_mode_voice": "🎤 Clavardage Vocal (Téléverser Fichier)",
            "chat_mode_creative": "✨ Studio Créatif",
            "chat_input_placeholder": "Tapez votre message ou une commande : Ex: 'Bonjour', 'recherche web: Streamlit', 'texte créatif: extraterrestres'...",
            "generating_response": "Génération de la réponse...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Merci pour vos commentaires!",
            "image_gen_title": "Générateur d'Images",
            "image_gen_input_label": "Décrivez l'image que vous voulez créer :",
            "image_gen_button": "Générer l'Image",
            "image_gen_warning_placeholder": "La fonction de génération d'images est actuellement un exemple et n'est pas connectée à une véritable API.",
            "image_gen_warning_prompt_missing": "Veuillez entrer une description d'image.",
            "voice_chat_title": "Clavardage Vocal",
            "voice_upload_label": "Téléverser un fichier audio (MP3, WAV)",
            "voice_upload_warning": "La fonction de transcription de fichier audio est actuellement un exemple.",
            "voice_live_input_title": "Entrée Vocale en Direct",
            "voice_mic_button": "Démarrer le Microphone",
            "voice_not_available": "Les fonctions de clavardage vocal sont indisponibles. Assurez-vous que les bibliothèques requises (pyttsx3, SpeechRecognition) sont installées.",
            "voice_listening": "Écoute en cours...",
            "voice_heard": "Vous avez dit : {text}",
            "voice_no_audio": "Aucun son détecté, veuillez réessayer.",
            "voice_api_error": "Impossible de joindre le service de reconnaissance vocale; {error}",
            "creative_studio_title": "Studio Créatif",
            "creative_studio_info": "Cette section est conçue pour des fonctionnalités avancées comme la génération de texte créatif.",
            "creative_studio_input_label": "Entrez votre demande de texte créatif :",
            "creative_studio_button": "Générer du Texte",
            "creative_studio_warning_prompt_missing": "Veuillez entrer une demande de texte créatif.",
            "settings_personalization_title": "Paramètres & Personnalisation",
            "settings_name_change_label": "Changer votre nom :",
            "settings_avatar_change_label": "Changer la photo de profil (optionnel)",
            "settings_update_profile_button": "Mettre à jour les informations du profil",
            "settings_profile_updated_toast": "Profil mis à jour!",
            "settings_chat_management_title": "Gestion du Clavardage",
            "settings_clear_chat_button": "🧹 Effacer l'historique du clavardage actif",
            "about_us_title": "ℹ️ À Propos de Nous",
            "about_us_text": "Hanogt AI a été créé par Oğuz Han Guluzade, propriétaire de HanStudios, en 2025. C'est un projet à code source ouvert, entraîné par Gemini, et tous les droits d'auteur sont réservés.",
            "footer_user": "Utilisateur : {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA : Active ({model_name}) | Journal : Actif",
            "model_init_success": "Modèle Gemini initialisé avec succès!",
            "model_init_error": "Une erreur s'est produite lors de l'initialisation du modèle Gemini : {error}. Veuillez vérifier que votre clé API est correcte et active.",
            "gemini_model_not_initialized": "Modèle Gemini non initialisé. Veuillez vérifier votre clé API.",
            "image_load_error": "Impossible de charger l'image : {error}",
            "image_not_convertible": "Ce contenu ne peut pas être converti en parole (non textuel).",
            "duckduckgo_error": "Une erreur s'est produite lors de la recherche DuckDuckGo : {error}",
            "wikipedia_network_error": "Erreur réseau lors de la recherche Wikipédia : {error}",
            "wikipedia_json_error": "Erreur lors de l'analyse de la réponse Wikipédia : {error}",
            "wikipedia_general_error": "Une erreur générale s'est produite lors de la recherche Wikipédia : {error}",
            "unexpected_response_error": "Une erreur inattendue s'est produite lors de l'obtention d'une réponse : {error}",
            "source_error": "Source : Erreur ({error})",
            "chat_cleared_toast": "Clavardage actif effacé!",
            "profile_image_load_error": "Impossible de charger l'image de profil : {error}",
            "web_search_results": "Résultats de la recherche Web :",
            "web_search_no_results": "Aucun résultat trouvé pour votre terme de recherche.",
            "wikipedia_search_results": "Résultats de la recherche Wikipédia :",
            "wikipedia_search_no_results": "Aucun résultat trouvé pour votre terme de recherche.",
            "image_generated_example": "Une image pour '{prompt}' a été générée (exemple).",
            "image_upload_caption": "Image Téléversée",
            "image_processing_error": "Une erreur s'est produite lors du traitement de l'image : {error}",
            "image_vision_query": "Que voyez-vous dans cette image?",
            "loading_audio_file": "Chargement du fichier audio...",
            "tts_sr_not_available": "Les fonctions de clavardage vocal et de synthèse vocale sont indisponibles. Assurez-vous que les bibliothèques requises sont installées.",
            "mic_listen_timeout": "Le temps d'attente pour la détection audio est écoulé.",
            "unexpected_audio_record_error": "Une erreur inattendue s'est produite lors de l'enregistrement audio : {error}",
            "gemini_response_error": "Une erreur inattendue s'est produite lors de l'obtention d'une réponse : {error}",
            "creative_text_generated": "Texte Créatif Généré : {text}",
            "turkish_voice_not_found": "Voix turque non trouvée, la voix par défaut sera utilisée. Veuillez vérifier les paramètres sonores de votre système d'exploitation."
        },
        "MX": { # İspanyolca (Meksika)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "¡Tu Nuevo Asistente Personal de IA!",
            "profile_title": "¿Cómo debo llamarte?",
            "profile_name_label": "Tu Nombre:",
            "profile_upload_label": "Subir Foto de Perfil (opcional)",
            "profile_save_button": "Guardar",
            "profile_greeting": "¡Hola, {name}!",
            "profile_edit_info": "Puedes editar tu perfil en la sección de Configuración y Personalización.",
            "ai_features_title": "Características de Hanogt AI:",
            "feature_general_chat": "Chat general",
            "feature_web_search": "Búsqueda web (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Respuestas de la base de conocimientos",
            "feature_creative_text": "Generación de texto creativo",
            "feature_image_generation": "Generación de imagen simple (ejemplo)",
            "feature_text_to_speech": "Texto a voz (TTS)",
            "feature_feedback": "Mecanismo de retroalimentación",
            "settings_button": "⚙️ Configuración y Personalización",
            "about_button": "ℹ️ Acerca de Nosotros",
            "app_mode_title": "Modo de la Aplicación",
            "chat_mode_text": "💬 Chat de Texto",
            "chat_mode_image": "🖼️ Generador de Imágenes",
            "chat_mode_voice": "🎤 Chat de Voz (Subir Archivo)",
            "chat_mode_creative": "✨ Estudio Creativo",
            "chat_input_placeholder": "Escribe tu mensaje o un comando: Ej: 'Hola', 'búsqueda web: Streamlit', 'texto creativo: alienígenas'...",
            "generating_response": "Generando respuesta...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "¡Gracias por tus comentarios!",
            "image_gen_title": "Generador de Imágenes",
            "image_gen_input_label": "Describe la imagen que quieres crear:",
            "image_gen_button": "Generar Imagen",
            "image_gen_warning_placeholder": "La función de generación de imágenes es actualmente un ejemplo y no está conectada a una API real.",
            "image_gen_warning_prompt_missing": "Por favor, ingresa una descripción de la imagen.",
            "voice_chat_title": "Chat de Voz",
            "voice_upload_label": "Subir archivo de audio (MP3, WAV)",
            "voice_upload_warning": "La función de transcripción de archivos de audio es actualmente un ejemplo.",
            "voice_live_input_title": "Entrada de Voz en Vivo",
            "voice_mic_button": "Iniciar Micrófono",
            "voice_not_available": "Las funciones de chat de voz no están disponibles. Asegúrate de que las librerías necesarias (pyttsx3, SpeechRecognition) estén instaladas.",
            "voice_listening": "Escuchando...",
            "voice_heard": "Dijiste: {text}",
            "voice_no_audio": "No se detectó audio, por favor, inténtalo de nuevo.",
            "voice_api_error": "No se pudo contactar al servicio de reconocimiento de voz; {error}",
            "creative_studio_title": "Estudio Creativo",
            "creative_studio_info": "Esta sección está diseñada para funciones avanzadas como la generación de texto creativo.",
            "creative_studio_input_label": "Ingresa tu solicitud de texto creativo:",
            "creative_studio_button": "Generar Texto",
            "creative_studio_warning_prompt_missing": "Por favor, ingresa una solicitud de texto creativo.",
            "settings_personalization_title": "Configuración y Personalización",
            "settings_name_change_label": "Cambiar tu Nombre:",
            "settings_avatar_change_label": "Cambiar Foto de Perfil (opcional)",
            "settings_update_profile_button": "Actualizar Información del Perfil",
            "settings_profile_updated_toast": "¡Perfil actualizado!",
            "settings_chat_management_title": "Gestión de Chat",
            "settings_clear_chat_button": "🧹 Limpiar Historial de Chat Activo",
            "about_us_title": "ℹ️ Acerca de Nosotros",
            "about_us_text": "Hanogt AI fue creado por Oğuz Han Guluzade, dueño de HanStudios, en 2025. Es de código abierto, entrenado por Gemini, y todos los derechos de autor están reservados.",
            "footer_user": "Usuario: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA: Activa ({model_name}) | Log: Activo",
            "model_init_success": "¡Modelo Gemini iniciado con éxito!",
            "model_init_error": "Ocurrió un error al iniciar el modelo Gemini: {error}. Por favor, asegúrate de que tu clave de API es correcta y está activa.",
            "gemini_model_not_initialized": "Modelo Gemini no iniciado. Por favor, revisa tu clave de API.",
            "image_load_error": "No se pudo cargar la imagen: {error}",
            "image_not_convertible": "Este contenido no se puede convertir a voz (no es texto).",
            "duckduckgo_error": "Ocurrió un error al realizar la búsqueda en DuckDuckGo: {error}",
            "wikipedia_network_error": "Ocurrió un error de red al buscar en Wikipedia: {error}",
            "wikipedia_json_error": "Ocurrió un error al procesar la respuesta de Wikipedia: {error}",
            "wikipedia_general_error": "Ocurrió un error general al buscar en Wikipedia: {error}",
            "unexpected_response_error": "Ocurrió un error inesperado al obtener una respuesta: {error}",
            "source_error": "Fuente: Error ({error})",
            "chat_cleared_toast": "¡Chat activo limpiado!",
            "profile_image_load_error": "No se pudo cargar la imagen de perfil: {error}",
            "web_search_results": "Resultados de Búsqueda Web:",
            "web_search_no_results": "No se encontraron resultados para tu término de búsqueda.",
            "wikipedia_search_results": "Resultados de Búsqueda de Wikipedia:",
            "wikipedia_search_no_results": "No se encontraron resultados para tu término de búsqueda.",
            "image_generated_example": "Se generó una imagen para '{prompt}' (ejemplo).",
            "image_upload_caption": "Imagen Subida",
            "image_processing_error": "Ocurrió un error al procesar la imagen: {error}",
            "image_vision_query": "¿Qué ves en esta imagen?",
            "loading_audio_file": "Cargando archivo de audio...",
            "tts_sr_not_available": "Las funciones de chat de voz y texto a voz no están disponibles. Asegúrate de que las librerías necesarias estén instaladas.",
            "mic_listen_timeout": "Se agotó el tiempo de espera para la detección de audio.",
            "unexpected_audio_record_error": "Ocurrió un error inesperado durante la grabación de audio: {error}",
            "gemini_response_error": "Ocurrió un error inesperado al obtener una respuesta: {error}",
            "creative_text_generated": "Texto Creativo Generado: {text}",
            "turkish_voice_not_found": "No se encontró voz en turco, se usará la voz predeterminada. Revisa la configuración de sonido de tu sistema operativo."
        },
        "AR": { # İspanyolca (Arjantin)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "¡Tu Nuevo Asistente Personal de IA!",
            "profile_title": "¿Cómo debería llamarte?",
            "profile_name_label": "Tu Nombre:",
            "profile_upload_label": "Subir Foto de Perfil (opcional)",
            "profile_save_button": "Guardar",
            "profile_greeting": "¡Hola, {name}!",
            "profile_edit_info": "Podés editar tu perfil en la sección de Configuración y Personalización.",
            "ai_features_title": "Características de Hanogt AI:",
            "feature_general_chat": "Chat general",
            "feature_web_search": "Búsqueda web (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Respuestas de la base de conocimientos",
            "feature_creative_text": "Generación de texto creativo",
            "feature_image_generation": "Generación de imagen simple (ejemplo)",
            "feature_text_to_speech": "Texto a voz (TTS)",
            "feature_feedback": "Mecanismo de opiniones",
            "settings_button": "⚙️ Configuración y Personalización",
            "about_button": "ℹ️ Sobre Nosotros",
            "app_mode_title": "Modo de la Aplicación",
            "chat_mode_text": "💬 Chat de Texto",
            "chat_mode_image": "🖼️ Generador de Imágenes",
            "chat_mode_voice": "🎤 Chat de Voz (Subir Archivo)",
            "chat_mode_creative": "✨ Estudio Creativo",
            "chat_input_placeholder": "Escribí tu mensaje o un comando: Ej: 'Hola', 'buscar en web: Streamlit', 'texto creativo: alienígenas'...",
            "generating_response": "Generando respuesta...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "¡Gracias por tus comentarios!",
            "image_gen_title": "Generador de Imágenes",
            "image_gen_input_label": "Describí la imagen que querés crear:",
            "image_gen_button": "Generar Imagen",
            "image_gen_warning_placeholder": "La función de generación de imágenes es actualmente un ejemplo y no está conectada a una API real.",
            "image_gen_warning_prompt_missing": "Por favor, ingresá una descripción de la imagen.",
            "voice_chat_title": "Chat de Voz",
            "voice_upload_label": "Subir archivo de audio (MP3, WAV)",
            "voice_upload_warning": "La función de transcripción de archivos de audio es actualmente un ejemplo.",
            "voice_live_input_title": "Entrada de Voz en Vivo",
            "voice_mic_button": "Iniciar Micrófono",
            "voice_not_available": "Las funciones de chat de voz no están disponibles. Asegurate de que las librerías necesarias (pyttsx3, SpeechRecognition) estén instaladas.",
            "voice_listening": "Escuchando...",
            "voice_heard": "Dijiste: {text}",
            "voice_no_audio": "No se detectó audio, por favor, intentá de nuevo.",
            "voice_api_error": "No se pudo contactar al servicio de reconocimiento de voz; {error}",
            "creative_studio_title": "Estudio Creativo",
            "creative_studio_info": "Esta sección está diseñada para funciones avanzadas como la generación de texto creativo.",
            "creative_studio_input_label": "Ingresá tu pedido de texto creativo:",
            "creative_studio_button": "Generar Texto",
            "creative_studio_warning_prompt_missing": "Por favor, ingresá un pedido de texto creativo.",
            "settings_personalization_title": "Configuración y Personalización",
            "settings_name_change_label": "Cambiar tu Nombre:",
            "settings_avatar_change_label": "Cambiar Foto de Perfil (opcional)",
            "settings_update_profile_button": "Actualizar Información del Perfil",
            "settings_profile_updated_toast": "¡Perfil actualizado!",
            "settings_chat_management_title": "Gestión de Chat",
            "settings_clear_chat_button": "🧹 Limpiar Historial de Chat Activo",
            "about_us_title": "ℹ️ Sobre Nosotros",
            "about_us_text": "Hanogt AI fue creado por Oğuz Han Guluzade, dueño de HanStudios, en 2025. Es de código abierto, entrenado por Gemini, y todos los derechos de autor están reservados.",
            "footer_user": "Usuario: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA: Activa ({model_name}) | Log: Activo",
            "model_init_success": "¡Modelo Gemini iniciado con éxito!",
            "model_init_error": "Ocurrió un error al iniciar el modelo Gemini: {error}. Por favor, asegurate de que tu clave de API sea correcta y esté activa.",
            "gemini_model_not_initialized": "Modelo Gemini no iniciado. Por favor, revisá tu clave de API.",
            "image_load_error": "No se pudo cargar la imagen: {error}",
            "image_not_convertible": "Este contenido no se puede convertir a voz (no es texto).",
            "duckduckgo_error": "Ocurrió un error al realizar la búsqueda en DuckDuckGo: {error}",
            "wikipedia_network_error": "Ocurrió un error de red al buscar en Wikipedia: {error}",
            "wikipedia_json_error": "Ocurrió un error al procesar la respuesta de Wikipedia: {error}",
            "wikipedia_general_error": "Ocurrió un error general al buscar en Wikipedia: {error}",
            "unexpected_response_error": "Ocurrió un error inesperado al obtener una respuesta: {error}",
            "source_error": "Fuente: Error ({error})",
            "chat_cleared_toast": "¡Chat activo limpiado!",
            "profile_image_load_error": "No se pudo cargar la imagen de perfil: {error}",
            "web_search_results": "Resultados de Búsqueda Web:",
            "web_search_no_results": "No se encontraron resultados para tu término de búsqueda.",
            "wikipedia_search_results": "Resultados de Búsqueda de Wikipedia:",
            "wikipedia_search_no_results": "No se encontraron resultados para tu término de búsqueda.",
            "image_generated_example": "Se generó una imagen para '{prompt}' (ejemplo).",
            "image_upload_caption": "Imagen Subida",
            "image_processing_error": "Ocurrió un error al procesar la imagen: {error}",
            "image_vision_query": "¿Qué ves en esta imagen?",
            "loading_audio_file": "Cargando archivo de audio...",
            "tts_sr_not_available": "Las funciones de chat de voz y texto a voz no están disponibles. Asegurate de que las librerías necesarias estén instaladas.",
            "mic_listen_timeout": "Se agotó el tiempo de espera para la detección de audio.",
            "unexpected_audio_record_error": "Ocurrió un error inesperado durante la grabación de audio: {error}",
            "gemini_response_error": "Ocurrió un error inesperado al obtener una respuesta: {error}",
            "creative_text_generated": "Texto Creativo Generado: {text}",
            "turkish_voice_not_found": "No se encontró voz en turco, se usará la voz predeterminada. Revisa la configuración de sonido de tu sistema operativo."
        },
        "PT": { # Portekizce (Portekiz)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "O Seu Novo Assistente Pessoal de IA!",
            "profile_title": "Como devo tratar-te?",
            "profile_name_label": "O seu Nome:",
            "profile_upload_label": "Carregar Foto de Perfil (opcional)",
            "profile_save_button": "Guardar",
            "profile_greeting": "Olá, {name}!",
            "profile_edit_info": "Pode editar o seu perfil na secção de Definições e Personalização.",
            "ai_features_title": "Funcionalidades do Hanogt AI:",
            "feature_general_chat": "Chat geral",
            "feature_web_search": "Pesquisa na web (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Respostas da base de conhecimento",
            "feature_creative_text": "Geração de texto criativo",
            "feature_image_generation": "Geração de imagem simples (exemplo)",
            "feature_text_to_speech": "Texto para fala (TTS)",
            "feature_feedback": "Mecanismo de feedback",
            "settings_button": "⚙️ Definições e Personalização",
            "about_button": "ℹ️ Sobre Nós",
            "app_mode_title": "Modo da Aplicação",
            "chat_mode_text": "💬 Chat de Texto",
            "chat_mode_image": "🖼️ Gerador de Imagens",
            "chat_mode_voice": "🎤 Chat de Voz (Carregar Ficheiro)",
            "chat_mode_creative": "✨ Estúdio Criativo",
            "chat_input_placeholder": "Escreva a sua mensagem ou um comando: Ex: 'Olá', 'pesquisa web: Streamlit', 'texto criativo: alienígenas'...",
            "generating_response": "A gerar resposta...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Obrigado pelo seu feedback!",
            "image_gen_title": "Gerador de Imagens",
            "image_gen_input_label": "Descreva a imagem que quer criar:",
            "image_gen_button": "Gerar Imagem",
            "image_gen_warning_placeholder": "A funcionalidade de geração de imagens é atualmente um exemplo e não está ligada a uma API real.",
            "image_gen_warning_prompt_missing": "Por favor, insira uma descrição da imagem.",
            "voice_chat_title": "Chat de Voz",
            "voice_upload_label": "Carregar ficheiro de áudio (MP3, WAV)",
            "voice_upload_warning": "A funcionalidade de transcrição de ficheiro de áudio é atualmente um exemplo.",
            "voice_live_input_title": "Entrada de Voz ao Vivo",
            "voice_mic_button": "Iniciar Microfone",
            "voice_not_available": "As funcionalidades de chat de voz estão indisponíveis. Certifique-se de que as bibliotecas necessárias (pyttsx3, SpeechRecognition) estão instaladas.",
            "voice_listening": "A ouvir...",
            "voice_heard": "Disse: {text}",
            "voice_no_audio": "Nenhum áudio detetado, por favor, tente novamente.",
            "voice_api_error": "Não foi possível contactar o serviço de reconhecimento de voz; {error}",
            "creative_studio_title": "Estúdio Criativo",
            "creative_studio_info": "Esta secção destina-se a funcionalidades avançadas como a geração de texto criativo.",
            "creative_studio_input_label": "Insira o seu pedido de texto criativo:",
            "creative_studio_button": "Gerar Texto",
            "creative_studio_warning_prompt_missing": "Por favor, insira um pedido de texto criativo.",
            "settings_personalization_title": "Definições e Personalização",
            "settings_name_change_label": "Mudar o Seu Nome:",
            "settings_avatar_change_label": "Mudar Foto de Perfil (opcional)",
            "settings_update_profile_button": "Atualizar Informações de Perfil",
            "settings_profile_updated_toast": "Perfil atualizado!",
            "settings_chat_management_title": "Gestão de Chat",
            "settings_clear_chat_button": "🧹 Limpar Histórico de Chat Ativo",
            "about_us_title": "ℹ️ Sobre Nós",
            "about_us_text": "O Hanogt AI foi criado por Oğuz Han Guluzade, proprietário da HanStudios, em 2025. É de código aberto, treinado pelo Gemini, e todos os direitos de autor estão reservados.",
            "footer_user": "Utilizador: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "IA: Ativa ({model_name}) | Registo: Ativo",
            "model_init_success": "Modelo Gemini iniciado com sucesso!",
            "model_init_error": "Ocorreu um erro ao iniciar o modelo Gemini: {error}. Por favor, certifique-se de que a sua chave de API está correta e ativa.",
            "gemini_model_not_initialized": "Modelo Gemini não iniciado. Por favor, verifique a sua chave de API.",
            "image_load_error": "Não foi possível carregar a imagem: {error}",
            "image_not_convertible": "Este conteúdo não pode ser convertido para fala (não é texto).",
            "duckduckgo_error": "Ocorreu um erro ao realizar a pesquisa no DuckDuckGo: {error}",
            "wikipedia_network_error": "Ocorreu um erro de rede ao pesquisar na Wikipedia: {error}",
            "wikipedia_json_error": "Ocorreu um erro ao processar a resposta da Wikipedia: {error}",
            "wikipedia_general_error": "Ocorreu um erro geral ao pesquisar na Wikipedia: {error}",
            "unexpected_response_error": "Ocorreu um erro inesperado ao obter uma resposta: {error}",
            "source_error": "Fonte: Erro ({error})",
            "chat_cleared_toast": "Chat ativo limpo!",
            "profile_image_load_error": "Não foi possível carregar a imagem de perfil: {error}",
            "web_search_results": "Resultados da Pesquisa na Web:",
            "web_search_no_results": "Nenhum resultado encontrado para o seu termo de pesquisa.",
            "wikipedia_search_results": "Resultados da Pesquisa na Wikipedia:",
            "wikipedia_search_no_results": "Nenhum resultado encontrado para o seu termo de pesquisa.",
            "image_generated_example": "Uma imagem para '{prompt}' foi gerada (exemplo).",
            "image_upload_caption": "Imagem Carregada",
            "image_processing_error": "Ocorreu um erro ao processar a imagem: {error}",
            "image_vision_query": "O que vê nesta imagem?",
            "loading_audio_file": "A carregar ficheiro de áudio...",
            "tts_sr_not_available": "As funcionalidades de chat de voz e texto para fala estão indisponíveis. Certifique-se de que as bibliotecas necessárias estão instaladas.",
            "mic_listen_timeout": "O tempo de deteção de áudio esgotou-se.",
            "unexpected_audio_record_error": "Ocorreu um erro inesperado durante a gravação de áudio: {error}",
            "gemini_response_error": "Ocorreu um erro inesperado ao obter uma resposta: {error}",
            "creative_text_generated": "Texto Criativo Gerado: {text}",
            "turkish_voice_not_found": "Voz em turco não encontrada, será usada a voz predefinida. Verifique as definições de som do seu sistema operativo."
        },
        "CN": { # Çince (Basitleştirilmiş)
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "您的新个人AI助手！",
            "profile_title": "我该如何称呼您？",
            "profile_name_label": "您的名字：",
            "profile_upload_label": "上传个人资料图片（可选）",
            "profile_save_button": "保存",
            "profile_greeting": "您好，{name}！",
            "profile_edit_info": "您可以在“设置与个性化”部分编辑您的个人资料。",
            "ai_features_title": "Hanogt AI 功能：",
            "feature_general_chat": "一般聊天",
            "feature_web_search": "网络搜索（DuckDuckGo, 维基百科）",
            "feature_knowledge_base": "知识库回答",
            "feature_creative_text": "创意文本生成",
            "feature_image_generation": "简单图像生成（示例）",
            "feature_text_to_speech": "文本转语音（TTS）",
            "feature_feedback": "反馈机制",
            "settings_button": "⚙️ 设置与个性化",
            "about_button": "ℹ️ 关于我们",
            "app_mode_title": "应用模式",
            "chat_mode_text": "💬 文字聊天",
            "chat_mode_image": "🖼️ 图像生成器",
            "chat_mode_voice": "🎤 语音聊天（上传文件）",
            "chat_mode_creative": "✨ 创意工作室",
            "chat_input_placeholder": "输入您的消息或命令：例如：“你好”、“网络搜索：Streamlit”、“创意文本：外星人”...",
            "generating_response": "正在生成回应...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "感谢您的反馈！",
            "image_gen_title": "图像生成器",
            "image_gen_input_label": "描述您想创建的图像：",
            "image_gen_button": "生成图像",
            "image_gen_warning_placeholder": "图像生成功能目前是一个占位符，并未连接到真实的API。",
            "image_gen_warning_prompt_missing": "请输入图像描述。",
            "voice_chat_title": "语音聊天",
            "voice_upload_label": "上传音频文件（MP3, WAV）",
            "voice_upload_warning": "音频文件转录功能目前是一个占位符。",
            "voice_live_input_title": "实时语音输入",
            "voice_mic_button": "启动麦克风",
            "voice_not_available": "语音聊天功能不可用。请确保已安装所需的库（pyttsx3, SpeechRecognition）。",
            "voice_listening": "正在聆听...",
            "voice_heard": "您说：{text}",
            "voice_no_audio": "未检测到音频，请重试。",
            "voice_api_error": "无法连接到语音识别服务；{error}",
            "creative_studio_title": "创意工作室",
            "creative_studio_info": "本部分设计用于创意文本生成等高级功能。",
            "creative_studio_input_label": "输入您的创意文本请求：",
            "creative_studio_button": "生成文本",
            "creative_studio_warning_prompt_missing": "请输入创意文本请求。",
            "settings_personalization_title": "设置与个性化",
            "settings_name_change_label": "更改您的姓名：",
            "settings_avatar_change_label": "更改个人资料图片（可选）",
            "settings_update_profile_button": "更新个人资料信息",
            "settings_profile_updated_toast": "个人资料已更新！",
            "settings_chat_management_title": "聊天管理",
            "settings_clear_chat_button": "🧹 清除当前聊天记录",
            "about_us_title": "ℹ️ 关于我们",
            "about_us_text": "Hanogt AI 由 HanStudios 的所有者 Oğuz Han Guluzade 于 2025 年创建。它由 Gemini 训练，是开源的，并保留所有版权。",
            "footer_user": "用户：{user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI：活动（{model_name}）| 日志：活动",
            "model_init_success": "Gemini 模型初始化成功！",
            "model_init_error": "初始化 Gemini 模型时出错：{error}。请确保您的 API 密钥正确且有效。",
            "gemini_model_not_initialized": "Gemini 模型未初始化。请检查您的 API 密钥。",
            "image_load_error": "无法加载图像：{error}",
            "image_not_convertible": "此内容无法转换为语音（不是文本）。",
            "duckduckgo_error": "执行 DuckDuckGo 搜索时出错：{error}",
            "wikipedia_network_error": "执行维基百科搜索时出现网络错误：{error}",
            "wikipedia_json_error": "解析维基百科响应时出错：{error}",
            "wikipedia_general_error": "执行维基百科搜索时出现一般性错误：{error}",
            "unexpected_response_error": "获取响应时发生意外错误：{error}",
            "source_error": "来源：错误 ({error})",
            "chat_cleared_toast": "当前聊天已清除！",
            "profile_image_load_error": "无法加载个人资料图片：{error}",
            "web_search_results": "网络搜索结果：",
            "web_search_no_results": "未找到与您搜索词相关的结果。",
            "wikipedia_search_results": "维基百科搜索结果：",
            "wikipedia_search_no_results": "未找到与您搜索词相关的结果。",
            "image_generated_example": "已为“{prompt}”生成图像（示例）。",
            "image_upload_caption": "上传的图像",
            "image_processing_error": "处理图像时出错：{error}",
            "image_vision_query": "您在这张图片中看到了什么？",
            "loading_audio_file": "正在加载音频文件...",
            "tts_sr_not_available": "语音聊天和文本转语音功能不可用。请确保已安装所需的库。",
            "mic_listen_timeout": "音频检测超时。",
            "unexpected_audio_record_error": "录音期间发生意外错误：{error}",
            "gemini_response_error": "获取响应时发生意外错误：{error}",
            "creative_text_generated": "创意文本已生成：{text}",
            "turkish_voice_not_found": "未找到土耳其语语音，将使用默认语音。请检查您操作系统的声音设置。"
        },
        "IN": { # Hintçe
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "आपका नया व्यक्तिगत एआई सहायक!",
            "profile_title": "मैं आपको कैसे संबोधित करूं?",
            "profile_name_label": "आपका नाम:",
            "profile_upload_label": "प्रोफ़ाइल चित्र अपलोड करें (वैकल्पिक)",
            "profile_save_button": "सहेजें",
            "profile_greeting": "नमस्ते, {name}!",
            "profile_edit_info": "आप सेटिंग्स और वैयक्तिकरण अनुभाग में अपनी प्रोफ़ाइल संपादित कर सकते हैं।",
            "ai_features_title": "Hanogt AI की विशेषताएं:",
            "feature_general_chat": "सामान्य चैट",
            "feature_web_search": "वेब खोज (DuckDuckGo, विकिपीडिया)",
            "feature_knowledge_base": "ज्ञान आधार प्रतिक्रियाएँ",
            "feature_creative_text": "रचनात्मक पाठ निर्माण",
            "feature_image_generation": "सरल छवि निर्माण (उदाहरण)",
            "feature_text_to_speech": "टेक्स्ट-टू-स्पीच (TTS)",
            "feature_feedback": "प्रतिक्रिया तंत्र",
            "settings_button": "⚙️ सेटिंग्स और वैयक्तिकरण",
            "about_button": "ℹ️ हमारे बारे में",
            "app_mode_title": "एप्लिकेशन मोड",
            "chat_mode_text": "💬 टेक्स्ट चैट",
            "chat_mode_image": "🖼️ छवि जेनरेटर",
            "chat_mode_voice": "🎤 वॉयस चैट (फ़ाइल अपलोड करें)",
            "chat_mode_creative": "✨ क्रिएटिव स्टूडियो",
            "chat_input_placeholder": "अपना संदेश या कमांड टाइप करें: जैसे: 'नमस्ते', 'वेब खोज: Streamlit', 'रचनात्मक पाठ: एलियंस'...",
            "generating_response": "प्रतिक्रिया उत्पन्न हो रही है...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
            "image_gen_title": "छवि जेनरेटर",
            "image_gen_input_label": "आप जो छवि बनाना चाहते हैं उसका वर्णन करें:",
            "image_gen_button": "छवि बनाएं",
            "image_gen_warning_placeholder": "छवि निर्माण सुविधा वर्तमान में एक प्लेसहोल्डर है और वास्तविक एपीआई से जुड़ी नहीं है।",
            "image_gen_warning_prompt_missing": "कृपया एक छवि विवरण दर्ज करें।",
            "voice_chat_title": "वॉयस चैट",
            "voice_upload_label": "ऑडियो फ़ाइल अपलोड करें (MP3, WAV)",
            "voice_upload_warning": "ऑडियो फ़ाइल ट्रांसक्रिप्शन सुविधा वर्तमान में एक प्लेसहोल्डर है।",
            "voice_live_input_title": "लाइव वॉयस इनपुट",
            "voice_mic_button": "माइक्रोफ़ोन प्रारंभ करें",
            "voice_not_available": "वॉयस चैट सुविधाएँ अनुपलब्ध हैं। सुनिश्चित करें कि आवश्यक लाइब्रेरी (pyttsx3, SpeechRecognition) स्थापित हैं।",
            "voice_listening": "सुन रहा है...",
            "voice_heard": "आपने कहा: {text}",
            "voice_no_audio": "कोई ऑडियो नहीं मिला, कृपया पुनः प्रयास करें।",
            "voice_api_error": "वाक् पहचान सेवा तक नहीं पहुंच सका; {error}",
            "creative_studio_title": "क्रिएटिव स्टूडियो",
            "creative_studio_info": "यह अनुभाग रचनात्मक पाठ निर्माण जैसी उन्नत सुविधाओं के लिए डिज़ाइन किया गया है।",
            "creative_studio_input_label": "अपना रचनात्मक पाठ अनुरोध दर्ज करें:",
            "creative_studio_button": "पाठ बनाएं",
            "creative_studio_warning_prompt_missing": "कृपया एक रचनात्मक पाठ अनुरोध दर्ज करें।",
            "settings_personalization_title": "सेटिंग्स और वैयक्तिकरण",
            "settings_name_change_label": "अपना नाम बदलें:",
            "settings_avatar_change_label": "प्रोफ़ाइल चित्र बदलें (वैकल्पिक)",
            "settings_update_profile_button": "प्रोफ़ाइल जानकारी अपडेट करें",
            "settings_profile_updated_toast": "प्रोफ़ाइल अपडेट की गई!",
            "settings_chat_management_title": "चैट प्रबंधन",
            "settings_clear_chat_button": "🧹 सक्रिय चैट इतिहास साफ़ करें",
            "about_us_title": "ℹ️ हमारे बारे में",
            "about_us_text": "Hanogt AI को 2025 में HanStudios के मालिक Oğuz Han Guluzade द्वारा बनाया गया था। यह ओपन-सोर्स है, जेमिनी द्वारा प्रशिक्षित है, और सभी कॉपीराइट आरक्षित हैं।",
            "footer_user": "उपयोगकर्ता: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: सक्रिय ({model_name}) | लॉग: सक्रिय",
            "model_init_success": "जेमिनी मॉडल सफलतापूर्वक प्रारंभ हो गया!",
            "model_init_error": "जेमिनी मॉडल प्रारंभ करते समय एक त्रुटि हुई: {error}। कृपया सुनिश्चित करें कि आपकी एपीआई कुंजी सही और सक्रिय है।",
            "gemini_model_not_initialized": "जेमिनी मॉडल प्रारंभ नहीं हुआ। कृपया अपनी एपीआई कुंजी जांचें।",
            "image_load_error": "छवि लोड नहीं हो सकी: {error}",
            "image_not_convertible": "इस सामग्री को भाषण में परिवर्तित नहीं किया जा सकता (पाठ नहीं)।",
            "duckduckgo_error": "DuckDuckGo खोज करते समय एक त्रुटि हुई: {error}",
            "wikipedia_network_error": "विकिपीडिया खोज करते समय नेटवर्क त्रुटि हुई: {error}",
            "wikipedia_json_error": "विकिपीडिया प्रतिक्रिया को पार्स करते समय त्रुटि हुई: {error}",
            "wikipedia_general_error": "विकिपीडिया खोज करते समय एक सामान्य त्रुटि हुई: {error}",
            "unexpected_response_error": "प्रतिक्रिया प्राप्त करते समय एक अप्रत्याशित त्रुटि हुई: {error}",
            "source_error": "स्रोत: त्रुटि ({error})",
            "chat_cleared_toast": "सक्रिय चैट साफ़ हो गई!",
            "profile_image_load_error": "प्रोफ़ाइल छवि लोड नहीं हो सकी: {error}",
            "web_search_results": "वेब खोज परिणाम:",
            "web_search_no_results": "आपके खोज शब्द के लिए कोई परिणाम नहीं मिला।",
            "wikipedia_search_results": "विकिपीडिया खोज परिणाम:",
            "wikipedia_search_no_results": "आपके खोज शब्द के लिए कोई परिणाम नहीं मिला।",
            "image_generated_example": "'{prompt}' के लिए एक छवि बनाई गई (उदाहरण)।",
            "image_upload_caption": "अपलोड की गई छवि",
            "image_processing_error": "छवि को संसाधित करते समय एक त्रुटि हुई: {error}",
            "image_vision_query": "आप इस छवि में क्या देखते हैं?",
            "loading_audio_file": "ऑडियो फ़ाइल लोड हो रही है...",
            "tts_sr_not_available": "वॉयस चैट और टेक्स्ट-टू-स्पीच सुविधाएँ अनुपलब्ध हैं। सुनिश्चित करें कि आवश्यक लाइब्रेरी स्थापित हैं।",
            "mic_listen_timeout": "ऑडियो पहचान का समय समाप्त हो गया।",
            "unexpected_audio_record_error": "ऑडियो रिकॉर्डिंग के दौरान एक अप्रत्याशit त्रुटि हुई: {error}",
            "gemini_response_error": "प्रतिक्रिया प्राप्त करते समय एक अप्रत्याशित त्रुटि हुई: {error}",
            "creative_text_generated": "रचनात्मक पाठ बनाया गया: {text}",
            "turkish_voice_not_found": "तुर्की आवाज नहीं मिली, डिफ़ॉल्ट आवाज का उपयोग किया जाएगा। कृपया अपने ऑपरेटिंग सिस्टम की ध्वनि सेटिंग्स जांचें।"
        },
        "PK": { # Urduca
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "آپ کا نیا ذاتی AI اسسٹنٹ!",
            "profile_title": "میں آپ کو کیسے مخاطب کروں؟",
            "profile_name_label": "آپ کا نام:",
            "profile_upload_label": "پروفائل تصویر اپ لوڈ کریں (اختیاری)",
            "profile_save_button": "محفوظ کریں",
            "profile_greeting": "ہیلو، {name}!",
            "profile_edit_info": "آپ سیٹنگز اور پرسنلائزیشن سیکشن میں اپنی پروفائل میں ترمیم کر سکتے ہیں۔",
            "ai_features_title": "Hanogt AI کی خصوصیات:",
            "feature_general_chat": "عمومی بات چیت",
            "feature_web_search": "ویب تلاش (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "علمی مرکز کے جوابات",
            "feature_creative_text": "تخلیقی متن کی تیاری",
            "feature_image_generation": "سادہ تصویر کی تیاری (مثال)",
            "feature_text_to_speech": "ٹیکسٹ ٹو اسپیچ (TTS)",
            "feature_feedback": "تاثرات کا طریقہ کار",
            "settings_button": "⚙️ سیٹنگز اور پرسنلائزیشن",
            "about_button": "ℹ️ ہمارے بارے میں",
            "app_mode_title": "ایپلیکیشن موڈ",
            "chat_mode_text": "💬 تحریری بات چیت",
            "chat_mode_image": "🖼️ امیج جنریٹر",
            "chat_mode_voice": "🎤 صوتی بات چیت (فائل اپ لوڈ کریں)",
            "chat_mode_creative": "✨ تخلیقی اسٹوڈیو",
            "chat_input_placeholder": "اپنا پیغام یا کمانڈ ٹائپ کریں: جیسے: 'ہیلو'، 'ویب تلاش: Streamlit'، 'تخلیقی متن: خلائی مخلوق'...",
            "generating_response": "جواب تیار کیا جا رہا ہے...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "آپ کے تاثرات کا شکریہ!",
            "image_gen_title": "امیج جنریٹر",
            "image_gen_input_label": "جس تصویر کو آپ بنانا چاہتے ہیں اس کی وضاحت کریں:",
            "image_gen_button": "تصویر بنائیں",
            "image_gen_warning_placeholder": "تصویر بنانے کی خصوصیت فی الحال ایک پلیس ہولڈر ہے اور کسی حقیقی API سے منسلک نہیں ہے۔",
            "image_gen_warning_prompt_missing": "براہ کرم تصویر کی تفصیل درج کریں۔",
            "voice_chat_title": "صوتی بات چیت",
            "voice_upload_label": "آڈیو فائل اپ لوڈ کریں (MP3, WAV)",
            "voice_upload_warning": "آڈیو فائل ٹرانسکرپشن کی خصوصیت فی الحال ایک پلیس ہولڈر ہے۔",
            "voice_live_input_title": "لائیو وائس ان پٹ",
            "voice_mic_button": "مائیکروفون شروع کریں",
            "voice_not_available": "صوتی بات چیت کی خصوصیات دستیاب نہیں ہیں۔ یقینی بنائیں کہ ضروری لائبریریاں (pyttsx3, SpeechRecognition) انسٹال ہیں۔",
            "voice_listening": "سن رہا ہے...",
            "voice_heard": "آپ نے کہا: {text}",
            "voice_no_audio": "کوئی آڈیو نہیں ملا، براہ کرم دوبارہ کوشش کریں۔",
            "voice_api_error": "تقریر کی شناخت کی سروس تک رسائی ممکن نہیں؛ {error}",
            "creative_studio_title": "تخلیقی اسٹوڈیو",
            "creative_studio_info": "یہ سیکشن تخلیقی متن کی تیاری جیسی جدید خصوصیات کے لیے ڈیزائن کیا گیا ہے۔",
            "creative_studio_input_label": "اپنی تخلیقی متن کی درخواست درج کریں:",
            "creative_studio_button": "متن بنائیں",
            "creative_studio_warning_prompt_missing": "براہ کرم تخلیقی متن کی درخواست درج کریں۔",
            "settings_personalization_title": "سیٹنگز اور پرسنلائزیشن",
            "settings_name_change_label": "اپنا نام تبدیل کریں:",
            "settings_avatar_change_label": "پروفائل تصویر تبدیل کریں (اختیاری)",
            "settings_update_profile_button": "پروفائل کی معلومات کو اپ ڈیٹ کریں",
            "settings_profile_updated_toast": "پروفائل اپ ڈیٹ ہو گئی!",
            "settings_chat_management_title": "چیٹ مینجمنٹ",
            "settings_clear_chat_button": "🧹 فعال چیٹ کی تاریخ صاف کریں",
            "about_us_title": "ℹ️ ہمارے بارے میں",
            "about_us_text": "Hanogt AI کو 2025 میں HanStudios کے مالک Oğuz Han Guluzade نے بنایا تھا۔ یہ اوپن سورس ہے، Gemini کے ذریعے تربیت یافتہ ہے، اور تمام کاپی رائٹس محفوظ ہیں۔",
            "footer_user": "صارف: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: فعال ({model_name}) | لاگ: فعال",
            "model_init_success": "Gemini ماڈل کامیابی سے شروع ہو گیا!",
            "model_init_error": "Gemini ماڈل شروع کرتے وقت ایک خرابی پیش آئی: {error}۔ براہ کرم یقینی بنائیں کہ آپ کی API کلید درست اور فعال ہے۔",
            "gemini_model_not_initialized": "Gemini ماڈل شروع نہیں ہوا۔ براہ کرم اپنی API کلید چیک کریں۔",
            "image_load_error": "تصویر لوڈ نہیں ہو سکی: {error}",
            "image_not_convertible": "اس مواد کو تقریر میں تبدیل نہیں کیا جا سکتا (متن نہیں)۔",
            "duckduckgo_error": "DuckDuckGo تلاش کرتے وقت ایک خرابی پیش آئی: {error}",
            "wikipedia_network_error": "Wikipedia تلاش کرتے وقت نیٹ ورک کی خرابی پیش آئی: {error}",
            "wikipedia_json_error": "Wikipedia جواب کو پارس کرتے وقت خرابی پیش آئی: {error}",
            "wikipedia_general_error": "Wikipedia تلاش کرتے وقت ایک عمومی خرابی پیش آئی: {error}",
            "unexpected_response_error": "جواب حاصل کرتے وقت ایک غیر متوقع خرابی پیش آئی: {error}",
            "source_error": "ماخذ: خرابی ({error})",
            "chat_cleared_toast": "فعال چیٹ صاف ہو گئی!",
            "profile_image_load_error": "پروفائل تصویر لوڈ نہیں ہو سکی: {error}",
            "web_search_results": "ویب تلاش کے نتائج:",
            "web_search_no_results": "آپ کی تلاش کی اصطلاح کے لیے کوئی نتیجہ نہیں ملا۔",
            "wikipedia_search_results": "ویکیپیڈیا تلاش کے نتائج:",
            "wikipedia_search_no_results": "آپ کی تلاش کی اصطلاح کے لیے کوئی نتیجہ نہیں ملا۔",
            "image_generated_example": "'{prompt}' کے لیے ایک تصویر بنائی گئی (مثال)۔",
            "image_upload_caption": "اپ لوڈ کردہ تصویر",
            "image_processing_error": "تصویر پر کارروائی کرتے وقت ایک خرابی پیش آئی: {error}",
            "image_vision_query": "آپ اس تصویر میں کیا دیکھتے ہیں؟",
            "loading_audio_file": "آڈیو فائل لوڈ ہو رہی ہے...",
            "tts_sr_not_available": "صوتی بات چیت اور ٹیکسٹ ٹو اسپیچ کی خصوصیات دستیاب نہیں ہیں۔ یقینی بنائیں کہ ضروری لائبریریاں انسٹال ہیں۔",
            "mic_listen_timeout": "آڈیو کا پتہ لگانے کا وقت ختم ہو گیا۔",
            "unexpected_audio_record_error": "آڈیو ریکارڈنگ کے دوران ایک غیر متوقع خرابی پیش آئی: {error}",
            "gemini_response_error": "جواب حاصل کرتے وقت ایک غیر متوقع خرابی پیش آئی: {error}",
            "creative_text_generated": "تخلیقی متن بنایا گیا: {text}",
            "turkish_voice_not_found": "ترکی آواز نہیں ملی، ڈیفالٹ آواز استعمال کی جائے گی۔ براہ کرم اپنے آپریٹنگ سسٹم کی آواز کی ترتیبات چیک کریں۔"
        },
        "UZ": { # Özbekçe
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Sizning yangi shaxsiy sun'iy intellekt yordamchingiz!",
            "profile_title": "Sizga qanday murojaat qilishim kerak?",
            "profile_name_label": "Ismingiz:",
            "profile_upload_label": "Profil rasmini yuklash (ixtiyoriy)",
            "profile_save_button": "Saqlash",
            "profile_greeting": "Salom, {name}!",
            "profile_edit_info": "Profilingizni Sozlamalar va Shaxsiylashtirish bo'limida tahrirlashingiz mumkin.",
            "ai_features_title": "Hanogt AI xususiyatlari:",
            "feature_general_chat": "Umumiy suhbat",
            "feature_web_search": "Veb-qidiruv (DuckDuckGo, Vikipediya)",
            "feature_knowledge_base": "Bilimlar bazasidan javoblar",
            "feature_creative_text": "Ijodiy matn yaratish",
            "feature_image_generation": "Oddiy rasm yaratish (namuna)",
            "feature_text_to_speech": "Matndan nutqqa (TTS)",
            "feature_feedback": "Fikr-mulohaza mexanizmi",
            "settings_button": "⚙️ Sozlamalar va Shaxsiylashtirish",
            "about_button": "ℹ️ Biz haqimizda",
            "app_mode_title": "Ilova rejimi",
            "chat_mode_text": "💬 Matnli suhbat",
            "chat_mode_image": "🖼️ Rasm generatori",
            "chat_mode_voice": "🎤 Ovozli suhbat (Fayl yuklash)",
            "chat_mode_creative": "✨ Ijodiy studiya",
            "chat_input_placeholder": "Xabaringizni yoki buyruqni yozing: Masalan: 'Salom', 'veb-qidiruv: Streamlit', 'ijodiy matn: o'zga sayyoraliklar'...",
            "generating_response": "Javob yaratilmoqda...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Fikr-mulohazangiz uchun rahmat!",
            "image_gen_title": "Rasm generatori",
            "image_gen_input_label": "Yaratmoqchi bo'lgan rasmni tasvirlab bering:",
            "image_gen_button": "Rasm yaratish",
            "image_gen_warning_placeholder": "Rasm yaratish xususiyati hozirda vaqtinchalik va haqiqiy API'ga ulanmagan.",
            "image_gen_warning_prompt_missing": "Iltimos, rasm tavsifini kiriting.",
            "voice_chat_title": "Ovozli suhbat",
            "voice_upload_label": "Audio fayl yuklang (MP3, WAV)",
            "voice_upload_warning": "Audio fayldan matnni transkripsiya qilish xususiyati hozirda vaqtinchalik.",
            "voice_live_input_title": "Jonli ovozli kiritish",
            "voice_mic_button": "Mikrofonni ishga tushirish",
            "voice_not_available": "Ovozli suhbat xususiyatlari mavjud emas. Kerakli kutubxonalar (pyttsx3, SpeechRecognition) o'rnatilganligiga ishonch hosil qiling.",
            "voice_listening": "Tinglanmoqda...",
            "voice_heard": "Siz aytdingiz: {text}",
            "voice_no_audio": "Ovoz aniqlanmadi, iltimos, qayta urinib ko'ring.",
            "voice_api_error": "Nutqni aniqlash xizmatiga ulanib bo'lmadi; {error}",
            "creative_studio_title": "Ijodiy studiya",
            "creative_studio_info": "Ushbu bo'lim ijodiy matn yaratish kabi ilg'or xususiyatlar uchun mo'ljallangan.",
            "creative_studio_input_label": "Ijodiy matn so'rovingizni kiriting:",
            "creative_studio_button": "Matn yaratish",
            "creative_studio_warning_prompt_missing": "Iltimos, ijodiy matn so'rovini kiriting.",
            "settings_personalization_title": "Sozlamalar va Shaxsiylashtirish",
            "settings_name_change_label": "Ismingizni o'zgartirish:",
            "settings_avatar_change_label": "Profil rasmini o'zgartirish (ixtiyoriy)",
            "settings_update_profile_button": "Profil ma'lumotlarini yangilash",
            "settings_profile_updated_toast": "Profil yangilandi!",
            "settings_chat_management_title": "Suhbatni boshqarish",
            "settings_clear_chat_button": "🧹 Faol suhbat tarixini tozalash",
            "about_us_title": "ℹ️ Biz haqimizda",
            "about_us_text": "Hanogt AI 2025 yilda HanStudios egasi Oğuz Han Guluzade tomonidan yaratilgan. U ochiq manbali, Gemini tomonidan o'qitilgan va barcha mualliflik huquqlari himoyalangan.",
            "footer_user": "Foydalanuvchi: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Faol ({model_name}) | Jurnal: Faol",
            "model_init_success": "Gemini modeli muvaffaqiyatli ishga tushirildi!",
            "model_init_error": "Gemini modelini ishga tushirishda xatolik yuz berdi: {error}. Iltimos, API kalitingiz to'g'ri va faol ekanligiga ishonch hosil qiling.",
            "gemini_model_not_initialized": "Gemini modeli ishga tushirilmagan. Iltimos, API kalitingizni tekshiring.",
            "image_load_error": "Rasmni yuklab bo'lmadi: {error}",
            "image_not_convertible": "Ushbu tarkibni nutqqa aylantirib bo'lmaydi (matn emas).",
            "duckduckgo_error": "DuckDuckGo qidiruvini amalga oshirishda xatolik yuz berdi: {error}",
            "wikipedia_network_error": "Vikipediya qidiruvini amalga oshirishda tarmoq xatoligi yuz berdi: {error}",
            "wikipedia_json_error": "Vikipediya javobini tahlil qilishda xatolik yuz berdi: {error}",
            "wikipedia_general_error": "Vikipediya qidiruvini amalga oshirishda umumiy xatolik yuz berdi: {error}",
            "unexpected_response_error": "Javob olishda kutilmagan xatolik yuz berdi: {error}",
            "source_error": "Manba: Xato ({error})",
            "chat_cleared_toast": "Faol suhbat tozalandi!",
            "profile_image_load_error": "Profil rasmini yuklab bo'lmadi: {error}",
            "web_search_results": "Veb-qidiruv natijalari:",
            "web_search_no_results": "Qidiruv so'rovingiz uchun natijalar topilmadi.",
            "wikipedia_search_results": "Vikipediya qidiruv natijalari:",
            "wikipedia_search_no_results": "Qidiruv so'rovingiz uchun natijalar topilmadi.",
            "image_generated_example": "'{prompt}' uchun rasm yaratildi (namuna).",
            "image_upload_caption": "Yuklangan rasm",
            "image_processing_error": "Rasmni qayta ishlashda xatolik yuz berdi: {error}",
            "image_vision_query": "Bu rasmda nimani ko'ryapsiz?",
            "loading_audio_file": "Audio fayl yuklanmoqda...",
            "tts_sr_not_available": "Ovozli suhbat va matndan nutqqa o'girish xususiyatlari mavjud emas. Kerakli kutubxonalar o'rnatilganligiga ishonch hosil qiling.",
            "mic_listen_timeout": "Ovozni aniqlash vaqti tugadi.",
            "unexpected_audio_record_error": "Ovoz yozish paytida kutilmagan xatolik yuz berdi: {error}",
            "gemini_response_error": "Javob olishda kutilmagan xatolik yuz berdi: {error}",
            "creative_text_generated": "Ijodiy matn yaratildi: {text}",
            "turkish_voice_not_found": "Turkcha ovoz topilmadi, standart ovoz ishlatiladi. Operatsion tizimingizning ovoz sozlamalarini tekshiring."
        },
        "KZ": { # Kazakça
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Сіздің жаңа жеке жасанды интеллект көмекшіңіз!",
            "profile_title": "Сізге қалай жүгінуім керек?",
            "profile_name_label": "Сіздің атыңыз:",
            "profile_upload_label": "Профиль суретін жүктеу (міндетті емес)",
            "profile_save_button": "Сақтау",
            "profile_greeting": "Сәлем, {name}!",
            "profile_edit_info": "Профиліңізді Баптаулар және Жекелендіру бөлімінде өңдей аласыз.",
            "ai_features_title": "Hanogt AI мүмкіндіктері:",
            "feature_general_chat": "Жалпы сөйлесу",
            "feature_web_search": "Веб-іздеу (DuckDuckGo, Уикипедия)",
            "feature_knowledge_base": "Білім базасынан жауаптар",
            "feature_creative_text": "Шығармашылық мәтін құру",
            "feature_image_generation": "Қарапайым сурет құру (мысал)",
            "feature_text_to_speech": "Мәтіннен сөйлеуге (TTS)",
            "feature_feedback": "Кері байланыс механизмі",
            "settings_button": "⚙️ Баптаулар және Жекелендіру",
            "about_button": "ℹ️ Біз туралы",
            "app_mode_title": "Қолданба режимі",
            "chat_mode_text": "💬 Мәтіндік сөйлесу",
            "chat_mode_image": "🖼️ Сурет генераторы",
            "chat_mode_voice": "🎤 Дауыстық сөйлесу (Файл жүктеу)",
            "chat_mode_creative": "✨ Шығармашылық студия",
            "chat_input_placeholder": "Хабарламаңызды немесе пәрменді жазыңыз: Мысалы: 'Сәлем', 'веб-іздеу: Streamlit', 'шығармашылық мәтін: бөтенғаламшарлықтар'...",
            "generating_response": "Жауап құрылуда...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Кері байланысыңыз үшін рахмет!",
            "image_gen_title": "Сурет генераторы",
            "image_gen_input_label": "Құрғыңыз келетін суретті сипаттаңыз:",
            "image_gen_button": "Сурет құру",
            "image_gen_warning_placeholder": "Сурет құру мүмкіндігі қазіргі уақытта орын толтырғыш болып табылады және нақты API-ге қосылмаған.",
            "image_gen_warning_prompt_missing": "Сурет сипаттамасын енгізіңіз.",
            "voice_chat_title": "Дауыстық сөйлесу",
            "voice_upload_label": "Аудио файлды жүктеу (MP3, WAV)",
            "voice_upload_warning": "Аудио файлды транскрипциялау мүмкіндігі қазіргі уақытта орын толтырғыш болып табылады.",
            "voice_live_input_title": "Тікелей дауыстық енгізу",
            "voice_mic_button": "Микрофонды бастау",
            "voice_not_available": "Дауыстық сөйлесу мүмкіндіктері қолжетімсіз. Қажетті кітапханалардың (pyttsx3, SpeechRecognition) орнатылғанына көз жеткізіңіз.",
            "voice_listening": "Тыңдалуда...",
            "voice_heard": "Сіз айттыңыз: {text}",
            "voice_no_audio": "Дауыс анықталмады, қайталап көріңіз.",
            "voice_api_error": "Сөйлеуді тану қызметіне қол жеткізу мүмкін болмады; {error}",
            "creative_studio_title": "Шығармашылық студия",
            "creative_studio_info": "Бұл бөлім шығармашылық мәтін құру сияқты кеңейтілген мүмкіндіктерге арналған.",
            "creative_studio_input_label": "Шығармашылық мәтін сұрауыңызды енгізіңіз:",
            "creative_studio_button": "Мәтін құру",
            "creative_studio_warning_prompt_missing": "Шығармашылық мәтін сұрауын енгізіңіз.",
            "settings_personalization_title": "Баптаулар және Жекелендіру",
            "settings_name_change_label": "Атыңызды өзгерту:",
            "settings_avatar_change_label": "Профиль суретін өзгерту (міндетті емес)",
            "settings_update_profile_button": "Профиль ақпаратын жаңарту",
            "settings_profile_updated_toast": "Профиль жаңартылды!",
            "settings_chat_management_title": "Сөйлесуді басқару",
            "settings_clear_chat_button": "🧹 Белсенді сөйлесу тарихын тазалау",
            "about_us_title": "ℹ️ Біз туралы",
            "about_us_text": "Hanogt AI 2025 жылы HanStudios иесі Oğuz Han Guluzade тарапынан жасалған. Ол ашық бастапқы кодты, Gemini арқылы оқытылған және барлық авторлық құқықтары қорғалған.",
            "footer_user": "Пайдаланушы: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Белсенді ({model_name}) | Журнал: Белсенді",
            "model_init_success": "Gemini моделі сәтті іске қосылды!",
            "model_init_error": "Gemini үлгісін бастау кезінде қате пайда болды: {error}. API кілтіңіздің дұрыс және белсенді екеніне көз жеткізіңіз.",
            "gemini_model_not_initialized": "Gemini моделі іске қосылмаған. API кілтіңізді тексеріңіз.",
            "image_load_error": "Суретті жүктеу мүмкін болмады: {error}",
            "image_not_convertible": "Бұл мазмұнды сөйлеуге айналдыру мүмкін емес (мәтін емес).",
            "duckduckgo_error": "DuckDuckGo іздеуі кезінде қате пайда болды: {error}",
            "wikipedia_network_error": "Уикипедия іздеуі кезінде желі қатесі пайда болды: {error}",
            "wikipedia_json_error": "Уикипедия жауабын талдау кезінде қате пайда болды: {error}",
            "wikipedia_general_error": "Уикипедия іздеуі кезінде жалпы қате пайда болды: {error}",
            "unexpected_response_error": "Жауап алу кезінде күтпеген қате пайда болды: {error}",
            "source_error": "Дереккөз: Қате ({error})",
            "chat_cleared_toast": "Белсенді сөйлесу тазаланды!",
            "profile_image_load_error": "Профиль суретін жүктеу мүмкін болмады: {error}",
            "web_search_results": "Веб-іздеу нәтижелері:",
            "web_search_no_results": "Іздеу терминіңізге сәйкес нәтижелер табылмады.",
            "wikipedia_search_results": "Уикипедия іздеу нәтижелері:",
            "wikipedia_search_no_results": "Іздеу терминіңізге сәйкес нәтижелер табылмады.",
            "image_generated_example": "'{prompt}' үшін сурет жасалды (мысал).",
            "image_upload_caption": "Жүктелген сурет",
            "image_processing_error": "Суретті өңдеу кезінде қате пайда болды: {error}",
            "image_vision_query": "Бұл суретте не көріп тұрсыз?",
            "loading_audio_file": "Аудио файл жүктелуде...",
            "tts_sr_not_available": "Дауыстық сөйлесу және мәтіннен сөйлеуге айналдыру мүмкіндіктері қолжетімсіз. Қажетті кітапханалардың орнатылғанына көз жеткізіңіз.",
            "mic_listen_timeout": "Дауысты анықтау уақыты аяқталды.",
            "unexpected_audio_record_error": "Аудио жазу кезінде күтпеген қате пайда болды: {error}",
            "gemini_response_error": "Жауап алу кезінде күтпеген қате пайда болды: {error}",
            "creative_text_generated": "Шығармашылық мәтін жасалды: {text}",
            "turkish_voice_not_found": "Түрік дауысы табылмады, әдепкі дауыс пайдаланылады. Операциялық жүйеңіздің дыбыс параметрлерін тексеріңіз."
        },
    }

    # İstenen dil kodunu al (örn: "BR", "CA", "MX", vb.)
    # Eğer dil kodu ana `texts` sözlüğünde yoksa, varsayılan olarak "TR" (Türkçe) kullanılır.
    # Bu yapı, eklediğiniz her yeni dilin sorunsuz çalışmasını sağlar.
    return texts.get(st.session_state.current_language, texts["TR"]).get(key, "TEXT_MISSING")

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
        st.session_state.chat_mode = "💬 Yazılı Sohbet" # Updated to include emoji
    if "current_mode_index" not in st.session_state:
        st.session_state.current_mode_index = 0
    if "show_settings" not in st.session_state: # Ayarlar bölümünü gösterme kontrolü
        st.session_state.show_settings = False
    if "show_about" not in st.session_state: # Hakkında bölümünü gösterme kontrolü
        st.session_state.show_about = False
    if "current_language" not in st.session_state:
        st.session_state.current_language = "TR" # Varsayılan dil Türkçe
    
    # EKLENEN KISIM: gemini_model'i burada kontrol et ve başlat
    # Bu kontrol, uygulamanın her yeniden yüklenmesinde modeli tekrar başlatmaktan kaçınır
    if "gemini_model" not in st.session_state or not st.session_state.models_initialized:
        initialize_gemini_model() # Modeli başlatma fonksiyonunu çağır

    load_chat_history()

def initialize_gemini_model():
    """Gemini modelini başlatır ve oturum durumuna kaydeder."""
    # Sadece 'gemini_model' None ise veya models_initialized False ise başlat
    if st.session_state.get("gemini_model") is None or not st.session_state.get("models_initialized", False):
        try:
            st.session_state.gemini_model = genai.GenerativeModel(
                model_name=GLOBAL_MODEL_NAME,
                # Düzeltme: 'Generation_config' yerine 'GenerationConfig' kullanıldı.
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
    
    if isinstance(content, Image.Image):
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='PNG')
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [img_byte_arr.getvalue()]})
    elif isinstance(content, bytes):
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    else:
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    
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
            del st.session_state.chat_session
        st.toast(get_text("chat_cleared_toast"), icon="🧹")
        logger.info(f"Aktif sohbet ({st.session_state.active_chat_id}) temizlendi.")
    st.rerun()

def text_to_speech(text):
    """Metni konuşmaya çevirir ve sesi oynatır."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return False
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        found_turkish_voice = False
        for voice in voices:
            # Türkçe ses arama mantığı
            if "turkish" in voice.name.lower() or "tr-tr" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                found_turkish_voice = True
                break
        if not found_turkish_voice:
            st.warning(get_text("turkish_voice_not_found"))

        engine.say(text)
        engine.runAndWait()
        logger.info("Metinden sese çevirme başarılı.")
        return True
    except Exception as e:
        st.error(get_text("unexpected_response_error").format(error=e))
        logger.error(f"Metinden sese çevirme hatası: {e}")
        return False

def record_audio():
    """Kullanıcıdan ses girişi alır."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(get_text("voice_listening"))
        try:
            # Dinleme süresini ve zaman aşımını artırabilirsiniz
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning(get_text("mic_listen_timeout"))
            return ""
        except Exception as e:
            st.error(get_text("unexpected_audio_record_error").format(error=e))
            return ""
            
    try:
        # Tanıma dilini dinamik olarak ayarlayabilirsiniz, ancak genellikle 'tr-TR' iyi çalışır.
        text = r.recognize_google(audio, language="tr-TR") 
        st.write(get_text("voice_heard").format(text=text))
        logger.info(f"Tanınan ses: {text}")
        return text
    except sr.UnknownValueError:
        st.warning(get_text("voice_unknown", "Ne dediğinizi anlayamadım.")) # 'voice_unknown' metni eklenmeli
        return ""
    except sr.RequestError as e:
        st.error(get_text("voice_api_error").format(error=e))
        return ""
    except Exception as e:
        st.error(get_text("unexpected_audio_record_error").format(error=e))
        return ""

@st.cache_data(ttl=3600)
def duckduckgo_search(query):
    """DuckDuckGo kullanarak web araması yapar."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results
    except Exception as e:
        st.error(get_text("duckduckgo_error").format(error=e))
        return []

@st.cache_data(ttl=3600)
def wikipedia_search(query):
    """Wikipedia'da arama yapar."""
    # Dil kodunu Wikipedia API'sine uygun formata çevir (örn: 'TR' -> 'tr')
    lang_code = st.session_state.current_language.lower().split('-')[0]
    try:
        response = requests.get(f"https://{lang_code}.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
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
    """Görsel oluşturma (örnek - placeholder)."""
    st.warning(get_text("image_gen_warning_placeholder"))
    placeholder_image_url = f"https://via.placeholder.com/400x300.png?text={prompt.replace(' ', '+')}"
    st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", get_text("image_generated_example").format(prompt=prompt))

# ... (UI Bileşenleri ve Ana Uygulama Mantığı aynı kalır)
# --- UI Bileşenleri ---

def display_welcome_and_profile_setup():
    """Hoş geldiniz mesajı ve profil oluşturma/düzenleme."""
    st.markdown(f"<h1 style='text-align: center;'>{get_text('welcome_title')}</h1>", unsafe_allow_html=True)
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
        st.subheader(get_text("profile_title"))
        
        # Profil resmi gösterimi
        if st.session_state.user_avatar:
            try:
                profile_image = Image.open(io.BytesIO(st.session_state.user_avatar))
                st.image(profile_image, caption=st.session_state.user_name if st.session_state.user_name else "Kullanıcı", width=150)
            except Exception as e:
                st.warning(get_text("profile_image_load_error").format(error=e))
                st.image("https://via.placeholder.com/150?text=Profil", width=150)
        else:
            st.image("https://via.placeholder.com/150?text=Profil", width=150)
        
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
    """Ayarlar ve Kişiselleştirme bölümünü gösterir."""
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
    """'Hakkımızda' bölümünü gösterir."""
    st.markdown(f"## {get_text('about_us_title')}")
    st.markdown(get_text("about_us_text"))
    st.write("---")

def display_main_chat_interface():
    """Ana sohbet arayüzünü gösterir."""
    
    col_settings, col_about = st.columns(2)
    with col_settings:
        if st.button(get_text("settings_button"), key="toggle_settings"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_about = False
    with col_about:
        if st.button(get_text("about_button"), key="toggle_about"):
            st.session_state.show_about = not st.session_state.show_about
            st.session_state.show_settings = False

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
    # Düzeltme: `chat_mode` session state'i mod değiştirildiğinde güncellenmeli.
    selected_mode = st.radio(
        "Mod Seçimi",
        mode_options,
        horizontal=True,
        index=mode_options.index(st.session_state.chat_mode) if st.session_state.chat_mode in mode_options else 0,
        key="main_mode_radio"
    )

    if selected_mode != st.session_state.chat_mode:
        st.session_state.chat_mode = selected_mode
        st.rerun()

    if st.session_state.chat_mode == get_text("chat_mode_text"):
        handle_text_chat()
    elif st.session_state.chat_mode == get_text("chat_mode_image"):
        handle_image_generation()
    elif st.session_state.chat_mode == get_text("chat_mode_voice"):
        handle_voice_chat()
    elif st.session_state.chat_mode == get_text("chat_mode_creative"):
        handle_creative_studio()

def handle_text_chat():
    """Yazılı sohbet modunu yönetir."""
    chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    for message_index, message in enumerate(chat_messages):
        avatar_src = None
        role_to_display = message["role"]
        if role_to_display == "model":
            role_to_display = "assistant" # Streamlit'in beklediği rol

        if message["role"] == "user" and st.session_state.user_avatar:
            try:
                avatar_src = Image.open(io.BytesIO(st.session_state.user_avatar))
            except Exception as e:
                logger.warning(f"Kullanıcı avatarı yüklenemedi: {e}")
                avatar_src = None
        
        with st.chat_message(role_to_display, avatar=avatar_src):
            content_part = message["parts"][0]
            if isinstance(content_part, str):
                st.markdown(content_part)
            elif isinstance(content_part, bytes):
                try:
                    image = Image.open(io.BytesIO(content_part))
                    st.image(image, caption=get_text("image_upload_caption"), use_column_width=True)
                except Exception as e:
                    st.warning(get_text("image_load_error").format(error=e))
            
            # Butonlar
            if message["role"] == "model": # Sadece model cevapları için buton göster
                cols = st.columns([0.1, 0.1, 0.8])
                with cols[0]:
                    if st.button(get_text("tts_button"), key=f"tts_btn_{st.session_state.active_chat_id}_{message_index}"):
                        if isinstance(content_part, str):
                            text_to_speech(content_part)
                        else:
                            st.warning(get_text("image_not_convertible"))
                with cols[1]:
                    if st.button(get_text("feedback_button"), key=f"fb_btn_{st.session_state.active_chat_id}_{message_index}"):
                        st.toast(get_text("feedback_toast"), icon="🙏")

    prompt = st.chat_input(get_text("chat_input_placeholder"))

    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        st.rerun() # Kullanıcı mesajını hemen göstermek için yeniden çalıştır

    # Son mesaj kullanıcıdan ise ve cevap bekleniyorsa AI cevabını al
    if chat_messages and chat_messages[-1]["role"] == "user":
        last_prompt = chat_messages[-1]["parts"][0]
        
        if last_prompt.lower().startswith("web ara:"):
            query = last_prompt[len("web ara:"):].strip()
            with st.spinner(get_text("generating_response")):
                results = duckduckgo_search(query)
                if results:
                    response_text = get_text("web_search_results") + "\n"
                    for i, r in enumerate(results):
                        response_text += f"{i+1}. **{r['title']}**\n   {r['body']}\n   [{r['href']}]({r['href']})\n\n"
                else:
                    response_text = get_text("web_search_no_results")
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif last_prompt.lower().startswith("wiki ara:"):
            query = last_prompt[len("wiki ara:"):].strip()
            with st.spinner(get_text("generating_response")):
                results = wikipedia_search(query)
                if results:
                    response_text = get_text("wikipedia_search_results") + "\n"
                    for i, r in enumerate(results):
                        page_id = r['pageid']
                        response_text += f"{i+1}. **{r['title']}**\n   [https://{st.session_state.current_language.lower().split('-')[0]}.wikipedia.org/?curid={page_id}](https://{st.session_state.current_language.lower().split('-')[0]}.wikipedia.org/?curid={page_id})\n\n"
                else:
                    response_text = get_text("wikipedia_search_no_results")
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif last_prompt.lower().startswith("görsel oluştur:"):
            image_prompt = last_prompt[len("görsel oluştur:"):].strip()
            generate_image(image_prompt)
        else:
            if st.session_state.gemini_model: 
                with st.spinner(get_text("generating_response")):
                    try:
                        # Geçmişi Gemini formatına uygun hale getir
                        gemini_history = []
                        for msg in chat_messages[:-1]: # Son kullanıcı mesajı hariç
                            role = "assistant" if msg["role"] == "model" else msg["role"]
                            gemini_history.append({"role": role, "parts": msg["parts"]})

                        chat_session = st.session_state.gemini_model.start_chat(history=gemini_history)
                        response = chat_session.send_message(last_prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text + "▌")
                        
                        response_placeholder.markdown(response_text)
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)

                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        
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

def handle_voice_chat():
    """Sesli sohbet modunu yönetir."""
    st.subheader(get_text("voice_chat_title"))
    
    if not TTS_SR_AVAILABLE:
        st.info(get_text("voice_not_available"))
    else:
        st.markdown("---")
        st.subheader(get_text("voice_live_input_title"))
        if st.button(get_text("voice_mic_button"), key="start_mic_button"):
            recognized_text = record_audio()
            if recognized_text:
                add_to_chat_history(st.session_state.active_chat_id, "user", recognized_text)
                st.rerun()

        # Handle response generation similar to text chat
        chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])
        if chat_messages and chat_messages[-1]["role"] == "user":
            last_prompt = chat_messages[-1]["parts"][0]
            if st.session_state.gemini_model:
                with st.spinner(get_text("generating_response")):
                    try:
                        gemini_history = []
                        for msg in chat_messages[:-1]:
                            role = "assistant" if msg["role"] == "model" else msg["role"]
                            gemini_history.append({"role": role, "parts": msg["parts"]})

                        chat_session = st.session_state.gemini_model.start_chat(history=gemini_history)
                        response = chat_session.send_message(last_prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text + "▌")
                        
                        response_placeholder.markdown(response_text)
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                        
                        # Otomatik olarak sesi oynat
                        text_to_speech(response_text)
                        st.rerun()

                    except Exception as e:
                        st.error(get_text("gemini_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))

def handle_creative_studio():
    """Yaratıcı stüdyo modunu yönetir."""
    st.subheader(get_text("creative_studio_title"))
    st.write(get_text("creative_studio_info"))
    
    creative_prompt = st.text_area(get_text("creative_studio_input_label"), height=150, key="creative_prompt_input")
    if st.button(get_text("creative_studio_button"), key="generate_creative_text_button"):
        if creative_prompt:
            if st.session_state.gemini_model:
                with st.spinner(get_text("generating_response")):
                    try:
                        creative_chat_session = st.session_state.gemini_model.start_chat(history=[])
                        response = creative_chat_session.send_message(f"Yaratıcı bir metin oluştur: {creative_prompt}", stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text + "▌")

                        response_placeholder.markdown(response_text)
                        st.success(get_text("creative_text_generated").format(text=""))
                        st.code(response_text, language=None)
                        
                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        else:
            st.warning(get_text("creative_studio_warning_prompt_missing"))


# --- Ana Uygulama Mantığı ---

def main():
    """Ana Streamlit uygulamasını çalıştırır."""
    st.set_page_config(
        page_title="Hanogt AI Asistan",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    initialize_session_state()

    st.markdown("""
        <style>
            header[data-testid="stHeader"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)


    # Dil Seçici Butonu
    _, col_lang, _ = st.columns([0.8, 0.15, 0.05])
    with col_lang:
        # Dil listesini `LANGUAGES` sözlüğünden dinamik olarak oluştur
        lang_options = list(LANGUAGES.keys())
        # Mevcut dilin index'ini bul
        try:
            current_lang_index = lang_options.index(st.session_state.current_language)
        except ValueError:
            current_lang_index = 0 # Eğer listede yoksa ilkini seç

        selected_lang_code = st.selectbox(
            label="Dil Seçimi",
            options=lang_options,
            index=current_lang_index,
            key="language_selector",
            format_func=lambda code: f"{LANGUAGES[code]['emoji']} {LANGUAGES[code]['name']}",
            label_visibility="collapsed"
        )
        
        if selected_lang_code != st.session_state.current_language:
            st.session_state.current_language = selected_lang_code
            # Dil değiştiğinde mod isimlerini de güncellemek için chat_mode'u sıfırla
            st.session_state.chat_mode = get_text("chat_mode_text")
            st.rerun()

    # Profil bilgisi girilmediyse, başlangıç ekranını göster
    if not st.session_state.user_name:
        display_welcome_and_profile_setup()
    else:
        st.markdown(f"<h1 style='text-align: center;'>{get_text('welcome_title')}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray;'>{get_text('profile_greeting').format(name=st.session_state.user_name)}</p>", unsafe_allow_html=True)
        
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
