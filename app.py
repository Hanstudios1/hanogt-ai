# --- Gerekli Kütüphaneler ---
# Temel Streamlit ve Veri İşleme
import streamlit as st
import requests # pip install requests
from bs4 import BeautifulSoup # pip install beautifulsoup4 lxml
import json
import re
import os
import random
import time
from io import BytesIO
from urllib.parse import urlparse, unquote
from datetime import datetime
import uuid

# Yapay Zeka ve Arama Motorları
import wikipedia # pip install wikipedia
from duckduckgo_search import DDGS # pip install -U duckduckgo_search
import google.generativeai as genai # pip install google-generativeai

# Multimedya ve Diğerleri
from PIL import Image, ImageDraw, ImageFont # pip install Pillow
import speech_recognition as sr # pip install SpeechRecognition pydub
#   -> Gerekirse: sudo apt-get install ffmpeg veya brew install ffmpeg
import pyttsx3 # pip install pyttsx3
#   -> Linux için: sudo apt-get update && sudo apt-get install espeak ffmpeg libespeak1

# İsteğe Bağlı: Token Sayımı için
try:
    import tiktoken # pip install tiktoken
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base") # Yaygın bir encoder
except ImportError:
    tiktoken = None
    tiktoken_encoder = None
    print("INFO: tiktoken library not found. Token counting will be disabled.")

# Supabase (isteğe bağlı, loglama/feedback için)
try:
    from supabase import create_client, Client # pip install supabase
    from postgrest.exceptions import APIError as SupabaseAPIError
except ImportError:
    print("ERROR: Supabase kütüphanesi bulunamadı. Loglama/Feedback devre dışı.")
    create_client = None
    Client = None
    SupabaseAPIError = Exception

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+ Enhanced",
    page_icon="✨", # Sayfa ikonu olarak yeni logoyu da kullanabilirsiniz: "ai_logo.png" (eğer dosya varsa)
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.1.5 Pro+ Enhanced (Logo & Button Update)" # Sürüm güncellendi
CURRENT_YEAR = datetime.now().year
CHAT_HISTORY_FILE = "chat_history_v2.json"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir sorun oluştu. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 20
SCRAPE_MAX_CHARS = 3800
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 {APP_NAME}/{APP_VERSION}"
SUPABASE_TABLE_LOGS = "chat_logs"
SUPABASE_TABLE_FEEDBACK = "user_feedback"
FONT_FILE = "arial.ttf"
AI_LOGO_PATH = "ai_logo.png" # YENİ LOGONUZUN DOSYA YOLU

# --- Dinamik Fonksiyonlar ---
DYNAMIC_FUNCTIONS_MAP = {
    "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
    "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})",
    "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})"
}

# --- Bilgi Tabanı ---
knowledge_base_load_error_global = None

@st.cache_data(ttl=3600)
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE, user_name_for_greeting="kullanıcı"):
    error_message = None
    default_knowledge = {
        "merhaba": [f"Merhaba {user_name_for_greeting}!", "Selam!", "Hoş geldin!", "Size nasıl yardımcı olabilirim?"],
        "selam": ["Merhaba!", "Selam sana da!", "Nasıl gidiyor?"],
        "nasılsın": ["İyiyim, teşekkürler! Siz nasılsınız?", "Harika hissediyorum!", "Sizin için ne yapabilirim?"],
        "hanogt kimdir": [f"Ben {APP_NAME} ({APP_VERSION}), Streamlit ve Python ile geliştirilmiş bir AI asistanıyım.", f"{APP_NAME}, sorularınızı yanıtlamak, metin üretmek ve basit görseller oluşturmak için tasarlandı."],
        "teşekkür ederim": ["Rica ederim!", "Ne demek!", "Yardımcı olabildiğime sevindim.", "Her zaman!"],
        "görüşürüz": ["Görüşmek üzere!", "Hoşça kal!", "İyi günler!", "Tekrar beklerim!"],
        "adın ne": [f"Ben {APP_NAME}, versiyon {APP_VERSION}.", f"Bana {APP_NAME} diyebilirsiniz."],
        "ne yapabilirsin": ["Sorularınızı yanıtlayabilir, web'de arama yapabilir, yaratıcı metinler üretebilir ve basit görseller çizebilirim.", "Size çeşitli konularda yardımcı olabilirim."],
        "saat kaç": ["Saat bilgisini alıyorum."], "bugün ayın kaçı": ["Tarih bilgisini alıyorum."], "tarih ne": ["Tarih bilgisini alıyorum."],
        "hava durumu": ["Üzgünüm, güncel hava durumu bilgisi sağlayamıyorum.", "Hava durumu servisim henüz aktif değil."]
    }
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                loaded_kb = json.load(f)
            merged_kb = {**default_knowledge, **loaded_kb}
            return merged_kb, None
        else:
            error_message = f"Bilgi tabanı ({filename}) bulunamadı. Varsayılan kullanılıyor."
            print(f"INFO: {error_message}")
            return default_knowledge, error_message
    except json.JSONDecodeError as e:
        error_message = f"Bilgi tabanı ({filename}) hatalı (JSONDecodeError: {e}). Varsayılan kullanılıyor."
        print(f"ERROR: {error_message}")
        return default_knowledge, error_message
    except Exception as e:
        error_message = f"Bilgi tabanı yüklenirken genel hata: {e}. Varsayılan kullanılıyor."
        print(f"ERROR: {error_message}")
        return default_knowledge, error_message

KNOWLEDGE_BASE = {}

# --- API Anahtarı ve Gemini Yapılandırması (Fonksiyon içeriği önceki gibi) ---
gemini_model = None
gemini_init_error_global = None
def initialize_gemini_model():
    global gemini_init_error_global 
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta bulunamadı! Gemini özellikleri devre dışı."
        return None, gemini_init_error_global
    try:
        genai.configure(api_key=api_key)
        safety = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        model_name = st.session_state.get('gemini_model_name', 'gemini-1.5-flash-latest')
        system_prompt = st.session_state.get('gemini_system_prompt', None)
        config = genai.types.GenerationConfig(
            temperature=st.session_state.get('gemini_temperature', 0.7),
            top_p=st.session_state.get('gemini_top_p', 0.95),
            top_k=st.session_state.get('gemini_top_k', 40),
            max_output_tokens=st.session_state.get('gemini_max_tokens', 4096)
        )
        model_args = {
            "model_name": model_name,
            "safety_settings": safety,
            "generation_config": config
        }
        if system_prompt and system_prompt.strip():
            model_args["system_instruction"] = system_prompt.strip()
        
        model = genai.GenerativeModel(**model_args)
        gemini_init_error_global = None 
        print(f"INFO: Gemini modeli ({model_name}) başarıyla yapılandırıldı ve yüklendi!")
        return model, None
    except Exception as e:
        gemini_init_error_global = f"🛑 Gemini yapılandırma sırasında kritik hata: {e}. Model kullanılamıyor."
        print(f"CRITICAL_ERROR: Gemini Init Failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None, gemini_init_error_global

# --- Supabase İstemcisini Başlatma (Fonksiyon içeriği önceki gibi) ---
supabase = None
supabase_init_error_global = None
@st.cache_resource(ttl=3600)
def init_supabase_client_cached():
    if not create_client: 
        error_msg = "Supabase kütüphanesi yüklenemediğinden Supabase başlatılamadı. Loglama devre dışı."
        print(f"ERROR: {error_msg}")
        return None, error_msg
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        error_msg = "Supabase URL veya Servis Anahtarı Secrets'ta bulunamadı. Loglama ve geri bildirim özellikleri devre dışı."
        print(f"ERROR: {error_msg}")
        return None, error_msg
    try:
        client: Client = create_client(url, key)
        print("INFO: Supabase client created successfully via cache function.")
        return client, None 
    except Exception as e:
        error_msg = f"Supabase bağlantısı sırasında kritik hata: {e}. Loglama ve geri bildirim devre dışı."
        print(f"CRITICAL_ERROR: Supabase connection failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None, error_msg

# --- AI Avatar Yükleme Fonksiyonu ---
@st.cache_data 
def load_ai_avatar(image_path=AI_LOGO_PATH, fallback_emoji="🤖"):
    """AI avatarını yükler, bulunamazsa fallback emoji döner."""
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        print(f"UYARI: AI Logo dosyası '{image_path}' bulunamadı. Varsayılan emoji '{fallback_emoji}' kullanılacak.")
        if 'ai_logo_warning_shown' not in st.session_state: # Uyarıyı sadece bir kere göster
            st.toast(f"AI logo dosyası ({image_path}) bulunamadı. Lütfen projeye ekleyin.", icon="🖼️")
            st.session_state.ai_logo_warning_shown = True
        return fallback_emoji
    except Exception as e:
        print(f"HATA: AI Logo yüklenirken sorun oluştu ({image_path}): {e}. Varsayılan emoji '{fallback_emoji}' kullanılacak.")
        return fallback_emoji

# Diğer yardımcı fonksiyonlar, TTS, metin temizleme, web kazıma, web arama, sohbet geçmişi yönetimi,
# Gemini yanıt alma, Supabase loglama, yanıt orkestrasyonu, yaratıcı modüller, görsel oluşturucu
# fonksiyonlarının içerikleri bir önceki yanıttaki (v5.1.4) gibi kalabilir.
# Önemli olan bu fonksiyonların doğru şekilde çağrılması ve global değişkenlere (KNOWLEDGE_BASE, gemini_model vb.)
# doğru şekilde erişmesidir. Uzunluk nedeniyle tekrar buraya eklemiyorum.
# Lütfen bu fonksiyonları bir önceki yanıttan kopyalayın.
# Aşağıda sadece değişiklik yapılan veya yeni eklenen bölümler vurgulanacaktır.

# --- ÖNCEKİ YANITTAN KOPYALANACAK FONKSİYONLAR (İÇERİKLERİ AYNI) ---
# kb_chatbot_response(query, knowledge_base_dict) (KNOWLEDGE_BASE globalini kullanır)
# _get_session_id()
# init_tts_engine_cached()
# speak(text)
# _clean_text(text)
# scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS)
# search_web(query)
# load_all_chats_cached(file_path=CHAT_HISTORY_FILE)
# save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE)
# get_gemini_response(prompt_text, history_list, stream_output=False)
# log_to_supabase(table_name, data_dict)
# log_interaction(prompt, ai_response, source, message_id, chat_id_val)
# log_feedback(message_id, user_prompt, ai_response, feedback_type, comment="")
# get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id_val, use_stream=False)
# creative_response_generator(prompt_text, length_mode="orta", style_mode="genel")
# generate_new_idea_creative(seed_text, style="genel")
# advanced_word_generator(base_word)
# generate_prompt_influenced_image(prompt) # Bu fonksiyonun içinde FONT_FILE kullanılıyor, varlığından emin olun.
# --- FONKSİYON KOPYALAMA SONU ---

# --- Session State Başlatma (Fonksiyon içeriği önceki gibi) ---
def initialize_session_state():
    defaults = {
        'all_chats': {}, 'active_chat_id': None, 'next_chat_id_counter': 0,
        'app_mode': "Yazılı Sohbet", 'user_name': None, 'user_avatar_bytes': None,
        'show_main_app': False, 'greeting_message_shown': False,
        'tts_enabled': True, 'gemini_stream_enabled': True,
        'gemini_temperature': 0.7, 'gemini_top_p': 0.95, 'gemini_top_k': 40,
        'gemini_max_tokens': 4096, 'gemini_model_name': 'gemini-1.5-flash-latest',
        'gemini_system_prompt': "",
        'message_id_counter': 0, 'last_ai_response_for_feedback': None,
        'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
        'feedback_comment_input': "", 'show_feedback_comment_form': False,
        'session_id': str(uuid.uuid4()), 'last_feedback_type': 'positive',
        'models_initialized': False,
        'ai_logo_warning_shown': False # AI logo uyarısı için
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# --- Modelleri ve İstemcileri Başlatma (Fonksiyon içeriği önceki gibi, KNOWLEDGE_BASE ataması dahil) ---
if not st.session_state.models_initialized:
    print("INFO: Uygulama kaynakları ilk kez başlatılıyor...")
    gemini_model, gemini_init_error_global = initialize_gemini_model()
    if gemini_model: st.toast(f"✨ Gemini modeli ({st.session_state.gemini_model_name}) başarıyla yüklendi!", icon="🤖")
    supabase, supabase_init_error_global = init_supabase_client_cached()
    if supabase: st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱")
    tts_engine, tts_init_error_global = init_tts_engine_cached()
    if tts_engine: st.toast("🔊 TTS motoru hazır.", icon="🗣️")
    all_chats_data, chat_load_errors = load_all_chats_cached()
    st.session_state.all_chats = all_chats_data
    if chat_load_errors:
        for msg_info in chat_load_errors:
            if msg_info['type'] == 'toast': st.toast(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'warning': st.warning(msg_info['text'], icon=msg_info.get('icon'))
            # ... diğer mesaj türleri
    if not st.session_state.active_chat_id and st.session_state.all_chats:
        try:
            valid_chat_ids = [cid for cid in st.session_state.all_chats.keys() if cid.startswith("chat_") and len(cid.split('_')) > 1 and cid.split('_')[-1].isdigit()]
            if valid_chat_ids: st.session_state.active_chat_id = sorted(valid_chat_ids, key=lambda x: int(x.split('_')[-1]), reverse=True)[0]
        except Exception as e: print(f"WARNING: Aktif sohbet ID'si belirlenirken sorun: {e}")
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    kb_data, kb_error = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    globals()['KNOWLEDGE_BASE'] = kb_data
    globals()['knowledge_base_load_error_global'] = kb_error
    st.session_state.models_initialized = True
    print("INFO: Uygulama kaynaklarının ilk başlatılması tamamlandı.")
else: # Sonraki çalıştırmalar
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    current_kb, kb_load_err_rerun = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    if kb_load_err_rerun and kb_load_err_rerun != globals().get('knowledge_base_load_error_global'):
        globals()['knowledge_base_load_error_global'] = kb_load_err_rerun
    elif not kb_load_err_rerun and globals().get('knowledge_base_load_error_global'):
        globals()['knowledge_base_load_error_global'] = None
        st.toast("Bilgi tabanı başarıyla güncellendi/yüklendi.", icon="📚")
    globals()['KNOWLEDGE_BASE'] = current_kb

# --- ARAYÜZ FONKSİYONLARI ---
def display_settings_section(): # İçerik önceki gibi, sadece sorunlu buton güncellendi
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**")
        # ... (Kullanıcı adı, avatar yükleme kodları önceki gibi) ...
        new_user_name = st.text_input("Adınız:", value=st.session_state.user_name, key="change_user_name_input", label_visibility="collapsed", placeholder="Görünür adınız...")
        if new_user_name != st.session_state.user_name and new_user_name.strip():
            st.session_state.user_name = new_user_name.strip(); load_knowledge_from_file.clear(); st.toast("Adınız güncellendi!", icon="✏️"); st.rerun()
        avatar_col1, avatar_col2 = st.columns([0.8, 0.2])
        with avatar_col1:
            uploaded_avatar_file = st.file_uploader("Avatar yükle (PNG, JPG - maks 2MB):", type=["png", "jpg", "jpeg"], key="upload_avatar_file",label_visibility="collapsed")
            if uploaded_avatar_file:
                if uploaded_avatar_file.size > 2 * 1024 * 1024: st.error("Dosya boyutu 2MB'den büyük olamaz!", icon="❌")
                else: st.session_state.user_avatar_bytes = uploaded_avatar_file.getvalue(); st.toast("Avatarınız güncellendi!", icon="🖼️"); st.rerun()
        with avatar_col2:
            if st.session_state.user_avatar_bytes:
                st.image(st.session_state.user_avatar_bytes, width=60)
                if st.button("🗑️ Kaldır", key="remove_avatar_button", help="Yüklü avatarı kaldır", use_container_width=True):
                    st.session_state.user_avatar_bytes = None; st.toast("Avatar kaldırıldı.", icon="🗑️"); st.rerun()
        st.caption("Avatarınız sadece bu tarayıcı oturumunda saklanır."); st.divider()
        st.subheader("🤖 Yapay Zeka ve Arayüz Ayarları")
        # ... (TTS, Stream toggle, Sistem Talimatı, Gelişmiş Yapılandırma kodları önceki gibi) ...
        tts_toggle_col, stream_toggle_col = st.columns(2); is_tts_engine_ok = globals().get('tts_engine') is not None
        with tts_toggle_col: st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not is_tts_engine_ok,help="Yanıtları sesli olarak oku (TTS motoru aktifse).")
        with stream_toggle_col: st.session_state.gemini_stream_enabled = st.toggle("Yanıt Akışı (Streaming)", value=st.session_state.gemini_stream_enabled, help="Yanıtları kelime kelime alarak daha hızlı gösterim sağla.")
        st.session_state.gemini_system_prompt = st.text_area("AI Sistem Talimatı (Opsiyonel):",value=st.session_state.get('gemini_system_prompt', ""),key="system_prompt_input_area",height=100,placeholder="Yapay zekanın genel davranışını veya rolünü tanımlayın...",help="Modelin yanıtlarını etkilemek için genel bir talimat girin.")
        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma"); gemini_config_col1, gemini_config_col2 = st.columns(2)
        available_gemini_models = ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']
        with gemini_config_col1:
            try: current_model_index = available_gemini_models.index(st.session_state.gemini_model_name)
            except ValueError: current_model_index = 0; st.session_state.gemini_model_name = available_gemini_models[0]
            st.session_state.gemini_model_name = st.selectbox("AI Modeli:", available_gemini_models, index=current_model_index, key="select_gemini_model", help="Kullanılacak Gemini modelini seçin.")
            st.session_state.gemini_temperature = st.slider("Sıcaklık (Temperature):", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="temperature_slider", help="Yaratıcılık seviyesi.")
            st.session_state.gemini_max_tokens = st.slider("Maksimum Yanıt Token:", 256, 8192,st.session_state.gemini_max_tokens, 128, key="max_tokens_slider", help="Bir yanıtta üretilecek maksimum token sayısı.")
        with gemini_config_col2:
            st.session_state.gemini_top_k = st.slider("Top K:", 1, 100,st.session_state.gemini_top_k, 1, key="top_k_slider", help="Kelime seçim çeşitliliği.")
            st.session_state.gemini_top_p = st.slider("Top P:", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="top_p_slider", help="Kelime seçim odaklılığı.")
            if st.button("⚙️ AI Ayarlarını Uygula & Modeli Yeniden Başlat", key="reload_ai_model_button", use_container_width=True, type="primary", help="Seçili AI modelini ve parametreleri yeniden yükler."):
                with st.spinner("AI modeli yeni ayarlarla yeniden başlatılıyor..."):
                    new_model, new_error = initialize_gemini_model(); globals()['gemini_model'] = new_model; globals()['gemini_init_error_global'] = new_error
                if not globals()['gemini_model']: st.error(f"AI modeli yüklenemedi: {globals()['gemini_init_error_global']}")
                else: st.success("AI ayarları başarıyla uygulandı ve model yeniden başlatıldı!", icon="⚙️")
                st.rerun()
        st.divider()

        st.subheader("🧼 Geçmiş Yönetimi")
        clear_current_col, clear_all_col = st.columns(2)
        with clear_current_col: # Aktif sohbeti temizleme (önceki gibi)
            active_chat_id_for_clear = st.session_state.get('active_chat_id')
            is_clear_current_disabled = not bool(active_chat_id_for_clear and st.session_state.all_chats.get(active_chat_id_for_clear))
            if st.button("🧹 Aktif Sohbetin İçeriğini Temizle", use_container_width=True, type="secondary", key="clear_current_chat_button", help="Sadece açık olan sohbetin içeriğini temizler.", disabled=is_clear_current_disabled):
                if active_chat_id_for_clear and active_chat_id_for_clear in st.session_state.all_chats:
                    st.session_state.all_chats[active_chat_id_for_clear] = []; save_all_chats(st.session_state.all_chats); st.toast("Aktif sohbetin içeriği temizlendi!", icon="🧹"); st.rerun()
        
        with clear_all_col:
            is_clear_all_disabled = not bool(st.session_state.all_chats)
            
            st.markdown("--- *Buton Hata Ayıklama Alanı* ---")
            st.write(f"DEBUG (Buton Öncesi): `is_clear_all_disabled` = {is_clear_all_disabled} (Tip: {type(is_clear_all_disabled)})")
            st.write(f"DEBUG (Buton Öncesi): `st.session_state.all_chats` boş mu? = {not bool(st.session_state.all_chats)} (Tip: {type(st.session_state.all_chats)})")

            st.write("Aşağıdaki **'Geliştirilmiş Test Butonu'** çalışıyorsa, sorun çözülmüş olabilir. Eğer hala `StreamlitAPIException` veriyorsa, sorun `use_container_width=True` parametresinde olabilir veya daha derin bir Streamlit sorunudur (Cloud loglarını kontrol edin!).")
            
            # Geliştirilmiş Test Butonu: Basit test butonu çalıştığı için, şimdi 'use_container_width' ekleyerek deneyelim.
            # 'type' ve 'help' parametreleri hala sorunlu olabilir diye eklenmedi.
            if st.button("🗑️ TÜM Sohbet Geçmişini Kalıcı Olarak Sil (Geliştirilmiş Test)", 
                           key="clear_all_chats_button_revised_v1", # Key'i güncelledik
                           disabled=is_clear_all_disabled,
                           use_container_width=True): # Bu parametre eklendi
                st.session_state.all_chats = {}
                st.session_state.active_chat_id = None
                save_all_chats({})
                st.toast("TÜM sohbet geçmişi silindi! (Geliştirilmiş Test ile)", icon="🗑️")
                st.rerun()
            
            st.caption("Eğer yukarıdaki buton hata verirse, `use_container_width=True` parametresini kaldırıp deneyin. O da hata verirse, Streamlit Cloud loglarındaki tam hata mesajı gereklidir.")
            st.markdown("--- *Buton Hata Ayıklama Alanı Sonu* ---")

# --- display_chat_message_with_feedback ---
# Bu fonksiyonda AI avatarını güncelleyeceğiz.
def display_chat_message_with_feedback(message_data, message_index, current_chat_id):
    role = message_data.get('role', 'model')
    content_text = str(message_data.get('parts', ''))
    is_user_message = (role == 'user')

    ai_default_avatar = load_ai_avatar() # YENİ LOGOYU YÜKLE (veya fallback emoji)

    if is_user_message:
        sender_display_name = st.session_state.get('user_name', 'Kullanıcı')
        avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes)) if st.session_state.user_avatar_bytes else "🧑"
    else: # AI mesajı
        sender_display_name = message_data.get('sender_display', APP_NAME)
        if "Gemini" in sender_display_name: avatar_icon = "✨" # Gemini için özel ikon kalabilir
        elif any(w in sender_display_name.lower() for w in ["web", "wiki", "arama", "ddg"]): avatar_icon = "🌐"
        elif any(w in sender_display_name.lower() for w in ["bilgi", "fonksiyon", "taban"]): avatar_icon = "📚"
        elif "Yaratıcı" in sender_display_name: avatar_icon = "🎨"
        elif "Görsel" in sender_display_name: avatar_icon = "🖼️"
        else: avatar_icon = ai_default_avatar # Varsayılan AI avatarı olarak YENİ LOGO veya fallback emoji

    with st.chat_message(role, avatar=avatar_icon):
        # ... (Mesaj içeriği, kod bloğu, markdown gösterimi önceki gibi) ...
        if "```" in content_text:
            text_parts = content_text.split("```")
            for i, part in enumerate(text_parts):
                if i % 2 == 1:
                    language_match = re.match(r"(\w+)\n", part); code_block_content = part[len(language_match.group(0)):] if language_match else part
                    actual_code_language = language_match.group(1).lower() if language_match else None
                    st.code(code_block_content.strip(), language=actual_code_language)
                    if st.button(f"📋 Kopyala", key=f"copy_code_{current_chat_id}_{message_index}_{i}", help="Kodu panoya kopyala", use_container_width=False):
                        st.write_to_clipboard(code_block_content.strip()); st.toast("Kod panoya kopyalandı!", icon="✅")
                elif part.strip(): st.markdown(part, unsafe_allow_html=True)
        elif content_text.strip(): st.markdown(content_text, unsafe_allow_html=True)
        else: st.caption("[Mesaj içeriği bulunmuyor]")

        if not is_user_message and content_text.strip(): # AI mesajı için kaynak, TTS, geri bildirim
            # ... (Token sayımı, TTS ve Geri Bildirim butonları önceki gibi) ...
            token_count_display_str = ""; ts_col, tts_col, fb_col = st.columns([0.75, 0.1, 0.15])
            if tiktoken_encoder:
                try: token_count = len(tiktoken_encoder.encode(content_text)); token_count_display_str = f" (~{token_count} token)"
                except Exception: pass
            with ts_col: source_name_only = sender_display_name.split('(')[-1].replace(')', '').strip() if '(' in sender_display_name else sender_display_name; st.caption(f"Kaynak: {source_name_only}{token_count_display_str}")
            with tts_col:
                if st.session_state.tts_enabled and globals().get('tts_engine'):
                    if st.button("🔊", key=f"tts_tts_{current_chat_id}_{message_index}", help="Yanıtı sesli oku", use_container_width=True): speak(content_text) # Key düzeltildi
            with fb_col:
                if st.button("✍️", key=f"fb_btn_{current_chat_id}_{message_index}", help="Yanıt hakkında geri bildirim ver", use_container_width=True): # Key düzeltildi
                    st.session_state.current_message_id_for_feedback = f"{current_chat_id}_{message_index}"
                    prev_prompt = "[İstem bulunamadı]"; idx_prev = message_index -1
                    if idx_prev >= 0 and st.session_state.all_chats[current_chat_id][idx_prev]['role'] == 'user': prev_prompt = st.session_state.all_chats[current_chat_id][idx_prev]['parts']
                    st.session_state.last_user_prompt_for_feedback = prev_prompt; st.session_state.last_ai_response_for_feedback = content_text
                    st.session_state.show_feedback_comment_form = True; st.session_state.feedback_comment_input = ""; st.rerun()

# --- display_chat_list_and_about, display_feedback_form_if_active, display_chat_interface_main ---
# Bu fonksiyonların içerikleri bir önceki yanıttaki (v5.1.4) gibidir.
# Uzunluk nedeniyle ve bu fonksiyonlarda logo veya butonla ilgili doğrudan bir değişiklik olmadığı için
# buraya tekrar eklenmemiştir. Lütfen bu fonksiyonları bir önceki yanıttan alın.
# ÖNEMLİ: Bu fonksiyonlar KNOWLEDGE_BASE, gemini_model gibi global değişkenleri kullanır, bunların doğru yüklendiğinden emin olun.

# --- ÖNCEKİ YANITTAN KOPYALANACAK FONKSİYONLAR (İÇERİKLERİ AYNI) ---
# display_chat_list_and_about(left_column_ref)
# display_feedback_form_if_active()
# display_chat_interface_main(main_column_container_ref=None)
# --- FONKSİYON KOPYALAMA SONU ---


# --- UYGULAMA ANA AKIŞI (Önceki gibi, sadece global hata gösterimini kontrol et) ---
st.markdown(f"<h1 style='text-align:center;color:#0078D4;'>{APP_NAME} <sup style='font-size:0.6em;color:#555;'>v{APP_VERSION}</sup></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-style:italic;color:#555;'>Yapay Zeka Destekli Kişisel Asistanınız</p>", unsafe_allow_html=True)
st.markdown("---")

# Global başlatma hatalarını göster
# Bu değişkenler initialize bloklarında veya sonrasında set ediliyor olmalı.
# Önceki yanıtlardaki gibi globals().get() ile erişmek daha güvenli olabilir.
if globals().get('gemini_init_error_global'): st.warning(globals().get('gemini_init_error_global'), icon="🗝️")
if globals().get('supabase_init_error_global'): st.warning(globals().get('supabase_init_error_global'), icon="🧱")
if globals().get('tts_init_error_global'): st.warning(globals().get('tts_init_error_global'), icon="🔇")
if globals().get('knowledge_base_load_error_global'): st.warning(globals().get('knowledge_base_load_error_global'), icon="📚")

# Giriş ekranı veya ana uygulama (önceki gibi)
if not st.session_state.show_main_app:
    # ... (Giriş ekranı kodu önceki gibi) ...
    st.subheader("👋 Merhaba! Başlamadan Önce Sizi Tanıyalım"); login_cols = st.columns([0.2, 0.6, 0.2])
    with login_cols[1]:
        with st.form("user_login_form"):
            user_entered_name = st.text_input("Size nasıl hitap etmemizi istersiniz?", placeholder="İsminiz...", key="login_name_input", value=st.session_state.get('user_name', ''))
            if st.form_submit_button("✨ Uygulamayı Başlat", use_container_width=True, type="primary"):
                if user_entered_name and user_entered_name.strip():
                    st.session_state.user_name = user_entered_name.strip(); st.session_state.show_main_app = True; st.session_state.greeting_message_shown = False; load_knowledge_from_file.clear(); st.rerun()
                else: st.error("Lütfen geçerli bir isim giriniz.")
else: # Ana Uygulama
    if not st.session_state.greeting_message_shown:
        st.success(f"Tekrar hoş geldiniz, **{st.session_state.user_name}**! Size nasıl yardımcı olabilirim?", icon="🎉"); st.session_state.greeting_message_shown = True
    app_left_column, app_main_column = st.columns([1, 3])
    # display_chat_list_and_about(app_left_column) # YUKARIDA KOPYALANACAK FONKSİYONLAR ARASINDA OLMALI
    # with app_main_column:
    #     display_settings_section()
    #     # ... (Mod seçimi ve modlara göre arayüz yükleme kodları önceki gibi) ...
    #     # display_chat_interface_main() # YUKARIDA KOPYALANACAK FONKSİYONLAR ARASINDA OLMALI
    #     # ... (Diğer modların yüklenmesi)
    #     # Footer (önceki gibi)

    # YUKARIDAKİ KOPYALANACAK FONKSİYONLARIN LİSTESİNE EK OLARAK,
    # BU KISIMDAKİ display_chat_list_and_about ve display_chat_interface_main ÇAĞRILARINI
    # VE MOD YÖNETİMİNİ DE BİR ÖNCEKİ YANITTAN ALMANIZ GEREKİR.
    # KODUN TAMAMINI BURAYA TEKRAR EKLEMEK YANITI ÇOK UZATACAĞINDAN KISALTILMIŞTIR.
    # LÜTFEN BU BÖLÜMÜ BİR ÖNCEKİ (v5.1.4) KODUNDAN TAMAMLAYIN.
    # Örnek olarak ana yapı bırakılmıştır:
    display_chat_list_and_about(app_left_column) # Bu fonksiyonu yukarıdaki listeden kopyalayın
    with app_main_column:
        display_settings_section() # Bu fonksiyon yukarıda güncellendi
        st.markdown("#### Uygulama Modu")
        app_modes = {"Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya Yükle)": "🎤", "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️"}
        mode_options_keys = list(app_modes.keys())
        try: current_mode_index = mode_options_keys.index(st.session_state.app_mode)
        except ValueError: current_mode_index = 0; st.session_state.app_mode = mode_options_keys[0]
        selected_app_mode = st.radio("Çalışma Modunu Seçin:",options=mode_options_keys,index=current_mode_index,format_func=lambda k: f"{app_modes[k]} {k}",horizontal=True,label_visibility="collapsed",key="app_mode_selection_radio")
        if selected_app_mode != st.session_state.app_mode: st.session_state.app_mode = selected_app_mode; st.rerun()
        st.markdown("<hr style='margin-top:0.1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
        current_app_mode = st.session_state.app_mode
        if current_app_mode == "Yazılı Sohbet": display_chat_interface_main() # Bu fonksiyonu yukarıdaki listeden kopyalayın
        elif current_app_mode == "Sesli Sohbet (Dosya Yükle)": pass # Bu modu da önceki yanıttan kopyalayın
        elif current_app_mode == "Yaratıcı Stüdyo": pass # Bu modu da önceki yanıttan kopyalayın
        elif current_app_mode == "Görsel Oluşturucu": pass # Bu modu da önceki yanıttan kopyalayın
        # Footer'ı da önceki yanıttan kopyalayın...
        st.markdown("<hr style='margin-top:1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        with footer_cols[0]: st.caption(f"Kullanıcı: **{st.session_state.get('user_name', 'Tanımlanmamış')}**")
        with footer_cols[1]: st.caption(f"<div style='text-align:center;'>{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}</div>", unsafe_allow_html=True)
        with footer_cols[2]:
            ai_model_name_display = st.session_state.gemini_model_name.split('/')[-1]
            ai_status_text = "Aktif" if globals().get('gemini_model') else "Devre Dışı"
            logging_status_text = "Aktif" if globals().get('supabase') else "Devre Dışı"
            st.caption(f"<div style='text-align:right;'>AI: {ai_status_text} ({ai_model_name_display}) | Log: {logging_status_text}</div>", unsafe_allow_html=True)

