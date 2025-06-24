# --- Gerekli Kütüphaneler ---
# Temel Streamlit ve Veri İşleme
import streamlit as st
import requests
from bs4 import BeautifulSoup
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
import wikipedia
from duckduckgo_search import DDGS
import google.generativeai as genai

# Multimedya ve Diğerleri
from PIL import Image, ImageDraw, ImageFont
import speech_recognition as sr
import pyttsx3

# İsteğe Bağlı: Token Sayımı için
try:
    import tiktoken
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    tiktoken_encoder = None
    print("INFO: tiktoken library not found. Token counting will be disabled.")

# Supabase (isteğe bağlı, loglama/feedback için)
try:
    from supabase import create_client, Client
    from postgrest.exceptions import APIError as SupabaseAPIError
except ImportError:
    print("ERROR: Supabase kütüphanesi bulunamadı. Loglama/Feedback devre dışı.")
    create_client = None
    Client = None
    SupabaseAPIError = Exception

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+ Enhanced",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.1.5 Pro+ Enhanced (Refactored)" # Sürüm güncellendi
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
AI_LOGO_PATH = "ai_logo.png"

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

KNOWLEDGE_BASE = {} # Will be loaded during initialization

# --- API Anahtarı ve Gemini Yapılandırması ---
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
        # Fixed model parameters as AI customization is removed
        model_name = 'gemini-1.5-flash-latest'
        system_prompt = "" # No system prompt by default when customization is removed
        config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096
        )
        model_args = {
            "model_name": model_name,
            "safety_settings": safety,
            "generation_config": config
        }
        # If a system prompt was ever needed, it would be added here
        # if system_prompt and system_prompt.strip():
        #     model_args["system_instruction"] = system_prompt.strip()

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

# --- Supabase İstemcisini Başlatma ---
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
        if 'ai_logo_warning_shown' not in st.session_state:
            st.toast(f"AI logo dosyası ({image_path}) bulunamadı. Lütfen projeye ekleyin.", icon="🖼️")
            st.session_state.ai_logo_warning_shown = True
        return fallback_emoji
    except Exception as e:
        print(f"HATA: AI Logo yüklenirken sorun oluştu ({image_path}): {e}. Varsayılan emoji '{fallback_emoji}' kullanılacak.")
        return fallback_emoji

# --- TTS Engine Initialization (moved from `initialize_session_state` to a cached function) ---
tts_engine = None
tts_init_error_global = None

@st.cache_resource(ttl=3600)
def init_tts_engine_cached():
    global tts_init_error_global
    try:
        engine = pyttsx3.init()
        # You might want to set properties like voice, rate, volume here
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[0].id) # Example: set to first voice
        print("INFO: TTS engine initialized.")
        tts_init_error_global = None
        return engine, None
    except Exception as e:
        tts_init_error_global = f"🛑 TTS motoru başlatılamadı: {e}. Sesli yanıtlar devre dışı."
        print(f"ERROR: TTS Init Failed: {e}")
        return None, tts_init_error_global

def speak(text):
    global tts_engine
    if st.session_state.tts_enabled and tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Metin okuma sırasında hata: {e}", icon="🔇")
            print(f"ERROR: TTS Say Failed: {e}")
    elif not tts_engine:
        st.warning("TTS motoru aktif değil veya başlatılamadı. Ayarları kontrol edin.", icon="🔇")

# --- Helper Functions ---
def _clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS):
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return clean_text[:max_chars] if len(clean_text) > max_chars else clean_text
    except requests.exceptions.RequestException as e:
        return f"Web içeriği çekilemedi ({url}): {e}"
    except Exception as e:
        return f"Web içeriği işlenirken hata oluştu ({url}): {e}"

def search_web(query):
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)
        if not results:
            wiki_search = wikipedia.search(query, results=3)
            if wiki_search:
                try:
                    wiki_page = wikipedia.page(wiki_search[0], auto_suggest=False, redirect=True, summary=True)
                    return {"source": f"Wikipedia: {wiki_page.url}", "content": wiki_page.summary}
                except wikipedia.exceptions.PageError:
                    return {"source": "Wikipedia", "content": "Üzgünüm, Wikipedia'da bu konuda bir sayfa bulunamadı."}
                except wikipedia.exceptions.DisambiguationError as e:
                    return {"source": "Wikipedia", "content": f"Arama çok genel, lütfen daha spesifik olun. Olası seçenekler: {e.options[:5]}"}
            return {"source": "DuckDuckGo/Wikipedia", "content": "Üzgünüm, web'de veya Wikipedia'da tatmin edici sonuç bulunamadı."}
        
        snippets = []
        for i, r in enumerate(results):
            snippets.append(f"Kaynak {i+1}: {r.get('title', 'N/A')} - {r.get('href', 'N/A')}\n{r.get('body', 'N/A')}\n")
        
        full_content = "\n".join(snippets)
        return {"source": "DuckDuckGo Search", "content": full_content}
    except Exception as e:
        return {"source": "Search Error", "content": f"Web araması sırasında hata oluştu: {e}"}

def kb_chatbot_response(query, knowledge_base_dict):
    query_lower = query.lower()
    for key, responses in knowledge_base_dict.items():
        if key in query_lower:
            return random.choice(responses)
    return None

def _get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def load_all_chats_cached(file_path=CHAT_HISTORY_FILE):
    """Loads all chat history from a JSON file, cached."""
    all_chats = {}
    errors = []
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_chats = json.load(f)
            # Ensure each chat is a list of dicts, and each message has 'role' and 'parts'
            for chat_id, messages in all_chats.items():
                if not isinstance(messages, list):
                    all_chats[chat_id] = []
                    errors.append({'type': 'warning', 'text': f"Sohbet ID '{chat_id}' geçersiz formatta, sıfırlandı.", 'icon': '⚠️'})
                    continue
                cleaned_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and 'role' in msg and 'parts' in msg:
                        cleaned_messages.append(msg)
                    else:
                        errors.append({'type': 'warning', 'text': f"Sohbet ID '{chat_id}' içinde bozuk mesajlar bulundu, bazıları atlandı.", 'icon': '⚠️'})
                all_chats[chat_id] = cleaned_messages
            if errors:
                errors.append({'type': 'toast', 'text': "Sohbet geçmişi yüklenirken bazı hatalar oluştu. Detaylar için uyarıları kontrol edin.", 'icon': '🚨'})
            return all_chats, errors
        else:
            errors.append({'type': 'toast', 'text': "Sohbet geçmişi dosyası bulunamadı. Yeni bir sohbet başlatılıyor.", 'icon': '🆕'})
            return {}, errors
    except json.JSONDecodeError as e:
        errors.append({'type': 'error', 'text': f"Sohbet geçmişi dosyası bozuk (JSON Hatası): {e}. Yeni bir sohbet başlatılıyor.", 'icon': '🗑️'})
        return {}, errors
    except Exception as e:
        errors.append({'type': 'error', 'text': f"Sohbet geçmişi yüklenirken beklenmeyen hata: {e}. Yeni bir sohbet başlatılıyor.", 'icon': '🚨'})
        return {}, errors

def save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE):
    """Saves all chat history to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chats_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Sohbet geçmişi kaydedilemedi: {e}", icon="💾")
        print(f"ERROR: Failed to save chat history: {e}")

def get_gemini_response(prompt_text, history_list, stream_output=False):
    global gemini_model, gemini_init_error_global
    if not gemini_model:
        return f"{GEMINI_ERROR_PREFIX} {gemini_init_error_global or 'Gemini modeli başlatılamadı.'}"

    try:
        chat_session = gemini_model.start_chat(history=history_list)
        if stream_output:
            response_chunks = []
            for chunk in chat_session.send_message(prompt_text, stream=True):
                response_chunks.append(chunk.text)
                yield "".join(response_chunks) # Yield cumulative response
            return # Ensure generator finishes
        else:
            response = chat_session.send_message(prompt_text)
            return response.text
    except genai.types.BlockedPromptException as e:
        return f"{GEMINI_ERROR_PREFIX} Güvenlik politikaları nedeniyle yanıt engellendi. Lütfen daha az hassas bir ifade kullanmayı deneyin. Detay: {e}"
    except Exception as e:
        print(f"Gemini yanıt hatası: {e}")
        import traceback
        print(traceback.format_exc())
        return f"{GEMINI_ERROR_PREFIX} Yapay zeka yanıtı alınamadı: {e}"

# Supabase logging functions
def log_to_supabase(table_name, data_dict):
    global supabase
    if supabase:
        try:
            response = supabase.table(table_name).insert(data_dict).execute()
            if response.data:
                print(f"INFO: Logged to Supabase table '{table_name}'.")
                return True
            else:
                print(f"WARNING: Supabase log returned no data. Status: {response.status}, Error: {response.status_code}")
                return False
        except SupabaseAPIError as e:
            print(f"ERROR: Supabase API Error logging to {table_name}: {e.message}")
            return False
        except Exception as e:
            print(f"ERROR: Supabase logging failed for table '{table_name}': {e}")
            return False
    return False

def log_interaction(prompt, ai_response, source, message_id, chat_id_val):
    if not supabase: return
    try:
        log_data = {
            "session_id": st.session_state.session_id,
            "chat_id": chat_id_val,
            "message_id": message_id,
            "user_prompt": prompt,
            "ai_response": ai_response,
            "response_source": source,
            "app_version": APP_VERSION,
            "model_name": st.session_state.get('gemini_model_name', 'N/A'),
            "temperature": st.session_state.get('gemini_temperature', 0.7),
            "timestamp": datetime.now().isoformat()
        }
        log_to_supabase(SUPABASE_TABLE_LOGS, log_data)
    except Exception as e:
        print(f"ERROR: Failed to prepare or log interaction to Supabase: {e}")

def log_feedback(message_id, user_prompt, ai_response, feedback_type, comment=""):
    if not supabase: return
    try:
        feedback_data = {
            "session_id": st.session_state.session_id,
            "message_id": message_id,
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            "feedback_type": feedback_type,
            "comment": comment,
            "app_version": APP_VERSION,
            "timestamp": datetime.now().isoformat()
        }
        log_to_supabase(SUPABASE_TABLE_FEEDBACK, feedback_data)
        st.toast(f"Geri bildiriminiz ({feedback_type}) için teşekkürler!", icon="💖")
    except Exception as e:
        st.error(f"Geri bildirim kaydedilemedi: {e}", icon="❌")
        print(f"ERROR: Failed to log feedback to Supabase: {e}")

def get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id_val, use_stream=False):
    prompt_lower = prompt.lower()
    response_text = None
    source = "Yapay Zeka"

    # 1. Check Dynamic Functions
    for func_key, func in DYNAMIC_FUNCTIONS_MAP.items():
        if func_key in prompt_lower:
            response_text = func()
            source = "Dinamik Fonksiyon"
            log_interaction(prompt, response_text, source, msg_id, chat_id_val)
            yield {"text": response_text, "source": source}
            return

    # 2. Check Knowledge Base
    kb_response = kb_chatbot_response(prompt_lower, KNOWLEDGE_BASE)
    if kb_response:
        response_text = kb_response
        source = "Bilgi Tabanı"
        log_interaction(prompt, response_text, source, msg_id, chat_id_val)
        yield {"text": response_text, "source": source}
        return

    # 3. Check for specific commands / modules
    if "web ara:" in prompt_lower or "web'de ara:" in prompt_lower or "ara:" in prompt_lower:
        search_query = prompt_lower.split("web ara:")[-1].split("web'de ara:")[-1].split("ara:")[-1].strip()
        st.info(f"🌐 Web'de arama yapılıyor: '{search_query}'...", icon="🔍")
        search_result = search_web(search_query)
        response_text = search_result['content']
        source = f"Web Arama ({search_result['source']})"
    elif "yaratıcı metin:" in prompt_lower:
        creative_seed = prompt_lower.split("yaratıcı metin:")[-1].strip()
        st.info("✍️ Yaratıcı metin üretiliyor...", icon="💡")
        response_text = creative_response_generator(creative_seed)
        source = "Yaratıcı Stüdyo"
    elif "görsel oluştur:" in prompt_lower:
        image_prompt = prompt_lower.split("görsel oluştur:")[-1].strip()
        st.info(f"🖼️ Görsel oluşturuluyor: '{image_prompt}'...", icon="🎨")
        generated_image_path = generate_prompt_influenced_image(image_prompt)
        if generated_image_path and os.path.exists(generated_image_path):
            with open(generated_image_path, "rb") as f:
                img_bytes = f.read()
            yield {"image": img_bytes, "source": "Görsel Oluşturucu", "text": "Oluşturulan Görsel"}
            return # Image handling is different, return directly
        else:
            response_text = "Görsel oluşturulamadı. Lütfen tekrar deneyin veya daha farklı bir açıklama girin."
            source = "Görsel Oluşturucu (Hata)"
    else:
        # 4. Fallback to Gemini Pro if no specific command/KB match
        try:
            gemini_response_generator = get_gemini_response(prompt, history, stream_output=use_stream)
            if use_stream:
                for chunk in gemini_response_generator:
                    if chunk.startswith(GEMINI_ERROR_PREFIX):
                        response_text = chunk
                        source = "Yapay Zeka (Hata)"
                        break
                    yield {"text": chunk, "source": "Yapay Zeka (Akış)"}
                if not response_text: # If no error, the last chunk is the full response
                    response_text = chunk
            else:
                response_text = gemini_response_generator
                if response_text.startswith(GEMINI_ERROR_PREFIX):
                    source = "Yapay Zeka (Hata)"

        except Exception as e:
            response_text = f"{GEMINI_ERROR_PREFIX} Yapay zeka ile iletişim hatası: {e}"
            source = "Yapay Zeka (Kritik Hata)"
    
    if response_text:
        log_interaction(prompt, response_text, source, msg_id, chat_id_val)
        yield {"text": response_text, "source": source}
    else:
        log_interaction(prompt, DEFAULT_ERROR_MESSAGE, "Sistem (Hata)", msg_id, chat_id_val)
        yield {"text": DEFAULT_ERROR_MESSAGE, "source": "Sistem (Hata)"}

# --- Creative Modules ---
def creative_response_generator(prompt_text, length_mode="orta", style_mode="genel"):
    """
    Generates creative text using Gemini.
    Length and style modes are simplified/removed in this version since AI customization is removed.
    """
    prompt = f"Aşağıdaki konuda yaratıcı ve özgün bir metin yaz: '{prompt_text}'. Konuyla ilgili ilgi çekici ve detaylı bir çıktı oluştur."
    
    # We still use the global model, but its config is now fixed
    global gemini_model
    if not gemini_model:
        return "Yaratıcı modül için yapay zeka modeli kullanılamıyor."

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Yaratıcı metin üretilirken hata oluştu: {e}"

def generate_new_idea_creative(seed_text, style="genel"):
    """Generates a new idea based on a seed text."""
    prompt = f"'{seed_text}' konusunda yenilikçi ve orijinal bir fikir üret. Detaylı ve uygulanabilir bir konsept sun."
    global gemini_model
    if not gemini_model:
        return "Fikir üretimi için yapay zeka modeli kullanılamıyor."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Yeni fikir üretilirken hata oluştu: {e}"

def advanced_word_generator(base_word):
    """Generates related words or concepts for a given word."""
    prompt = f"'{base_word}' kelimesiyle ilişkili 10-15 tane farklı kelime veya kavram listesi oluştur. Açıklama yapma, sadece kelimeleri virgülle ayırarak listele."
    global gemini_model
    if not gemini_model:
        return "Kelime üretimi için yapay zeka modeli kullanılamıyor."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Kelime üretilirken hata oluştu: {e}"

def generate_prompt_influenced_image(prompt):
    """
    Generates a simple image based on a prompt.
    This is a dummy function as real image generation requires an external API (e.g., DALL-E, Stable Diffusion).
    For demonstration, it creates a placeholder image with the prompt text.
    """
    try:
        # Create a dummy image based on the prompt
        img_size = (500, 300)
        img = Image.new('RGB', img_size, color = (73, 109, 137))
        d = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_FILE, 20)
        except IOError:
            font = ImageFont.load_default()
            print(f"WARNING: Font file '{FONT_FILE}' not found. Using default font.")

        text_content = f"'{prompt}' ile ilgili bir görsel hayal edin."
        
        # Wrap text for better display
        lines = []
        words = text_content.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            if font.getlength(test_line) < img_size[0] - 40: # 20px padding on each side
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))
        
        # Calculate text position
        y_text = 50
        for line in lines:
            text_width = font.getlength(line)
            x_text = (img_size[0] - text_width) / 2
            d.text((x_text, y_text), line, font=font, fill=(255, 255, 255))
            y_text += 25 # Line height

        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        img_filename = f"{output_dir}/image_{uuid.uuid4().hex[:8]}.png"
        img.save(img_filename)
        return img_filename
    except Exception as e:
        st.error(f"Görsel oluşturma hatası: {e}")
        return None

# --- Session State Başlatma ---
def initialize_session_state():
    defaults = {
        'all_chats': {}, 'active_chat_id': None,
        'app_mode': "Yazılı Sohbet", 'user_name': None, 'user_avatar_bytes': None,
        'show_main_app': False, 'greeting_message_shown': False,
        'tts_enabled': True, 'gemini_stream_enabled': True,
        'message_id_counter': 0, 'last_ai_response_for_feedback': None,
        'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
        'feedback_comment_input': "", 'show_feedback_comment_form': False,
        'session_id': str(uuid.uuid4()), 'last_feedback_type': 'positive',
        'models_initialized': False,
        'ai_logo_warning_shown': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# --- Modelleri ve İstemcileri Başlatma ---
if not st.session_state.models_initialized:
    print("INFO: Uygulama kaynakları ilk kez başlatılıyor...")
    gemini_model, gemini_init_error_global = initialize_gemini_model()
    if gemini_model: st.toast(f"✨ Gemini modeli başarıyla yüklendi!", icon="🤖")
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

    # If no active chat, and there are existing chats, set the most recent one as active
    if not st.session_state.active_chat_id and st.session_state.all_chats:
        try:
            # Assuming chat_ids are "chat_X" where X is incremental
            # Find the highest X
            max_chat_num = -1
            for chat_id in st.session_state.all_chats.keys():
                try:
                    num = int(chat_id.split('_')[-1])
                    if num > max_chat_num:
                        max_chat_num = num
                except ValueError:
                    continue # Skip invalid chat IDs
            if max_chat_num != -1:
                st.session_state.active_chat_id = f"chat_{max_chat_num}"
            else: # If no valid chat IDs found, create a default one
                st.session_state.active_chat_id = "chat_0"
                if not st.session_state.all_chats.get("chat_0"):
                    st.session_state.all_chats["chat_0"] = []
        except Exception as e:
            print(f"WARNING: Aktif sohbet ID'si belirlenirken sorun: {e}. Varsayılan 'chat_0' ayarlanıyor.")
            st.session_state.active_chat_id = "chat_0"
            if "chat_0" not in st.session_state.all_chats:
                st.session_state.all_chats["chat_0"] = []
    elif not st.session_state.active_chat_id: # No chats exist at all
        st.session_state.active_chat_id = "chat_0"
        st.session_state.all_chats["chat_0"] = []
        save_all_chats(st.session_state.all_chats) # Save the initial empty chat

    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    kb_data, kb_error = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    globals()['KNOWLEDGE_BASE'] = kb_data
    globals()['knowledge_base_load_error_global'] = kb_error
    st.session_state.models_initialized = True
    print("INFO: Uygulama kaynaklarının ilk başlatılması tamamlandı.")
else:
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    current_kb, kb_load_err_rerun = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    if kb_load_err_rerun and kb_load_err_rerun != globals().get('knowledge_base_load_error_global'):
        globals()['knowledge_base_load_error_global'] = kb_load_err_rerun
    elif not kb_load_err_rerun and globals().get('knowledge_base_load_error_global'):
        globals()['knowledge_base_load_error_global'] = None
        st.toast("Bilgi tabanı başarıyla güncellendi/yüklendi.", icon="📚")
    globals()['KNOWLEDGE_BASE'] = current_kb

# --- ARAYÜZ FONKSİYONLARI ---

def display_chat_message_with_feedback(message_data, message_index, current_chat_id):
    role = message_data.get('role', 'model')
    content_text = str(message_data.get('parts', ''))
    is_user_message = (role == 'user')

    ai_default_avatar = load_ai_avatar()

    if is_user_message:
        sender_display_name = st.session_state.get('user_name', 'Kullanıcı')
        avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes)) if st.session_state.user_avatar_bytes else "🧑"
    else: # AI mesajı
        sender_display_name = message_data.get('sender_display', APP_NAME)
        if "Gemini" in sender_display_name: avatar_icon = "✨"
        elif any(w in sender_display_name.lower() for w in ["web", "wiki", "arama", "ddg"]): avatar_icon = "🌐"
        elif any(w in sender_display_name.lower() for w in ["bilgi", "fonksiyon", "taban"]): avatar_icon = "📚"
        elif "Yaratıcı" in sender_display_name: avatar_icon = "🎨"
        elif "Görsel" in sender_display_name: avatar_icon = "🖼️"
        else: avatar_icon = ai_default_avatar

    with st.chat_message(role, avatar=avatar_icon):
        if "```" in content_text:
            text_parts = content_text.split("```")
            for i, part in enumerate(text_parts):
                if i % 2 == 1:
                    language_match = re.match(r"(\w+)\n", part)
                    code_block_content = part[len(language_match.group(0)):] if language_match else part
                    actual_code_language = language_match.group(1).lower() if language_match else None
                    st.code(code_block_content.strip(), language=actual_code_language)
                    if st.button(f"📋 Kopyala", key=f"copy_code_{current_chat_id}_{message_index}_{i}", help="Kodu panoya kopyala", use_container_width=False):
                        st.write_to_clipboard(code_block_content.strip())
                        st.toast("Kod panoya kopyalandı!", icon="✅")
                elif part.strip():
                    st.markdown(part, unsafe_allow_html=True)
        elif content_text.strip():
            st.markdown(content_text, unsafe_allow_html=True)
        else:
            st.caption("[Mesaj içeriği bulunmuyor]")

        if not is_user_message and content_text.strip():
            token_count_display_str = ""
            ts_col, tts_col, fb_col = st.columns([0.75, 0.1, 0.15])
            if tiktoken_encoder:
                try:
                    token_count = len(tiktoken_encoder.encode(content_text))
                    token_count_display_str = f" (~{token_count} token)"
                except Exception:
                    pass
            with ts_col:
                source_name_only = sender_display_name.split('(')[-1].replace(')', '').strip() if '(' in sender_display_name else sender_display_name
                st.caption(f"Kaynak: {source_name_only}{token_count_display_str}")
            with tts_col:
                if st.session_state.tts_enabled and globals().get('tts_engine'):
                    # Unique key for TTS button
                    if st.button("🔊", key=f"tts_btn_{current_chat_id}_{message_index}", help="Yanıtı sesli oku", use_container_width=True):
                        speak(content_text)
            with fb_col:
                # Unique key for Feedback button
                if st.button("✍️", key=f"feedback_btn_{current_chat_id}_{message_index}", help="Yanıt hakkında geri bildirim ver", use_container_width=True):
                    st.session_state.current_message_id_for_feedback = f"{current_chat_id}_{message_index}"
                    prev_prompt = "[İstem bulunamadı]"
                    idx_prev = message_index - 1
                    if idx_prev >= 0 and st.session_state.all_chats[current_chat_id][idx_prev]['role'] == 'user':
                        prev_prompt = st.session_state.all_chats[current_chat_id][idx_prev]['parts']
                    st.session_state.last_user_prompt_for_feedback = prev_prompt
                    st.session_state.last_ai_response_for_feedback = content_text
                    st.session_state.show_feedback_comment_form = True
                    st.session_state.feedback_comment_input = ""
                    st.rerun()

def display_settings_section():
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**")
        new_user_name = st.text_input("Adınız:", value=st.session_state.user_name, key="change_user_name_input", label_visibility="collapsed", placeholder="Görünür adınız...")
        if new_user_name != st.session_state.user_name and new_user_name.strip():
            st.session_state.user_name = new_user_name.strip()
            load_knowledge_from_file.clear()
            st.toast("Adınız güncellendi!", icon="✏️")
            st.rerun()

        avatar_col1, avatar_col2 = st.columns([0.8, 0.2])
        with avatar_col1:
            uploaded_avatar_file = st.file_uploader("Avatar yükle (PNG, JPG - maks 2MB):", type=["png", "jpg", "jpeg"], key="upload_avatar_file",label_visibility="collapsed")
            if uploaded_avatar_file:
                if uploaded_avatar_file.size > 2 * 1024 * 1024:
                    st.error("Dosya boyutu 2MB'den büyük olamaz!", icon="❌")
                else:
                    st.session_state.user_avatar_bytes = uploaded_avatar_file.getvalue()
                    st.toast("Avatarınız güncellendi!", icon="🖼️")
                    st.rerun()
        with avatar_col2:
            if st.session_state.user_avatar_bytes:
                st.image(st.session_state.user_avatar_bytes, width=60)
                if st.button("🗑️ Kaldır", key="remove_avatar_button", help="Yüklü avatarı kaldır", use_container_width=True):
                    st.session_state.user_avatar_bytes = None
                    st.toast("Avatar kaldırıldı.", icon="🗑️")
                    st.rerun()
        st.caption("Avatarınız sadece bu tarayıcı oturumunda saklanır.")
        st.divider()

        st.subheader("🤖 Yapay Zeka ve Arayüz Ayarları")
        tts_toggle_col, stream_toggle_col = st.columns(2)
        is_tts_engine_ok = globals().get('tts_engine') is not None
        with tts_toggle_col:
            st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not is_tts_engine_ok, help="Yanıtları sesli olarak oku (TTS motoru aktifse).")
        with stream_toggle_col:
            st.session_state.gemini_stream_enabled = st.toggle("Yanıt Akışı (Streaming)", value=st.session_state.gemini_stream_enabled, help="Yanıtları kelime kelime alarak daha hızlı gösterim sağla.")
        st.divider()

        st.subheader("🧼 Geçmiş Yönetimi")
        # --- NEW: Single button to clear ALL chat history ---
        is_clear_all_disabled = not bool(st.session_state.all_chats.get(st.session_state.active_chat_id))

        if st.button("🧹 Sohbet Geçmişini Temizle", use_container_width=True, type="secondary", key="clear_all_chats_final_button", help="Mevcut sohbetin tüm geçmişini kalıcı olarak siler.", disabled=is_clear_all_disabled):
            st.session_state.all_chats[st.session_state.active_chat_id] = [] # Clear the current active chat
            save_all_chats(st.session_state.all_chats)
            st.toast("Sohbet geçmişi temizlendi!", icon="🗑️")
            st.rerun()

def display_chat_list_and_about(left_column_ref):
    with left_column_ref:
        st.markdown("### Sohbet Geçmişi")
        # Display the single, current chat. No list of chats or new chat button.
        st.info(f"Aktif Sohbet: `{st.session_state.active_chat_id}`", icon="💬")
        st.markdown("---")
        
        st.markdown("### Hakkında")
        st.info(f"**{APP_NAME}** v{APP_VERSION}\n\n"
                f"Streamlit ve Google Gemini Pro ile geliştirilmiştir.\n\n"
                "Desteklenen Özellikler:\n"
                "- 💬 Genel sohbet\n"
                "- 🌐 Web araması (DuckDuckGo, Wikipedia)\n"
                "- 📚 Bilgi tabanı yanıtları\n"
                "- ✍️ Yaratıcı metin üretimi\n"
                "- 🖼️ Basit görsel oluşturma (örnek)\n"
                "- 🗣️ Metin okuma (TTS)\n"
                "- ⬆️ Geri bildirim mekanizması (Supabase)\n"
                f"© {CURRENT_YEAR}")
        st.markdown("---")

def display_feedback_form_if_active():
    if st.session_state.show_feedback_comment_form and st.session_state.current_message_id_for_feedback:
        message_id = st.session_state.current_message_id_for_feedback
        user_prompt = st.session_state.last_user_prompt_for_feedback
        ai_response = st.session_state.last_ai_response_for_feedback

        st.markdown("---")
        st.markdown("#### Geri Bildiriminiz")
        st.info(f"**Soru:** {user_prompt[:150]}{'...' if len(user_prompt) > 150 else ''}")
        st.info(f"**Yanıt:** {ai_response[:150]}{'...' if len(ai_response) > 150 else ''}")

        feedback_type = st.radio("Bu yanıttan memnun musunuz?", ["Olumlu 👍", "Olumsuz 👎"], index=0, key=f"feedback_radio_{message_id}")
        st.session_state.last_feedback_type = "positive" if feedback_type == "Olumlu 👍" else "negative"

        comment_placeholder = "İyileştirme önerileriniz veya neden olumlu/olumsuz bulduğunuzu belirtin..."
        st.session_state.feedback_comment_input = st.text_area("Yorum (isteğe bağlı):", value=st.session_state.feedback_comment_input, placeholder=comment_placeholder, height=80, key=f"feedback_comment_area_{message_id}")

        col_submit_fb, col_cancel_fb = st.columns([0.6, 0.4])
        with col_submit_fb:
            if st.button("Geri Bildirimi Gönder", key=f"submit_feedback_btn_{message_id}", type="primary", use_container_width=True):
                log_feedback(
                    message_id=message_id,
                    user_prompt=user_prompt,
                    ai_response=ai_response,
                    feedback_type=st.session_state.last_feedback_type,
                    comment=st.session_state.feedback_comment_input
                )
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.last_ai_response_for_feedback = None
                st.session_state.last_user_prompt_for_feedback = None
                st.rerun()
        with col_cancel_fb:
            if st.button("İptal", key=f"cancel_feedback_btn_{message_id}", use_container_width=True):
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.last_ai_response_for_feedback = None
                st.session_state.last_user_prompt_for_feedback = None
                st.rerun()
        st.markdown("---")

def display_chat_interface_main():
    current_chat_id = st.session_state.active_chat_id
    if current_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[current_chat_id] = []
        save_all_chats(st.session_state.all_chats)

    chat_messages = st.session_state.all_chats[current_chat_id]

    # Display existing messages
    for i, message in enumerate(chat_messages):
        display_chat_message_with_feedback(message, i, current_chat_id)

    display_feedback_form_if_active()

    # Chat input
    if st.session_state.show_feedback_comment_form:
        st.stop() # Prevent new input if feedback form is active

    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_area("Mesajınızı yazın veya bir komut girin:", key="user_chat_input", height=100, placeholder="Örn: 'Merhaba', 'web ara: Streamlit', 'yaratıcı metin: uzaylılar'...", disabled=globals().get('gemini_init_error_global') is not None)
        send_button = st.form_submit_button("Gönder", type="primary", use_container_width=True, disabled=globals().get('gemini_init_error_global') is not None)

        if send_button and user_input:
            prompt = _clean_text(user_input)
            if not prompt:
                st.warning("Lütfen boş mesaj göndermeyin.", icon="⚠️")
                return

            st.session_state.message_id_counter += 1
            current_message_index = len(chat_messages)
            current_message_id = f"{current_chat_id}_{current_message_index}"

            # Append user message
            st.session_state.all_chats[current_chat_id].append({"role": "user", "parts": prompt, "sender_display": st.session_state.user_name})
            save_all_chats(st.session_state.all_chats)
            st.rerun() # Rerun to display user message immediately

    if chat_messages and chat_messages[-1]['role'] == 'user':
        # This block runs after the user message is displayed, to get AI response
        with st.spinner("Hanogt AI yanıtlıyor..."):
            history_for_gemini = [
                {"role": m["role"], "parts": m["parts"]}
                for m in chat_messages if m["role"] in ["user", "model"]
            ]

            full_ai_response = ""
            ai_source = "Yapay Zeka" # Default

            try:
                # Use the generator directly for streamed or non-streamed output
                response_generator = get_hanogt_response_orchestrator(
                    prompt=chat_messages[-1]['parts'], # Use the last user message
                    history=history_for_gemini[:-1], # History excluding the very last user prompt
                    msg_id=current_message_id,
                    chat_id_val=current_chat_id,
                    use_stream=st.session_state.gemini_stream_enabled
                )

                if st.session_state.gemini_stream_enabled:
                    response_placeholder = st.empty()
                    for response_part in response_generator:
                        if "image" in response_part:
                            st.image(response_part["image"], caption=response_part["text"], use_column_width=True)
                            full_ai_response = response_part["text"] + " (Görsel oluşturuldu)" # Indicate image was generated
                            ai_source = response_part["source"]
                            break # No further text chunks for image response
                        else:
                            full_ai_response += response_part["text"]
                            ai_source = response_part["source"]
                            response_placeholder.markdown(full_ai_response + "▌", unsafe_allow_html=True) # Add blinking cursor
                    response_placeholder.markdown(full_ai_response, unsafe_allow_html=True) # Final render without cursor
                else:
                    # For non-streaming, generator will yield one dictionary with 'text' and 'source'
                    single_response = next(response_generator)
                    if "image" in single_response:
                        st.image(single_response["image"], caption=single_response["text"], use_column_width=True)
                        full_ai_response = single_response["text"] + " (Görsel oluşturuldu)"
                        ai_source = single_response["source"]
                    else:
                        full_ai_response = single_response["text"]
                        ai_source = single_response["source"]
                        st.markdown(full_ai_response, unsafe_allow_html=True)

            except Exception as e:
                full_ai_response = f"Yanıt alınırken beklenmeyen bir hata oluştu: {e}"
                ai_source = "Sistem (Hata)"
                st.error(full_ai_response, icon="❌")
                print(f"ERROR: Orchestrator failed: {e}")

            # Append AI response
            st.session_state.all_chats[current_chat_id].append({
                "role": "model",
                "parts": full_ai_response,
                "sender_display": ai_source
            })
            save_all_chats(st.session_state.all_chats)
            st.rerun() # Rerun to display the AI response and clear input

    # Auto-scroll to the bottom of the chat
    st.markdown("<div id='end_of_chat'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            var element = document.getElementById('end_of_chat');
            if (element) {
                element.scrollIntoView({behavior: "smooth", block: "end"});
            }
        </script>
        """,
        unsafe_allow_html=True
    )

# --- UYGULAMA ANA AKIŞI ---
st.markdown(f"<h1 style='text-align:center;color:#0078D4;'>{APP_NAME} <sup style='font-size:0.6em;color:#555;'>v{APP_VERSION}</sup></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-style:italic;color:#555;'>Yapay Zeka Destekli Kişisel Asistanınız</p>", unsafe_allow_html=True)
st.markdown("---")

# Global başlatma hatalarını göster
if globals().get('gemini_init_error_global'): st.warning(globals().get('gemini_init_error_global'), icon="🗝️")
if globals().get('supabase_init_error_global'): st.warning(globals().get('supabase_init_error_global'), icon="🧱")
if globals().get('tts_init_error_global'): st.warning(globals().get('tts_init_error_global'), icon="🔇")
if globals().get('knowledge_base_load_error_global'): st.warning(globals().get('knowledge_base_load_error_global'), icon="📚")

# Giriş ekranı veya ana uygulama
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce Sizi Tanıyalım")
    login_cols = st.columns([0.2, 0.6, 0.2])
    with login_cols[1]:
        with st.form("user_login_form"):
            user_entered_name = st.text_input("Size nasıl hitap etmemizi istersiniz?", placeholder="İsminiz...", key="login_name_input", value=st.session_state.get('user_name', ''))
            if st.form_submit_button("✨ Uygulamayı Başlat", use_container_width=True, type="primary"):
                if user_entered_name and user_entered_name.strip():
                    st.session_state.user_name = user_entered_name.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False
                    load_knowledge_from_file.clear()
                    st.rerun()
                else:
                    st.error("Lütfen geçerli bir isim giriniz.")
else: # Ana Uygulama
    if not st.session_state.greeting_message_shown:
        st.success(f"Tekrar hoş geldiniz, **{st.session_state.user_name}**! Size nasıl yardımcı olabilirim?", icon="🎉")
        st.session_state.greeting_message_shown = True

    app_left_column, app_main_column = st.columns([1, 3])

    display_chat_list_and_about(app_left_column)

    with app_main_column:
        display_settings_section()
        st.markdown("#### Uygulama Modu")
        app_modes = {"Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya Yükle)": "🎤", "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️"}
        mode_options_keys = list(app_modes.keys())
        try:
            current_mode_index = mode_options_keys.index(st.session_state.app_mode)
        except ValueError:
            current_mode_index = 0
            st.session_state.app_mode = mode_options_keys[0]

        selected_app_mode = st.radio("Çalışma Modunu Seçin:",
                                     options=mode_options_keys,
                                     index=current_mode_index,
                                     format_func=lambda k: f"{app_modes[k]} {k}",
                                     horizontal=True,
                                     label_visibility="collapsed",
                                     key="app_mode_selection_radio")

        if selected_app_mode != st.session_state.app_mode:
            st.session_state.app_mode = selected_app_mode
            st.rerun()
        st.markdown("<hr style='margin-top:0.1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)

        current_app_mode = st.session_state.app_mode

        if current_app_mode == "Yazılı Sohbet":
            display_chat_interface_main()
        elif current_app_mode == "Sesli Sohbet (Dosya Yükle)":
            st.warning("Bu mod henüz aktif değil. Lütfen 'Yazılı Sohbet' modunu kullanın.", icon="⚠️")
            st.caption("Sesli sohbet özelliği geliştirme aşamasındadır.")
        elif current_app_mode == "Yaratıcı Stüdyo":
            st.subheader("🎨 Yaratıcı Stüdyo")
            st.write("Burada yaratıcı metinler üretebilirsiniz.")
            creative_prompt = st.text_area("Yaratıcı metin için konu:", placeholder="Örn: 'uzaylıların dünyayı ziyareti'", height=100)
            if st.button("Metin Oluştur", use_container_width=True, type="primary"):
                if creative_prompt:
                    with st.spinner("Yaratıcı metin oluşturuluyor..."):
                        generated_text = creative_response_generator(creative_prompt)
                        st.markdown("#### Oluşturulan Metin:")
                        st.write(generated_text)
                else:
                    st.warning("Lütfen bir konu girin.")
            st.markdown("---")
            st.subheader("💡 Yeni Fikir Üretici")
            idea_seed = st.text_input("Fikir için başlangıç noktası:", placeholder="Örn: 'geleceğin ulaşımı'")
            if st.button("Fikir Üret", use_container_width=True, type="primary", key="generate_idea_button"):
                if idea_seed:
                    with st.spinner("Yeni fikir üretiliyor..."):
                        generated_idea = generate_new_idea_creative(idea_seed)
                        st.markdown("#### Oluşturulan Fikir:")
                        st.write(generated_idea)
                else:
                    st.warning("Lütfen bir başlangıç noktası girin.")
            st.markdown("---")
            st.subheader("📝 Kelime İlişkilendirici")
            word_to_associate = st.text_input("İlişkili kelimeler üretmek için kelime:", placeholder="Örn: 'yapay zeka'")
            if st.button("Kelime Üret", use_container_width=True, type="primary", key="generate_words_button"):
                if word_to_associate:
                    with st.spinner("İlişkili kelimeler bulunuyor..."):
                        associated_words = advanced_word_generator(word_to_associate)
                        st.markdown("#### İlişkili Kelimeler:")
                        st.write(associated_words)
                else:
                    st.warning("Lütfen bir kelime girin.")
        elif current_app_mode == "Görsel Oluşturucu":
            st.subheader("🖼️ Görsel Oluşturucu (Örnek)")
            st.write("Bu mod, verdiğiniz açıklamalara göre basit temsili görseller oluşturur. Gerçek bir AI görsel üreticisi entegrasyonu değildir.")
            image_prompt_input = st.text_area("Oluşturulacak görseli açıklayın:", placeholder="Örn: 'güneşli bir kıyıda duran bir robot'", height=100)
            if st.button("Görsel Oluştur", use_container_width=True, type="primary"):
                if image_prompt_input:
                    with st.spinner("Görsel oluşturuluyor..."):
                        image_path = generate_prompt_influenced_image(image_prompt_input)
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, caption=f"Oluşturulan Görsel: '{image_prompt_input}'", use_column_width=True)
                        else:
                            st.error("Görsel oluşturulamadı. Lütfen farklı bir açıklama deneyin.")
                else:
                    st.warning("Lütfen bir görsel açıklaması girin.")

    st.markdown("<hr style='margin-top:1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    footer_cols = st.columns(3)
    with footer_cols[0]: st.caption(f"Kullanıcı: **{st.session_state.get('user_name', 'Tanımlanmamış')}**")
    with footer_cols[1]: st.caption(f"<div style='text-align:center;'>{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}</div>", unsafe_allow_html=True)
    with footer_cols[2]:
        ai_model_name_display = 'gemini-1.5-flash-latest'.split('/')[-1] # Fixed model name
        ai_status_text = "Aktif" if globals().get('gemini_model') else "Devre Dışı"
        logging_status_text = "Aktif" if globals().get('supabase') else "Devre Dışı"
        st.caption(f"<div style='text-align:right;'>AI: {ai_status_text} ({ai_model_name_display}) | Log: {logging_status_text}</div>", unsafe_allow_html=True)
