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
    from postgrest import APIError as SupabaseAPIError # SupabaseAPIError olarak düzeltildi
except ImportError:
    print("ERROR: Supabase kütüphanesi bulunamadı. Loglama/Feedback devre dışı.")
    create_client = None
    Client = None
    SupabaseAPIError = None # None olarak ayarla

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+ Enhanced",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.1.2 Pro+ FeatureRich" # Sürüm güncellendi (CacheReplay düzeltmesi)
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

# --- Dinamik Fonksiyonlar ---
DYNAMIC_FUNCTIONS_MAP = {
    "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
    "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})",
    "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})"
}

# --- Bilgi Tabanı ---
knowledge_base_load_error_global = None # Global olarak tanımla

@st.cache_data(ttl=3600)
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE, user_name_for_greeting="kullanıcı"):
    """Bilgi tabanını dosyadan yükler veya varsayılanı kullanır. UI elemanları içermez."""
    # global knowledge_base_load_error_global # Global değişkeni fonksiyon içinde değiştirmek için
    error_message = None # Yerel hata mesajı
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
            return merged_kb, None # Başarılı olursa None dön
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

def kb_chatbot_response(query, knowledge_base_dict):
    query_lower = query.lower().strip()
    if query_lower in DYNAMIC_FUNCTIONS_MAP:
        try:
            return DYNAMIC_FUNCTIONS_MAP[query_lower]()
        except Exception as e:
            st.error(f"Fonksiyon hatası ({query_lower}): {e}") # Bu st.error kalabilir, çünkü kb_chatbot_response cache'li değil.
            return DEFAULT_ERROR_MESSAGE
    if query_lower in knowledge_base_dict:
        resp = knowledge_base_dict[query_lower]
        return random.choice(resp) if isinstance(resp, list) else resp
    partial_matches = [resp for key, resp_list in knowledge_base_dict.items() if key in query_lower for resp in (resp_list if isinstance(resp_list, list) else [resp_list])]
    if partial_matches:
        return random.choice(list(set(partial_matches)))
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_score, best_responses = 0, []
    for key, resp_list in knowledge_base_dict.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words:
            continue
        score = len(query_words.intersection(key_words)) / len(key_words) if key_words else 0
        if score > 0.6:
            options = resp_list if isinstance(resp_list, list) else [resp_list]
            if score > best_score:
                best_score, best_responses = score, options
            elif score == best_score:
                best_responses.extend(options)
    if best_responses:
        return random.choice(list(set(best_responses)))
    return None

# --- API Anahtarı ve Gemini Yapılandırması ---
gemini_model = None
gemini_init_error_global = None

def initialize_gemini_model():
    global gemini_init_error_global
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta bulunamadı!"
        return None, gemini_init_error_global # Hata mesajını da dön
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
        #st.toast(f"✨ Gemini modeli ({model_name}) yüklendi!", icon="🤖") # Cache_replay_error önlemek için kaldırıldı
        print(f"INFO: Gemini modeli ({model_name}) yüklendi!")
        return model, None # Başarılı ise None dön
    except Exception as e:
        gemini_init_error_global = f"🛑 Gemini yapılandırma hatası: {e}."
        print(f"ERROR: Gemini Init Failed: {e}")
        return None, gemini_init_error_global # Hata mesajını dön

# --- Supabase İstemcisini Başlatma ---
supabase = None
supabase_init_error_global = None # Global olarak tanımla

@st.cache_resource(ttl=3600)
def init_supabase_client_cached():
    """Supabase istemcisini başlatır ve cache'ler. UI elemanları içermez."""
    if not create_client:
        error_msg = "Supabase kütüphanesi yüklenemedi."
        print(f"ERROR: {error_msg}")
        return None, error_msg
    url, key = st.secrets.get("SUPABASE_URL"), st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        error_msg = "Supabase URL veya Key Secrets'ta bulunamadı."
        print(f"ERROR: {error_msg}")
        return None, error_msg
    try:
        client: Client = create_client(url, key)
        print("INFO: Supabase client created successfully via cache function.")
        return client, None # Başarılı ise None dön
    except Exception as e:
        error_msg = f"Supabase bağlantısı sırasında hata: {e}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

# --- YARDIMCI FONKSİYONLAR ---
def _get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# --- TTS Motoru ---
tts_engine = None
tts_init_error_global = None

@st.cache_resource
def init_tts_engine_cached():
    """TTS motorunu başlatır. UI elemanları içermez."""
    # global tts_init_error_global # Globali değiştirmek için gerek yok, dönüş değeri kullanılacak
    try:
        engine = pyttsx3.init()
        #st.toast("🔊 TTS motoru hazır.", icon="🗣️") # CacheReplayClosureError önlemek için kaldırıldı
        print("INFO: TTS motoru başarıyla başlatıldı.")
        return engine, None # Başarılı ise None dön
    except Exception as e:
        error_message = f"TTS motoru başlatılamadı: {e}."
        print(f"ERROR: TTS Init Failed: {e}")
        return None, error_message # Hata mesajını dön

def speak(text):
    engine = globals().get('tts_engine')
    if not engine:
        st.toast("TTS motoru aktif değil.", icon="🔇") # Bu st.toast kalabilir, speak cache'li değil
        return
    if not st.session_state.get('tts_enabled', True):
        st.toast("TTS ayarlardan kapalı.", icon="🔇")
        return
    try:
        cleaned = re.sub(r'[^\w\s.,!?-]', '', text)
        engine.say(cleaned)
        engine.runAndWait()
    except RuntimeError as e:
        st.warning(f"TTS çalışma zamanı sorunu: {e}.", icon="🔊")
    except Exception as e:
        st.error(f"TTS hatası: {e}", icon="🔥")
        print(f"ERROR: TTS Speak Failed: {e}")

# --- Metin Temizleme ---
def _clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# --- Web Kazıma (Cache'li)---
@st.cache_data(ttl=600)
def scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS):
    """Bir URL'den içerik kazır. UI elemanlarını doğrudan kullanmaz, print ile loglar."""
    print(f"INFO: Scraping URL: {url}")
    messages_to_show_outside = [] # Fonksiyon dışında gösterilecek mesajlar
    try:
        parsed = urlparse(url)
        headers = {'User-Agent': USER_AGENT, 'Accept-Language': 'tr-TR,tr;q=0.9', 'Accept': 'text/html', 'DNT': '1'}
        if not all([parsed.scheme, parsed.netloc]) or parsed.scheme not in ['http', 'https']:
            messages_to_show_outside.append({'type': 'warning', 'text': f"Geçersiz URL: {url}", 'icon': "🔗"})
            return None, messages_to_show_outside
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get('content-type', '').lower()
        if 'html' not in ctype:
            messages_to_show_outside.append({'type': 'info', 'text': f"HTML değil ('{ctype}'). Atlanıyor: {url}", 'icon': "📄"})
            resp.close()
            return None, messages_to_show_outside
        html = ""
        size = 0
        max_size = max_chars * 12
        try:
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True, errors='ignore'):
                if chunk:
                    html += chunk
                    size += len(chunk.encode('utf-8', 'ignore'))
                if size > max_size:
                    messages_to_show_outside.append({'type': 'warning', 'text': f"HTML içeriği {max_size // 1024}KB'dan büyük, kesiliyor: {url}", 'icon': "✂️"})
                    break
        finally:
            resp.close()
        if not html:
            messages_to_show_outside.append({'type': 'warning', 'text': f"Boş içerik alındı: {url}", 'icon': "📄"})
            return None, messages_to_show_outside

        soup = BeautifulSoup(html, 'lxml')
        tags_to_remove = ["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio", "figure", "input", "select", "textarea"]
        for tag in soup.find_all(tags_to_remove):
            tag.decompose()
        content_parts = []
        selectors = ['article[class*="content"]', 'article[class*="post"]', 'main[id*="content"]', 'main', 'div[class*="post-body"]', 'div[itemprop="articleBody"]', 'article', '.content', '#content']
        container = next((found[0] for sel in selectors if (found := soup.select(sel, limit=1))), None)
        min_text_length = 80
        min_indicator_count = 1
        if container:
            for p_tag in container.find_all('p', limit=60):
                text = _clean_text(p_tag.get_text(separator=' ', strip=True))
                if len(text) > min_text_length and (text.count('.') + text.count('?') + text.count('!')) >= min_indicator_count:
                    content_parts.append(text)
        if not content_parts or len(" ".join(content_parts)) < 300:
            body_tag = soup.body
            if body_tag:
                raw_body_text = _clean_text(body_tag.get_text(separator='\n', strip=True))
                potential_parts = [p.strip() for p in raw_body_text.split('\n') if len(p.strip()) > min_text_length]
                if len(" ".join(potential_parts)) > 200:
                    messages_to_show_outside.append({'type': 'info', 'text': f"Genel body metni kullanıldı: {url}", 'icon': "ℹ️"})
                    content_parts = potential_parts[:40]
                else:
                    messages_to_show_outside.append({'type': 'info', 'text': f"Anlamlı içerik bulunamadı: {url}", 'icon': "📄"})
                    return None, messages_to_show_outside
            else:
                messages_to_show_outside.append({'type': 'info', 'text': f"Anlamlı içerik bulunamadı (body yok): {url}", 'icon': "📄"})
                return None, messages_to_show_outside
        cleaned_content = _clean_text("\n\n".join(content_parts))
        if not cleaned_content:
            messages_to_show_outside.append({'type': 'info', 'text': f"Kazıma sonucu boş içerik: {url}", 'icon': "📄"})
            return None, messages_to_show_outside
        final_content = cleaned_content[:max_chars] + ("..." if len(cleaned_content) > max_chars else "")
        messages_to_show_outside.append({'type': 'toast', 'text': f"'{urlparse(url).netloc}' içeriği başarıyla alındı.", 'icon': "✅"})
        return final_content, messages_to_show_outside
    except requests.exceptions.RequestException as e:
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Ağ hatası oluştu: {url} - {e}", 'icon': '🌐'})
        print(f"ERROR: Network error scraping '{url}': {e}")
        return None, messages_to_show_outside
    except Exception as e:
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Kazıma sırasında bir hata oluştu: {e}", 'icon': '🔥'})
        print(f"ERROR: Scraping '{url}' failed: {e}")
        return None, messages_to_show_outside

# --- Web Arama (Cache'li) ---
@st.cache_data(ttl=600)
def search_web(query):
    """Web'de arama yapar. UI elemanlarını doğrudan kullanmaz, print ile loglar."""
    print(f"INFO: Searching web for: {query}")
    messages_to_show_outside = []
    wikipedia.set_lang("tr")
    search_result = None
    try:
        wp_page = wikipedia.page(query, auto_suggest=False, redirect=True)
        wp_summary = wikipedia.summary(query, sentences=6, auto_suggest=False, redirect=True)
        search_result = f"**Wikipedia ({wp_page.title}):**\n\n{_clean_text(wp_summary)}\n\nKaynak: {wp_page.url}"
        messages_to_show_outside.append({'type': 'toast', 'text': f"✅ Wikipedia'dan bulundu: '{wp_page.title}'", 'icon': "📚"})
        return search_result, messages_to_show_outside
    except wikipedia.exceptions.PageError:
        messages_to_show_outside.append({'type': 'toast', 'text': f"ℹ️ Wikipedia'da '{query}' için sonuç bulunamadı.", 'icon': "🤷"})
    except wikipedia.exceptions.DisambiguationError as e:
        search_result = f"**Wikipedia Çok Anlamlı ({query}):**\n{e.options[:3]}..."
        messages_to_show_outside.append({'type': 'toast', 'text': f"ℹ️ Wikipedia'da birden fazla sonuç: '{query}'.", 'icon': "📚"})
    except Exception as e:
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Wikipedia araması sırasında hata: {e}", 'icon': "🔥"})
        print(f"ERROR: Wikipedia search error: {e}")

    ddg_url = None
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs:
            ddg_results = list(ddgs.text(query, region='tr-tr', safesearch='moderate', max_results=3))
            if ddg_results:
                snippet, href = ddg_results[0].get('body'), ddg_results[0].get('href')
                if href:
                    ddg_url = unquote(href)
                    domain_name = urlparse(ddg_url).netloc
                    if snippet and (not search_result or len(search_result) < 200):
                        search_result = f"**Web Özeti (DDG - {domain_name}):**\n\n{_clean_text(snippet)}\n\nKaynak: {ddg_url}"
                        messages_to_show_outside.append({'type': 'toast', 'text': f"ℹ️ DDG web özeti bulundu.", 'icon': "🦆"})
    except Exception as e:
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ DuckDuckGo araması sırasında hata: {e}", 'icon': "🔥"})
        print(f"ERROR: DDG search error: {e}")

    if ddg_url:
        scraped_content, scrape_messages = scrape_url_content(ddg_url) # scrape_url_content'den mesajları al
        messages_to_show_outside.extend(scrape_messages) # Kazıma mesajlarını ekle
        if scraped_content:
            domain_name = urlparse(ddg_url).netloc
            result_prefix = f"**Web Sayfası ({domain_name}):**\n\n"
            full_scraped_text = f"{result_prefix}{scraped_content}\n\nKaynak: {ddg_url}"
            if search_result and "Wikipedia" in search_result and len(search_result) > 300:
                 search_result += f"\n\n---\n\n{full_scraped_text}"
            else:
                search_result = full_scraped_text
            # scrape_url_content zaten kendi toast mesajını ekliyor
    if not search_result:
        messages_to_show_outside.append({'type': 'toast', 'text': f"'{query}' için web'de anlamlı bir sonuç bulunamadı.", 'icon': "❌"})
        return None, messages_to_show_outside
    return search_result, messages_to_show_outside

# --- Sohbet Geçmişi Yönetimi ---
@st.cache_data(ttl=86400)
def load_all_chats_cached(file_path=CHAT_HISTORY_FILE):
    """Sohbet geçmişini dosyadan yükler. UI elemanları içermez, hata mesajlarını döner."""
    error_messages_for_outside = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content and content.strip():
                data = json.loads(content)
                if isinstance(data, dict):
                    return {str(k): v for k, v in data.items()}, None
                else:
                    err_msg = f"Sohbet geçmişi dosyası ({file_path}) beklenmedik formatta. Yeniden adlandırılıyor."
                    print(f"WARNING: {err_msg}")
                    error_messages_for_outside.append({'type': 'warning', 'text': err_msg, 'icon': "⚠️"})
                    timestamp = int(time.time())
                    err_file_name = f"{file_path}.err_format_{timestamp}"
                    try:
                        os.rename(file_path, err_file_name)
                        info_msg = f"Formatı bozuk sohbet dosyası '{err_file_name}' olarak yeniden adlandırıldı."
                        print(f"INFO: {info_msg}")
                        error_messages_for_outside.append({'type': 'info', 'text': info_msg, 'icon': "ℹ️"})
                    except OSError as os_e:
                        err_msg_os = f"Formatı bozuk sohbet dosyasını yeniden adlandırma başarısız: {os_e}"
                        print(f"ERROR: {err_msg_os}")
                        error_messages_for_outside.append({'type': 'error', 'text': err_msg_os, 'icon': "🔥"})
                    return {}, error_messages_for_outside
            else:
                return {}, None # Dosya var ama boş, hata yok
        except json.JSONDecodeError as json_e:
            err_msg = f"Sohbet geçmişi dosyası ({file_path}) çözümlenemedi (JSON): {json_e}. Yeniden adlandırılıyor."
            print(f"ERROR: {err_msg}")
            error_messages_for_outside.append({'type': 'error', 'text': err_msg, 'icon': "🔥"})
            timestamp = int(time.time())
            err_file_name = f"{file_path}.err_json_{timestamp}"
            try:
                os.rename(file_path, err_file_name)
                info_msg = f"Bozuk JSON dosyası '{err_file_name}' olarak yeniden adlandırıldı."
                print(f"INFO: {info_msg}")
                error_messages_for_outside.append({'type': 'info', 'text': info_msg, 'icon': "ℹ️"})
            except OSError as os_e:
                err_msg_os = f"Bozuk JSON dosyasını yeniden adlandırma başarısız: {os_e}"
                print(f"ERROR: {err_msg_os}")
                error_messages_for_outside.append({'type': 'error', 'text': err_msg_os, 'icon': "🔥"})
            return {}, error_messages_for_outside
        except Exception as e:
            err_msg = f"Sohbet geçmişi ({file_path}) yüklenirken genel bir hata oluştu: {e}. Yeniden adlandırılıyor."
            print(f"ERROR: {err_msg}")
            error_messages_for_outside.append({'type': 'error', 'text': err_msg, 'icon': "🔥"})
            timestamp = int(time.time())
            err_file_name = f"{file_path}.err_generic_{timestamp}"
            try:
                os.rename(file_path, err_file_name)
                info_msg = f"Sorunlu sohbet dosyası '{err_file_name}' olarak yeniden adlandırıldı."
                print(f"INFO: {info_msg}")
                error_messages_for_outside.append({'type': 'info', 'text': info_msg, 'icon': "ℹ️"})
            except OSError as os_e:
                err_msg_os = f"Sorunlu sohbet dosyasını yeniden adlandırma başarısız: {os_e}"
                print(f"ERROR: {err_msg_os}")
                error_messages_for_outside.append({'type': 'error', 'text': err_msg_os, 'icon': "🔥"})
            return {}, error_messages_for_outside
    return {}, None # Dosya yoksa, hata yok

def save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chats_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Sohbet kaydedilemedi: {e}", icon="🔥")
        print(f"ERROR: Save chats failed: {e}")

# --- Gemini Yanıt Alma ---
def get_gemini_response_cached(prompt, history, stream=False):
    model = globals().get('gemini_model')
    if not model:
        return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    valid_history = [{'role': msg['role'], 'parts': [msg['parts']]}
                     for msg in history
                     if msg.get('role') in ['user', 'model'] and isinstance(msg.get('parts'), str) and msg['parts'].strip()]
    try:
        chat_session = model.start_chat(history=valid_history)
        response = chat_session.send_message(prompt, stream=stream)
        if stream:
            return response
        else:
            if response.parts:
                return "".join(p.text for p in response.parts if hasattr(p, 'text'))
            else:
                block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                finish_reason = getattr(response.candidates[0], 'finish_reason', '?') if response.candidates else '?'
                error_message = f"Engellendi ({block_reason})." if block_reason else f"Tamamlanmadı ({finish_reason})." if response.candidates else "Boş yanıt."
                st.warning(error_message, icon="🛡️" if block_reason else "⚠️") # Bu st.warning kalabilir, get_gemini_response_cached cache'li değil.
                return f"{GEMINI_ERROR_PREFIX} {error_message}"
    except Exception as e:
        st.error(f"Gemini API Hatası: {e}", icon="🔥")
        print(f"ERROR: Gemini API failed: {e}")
        return f"{GEMINI_ERROR_PREFIX} {e}"

# --- Supabase Loglama ---
def log_to_supabase(table_name, data_dict):
    client = globals().get('supabase')
    if not client:
        print(f"INFO: Supabase unavailable, skipping log to table: {table_name}")
        return False
    try:
        default_data = {
            'user_name': st.session_state.get('user_name', 'N/A'),
            'session_id': _get_session_id(),
            'app_version': APP_VERSION,
            'chat_id': st.session_state.get('active_chat_id', 'N/A')
        }
        response = client.table(table_name).insert({**default_data, **data_dict}).execute()
        if hasattr(response, 'error') and response.error:
             st.toast(f"⚠️ Loglama hatası ({table_name}): {response.error.message}", icon="💾")
             print(f"ERROR: Supabase log ({table_name}): {response.error.message}")
             return False
        return True
    except Exception as e:
        st.toast(f"⚠️ Loglama sırasında genel hata ({table_name}): {e}", icon="💾")
        print(f"ERROR: Supabase log ({table_name}) general exception: {e}")
        return False

def log_interaction(prompt, ai_response, source, message_id, chat_id_val):
    return log_to_supabase(SUPABASE_TABLE_LOGS, {
        "user_prompt": prompt,
        "ai_response": ai_response,
        "response_source": source,
        "message_id": message_id,
        "chat_id": chat_id_val
    })

def log_feedback(message_id, user_prompt, ai_response, feedback_type, comment=""):
    data = {
        "message_id": message_id,
        "user_prompt": user_prompt,
        "ai_response": ai_response,
        "feedback_type": feedback_type,
        "comment": comment
    }
    success = log_to_supabase(SUPABASE_TABLE_FEEDBACK, data)
    st.toast("Geri bildiriminiz alındı!" if success else "Geri bildirim gönderilemedi.", icon="💌" if success else "😔")
    return success

# --- Yanıt Orkestrasyonu ---
def get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id_val, use_stream=False):
    response_text, source_display_name = None, "Bilinmiyor"
    kb_response = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
    if kb_response:
        source_type = "Fonksiyonel" if prompt.lower() in DYNAMIC_FUNCTIONS_MAP else "Bilgi Tabanı"
        log_interaction(prompt, kb_response, source_type, msg_id, chat_id_val)
        return kb_response, f"{APP_NAME} ({source_type})"
    if globals().get('gemini_model'):
        gemini_response = get_gemini_response_cached(prompt, history, stream=use_stream)
        if gemini_response:
            if use_stream and not isinstance(gemini_response, str):
                return gemini_response, f"{APP_NAME} (Gemini Stream)"
            elif isinstance(gemini_response, str) and not gemini_response.startswith(GEMINI_ERROR_PREFIX):
                log_interaction(prompt, gemini_response, "Gemini", msg_id, chat_id_val)
                return gemini_response, f"{APP_NAME} (Gemini)"
            print(f"INFO: Gemini returned an error or non-string response: {gemini_response}")

    is_question_like = "?" in prompt or any(keyword in prompt.lower() for keyword in ["nedir", "kimdir", "nasıl", "bilgi", "araştır", "haber"])
    if not response_text and is_question_like and len(prompt.split()) > 2:
        web_search_response, web_messages = search_web(query) # Mesajları da al
        for msg_info in web_messages: # Dönen mesajları göster
            if msg_info['type'] == 'toast': st.toast(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'warning': st.warning(msg_info['text'], icon=msg_info.get('icon'))
            # Diğer mesaj türleri eklenebilir (info, error vs.)
        if web_search_response:
            log_interaction(prompt, web_search_response, "Web Search", msg_id, chat_id_val)
            return web_search_response, f"{APP_NAME} (Web)"

    default_responses = [
        f"Üzgünüm {st.session_state.get('user_name', '')}, bu konuda size yardımcı olamıyorum.",
        "Bu soruyu tam olarak anlayamadım, farklı bir şekilde sorabilir misiniz?",
        "Bu konuda henüz bir bilgim yok."
    ]
    response_text = random.choice(default_responses)
    log_interaction(prompt, response_text, "Varsayılan Yanıt", msg_id, chat_id_val)
    return response_text, f"{APP_NAME} (Varsayılan)"

# --- Yaratıcı Modüller ---
def creative_response_generator(prompt_text, length_mode="orta", style_mode="genel"):
    templates = {
        "genel": ["İşte bir fikir: {}", "Şöyle bir düşünce: {}", "Belki de: {}"],
        "şiirsel": ["Kalbimden dökülenler: {}", "Mısralarla: {}", "İlham perisi fısıldadı: {}"],
        "hikaye": ["Bir varmış bir yokmuş, {}...", "Hikayemiz başlar: {}...", "Ve sonra olanlar oldu: {}..."]
    }
    creative_idea = generate_new_idea_creative(prompt_text, style_mode)
    sentences = [s.strip() for s in creative_idea.split('.') if s.strip()]
    num_sentences = len(sentences)
    if length_mode == "kısa":
        final_idea = ". ".join(sentences[:max(1, num_sentences // 3)]) + "." if num_sentences > 0 else creative_idea
    elif length_mode == "uzun":
        additional_idea = generate_new_idea_creative(prompt_text[::-1], style_mode)
        final_idea = creative_idea + f"\n\nDahası, bir de şu var: {additional_idea}"
    else:
        final_idea = creative_idea
    selected_template = random.choice(templates.get(style_mode, templates["genel"]))
    return selected_template.format(final_idea)

def generate_new_idea_creative(seed_text, style="genel"):
    elements = ["zamanın dokusu", "kayıp orman", "kırık bir rüya", "kuantum dalgaları", "gölgelerin dansı", "yıldız tozu"]
    actions = ["gizemi çözer", "sınırları yeniden çizer", "unutulmuş şarkıları fısıldar", "kaderi yeniden yazar", "sessizliği boyar"]
    objects = ["evrenin kalbi", "saklı bir gerçek", "sonsuzluğun melodisi", "kayıp bir hatıra", "umudun ışığı"]
    words_from_seed = re.findall(r'\b\w{4,}\b', seed_text.lower())
    chosen_seed_word = random.choice(words_from_seed) if words_from_seed else "gizem"
    e1, a1, o1 = random.choice(elements), random.choice(actions), random.choice(objects)
    return f"{chosen_seed_word.capitalize()}, {e1} içinde {a1} ve {o1} ortaya çıkar."

def advanced_word_generator(base_word):
    base = base_word or "kelime"
    cleaned_base = "".join(filter(str.isalpha, base.lower()))
    vowels = "aeıioöuü"
    consonants = "bcçdfgğhjklmnprsştvyz"
    prefixes = ["bio", "krono", "neo", "mega", "poli", "meta", "xeno", "astro", "hidro"]
    suffixes = ["genez", "sfer", "loji", "tronik", "morf", "matik", "skop", "nomi", "tek"]
    core_part = ""
    if len(cleaned_base) > 2 and random.random() < 0.7:
        start_index = random.randint(0, max(0, len(cleaned_base) - 3))
        core_part = cleaned_base[start_index : start_index + random.randint(2,3)]
    else:
        core_part = "".join(random.choice(consonants if i % 2 else vowels) for i in range(random.randint(2,4)))
    new_word = core_part
    if random.random() > 0.3:
        new_word = random.choice(prefixes) + new_word
    if random.random() > 0.3:
        new_word += random.choice(suffixes)
    return new_word.capitalize() if len(new_word) > 1 else "KelimeX"

# --- Görsel Oluşturucu ---
def generate_prompt_influenced_image(prompt):
    width, height = 512, 512
    prompt_lower = prompt.lower()
    themes = {
        "güneş": {"bg": [(255, 230, 150), (255, 160, 0)], "sh": [{"t": "circle", "c": (255, 255, 0, 220), "p": (0.25, 0.25), "s": 0.2, "l": 1}]},
        "ay": {"bg": [(10, 10, 50), (40, 40, 100)], "sh": [{"t": "circle", "c": (240, 240, 240, 200), "p": (0.75, 0.2), "s": 0.15, "l": 1}]},
        "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "sh": []},
        "bulut": {"bg": None, "sh": [{"t": "ellipse", "c": (255, 255, 255, 180), "p": (random.uniform(0.2, 0.8), random.uniform(0.1, 0.4)), "swh": (random.uniform(0.15, 0.35), random.uniform(0.08, 0.15)), "l": 1} for _ in range(random.randint(2, 4))]},
        "deniz": {"bg": [(0, 105, 148), (0, 0, 100)], "sh": [{"t": "rect", "c": (60, 120, 180, 150), "p": (0.5, 0.75), "swh": (1.0, 0.5), "l": 0}]},
        "nehir": {"bg": None, "sh": [{"t": "line", "c": (100, 149, 237, 180), "pts": [(0, random.uniform(0.6, 0.8)), (0.3, random.uniform(0.65, 0.75)), (0.7, random.uniform(0.6, 0.7)), (1, random.uniform(0.55, 0.75))], "w": 15, "l": 0}]},
        "orman": {"bg": [(34, 139, 34), (0, 100, 0)], "sh": [{"t": "tri", "c": (random.randint(0, 30), random.randint(70, 100), random.randint(0, 30), 200), "p": (random.uniform(0.1, 0.9), random.uniform(0.65, 0.9)), "s": random.uniform(0.07, 0.20), "l": 2} for _ in range(random.randint(9, 16))]},
        "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "sh": [{"t": "rect", "c": (139, 69, 19, 255), "p": (rx := random.uniform(0.2, 0.8), 0.8), "swh": (0.05, 0.3), "l": 2}, {"t": "ellipse", "c": (34, 139, 34, 200), "p": (rx, 0.6), "swh": (0.25, 0.2), "l": 2}]},
        "ev": {"bg": None, "sh": [{"t": "rect", "c": (200, 180, 150, 240), "p": (ex := random.uniform(0.2, 0.8), 0.8), "swh": (0.15, 0.2), "l": 2}, {"t": "poly", "c": (139, 0, 0, 240), "pts": [(ex - 0.075, 0.7), (ex + 0.075, 0.7), (ex, 0.6)], "l": 2}]},
        "dağ": {"bg": [(200, 200, 200), (100, 100, 100)], "sh": [{"t": "poly", "c": (random.randint(100, 160),) * 3 + (230,), "pts": [(random.uniform(0.1, 0.4), 0.85), (0.5, random.uniform(0.2, 0.5)), (random.uniform(0.6, 0.9), 0.85)], "l": 0} for _ in range(random.randint(1, 2))]},
        "şehir": {"bg": [(100, 100, 120), (50, 50, 70)], "sh": [{"t": "rect", "c": (random.randint(60, 100),) * 3 + (random.randint(190, 230),), "p": (random.uniform(0.1, 0.9), random.uniform(0.5, 0.9)), "swh": (random.uniform(0.04, 0.12), random.uniform(0.1, 0.55)), "l": 1} for _ in range(random.randint(10, 18))]},
        "çiçek": {"bg": None, "sh": [{"t": "circle", "c": (random.randint(200, 255), random.randint(100, 200), random.randint(150, 255), 210), "p": (random.uniform(0.1, 0.9), random.uniform(0.8, 0.95)), "s": 0.015, "l": 3} for _ in range(random.randint(5, 10))]},
        "kar": {"bg": None, "sh": [{"t": "circle", "c": (255, 255, 255, 150), "p": (random.random(), random.random()), "s": 0.005, "l": 3}]},
        "yıldız": {"bg": None, "sh": [{"t": "circle", "c": (255, 255, 200, 200), "p": (random.random(), random.uniform(0, 0.5)), "s": 0.003, "l": 1}]},
    }
    bg_color1, bg_color2 = (random.randint(30, 120),) * 3, (random.randint(120, 220),) * 3
    applied_shapes = []
    themes_applied_count = 0
    for keyword, theme_details in themes.items():
        if keyword in prompt_lower:
            if theme_details["bg"] and themes_applied_count == 0:
                bg_color1, bg_color2 = theme_details["bg"]
            applied_shapes.extend(theme_details["sh"])
            themes_applied_count += 1
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y_coord in range(height):
        ratio = y_coord / height
        r_val = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio)
        g_val = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio)
        b_val = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio)
        draw.line([(0, y_coord), (width, y_coord)], fill=(r_val, g_val, b_val, 255))
    applied_shapes.sort(key=lambda s: s.get("l", 2))
    for shape_info in applied_shapes:
        try:
            shape_type = shape_info["t"]
            shape_color = shape_info["c"]
            outline_color = (0,0,0,40) if len(shape_color) == 4 and shape_color[3] < 250 else None
            if shape_info.get("p"):
                center_x, center_y = int(shape_info["p"][0] * width), int(shape_info["p"][1] * height)
            if shape_type == "circle":
                radius = int(shape_info["s"] * min(width, height) / 2)
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=shape_color, outline=outline_color)
            elif shape_type in ["rect", "ellipse"]:
                shape_w, shape_h = shape_info["swh"]
                pixel_w, pixel_h = int(shape_w * width), int(shape_h * height)
                box = (center_x - pixel_w // 2, center_y - pixel_h // 2, center_x + pixel_w // 2, center_y + pixel_h // 2)
                if shape_type == "rect": draw.rectangle(box, fill=shape_color, outline=outline_color)
                else: draw.ellipse(box, fill=shape_color, outline=outline_color)
            elif shape_type == "tri":
                size = int(shape_info["s"] * min(width, height))
                points = [(center_x, center_y - int(size * 0.58)), (center_x - size // 2, center_y + int(size * 0.3)), (center_x + size // 2, center_y + int(size * 0.3))]
                draw.polygon(points, fill=shape_color, outline=outline_color)
            elif shape_type == "poly":
                pixel_points = [(int(p[0] * width), int(p[1] * height)) for p in shape_info["pts"]]
                draw.polygon(pixel_points, fill=shape_color, outline=outline_color)
            elif shape_type == "line":
                pixel_points = [(int(p[0] * width), int(p[1] * height)) for p in shape_info["pts"]]
                line_width = shape_info.get("w", 5)
                draw.line(pixel_points, fill=shape_color, width=line_width, joint="curve")
        except Exception as e:
            print(f"DEBUG: Shape drawing error {shape_info}: {e}")
            continue
    if themes_applied_count == 0:
        for _ in range(random.randint(4, 7)):
            x_pos, y_pos = random.randint(0, width), random.randint(0, height)
            clr = tuple(random.randint(50, 250) for _ in range(3)) + (random.randint(150, 220),)
            radius = random.randint(20, 70)
            if random.random() > 0.5: draw.ellipse((x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius), fill=clr)
            else: draw.rectangle((x_pos - radius // 2, y_pos - radius // 2, x_pos + radius // 2, y_pos + radius // 2), fill=clr)
    try:
        font = ImageFont.load_default()
        text_to_draw = prompt[:80]
        if os.path.exists(FONT_FILE):
            try:
                font_size = max(14, min(28, int(width / (len(text_to_draw) * 0.3 + 10) if len(text_to_draw) > 0 else width / 10)))
                font = ImageFont.truetype(FONT_FILE, font_size)
            except (IOError, ZeroDivisionError): pass
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text_to_draw, font=font, anchor="lt")
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text_to_draw, font=font)
        pos_x = (width - text_width) / 2
        pos_y = height * 0.95 - text_height
        draw.text((pos_x + 1, pos_y + 1), text_to_draw, font=font, fill=(0, 0, 0, 150))
        draw.text((pos_x, pos_y), text_to_draw, font=font, fill=(255, 255, 255, 230))
    except Exception as e:
        st.toast(f"Görsel metni yazılamadı: {e}", icon="📝") # Bu toast kalabilir, generate_prompt_influenced_image cache'li değil
    return image.convert("RGB")

# --- Session State Başlatma ---
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
        'models_initialized': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Modelleri ve İstemcileri Başlatma ---
# Bu blok sadece ilk çalıştırmada veya state resetlendiğinde çalışır.
if not st.session_state.models_initialized:
    print("INFO: Initializing resources for the first time...")
    gemini_model, gemini_init_error_global = initialize_gemini_model()
    if gemini_model: st.toast(f"✨ Gemini modeli ({st.session_state.gemini_model_name}) başarıyla yüklendi!", icon="🤖")
    elif gemini_init_error_global: st.warning(gemini_init_error_global, icon="🗝️") # Hata varsa göster

    supabase, supabase_init_error_global = init_supabase_client_cached()
    if supabase: st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱")
    elif supabase_init_error_global: st.warning(f"Supabase başlatılamadı: {supabase_init_error_global}", icon="🧱")


    tts_engine, tts_init_error_global = init_tts_engine_cached()
    if tts_engine: st.toast("🔊 TTS motoru hazır.", icon="🗣️")
    elif tts_init_error_global: st.warning(tts_init_error_global, icon="🔇")


    all_chats_data, chat_load_errors = load_all_chats_cached()
    st.session_state.all_chats = all_chats_data
    if chat_load_errors: # Dönen mesaj listesini işle
        for msg_info in chat_load_errors:
            if msg_info['type'] == 'toast': st.toast(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'warning': st.warning(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'info': st.info(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'error': st.error(msg_info['text'], icon=msg_info.get('icon'))


    if not st.session_state.active_chat_id and st.session_state.all_chats:
        try:
            st.session_state.active_chat_id = sorted(st.session_state.all_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)[0]
        except (IndexError, ValueError):
             st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE, knowledge_base_load_error_global = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    if knowledge_base_load_error_global: # KB yükleme hatasını göster
        st.warning(knowledge_base_load_error_global, icon="📚")


    st.session_state.models_initialized = True
    print("INFO: Initialization complete.")
else:
    # Sonraki çalıştırmalar için global değişkenleri doğrudan kullan (zaten atanmış olmalılar)
    # Eğer initialize_gemini_model gibi fonksiyonlar globale atama yapıyorsa,
    # burada tekrar globals().get() kullanmaya gerek yok.
    # Sadece KNOWLEDGE_BASE'i kullanıcı adı değişmiş olabileceği için tekrar yükleyebiliriz.
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE, kb_load_err = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    if kb_load_err and kb_load_err != knowledge_base_load_error_global: # Yeni bir hata varsa veya önceki hata çözüldüyse
        knowledge_base_load_error_global = kb_load_err # Globali güncelle
        if knowledge_base_load_error_global: st.warning(knowledge_base_load_error_global, icon="📚")
        else: st.toast("Bilgi tabanı başarıyla güncellendi/yüklendi.", icon="📚") # Çözülme durumu


# --- ARAYÜZ FONKSİYONLARI ---
def display_settings_section():
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**")
        new_user_name = st.text_input("Adınız:", value=st.session_state.user_name, key="change_user_name_input", label_visibility="collapsed")
        if new_user_name != st.session_state.user_name and new_user_name.strip():
            st.session_state.user_name = new_user_name.strip()
            load_knowledge_from_file.clear()
            st.toast("Adınız güncellendi!", icon="✏️")
            st.rerun()

        avatar_col1, avatar_col2 = st.columns([0.8, 0.2])
        with avatar_col1:
            uploaded_avatar_file = st.file_uploader("Avatar yükle (PNG, JPG - maks 2MB):", type=["png", "jpg"], key="upload_avatar_file", label_visibility="collapsed")
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
                if st.button("🗑️ Kaldır", key="remove_avatar_button", help="Avatarı kaldır", use_container_width=True):
                    st.session_state.user_avatar_bytes = None
                    st.toast("Avatar kaldırıldı.", icon="🗑️")
                    st.rerun()
        st.caption("Avatarınız sadece bu oturumda saklanır.")
        st.divider()

        st.subheader("🤖 Yapay Zeka ve Arayüz")
        tts_toggle_col, stream_toggle_col = st.columns(2)
        is_tts_engine_ok = globals().get('tts_engine') is not None
        with tts_toggle_col:
            st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not is_tts_engine_ok, help="Yanıtları sesli olarak oku.")
        with stream_toggle_col:
            st.session_state.gemini_stream_enabled = st.toggle("Yanıt Akışı (Streaming)", value=st.session_state.gemini_stream_enabled, help="Yanıtları kelime kelime alarak daha hızlı gösterim sağla.")
        st.session_state.gemini_system_prompt = st.text_area(
            "AI Sistem Talimatı (Opsiyonel):",
            value=st.session_state.get('gemini_system_prompt', ""),
            key="system_prompt_input_area",
            height=100,
            placeholder="Yapay zekanın genel davranışını veya rolünü tanımlayın (örn: 'Sen esprili bir asistansın.', 'Kısa ve öz cevap ver.', 'Bir uzay kaşifi gibi konuş.')",
            help="Modelin yanıtlarını etkilemek için genel bir talimat girin. (Modelin desteklemesi gerekir)"
        )
        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma")
        gemini_config_col1, gemini_config_col2 = st.columns(2)
        with gemini_config_col1:
            current_model_index = 0 if st.session_state.gemini_model_name == 'gemini-1.5-flash-latest' else 1
            st.session_state.gemini_model_name = st.selectbox("AI Modeli:", ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'], index=current_model_index, key="select_gemini_model", help="Model yetenekleri ve maliyetleri farklılık gösterebilir.")
            st.session_state.gemini_temperature = st.slider("Sıcaklık (Temperature):", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="temperature_slider", help="Yaratıcılık seviyesi (0=Daha kesin, 1=Daha yaratıcı).")
            st.session_state.gemini_max_tokens = st.slider("Maksimum Token:", 256, 8192, st.session_state.gemini_max_tokens, 128, key="max_tokens_slider", help="Bir yanıtta üretilecek maksimum token (kelime/parça) sayısı.")
        with gemini_config_col2:
            st.session_state.gemini_top_k = st.slider("Top K:", 1, 100, st.session_state.gemini_top_k, 1, key="top_k_slider", help="Kelime seçim çeşitliliği.")
            st.session_state.gemini_top_p = st.slider("Top P:", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="top_p_slider", help="Kelime seçim odaklılığı (daha düşük değerler daha odaklı).")
            if st.button("⚙️ AI Ayarlarını Uygula & Modeli Yeniden Başlat", key="reload_ai_model_button", use_container_width=True, type="primary", help="Seçili AI modelini ve parametreleri yeniden yükler."):
                global gemini_model, gemini_init_error_global # Global değişkenleri güncelle
                with st.spinner("AI modeli yeniden başlatılıyor..."):
                    gemini_model, gemini_init_error_global = initialize_gemini_model()
                if not gemini_model:
                    st.error(f"AI modeli yüklenemedi: {gemini_init_error_global}")
                else:
                    st.toast("AI ayarları başarıyla uygulandı ve model yeniden başlatıldı!", icon="⚙️")
                st.rerun()
        st.divider()
        st.subheader("🧼 Geçmiş Yönetimi")
        clear_current_col, clear_all_col = st.columns(2)
        with clear_current_col:
            active_chat_id_for_clear = st.session_state.get('active_chat_id')
            is_clear_current_disabled = not bool(active_chat_id_for_clear and st.session_state.all_chats.get(active_chat_id_for_clear))
            if st.button("🧹 Aktif Sohbeti Temizle", use_container_width=True, type="secondary", key="clear_current_chat_button", help="Sadece şu an açık olan sohbetin içeriğini temizler.", disabled=is_clear_current_disabled):
                if active_chat_id_for_clear and active_chat_id_for_clear in st.session_state.all_chats:
                    st.session_state.all_chats[active_chat_id_for_clear] = []
                    save_all_chats(st.session_state.all_chats)
                    st.toast("Aktif sohbet temizlendi!", icon="🧹")
                    st.rerun()
        with clear_all_col:
            is_clear_all_disabled = not st.session_state.all_chats
            if st.button("🗑️ TÜM Geçmişi Kalıcı Olarak Sil", use_container_width=True, type="danger", key="clear_all_chats_button", help="Dikkat! Tüm sohbet geçmişini kalıcı olarak siler.", disabled=is_clear_all_disabled):
                st.session_state.all_chats = {}
                st.session_state.active_chat_id = None
                save_all_chats({})
                st.toast("TÜM sohbet geçmişi silindi!", icon="🗑️")
                st.rerun()

def display_chat_list_and_about(left_column_ref):
    with left_column_ref:
        st.markdown("#### Sohbetler")
        if st.button("➕ Yeni Sohbet Oluştur", use_container_width=True, key="new_chat_button"):
            st.session_state.next_chat_id_counter += 1
            timestamp = int(time.time())
            new_chat_id = f"chat_{st.session_state.next_chat_id_counter}_{timestamp}"
            st.session_state.all_chats[new_chat_id] = []
            st.session_state.active_chat_id = new_chat_id
            save_all_chats(st.session_state.all_chats)
            st.rerun()
        st.markdown("---")
        chat_list_container = st.container(height=450, border=False)
        with chat_list_container:
            current_chats = st.session_state.all_chats
            sorted_chat_ids = sorted(current_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
            if not sorted_chat_ids:
                st.caption("Henüz bir sohbet başlatılmamış.")
            else:
                active_chat_id_display = st.session_state.get('active_chat_id')
                for chat_id_item in sorted_chat_ids:
                    chat_history = current_chats.get(chat_id_item, [])
                    first_user_message = next((msg.get('parts', '') for msg in chat_history if msg.get('role') == 'user'), None)
                    chat_title = f"Sohbet {chat_id_item.split('_')[1]}"
                    if first_user_message:
                        chat_title = first_user_message[:30] + ("..." if len(first_user_message) > 30 else "")
                    chat_title = chat_title if chat_history else "Boş Sohbet"
                    select_col, download_col, delete_col = st.columns([0.7, 0.15, 0.15])
                    button_style_type = "primary" if active_chat_id_display == chat_id_item else "secondary"
                    if select_col.button(chat_title, key=f"select_chat_{chat_id_item}", use_container_width=True, type=button_style_type, help=f"'{chat_title}' adlı sohbeti aç"):
                        if active_chat_id_display != chat_id_item:
                            st.session_state.active_chat_id = chat_id_item
                            st.rerun()
                    chat_content_for_download = ""
                    for message_item in chat_history:
                        sender_name = 'Kullanıcı' if message_item.get('role') == 'user' else message_item.get('sender_display', 'AI')
                        chat_content_for_download += f"{sender_name}: {message_item.get('parts', '')}\n\n"
                    download_col.download_button("⬇️", data=chat_content_for_download.encode('utf-8'), file_name=f"{chat_title.replace(' ', '_')}_{chat_id_item}.txt", mime="text/plain", key=f"download_chat_{chat_id_item}", help=f"'{chat_title}' sohbetini indir (.txt)", use_container_width=True, disabled=not chat_history)
                    if delete_col.button("🗑️", key=f"delete_chat_{chat_id_item}", use_container_width=True, help=f"'{chat_title}' adlı sohbeti sil", type="secondary"):
                        if chat_id_item in current_chats:
                            del current_chats[chat_id_item]
                            if active_chat_id_display == chat_id_item:
                                remaining_ids = sorted(current_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
                                st.session_state.active_chat_id = remaining_ids[0] if remaining_ids else None
                            save_all_chats(current_chats)
                            st.toast(f"'{chat_title}' sohbeti silindi.", icon="🗑️")
                            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Uygulama Hakkında", expanded=False):
            st.markdown(f"**{APP_NAME} v{APP_VERSION}**\n\nAI Destekli Kişisel Asistan\n\nGeliştirici: **Hanogt**\n\n© 2024-{CURRENT_YEAR}")
            st.caption(f"Aktif Oturum ID: {_get_session_id()[:8]}...")

def display_chat_message_with_feedback(message_data, message_index, current_chat_id):
    role = message_data.get('role', 'model')
    content_text = str(message_data.get('parts', ''))
    sender_display = message_data.get('sender_display', APP_NAME if role == 'model' else st.session_state.user_name)
    is_user_message = (role == 'user')
    avatar_icon = "🧑"
    if is_user_message:
        if st.session_state.user_avatar_bytes:
            avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes))
    else:
        if "Gemini" in sender_display: avatar_icon = "✨"
        elif any(w in sender_display for w in ["Web", "Wiki"]): avatar_icon = "🌐"
        elif any(w in sender_display for w in ["Bilgi", "Fonksiyon"]): avatar_icon = "📚"
        else: avatar_icon = "🤖"

    with st.chat_message(role, avatar=avatar_icon):
        if "```" in content_text:
            text_parts = content_text.split("```")
            for i, part in enumerate(text_parts):
                if i % 2 == 1:
                    language_match = re.match(r"(\w+)\n", part)
                    code_block_content = part[len(language_match.group(1)) + 1:] if language_match else part
                    actual_code_language = language_match.group(1) if language_match else None
                    st.code(code_block_content, language=actual_code_language)
                    if st.button("📋 Kopyala", key=f"copy_code_{current_chat_id}_{message_index}_{i}", help="Kodu panoya kopyala"):
                        st.write_to_clipboard(code_block_content)
                        st.toast("Kod panoya kopyalandı!", icon="✅")
                elif part.strip():
                    st.markdown(part, unsafe_allow_html=True)
        elif content_text.strip():
            st.markdown(content_text, unsafe_allow_html=True)
        else:
            st.caption("[Boş Mesaj]")
        token_count_display = None
        if tiktoken_encoder and content_text.strip():
            try:
                token_count_display = len(tiktoken_encoder.encode(content_text))
            except Exception:
                pass
        if not is_user_message and content_text.strip():
            source_col, tts_col, feedback_col = st.columns([0.75, 0.1, 0.15])
            with source_col:
                source_text = sender_display.split('(')[-1].replace(')', '') if '(' in sender_display else sender_display
                token_info = f" | ~{token_count_display} token" if token_count_display else ""
                st.caption(f"Kaynak: {source_text}{token_info}")
            with tts_col:
                if st.session_state.tts_enabled and globals().get('tts_engine'):
                    if st.button("🔊", key=f"tts_button_{current_chat_id}_{message_index}", help="Yanıtı sesli oku", use_container_width=True):
                        speak(content_text)
            with feedback_col:
                if st.button("✍️ G.Bildirim", key=f"feedback_button_{current_chat_id}_{message_index}", help="Bu yanıt hakkında geri bildirim ver", use_container_width=True):
                    st.session_state.current_message_id_for_feedback = f"{current_chat_id}_{message_index}"
                    previous_user_prompt = "[Kullanıcı istemi bulunamadı]"
                    if message_index > 0 and st.session_state.all_chats[current_chat_id][message_index - 1]['role'] == 'user':
                        previous_user_prompt = st.session_state.all_chats[current_chat_id][message_index - 1]['parts']
                    st.session_state.last_user_prompt_for_feedback = previous_user_prompt
                    st.session_state.last_ai_response_for_feedback = content_text
                    st.session_state.show_feedback_comment_form = True
                    st.session_state.feedback_comment_input = ""
                    st.rerun()

def display_feedback_form_if_active():
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        form_unique_key = f"feedback_form_{st.session_state.current_message_id_for_feedback}"
        with st.form(key=form_unique_key):
            st.markdown("#### Yanıt Geri Bildirimi")
            st.caption(f"**Değerlendirilen İstem:** `{str(st.session_state.last_user_prompt_for_feedback)[:80]}...`")
            st.caption(f"**Değerlendirilen Yanıt:** `{str(st.session_state.last_ai_response_for_feedback)[:80]}...`")
            feedback_rating_type = st.radio(
                "Değerlendirmeniz:",
                ["👍 Beğendim", "👎 Beğenmedim"],
                horizontal=True,
                key=f"rating_type_{form_unique_key}",
                index=0 if st.session_state.last_feedback_type == 'positive' else 1
            )
            feedback_user_comment = st.text_area(
                "Yorumunuz (isteğe bağlı):",
                value=st.session_state.feedback_comment_input,
                key=f"comment_input_{form_unique_key}",
                height=100,
                placeholder="Yanıtla ilgili düşüncelerinizi paylaşın..."
            )
            st.session_state.feedback_comment_input = feedback_user_comment
            submit_col, cancel_col = st.columns(2)
            submitted_feedback = submit_col.form_submit_button("✅ Geri Bildirimi Gönder", use_container_width=True, type="primary")
            cancelled_feedback = cancel_col.form_submit_button("❌ Vazgeç", use_container_width=True)
            if submitted_feedback:
                parsed_feedback_type = "positive" if feedback_rating_type == "👍 Beğendim" else "negative"
                st.session_state.last_feedback_type = parsed_feedback_type
                log_feedback(
                    st.session_state.current_message_id_for_feedback,
                    st.session_state.last_user_prompt_for_feedback,
                    st.session_state.last_ai_response_for_feedback,
                    parsed_feedback_type,
                    feedback_user_comment
                )
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.feedback_comment_input = ""
                st.rerun()
            elif cancelled_feedback:
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.feedback_comment_input = ""
                st.rerun()
        st.markdown("---")

def display_chat_interface_main(main_column_container):
    active_chat_id_main = st.session_state.get('active_chat_id')
    if active_chat_id_main is None:
        st.info("💬 Başlamak için sol menüden **'➕ Yeni Sohbet Oluştur'** butonuna tıklayın veya var olan bir sohbeti seçin.", icon="👈")
        return
    current_chat_history = st.session_state.all_chats.get(active_chat_id_main, [])
    chat_messages_container = st.container(height=600, border=False)
    with chat_messages_container:
        if not current_chat_history:
            st.info(f"Merhaba {st.session_state.user_name}! Bu yeni sohbetinize hoş geldiniz. Ne merak ediyorsunuz?", icon="👋")
        for idx, message in enumerate(current_chat_history):
            display_chat_message_with_feedback(message, idx, active_chat_id_main)
    display_feedback_form_if_active()
    user_chat_prompt = st.chat_input(
        f"{st.session_state.user_name}, ne sormak istersin?",
        key=f"chat_input_{active_chat_id_main}"
    )
    if user_chat_prompt:
        user_message_data = {'role': 'user', 'parts': user_chat_prompt}
        st.session_state.all_chats[active_chat_id_main].append(user_message_data)
        save_all_chats(st.session_state.all_chats)
        message_unique_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"
        st.session_state.message_id_counter += 1
        history_for_model_request = st.session_state.all_chats[active_chat_id_main][-20:-1]
        with st.chat_message("assistant", avatar="⏳"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 _Yanıtınız hazırlanıyor..._")
        ai_response_content, ai_sender_name = get_hanogt_response_orchestrator(
            user_chat_prompt,
            history_for_model_request,
            message_unique_id,
            active_chat_id_main,
            use_stream=st.session_state.gemini_stream_enabled
        )
        final_ai_response_text = ""
        if st.session_state.gemini_stream_enabled and "Stream" in ai_sender_name and not isinstance(ai_response_content, str):
            stream_display_container = thinking_placeholder
            streamed_text_so_far = ""
            try:
                for chunk in ai_response_content:
                    if chunk.parts:
                        text_chunk = "".join(p.text for p in chunk.parts if hasattr(p, 'text'))
                        streamed_text_so_far += text_chunk
                        stream_display_container.markdown(streamed_text_so_far + "▌")
                        time.sleep(0.005)
                stream_display_container.markdown(streamed_text_so_far)
                final_ai_response_text = streamed_text_so_far
                log_interaction(user_chat_prompt, final_ai_response_text, "Gemini Stream", message_unique_id, active_chat_id_main)
            except Exception as e:
                error_message_stream = f"Stream sırasında hata: {e}"
                stream_display_container.error(error_message_stream)
                final_ai_response_text = error_message_stream
                ai_sender_name = f"{APP_NAME} (Stream Hatası)"
                log_interaction(user_chat_prompt, final_ai_response_text, "Stream Hatası", message_unique_id, active_chat_id_main)
        else:
            thinking_placeholder.empty()
            final_ai_response_text = str(ai_response_content)
        ai_message_data = {'role': 'model', 'parts': final_ai_response_text, 'sender_display': ai_sender_name}
        st.session_state.all_chats[active_chat_id_main].append(ai_message_data)
        save_all_chats(st.session_state.all_chats)
        if st.session_state.tts_enabled and globals().get('tts_engine') and isinstance(final_ai_response_text, str) and "Stream" not in ai_sender_name:
            speak(final_ai_response_text)
        st.rerun()

# --- UYGULAMA ANA AKIŞI ---
st.markdown(f"<h1 style='text-align:center;color:#0078D4;'>{APP_NAME} <span style='font-size:0.8em;color:#555;'>{APP_VERSION}</span></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-style:italic;color:#555;'>Yapay zeka destekli kişisel asistanınız</p>", unsafe_allow_html=True)

# Başlatma Hatalarını Göster (models_initialized bloğundan sonra, her zaman)
# Bu global değişkenler initialize bloklarında set ediliyor.
if gemini_init_error_global: st.warning(gemini_init_error_global, icon="🗝️")
if supabase_init_error_global: st.warning(f"Supabase başlatılamadı: {supabase_init_error_global}", icon="🧱")
if tts_init_error_global: st.warning(tts_init_error_global, icon="🔇")
if knowledge_base_load_error_global: st.warning(knowledge_base_load_error_global, icon="📚")


# --- Giriş Ekranı ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce...")
    login_cols = st.columns([0.2, 0.6, 0.2])
    with login_cols[1]:
        with st.form("user_login_form"):
            user_entered_name = st.text_input("Size nasıl hitap etmemizi istersiniz?", placeholder="İsminiz...", key="login_name_input")
            if st.form_submit_button("✨ Uygulamayı Başlat", use_container_width=True, type="primary"):
                if user_entered_name and user_entered_name.strip():
                    st.session_state.user_name = user_entered_name.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False
                    load_knowledge_from_file.clear()
                    st.rerun()
                else:
                    st.error("Lütfen geçerli bir isim giriniz.")
else:
    # --- Ana Uygulama Arayüzü ---
    if not st.session_state.greeting_message_shown:
        st.success(f"Tekrar hoş geldiniz, {st.session_state.user_name}! Size nasıl yardımcı olabilirim?", icon="🎉")
        st.session_state.greeting_message_shown = True
    app_left_column, app_main_column = st.columns([1, 3])
    display_chat_list_and_about(app_left_column)
    with app_main_column:
        display_settings_section()
        st.markdown("#### Uygulama Modu")
        app_modes = {
            "Yazılı Sohbet": "💬",
            "Sesli Sohbet (Dosya)": "🎤",
            "Yaratıcı Stüdyo": "🎨",
            "Görsel Oluşturucu": "🖼️"
        }
        mode_options_keys = list(app_modes.keys())
        current_mode_index = mode_options_keys.index(st.session_state.app_mode) if st.session_state.app_mode in mode_options_keys else 0
        selected_app_mode = st.radio(
            "Çalışma Modunu Seçin:",
            options=mode_options_keys,
            index=current_mode_index,
            format_func=lambda k: f"{app_modes[k]} {k}",
            horizontal=True,
            label_visibility="collapsed",
            key="app_mode_selection_radio"
        )
        if selected_app_mode != st.session_state.app_mode:
            st.session_state.app_mode = selected_app_mode
            st.rerun()
        st.markdown("<hr style='margin-top:0.1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
        current_app_mode = st.session_state.app_mode

        if current_app_mode == "Yazılı Sohbet":
            display_chat_interface_main(app_main_column)
        elif current_app_mode == "Sesli Sohbet (Dosya)":
            st.info("Yanıt almak istediğiniz ses dosyasını yükleyin (WAV, MP3, OGG, FLAC, M4A).", icon="📢")
            audio_file_uploaded = st.file_uploader("Ses Dosyası:", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed", key="audio_file_uploader")
            if audio_file_uploaded:
                st.audio(audio_file_uploaded, format=audio_file_uploaded.type)
                active_chat_id_for_audio = st.session_state.get('active_chat_id')
                if not active_chat_id_for_audio:
                    st.warning("Lütfen önce bir sohbet seçin veya yeni bir sohbet başlatın.", icon="⚠️")
                else:
                    transcribed_text = None # Başa al, spinner bloğundan önce tanımlı olsun
                    with st.spinner(f"🔊 '{audio_file_uploaded.name}' ses dosyası işleniyor..."):
                        recognizer_instance = sr.Recognizer()
                        try:
                            with sr.AudioFile(BytesIO(audio_file_uploaded.getvalue())) as audio_source:
                                audio_data = recognizer_instance.record(audio_source)
                                transcribed_text = recognizer_instance.recognize_google(audio_data, language="tr-TR")
                            st.success(f"**🎙️ Algılanan Metin:**\n> {transcribed_text}")
                        except sr.UnknownValueError:
                            st.error("Ses anlaşılamadı veya boş. Lütfen farklı bir dosya deneyin.", icon="🔇")
                        except sr.RequestError as e:
                            st.error(f"Google Speech Recognition servisine ulaşılamadı; {e}", icon="🌐")
                        except Exception as e:
                            st.error(f"Ses işleme sırasında beklenmedik bir hata oluştu: {e}")
                            print(f"ERROR: Audio processing failed: {e}")
                    if transcribed_text:
                        user_msg_audio = {'role': 'user', 'parts': f"(Yüklenen Ses Dosyası: {audio_file_uploaded.name}) {transcribed_text}"}
                        st.session_state.all_chats[active_chat_id_for_audio].append(user_msg_audio)
                        audio_msg_id = f"audio_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                        st.session_state.message_id_counter += 1
                        history_for_audio_prompt = st.session_state.all_chats[active_chat_id_for_audio][-20:-1]
                        with st.spinner("🤖 AI yanıtı hazırlanıyor..."):
                            ai_response_audio, sender_name_audio = get_hanogt_response_orchestrator(transcribed_text, history_for_audio_prompt, audio_msg_id, active_chat_id_for_audio, False)
                        st.markdown(f"#### {sender_name_audio} Yanıtı:")
                        st.markdown(str(ai_response_audio))
                        ai_msg_audio = {'role': 'model', 'parts': str(ai_response_audio), 'sender_display': sender_name_audio}
                        st.session_state.all_chats[active_chat_id_for_audio].append(ai_msg_audio)
                        save_all_chats(st.session_state.all_chats)
                        st.success("✅ Sesli istem ve AI yanıtı sohbete eklendi!")
                        if st.session_state.tts_enabled and globals().get('tts_engine'): speak(str(ai_response_audio))

        elif current_app_mode == "Yaratıcı Stüdyo":
            st.markdown("💡 Bir fikir verin, yapay zeka sizin için yaratıcı metinler üretsin!")
            creative_prompt_input = st.text_area("Yaratıcı Metin Tohumu:", key="creative_prompt_area", placeholder="Örn: 'Geceleri parlayan bir çiçek ve onun sırrı'", height=100)
            col_len, col_style = st.columns(2)
            length_selection = col_len.selectbox("Metin Uzunluğu:", ["kısa", "orta", "uzun"], index=1, key="creative_length_select")
            style_selection = col_style.selectbox("Metin Stili:", ["genel", "şiirsel", "hikaye"], index=0, key="creative_style_select")
            if st.button("✨ Yaratıcı Metin Üret!", key="generate_creative_text_button", type="primary", use_container_width=True):
                if creative_prompt_input and creative_prompt_input.strip():
                    active_chat_id_creative = st.session_state.get('active_chat_id', 'creative_mode_no_chat')
                    creative_msg_id = f"creative_{st.session_state.message_id_counter}_{int(time.time())}"
                    st.session_state.message_id_counter += 1
                    generated_response, response_sender_name = None, f"{APP_NAME} (Yaratıcı Modül)"
                    if globals().get('gemini_model'):
                        with st.spinner("✨ Gemini ilham perilerini çağırıyor..."):
                            gemini_system_instruction = f"Sen yaratıcı bir metin yazarısın. Kullanıcının verdiği '{creative_prompt_input}' tohumundan yola çıkarak '{style_selection}' stilinde ve '{length_selection}' uzunluğunda bir metin üret."
                            gemini_creative_response = get_gemini_response_cached(gemini_system_instruction, [], False)
                            if isinstance(gemini_creative_response, str) and not gemini_creative_response.startswith(GEMINI_ERROR_PREFIX):
                                generated_response = gemini_creative_response
                                response_sender_name = f"{APP_NAME} (Gemini Yaratıcı)"
                            else:
                                st.toast("Gemini yaratıcı yanıtı alınamadı, yerel üretici denenecek.", icon="ℹ️")
                    if not generated_response:
                        with st.spinner("✨ Hayal gücü motoru çalışıyor..."):
                            generated_response = creative_response_generator(creative_prompt_input, length_selection, style_selection)
                            first_word_of_prompt = creative_prompt_input.split()[0] if creative_prompt_input else "yaratıcı"
                            new_generated_word = advanced_word_generator(first_word_of_prompt)
                            generated_response += f"\n\n---\n🔮 **Kelimatör Önerisi:** {new_generated_word}"
                            response_sender_name = f"{APP_NAME} (Yerel Yaratıcı)"
                    st.markdown(f"#### {response_sender_name} İlhamı:")
                    st.markdown(generated_response)
                    log_interaction(f"Yaratıcı Stüdyo: {creative_prompt_input} (Stil: {style_selection}, Uzunluk: {length_selection})", generated_response, response_sender_name, creative_msg_id, active_chat_id_creative)
                    st.success("✨ Yaratıcı metin başarıyla oluşturuldu!")
                    if st.session_state.tts_enabled and globals().get('tts_engine'): speak(generated_response)
                else:
                    st.warning("Lütfen yaratıcı bir metin tohumu girin.", icon="✍️")

        elif current_app_mode == "Görsel Oluşturucu":
            st.markdown("🎨 Hayalinizi kelimelerle tarif edin, yapay zeka sizin için (basit) bir görsel çizsin!")
            st.info("ℹ️ Not: Bu mod sembolik ve basit çizimler üretir, karmaşık fotogerçekçi görseller beklemeyin.", icon="💡")
            image_prompt_input = st.text_input("Görsel Tarifi:", key="image_generation_prompt_input", placeholder="Örn: 'Karlı bir dağın zirvesinde tek bir ağaç'")
            if st.button("🖼️ Görsel Oluştur!", key="generate_image_button", type="primary", use_container_width=True):
                if image_prompt_input and image_prompt_input.strip():
                    with st.spinner("🖌️ Sanatçı fırçaları çalışıyor..."):
                        generated_image = generate_prompt_influenced_image(image_prompt_input)
                        st.image(generated_image, caption=f"'{image_prompt_input[:60]}' isteminin yorumu", use_container_width=True)
                    try:
                        image_buffer = BytesIO()
                        generated_image.save(image_buffer, format="PNG")
                        image_bytes = image_buffer.getvalue()
                        safe_filename_prompt_part = re.sub(r'[^\w\s-]', '', image_prompt_input.lower())[:30].replace(' ', '_')
                        image_file_name = f"hanogt_gorsel_{safe_filename_prompt_part or 'tarif'}_{int(time.time())}.png"
                        st.download_button("🖼️ Oluşturulan Görseli İndir", data=image_bytes, file_name=image_file_name, mime="image/png", use_container_width=True)
                        active_chat_id_image = st.session_state.get('active_chat_id')
                        if active_chat_id_image and active_chat_id_image in st.session_state.all_chats:
                            user_msg_image = {'role': 'user', 'parts': f"(Görsel Oluşturma İstemi: {image_prompt_input})"}
                            ai_msg_image = {'role': 'model', 'parts': f"'{image_prompt_input}' istemi için yukarıdaki görsel oluşturuldu. (İndirme butonu da mevcut)", 'sender_display': f"{APP_NAME} (Görsel Oluşturucu)"}
                            st.session_state.all_chats[active_chat_id_image].extend([user_msg_image, ai_msg_image])
                            save_all_chats(st.session_state.all_chats)
                            st.info("Görsel oluşturma istemi ve bilgisi aktif sohbete eklendi.", icon="💾")
                    except Exception as e:
                        st.error(f"Görsel indirme veya sohbete kaydetme sırasında hata: {e}")
                else:
                    st.warning("Lütfen bir görsel tarifi girin.", icon="✍️")

        # Footer
        st.markdown("<hr style='margin-top:1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        with footer_cols[0]:
            st.caption(f"Kullanıcı: {st.session_state.get('user_name', 'Tanımlanmamış')}")
        with footer_cols[1]:
            st.caption(f"{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}")
        with footer_cols[2]:
            ai_status_text = "Aktif" if globals().get('gemini_model') else "Kapalı"
            logging_status_text = "Aktif" if globals().get('supabase') else "Kapalı"
            st.caption(f"AI Durumu: {ai_status_text} | Loglama: {logging_status_text}", help=f"Kullanılan AI Modeli: {st.session_state.gemini_model_name}")
