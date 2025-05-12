# app.py

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

# Supabase (isteğe bağlı, loglama/feedback için)
try:
    from supabase import create_client, Client # pip install supabase
    from postgrest import APIError as SupabaseAPIError
except ImportError:
    st.toast("Supabase kütüphanesi bulunamadı. Loglama/Feedback devre dışı.", icon="ℹ️")
    create_client = None
    Client = None
    SupabaseAPIError = None

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.0.1 Pro+ Stable" # Sürüm güncellendi (Hata düzeltmeleri)
CURRENT_YEAR = datetime.now().year
CHAT_HISTORY_FILE = "chat_history_v2.json"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir sorun oluştu. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 20
SCRAPE_MAX_CHARS = 3500
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 {APP_NAME}/{APP_VERSION}" # Güncel User Agent
SUPABASE_TABLE_LOGS = "chat_logs"
SUPABASE_TABLE_FEEDBACK = "user_feedback"
FONT_FILE = "arial.ttf" # Mevcutsa kullanılacak font

# --- Dinamik Fonksiyonlar ---
DYNAMIC_FUNCTIONS_MAP = {
    "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
    "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})",
    "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')} ({datetime.now().year})"
}

# --- Bilgi Tabanı ---
knowledge_base_load_error = None

@st.cache_data(ttl=3600)
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE, user_name_for_greeting="kullanıcı"):
    """Bilgi tabanını dosyadan yükler veya varsayılanı kullanır."""
    global knowledge_base_load_error
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
            with open(filename, "r", encoding="utf-8") as f: loaded_kb = json.load(f)
            merged_kb = {**default_knowledge, **loaded_kb}
            knowledge_base_load_error = None
            return merged_kb
        else:
            knowledge_base_load_error = f"Bilgi tabanı ({filename}) bulunamadı. Varsayılan kullanılıyor."
            st.toast(knowledge_base_load_error, icon="ℹ️")
            return default_knowledge
    except json.JSONDecodeError:
        knowledge_base_load_error = f"Bilgi tabanı ({filename}) hatalı. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="⚠️")
        return default_knowledge
    except Exception as e:
        knowledge_base_load_error = f"Bilgi tabanı yüklenirken hata: {e}. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="🔥")
        return default_knowledge

def kb_chatbot_response(query, knowledge_base_dict):
    """Bilgi tabanından veya dinamik fonksiyonlardan yanıt döndürür."""
    query_lower = query.lower().strip()
    # 1. Dinamik Fonksiyon
    if query_lower in DYNAMIC_FUNCTIONS_MAP:
        try: return DYNAMIC_FUNCTIONS_MAP[query_lower]()
        except Exception as e: st.error(f"Fonksiyon hatası ({query_lower}): {e}"); return DEFAULT_ERROR_MESSAGE
    # 2. Tam Eşleşme
    if query_lower in knowledge_base_dict:
        resp = knowledge_base_dict[query_lower]
        return random.choice(resp) if isinstance(resp, list) else resp
    # 3. Kısmi Eşleşme (İçerme)
    partial_matches = [resp for key, resp_list in knowledge_base_dict.items() if key in query_lower for resp in (resp_list if isinstance(resp_list, list) else [resp_list])]
    if partial_matches: return random.choice(list(set(partial_matches)))
    # 4. Benzerlik Skoru (Kelime Kesişimi) - Basit versiyon
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_score, best_responses = 0, []
    for key, resp_list in knowledge_base_dict.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words: continue
        score = len(query_words.intersection(key_words)) / len(key_words) if key_words else 0
        if score > 0.6: # Benzerlik eşiği
            options = resp_list if isinstance(resp_list, list) else [resp_list]
            if score > best_score: best_score, best_responses = score, options
            elif score == best_score: best_responses.extend(options)
    if best_responses: return random.choice(list(set(best_responses)))
    return None

# --- API Anahtarı ve Gemini Yapılandırması ---
gemini_model = None
gemini_init_error_global = None

def initialize_gemini_model():
    """Google Generative AI modelini session state'deki ayarlarla başlatır."""
    global gemini_init_error_global
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta bulunamadı! (st.secrets['GOOGLE_API_KEY'])"
        return None
    try:
        genai.configure(api_key=api_key)
        safety = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        model_name = st.session_state.get('gemini_model_name', 'gemini-1.5-flash-latest')
        config = genai.types.GenerationConfig(
            temperature=st.session_state.get('gemini_temperature', 0.7),
            top_p=st.session_state.get('gemini_top_p', 0.95),
            top_k=st.session_state.get('gemini_top_k', 40),
            max_output_tokens=st.session_state.get('gemini_max_tokens', 4096)
        )
        model = genai.GenerativeModel(model_name=model_name, safety_settings=safety, generation_config=config)
        gemini_init_error_global = None
        st.toast(f"✨ Gemini modeli ({model_name}) yüklendi!", icon="🤖")
        return model
    except Exception as e:
        gemini_init_error_global = f"🛑 Gemini yapılandırma hatası: {e}."
        print(f"ERROR: Gemini Init Failed: {e}")
        return None

# --- Supabase İstemcisini Başlatma ---
supabase = None
supabase_error_global = None

@st.cache_resource(ttl=3600)
def init_supabase_client_cached():
    """Supabase istemcisini başlatır ve cache'ler."""
    global supabase_error_global
    if not create_client:
        supabase_error_global = "Supabase kütüphanesi yüklenemedi. Loglama/Feedback devre dışı."
        return None
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        supabase_error_global = "Supabase URL/Key Secrets'ta bulunamadı! Loglama/Feedback devre dışı."
        return None
    try:
        client: Client = create_client(url, key)
        supabase_error_global = None
        st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱")
        return client
    except Exception as e:
        error = f"Supabase bağlantı hatası: {e}."
        if "invalid url" in str(e).lower(): error += " URL formatını kontrol edin."
        elif "invalid key" in str(e).lower(): error += " Service Key'i kontrol edin."
        supabase_error_global = error
        print(f"ERROR: Supabase Connection Failed: {e}")
        return None

# --- YARDIMCI FONKSİYONLAR ---
def _get_session_id():
    """Oturum ID'sini alır veya oluşturur."""
    if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

tts_engine = None
tts_init_error_global = None

@st.cache_resource
def init_tts_engine_cached():
    """Metin okuma (TTS) motorunu başlatır."""
    global tts_init_error_global
    try:
        engine = pyttsx3.init()
        tts_init_error_global = None
        st.toast("🔊 TTS motoru hazır.", icon="🗣️")
        return engine
    except Exception as e:
        tts_init_error_global = f"⚠️ TTS motoru başlatılamadı: {e}."
        print(f"ERROR: TTS Init Failed: {e}")
        return None

def speak(text):
    """Verilen metni sesli okur."""
    engine = globals().get('tts_engine')
    if not engine: st.toast("TTS motoru aktif değil.", icon="🔇"); return
    if not st.session_state.get('tts_enabled', True): st.toast("TTS ayarlardan kapalı.", icon="🔇"); return
    try:
        cleaned = re.sub(r'[^\w\s.,!?-]', '', text) # Basit temizleme
        if not cleaned.strip(): st.toast("Okunacak metin yok.", icon="ℹ️"); return
        engine.say(cleaned)
        engine.runAndWait()
    except RuntimeError as e: st.warning(f"TTS çalışma zamanı sorunu: {e}.", icon="🔊")
    except Exception as e: st.error(f"TTS hatası: {e}", icon="🔥"); print(f"ERROR: TTS Speak Failed: {e}")

def _clean_text(text):
    """Metindeki fazla boşlukları/satırları temizler."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

@st.cache_data(ttl=600) # Kazıma sonuçlarını 10dk cache'le
def scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS):
    """URL'den ana metin içeriğini kazır (cache'li)."""
    st.toast(f"🌐 '{urlparse(url).netloc}' alınıyor...", icon="⏳")
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]) or parsed.scheme not in ['http', 'https']:
            st.warning(f"Geçersiz URL: {url}", icon="🔗"); return None
        headers = {'User-Agent': USER_AGENT, 'Accept-Language': 'tr-TR,tr;q=0.9', 'Accept': 'text/html', 'DNT': '1'}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get('content-type', '').lower()
        if 'html' not in ctype:
            st.info(f"HTML değil ('{ctype}'). Atlanıyor: {url}", icon="📄"); resp.close(); return None

        html = ""; size = 0; max_size = max_chars * 12 # Max HTML boyutu
        try:
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True, errors='ignore'):
                if chunk: html += chunk; size += len(chunk.encode('utf-8', 'ignore'))
                if size > max_size: st.warning(f"HTML çok büyük (> {max_size//1024}KB), kesiliyor.", icon="✂️"); break
        finally: resp.close()
        if not html: st.warning("Boş içerik alındı.", icon="📄"); return None

        soup = BeautifulSoup(html, 'lxml')
        tags_remove = ["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio", "figure", "input", "select"]
        for tag in soup.find_all(tags_remove): tag.decompose()

        content = []
        selectors = ['article[class*="content"]', 'article[class*="post"]', 'main[id*="content"]', 'main', 'div[class*="post-body"]', 'div[class*="article-body"]', 'div[itemprop="articleBody"]', 'article', '.content', '#content']
        container = None
        for sel in selectors:
            found = soup.select(sel, limit=1)
            if found: container = found[0]; break

        min_len = 80; min_indicator = 1
        if container:
            paragraphs = container.find_all('p', limit=60)
            for p in paragraphs:
                text = _clean_text(p.get_text(separator=' ', strip=True))
                if len(text) > min_len and (text.count('.') + text.count('!') + text.count('?')) >= min_indicator: content.append(text)
        if not content or len(" ".join(content)) < 300: # Body fallback
             body = soup.body
             if body:
                 body_text = _clean_text(body.get_text(separator='\n', strip=True))
                 parts = [p.strip() for p in body_text.split('\n') if len(p.strip()) > min_len]
                 if len(" ".join(parts)) > 200:
                      st.toast("Özel alan bulunamadı, genel metin kullanıldı.", icon="ℹ️"); content = parts[:40]
                 else: st.toast("Sayfada anlamlı içerik bulunamadı.", icon="📄"); return None
             else: st.toast("Body etiketi bulunamadı.", icon="📄"); return None

        full_text = "\n\n".join(content); cleaned = _clean_text(full_text)
        if not cleaned: st.toast("Kazıma sonrası boş içerik.", icon="📄"); return None
        final = cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")
        st.toast(f"'{urlparse(url).netloc}' içeriği alındı.", icon="✅")
        return final
    except requests.exceptions.RequestException as e: st.toast(f"⚠️ Ağ hatası ({type(e).__name__}): {url}", icon='🌐')
    except Exception as e: st.toast(f"⚠️ Kazıma hatası: {e}", icon='🔥'); print(f"ERROR: Scraping '{url}' failed: {e}")
    return None

@st.cache_data(ttl=600) # Arama sonuçlarını 10dk cache'le
def search_web(query):
    """Web'de arama yapar (Wikipedia, DDG) ve sonuçları döndürür."""
    st.toast(f"🔍 '{query}' aranıyor...", icon="⏳")
    wikipedia.set_lang("tr"); result = None
    # 1. Wikipedia
    try:
        wp_page = wikipedia.page(query, auto_suggest=False, redirect=True)
        summary = wikipedia.summary(query, sentences=6, auto_suggest=False, redirect=True)
        result = f"**Wikipedia ({wp_page.title}):**\n\n{_clean_text(summary)}\n\nKaynak: {wp_page.url}"
        st.toast(f"✅ Wikipedia'dan '{wp_page.title}' bulundu.", icon="📚"); return result
    except wikipedia.exceptions.PageError: st.toast(f"ℹ️ Wikipedia'da '{query}' bulunamadı.", icon="🤷")
    except wikipedia.exceptions.DisambiguationError as e: result = f"**Wikipedia'da Çok Anlamlı ({query}):**\n{e.options[:3]}..." # Devam et
    except Exception as e: st.toast(f"⚠️ Wikipedia hatası: {e}", icon="🔥")
    # 2. DuckDuckGo
    ddg_url = None
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs:
            ddg_results = list(ddgs.text(query, region='tr-tr', safesearch='moderate', max_results=3))
            if ddg_results:
                best_res = ddg_results[0]
                snippet = best_res.get('body'); url = best_res.get('href')
                if snippet and url:
                    ddg_url = unquote(url); domain = urlparse(ddg_url).netloc
                    st.toast(f"ℹ️ DDG'den '{domain}' özeti bulundu.", icon="🦆")
                    result = f"**Web Özeti (DDG - {domain}):**\n\n{_clean_text(snippet)}\n\nKaynak: {ddg_url}"
    except Exception as e: st.toast(f"⚠️ DDG hatası: {e}", icon="🔥")
    # 3. Kazıma (DDG URL varsa)
    if ddg_url:
        scraped = scrape_url_content(ddg_url) # Cache'li fonksiyonu çağır
        if scraped:
            domain = urlparse(ddg_url).netloc
            result = f"**Web Sayfasından ({domain}):**\n\n{scraped}\n\nKaynak: {ddg_url}" # Kazınan içerik öncelikli
            st.toast(f"✅ '{domain}' içeriği kazındı.", icon="📄")
        elif result: st.toast("ℹ️ Sayfa kazınamadı, DDG özeti kullanılıyor.", icon="📝")
        else: result = f"Detay için: {ddg_url}" # Sadece URL kaldıysa
    if not result: st.toast(f"'{query}' için web sonucu bulunamadı.", icon="❌")
    return result

# --- Sohbet Geçmişi Yönetimi ---
@st.cache_data(ttl=86400)
def load_all_chats_cached(file_path=CHAT_HISTORY_FILE):
    """Tüm sohbet geçmişlerini dosyadan yükler (cache'li)."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            if content and content.strip():
                data = json.loads(content)
                if isinstance(data, dict): return {str(k): v for k, v in data.items()}
                else: # Eski format veya bozuk
                    st.warning(f"Geçersiz format ({file_path}). Yeni yapıya geçiliyor.", icon="⚠️")
                    try: os.rename(file_path, f"{file_path}.backup_{int(time.time())}")
                    except OSError: pass; return {}
            else: return {} # Boş dosya
        except json.JSONDecodeError:
            st.error(f"Sohbet dosyası ({file_path}) bozuk. Yeni başlatılıyor.", icon="🔥")
            try: os.rename(file_path, f"{file_path}.corrupt_{int(time.time())}")
            except OSError: pass; return {}
        except Exception as e: st.error(f"Sohbet yüklenirken hata: {e}", icon="🔥"); return {}
    return {}

def save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE):
    """Tüm sohbetleri dosyaya kaydeder."""
    try:
        with open(file_path, "w", encoding="utf-8") as f: json.dump(chats_dict, f, ensure_ascii=False, indent=2)
    except Exception as e: st.error(f"Sohbet kaydedilemedi: {e}", icon="🔥"); print(f"ERROR: Save chats failed: {e}")

# --- Gemini Yanıt Alma ---
def get_gemini_response_cached(prompt, history, stream=False):
    """Gemini API'den yanıt alır."""
    model = globals().get('gemini_model')
    if not model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    validated_history = []
    for msg in history: # API formatını doğrula/düzelt
        role, parts = msg.get('role'), msg.get('parts')
        if role in ['user', 'model'] and isinstance(parts, str) and parts.strip():
             validated_history.append({'role': role, 'parts': [parts]}) # API 'parts'ı liste bekler
        elif role in ['user', 'model'] and isinstance(parts, list) and parts and isinstance(parts[0], str):
             validated_history.append(msg) # Zaten doğru formatta
    try:
        chat = model.start_chat(history=validated_history)
        response = chat.send_message(prompt, stream=stream)
        if stream: return response
        else: # Stream değilse içeriği kontrol et
             if response.parts: return "".join(p.text for p in response.parts if hasattr(p, 'text'))
             else: # Neden boş geldi?
                 reason = getattr(response.prompt_feedback, 'block_reason', None)
                 if reason: msg = f"Yanıt engellendi ({reason})."
                 else: reason = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None; msg = f"Yanıt tam değil ({reason})." if reason != 'STOP' else "Boş yanıt."
                 st.warning(msg, icon="🛡️" if "block" in msg.lower() else "⚠️"); return f"{GEMINI_ERROR_PREFIX} {msg}"
    except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as e: st.error(f"Gemini Hatası: {e}", icon="🛑"); return f"{GEMINI_ERROR_PREFIX} API Kısıtlaması: {e}"
    except requests.exceptions.RequestException as e: st.error(f"Gemini Ağ Hatası: {e}", icon="📡"); return f"{GEMINI_ERROR_PREFIX} Ağ Hatası: {e}"
    except Exception as e: st.error(f"Gemini API Hatası: {e}", icon="🔥"); print(f"ERROR: Gemini API failed: {e}"); return f"{GEMINI_ERROR_PREFIX} API Hatası: {e}"

# --- Supabase Loglama ---
def log_to_supabase(table, data):
    """Veriyi Supabase'e loglar."""
    client = globals().get('supabase')
    if not client: print(f"INFO: Supabase unavailable, skip log to '{table}'."); return False
    try:
        defaults = {'user_name': st.session_state.get('user_name', 'N/A'), 'session_id': _get_session_id(), 'app_version': APP_VERSION, 'chat_id': st.session_state.get('active_chat_id', 'N/A')}
        log_data = {**defaults, **data} # Varsayılanları ekle
        client.table(table).insert(log_data).execute()
        # print(f"DEBUG: Supabase log success to '{table}'.") # Başarı logu
        return True
    except SupabaseAPIError as e: st.toast(f"⚠️ Loglama hatası: {e.message}", icon="💾"); print(f"ERROR: Supabase API Error ({table}): {e}"); return False
    except Exception as e: st.error("Loglama sırasında kritik hata!"); print(f"ERROR: Supabase Log Critical ({table}): {e}"); return False

def log_interaction(prompt, response, source, msg_id, chat_id):
    """Etkileşimi loglar."""
    return log_to_supabase(SUPABASE_TABLE_LOGS, {"user_prompt": prompt, "ai_response": response, "response_source": source, "message_id": msg_id, "chat_id": chat_id})

def log_feedback(msg_id, prompt, response, f_type, comment=""):
    """Geri bildirimi loglar."""
    data = {"message_id": msg_id, "user_prompt": prompt, "ai_response": response, "feedback_type": f_type, "comment": comment}
    if log_to_supabase(SUPABASE_TABLE_FEEDBACK, data): st.toast("Geri bildiriminiz için teşekkürler!", icon="💌"); return True
    else: st.toast("Geri bildirim gönderilemedi.", icon="😔"); return False

# --- Yanıt Orkestrasyonu ---
def get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id, use_stream=False):
    """Farklı kaynaklardan yanıt alır."""
    response, source_tag = None, "Bilinmiyor"
    # 1. KB / Fonksiyon
    kb_resp = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
    if kb_resp:
        source_tag = "Fonksiyonel" if prompt.lower() in DYNAMIC_FUNCTIONS_MAP else "Bilgi Tabanı"
        log_interaction(prompt, kb_resp, source_tag, msg_id, chat_id)
        return kb_resp, f"{APP_NAME} ({source_tag})"
    # 2. Gemini
    if globals().get('gemini_model'):
        gemini_resp = get_gemini_response_cached(prompt, history, stream=use_stream)
        if gemini_resp:
            if use_stream: return gemini_resp, f"{APP_NAME} (Gemini Stream)" # Loglama stream sonrası yapılır
            elif isinstance(gemini_resp, str) and not gemini_resp.startswith(GEMINI_ERROR_PREFIX):
                 source_tag = "Gemini"; log_interaction(prompt, gemini_resp, source_tag, msg_id, chat_id); return gemini_resp, f"{APP_NAME} ({source_tag})"
            # else: Gemini hatası veya boş yanıt, devam et
    # 3. Web Arama (Gerekliyse)
    is_q = "?" in prompt or any(k in prompt.lower() for k in ["nedir", "kimdir", "nasıl", "bilgi", "araştır", "haber"])
    if not response and is_q and len(prompt.split()) > 2:
        web_resp = search_web(prompt) # Cache'li fonksiyon
        if web_resp:
            if "Wikipedia" in web_resp: source_tag = "Wikipedia"
            elif "Web Sayfasından" in web_resp: source_tag = "Web Kazıma"
            elif "Web Özeti" in web_resp: source_tag = "Web Özeti (DDG)"
            else: source_tag = "Web Arama"
            log_interaction(prompt, web_resp, source_tag, msg_id, chat_id); return web_resp, f"{APP_NAME} ({source_tag})"
    # 4. Varsayılan Yanıt
    defaults = [f"Üzgünüm {st.session_state.get('user_name', '')}, yardımcı olamıyorum.", "Anlayamadım, farklı sorar mısınız?", "Bu konuda bilgim yok.", "Öğreniyorum..."]
    response = random.choice(defaults); source_tag = "Varsayılan"
    log_interaction(prompt, response, source_tag, msg_id, chat_id)
    return response, f"{APP_NAME} ({source_tag})"

# --- Yaratıcı Modüller ---
def creative_response_generator(prompt, length="orta", style="genel"):
    """Yerel basit yaratıcı metin üretir."""
    templates = {"genel": ["İşte bir fikir: {}", "Hayal edelim: {}"], "şiirsel": ["Kalbimden: {}", "Sözcüklerle: {}"], "hikaye": ["Bir varmış: {}", "Sahne sizin: {}"]}
    template = random.choice(templates.get(style, templates["genel"]))
    idea = generate_new_idea_creative(prompt, style)
    sentences = [s.strip() for s in idea.split('.') if s.strip()]
    n = len(sentences)
    if length == "kısa" and n > 1: idea = ". ".join(sentences[:max(1, n // 3)]) + "."
    elif length == "uzun" and n > 0: idea += f"\n\nDahası, {generate_new_idea_creative(prompt[::-1], style)}"
    return template.format(idea)

def generate_new_idea_creative(seed, style="genel"):
    """Rastgele kelimelerle fikir üretir."""
    elems = ["zaman kristalleri", "psişik ormanlar", "rüya mimarisi", "kuantum köpüğü", "gölge enerjisi"]
    acts = ["dokur", "çözer", "yansıtır", "inşa eder", "fısıldar"]
    outs = ["kaderi", "varoluşun kodunu", "bilincin sınırlarını", "kadim sırları", "evrenin melodisini"]
    words = re.findall(r'\b\w{4,}\b', seed.lower())
    seeds = random.sample(words, k=min(len(words), 1)) + ["gizem"]
    e1, a1, o1 = random.choice(elems), random.choice(acts), random.choice(outs)
    return f"{seeds[0].capitalize()} {a1}, {e1} aracılığıyla {o1}."

def advanced_word_generator(base):
    """Yeni 'teknik' kelimeler türetir."""
    if not base or len(base) < 2: return "Kelimatör"
    v="aeıioöuü"; c="bcçdfgğhjklmnprsştvyz"; cln = "".join(filter(str.isalpha, base.lower()))
    if not cln: return "SözcükMimar"
    pre = ["bio", "krono", "psiko", "neo", "mega", "nano", "astro", "poli", "meta", "trans", "ultra", "xeno"]
    suf = ["genez", "sfer", "nomi", "tek", "loji", "tronik", "morf", "vers", "dinamik", "matik", "kinezis", "skop"]
    core = cln[random.randint(0, max(0, len(cln)-3)):][:random.randint(2,3)] if len(cln)>2 and random.random()<0.6 else "".join(random.choice(c if i%2 else v) for i in range(random.randint(3,4)))
    word = core
    if random.random()>0.4: word = random.choice(pre) + word
    if random.random()>0.4: word += random.choice(suf)
    return word.capitalize() if len(word)>1 else word

# --- Görsel Oluşturucu ---
def generate_prompt_influenced_image(prompt):
    """Prompt'a göre basit kural tabanlı görsel oluşturur."""
    w, h = 512, 512; p_lower = prompt.lower()
    themes = { # Daha fazla tema eklenebilir
        "güneş": {"bg": [(255,230,150),(255,160,0)], "sh": [{"t":"circle","c":(255,255,0,220),"p":(0.25,0.25),"s":0.2}]},
        "ay": {"bg": [(10,10,50),(40,40,100)], "sh": [{"t":"circle","c":(240,240,240,200),"p":(0.75,0.2),"s":0.15}]},
        "gökyüzü": {"bg": [(135,206,250),(70,130,180)], "sh": []},
        "bulut": {"bg":None, "sh": [{"t":"ellipse","c":(255,255,255,180),"p":(random.uniform(0.2,0.8),random.uniform(0.1,0.4)),"swh":(random.uniform(0.15,0.35),random.uniform(0.08,0.15))} for _ in range(random.randint(2,4))]},
        "deniz": {"bg": [(0,105,148),(0,0,100)], "sh": [{"t":"rect","c":(60,120,180,150),"p":(0.5,0.75),"swh":(1.0,0.5)}]},
        "orman": {"bg": [(34,139,34),(0,100,0)], "sh": [{"t":"tri","c":(random.randint(0,30),random.randint(70,100),random.randint(0,30),200),"p":(random.uniform(0.1,0.9),random.uniform(0.55,0.85)),"s":random.uniform(0.08,0.25)} for _ in range(random.randint(7,12))]},
        # --- DÜZELTİLMİŞ "ağaç" TANIMI (SyntaxError Fix) ---
        "ağaç": {
            "bg": [(180, 220, 180), (140, 190, 140)],
            "sh": [ # 'sh' kısaltması kullanıldı (shapes yerine)
                {"t": "rect", "c": (139, 69, 19, 255), "p": (random.uniform(0.2, 0.8), 0.75), "swh": (0.06, 0.4)}, # Gövde
                {"t": "ellipse", "c": (34, 139, 34, 200), "p": (random.uniform(0.2, 0.8), 0.45), "swh": (0.3, 0.25)}  # Tepe (Basit pos)
            ]
        },
        # --- DÜZELTME BİTTİ ---
        "dağ": {"bg": [(200,200,200),(100,100,100)], "sh": [{"t":"poly","c":(random.randint(130,170),random.randint(130,170),random.randint(130,170),230),"pts":[(random.uniform(0.1,0.4),0.85),(0.5,random.uniform(0.1,0.4)),(random.uniform(0.6,0.9),0.85)]} for _ in range(random.randint(1,3))]},
        "şehir": {"bg": [(100,100,120),(50,50,70)], "sh": [{"t":"rect","c":(random.randint(60,100),random.randint(60,100),random.randint(70,110),random.randint(180,220)),"p":(random.uniform(0.1,0.9),random.uniform(0.4,0.85)),"swh":(random.uniform(0.04,0.15),random.uniform(0.15,0.65))} for _ in range(random.randint(8,15))]},
        "kar": {"bg":None, "sh": [{"t":"circle","c":(255,255,255,150),"p":(random.random(),random.random()),"s":0.005} for _ in range(100)]},
        "yıldız": {"bg":None, "sh": [{"t":"circle","c":(255,255,200,200),"p":(random.random(),random.uniform(0,0.5)),"s":0.003} for _ in range(70)]},
    }
    bg1, bg2 = (random.randint(30,120),)*3, (random.randint(120,220),)*3
    shapes = []; themes_applied = 0
    for kw, theme in themes.items():
        if kw in p_lower:
            if theme["bg"] and themes_applied==0: bg1, bg2 = theme["bg"]
            shapes.extend(theme["sh"]); themes_applied+=1

    img = Image.new('RGBA', (w,h), (0,0,0,0)); draw = ImageDraw.Draw(img)
    for y in range(h): # Arka plan gradient
        r = y/h; R,G,B = [int(bg1[i]*(1-r)+bg2[i]*r) for i in range(3)]; draw.line([(0,y),(w,y)],fill=(R,G,B,255))
    # Şekiller (kısaltılmış isimler kullanıldı: t, c, p, s, swh, pts)
    for s in shapes:
        try:
            st, sc, sp = s["t"], s["c"], s.get("p"); out = (0,0,0,50) if len(sc)==4 and sc[3]<250 else None
            if sp: cx, cy = int(sp[0]*w), int(sp[1]*h)
            if st=="circle": r=int(s["s"]*min(w,h)/2); draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=sc,outline=out)
            elif st=="rect" or st=="ellipse": wr,hr=s["swh"]; wp,hp=int(wr*w),int(hr*h); draw.rectangle((cx-wp//2,cy-hp//2,cx+wp//2,cy+hp//2),fill=sc,outline=out) if st=="rect" else draw.ellipse((cx-wp//2,cy-hp//2,cx+wp//2,cy+hp//2),fill=sc,outline=out)
            elif st=="tri": sz=int(s["s"]*min(w,h)); p1,p2,p3=(cx,cy-int(sz*0.58)),(cx-sz//2,cy+int(sz*0.3)),(cx+sz//2,cy+int(sz*0.3)); draw.polygon([p1,p2,p3],fill=sc,outline=out)
            elif st=="poly": pts_px=[(int(p[0]*w),int(p[1]*h)) for p in s["pts"]]; draw.polygon(pts_px,fill=sc,outline=out)
        except Exception as e: print(f"DEBUG: Shape draw error {s}: {e}"); continue
    if themes_applied==0: # Rastgele şekiller
        for _ in range(random.randint(4,7)): x,y=random.randint(0,w),random.randint(0,h); clr=tuple(random.randint(50,250) for _ in range(3))+(random.randint(150,220),); r=random.randint(20,70); draw.ellipse((x-r,y-r,x+r,y+r),fill=clr) if random.random()>0.5 else draw.rectangle((x-r//2,y-r//2,x+r//2,y+r//2),fill=clr)
    # Metin yazdırma (hata kontrolü içinde)
    try:
        font=ImageFont.load_default(); txt=prompt[:80]
        if os.path.exists(FONT_FILE):
             try: fsize=max(14,min(28,int(w/(len(txt)*0.3+10)))); font=ImageFont.truetype(FONT_FILE,fsize)
             except IOError: st.toast(f"Font ({FONT_FILE}) yüklenemedi.",icon="⚠️")
        bb=draw.textbbox((0,0),txt,font=font,anchor="lt") if hasattr(draw,'textbbox') else draw.textsize(txt,font=font); tw,th=bb[2]-bb[0] if hasattr(draw,'textbbox') else bb[0], bb[3]-bb[1] if hasattr(draw,'textbbox') else bb[1]
        tx,ty=(w-tw)/2, h*0.95-th; draw.text((tx+1,ty+1),txt,font=font,fill=(0,0,0,150)); draw.text((tx,ty),txt,font=font,fill=(255,255,255,230))
    except Exception as e: st.toast(f"Metin yazılamadı: {e}",icon="📝")
    return img.convert("RGB")

# --- Session State Başlatma ---
def initialize_session_state():
    """Session State için varsayılan değerleri ayarlar."""
    defaults = {
        'all_chats': {}, 'active_chat_id': None, 'next_chat_id_counter': 0,
        'app_mode': "Yazılı Sohbet", 'user_name': None, 'user_avatar_bytes': None,
        'show_main_app': False, 'greeting_message_shown': False,
        'tts_enabled': True, 'gemini_stream_enabled': True,
        'gemini_temperature': 0.7, 'gemini_top_p': 0.95, 'gemini_top_k': 40,
        'gemini_max_tokens': 4096, 'gemini_model_name': 'gemini-1.5-flash-latest',
        'message_id_counter': 0, 'last_ai_response_for_feedback': None,
        'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
        'feedback_comment_input': "", 'show_feedback_comment_form': False,
        'session_id': str(uuid.uuid4()), 'last_feedback_type': 'positive',
        'models_initialized': False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

initialize_session_state()

# --- Modelleri ve İstemcileri Başlatma (Sadece İlk Çalıştırmada) ---
if not st.session_state.models_initialized:
    print("INFO: Initializing models, clients, and loading data...")
    gemini_model = initialize_gemini_model()
    supabase = init_supabase_client_cached()
    tts_engine = init_tts_engine_cached()
    st.session_state.all_chats = load_all_chats_cached()
    if not st.session_state.active_chat_id and st.session_state.all_chats:
        st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[-1] # En son sohbeti aktif yap
    user_greeting = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting)
    st.session_state.models_initialized = True
    print("INFO: Initialization complete.")
else: # Sonraki çalıştırmalarda global değişkenlerden al
    gemini_model = globals().get('gemini_model')
    supabase = globals().get('supabase')
    tts_engine = globals().get('tts_engine')
    user_greeting = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting) # KB'yi güncel tut

# Global hata mesajlarını al
gemini_init_error = globals().get('gemini_init_error_global')
supabase_error = globals().get('supabase_error_global')
tts_init_error = globals().get('tts_init_error_global')

# Giriş yapıldıysa ana uygulamayı göster
if st.session_state.user_name and not st.session_state.show_main_app:
    st.session_state.show_main_app = True

# --- ARAYÜZ BÖLÜMLERİ --- (Fonksiyon Tanımları)
# display_settings_section, display_chat_list_and_about,
# display_chat_message_with_feedback, display_feedback_form_if_active,
# display_chat_interface_main fonksiyonları önceki yanıttaki gibi
# (içerikleri buraya tekrar eklenmedi, sadece fonksiyon isimleri referans olarak bırakıldı)
# ÖNEMLİ: Bu fonksiyonların içeriğini bir önceki yanıttan alıp buraya eklemeniz gerekir.
# Sadece `display_settings_section` içindeki Gemini ayarları bölümünün
# nested expander olmadan yapıldığı versiyonu kullanın.

# Önceki Yanıttan Kopyalanacak Fonksiyonların Tanımları:
def display_settings_section():
    """Ayarlar ve Kişiselleştirme bölümünü ana alanda (expander içinde) gösterir."""
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        col1, col2 = st.columns([0.8, 0.2]) # Profil ve Avatar için kolonlar
        with col1:
            st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**")
            new_user_name = st.text_input("Adınızı Değiştirin:", value=st.session_state.user_name, key="change_name_main_input", label_visibility="collapsed")
            if new_user_name != st.session_state.user_name and new_user_name.strip():
                st.session_state.user_name = new_user_name.strip()
                load_knowledge_from_file.clear() # KB cache'ini temizle
                st.toast("Adınız güncellendi!", icon="✏️"); st.rerun()
        with col2:
            if st.session_state.user_avatar_bytes:
                st.image(st.session_state.user_avatar_bytes, width=60, use_column_width='auto')
                if st.button("🗑️", key="remove_avatar_main_button", help="Avatarı kaldır", use_container_width=True):
                    st.session_state.user_avatar_bytes = None
                    st.toast("Avatar kaldırıldı.", icon="🗑️"); st.rerun()
            else: st.caption("Avatar Yok") # Daha kompakt

        uploaded_avatar_file = st.file_uploader("Avatar Yükle (Max 2MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader_main_file", label_visibility="collapsed")
        if uploaded_avatar_file:
            if uploaded_avatar_file.size > 2 * 1024 * 1024: st.error("Dosya > 2MB!", icon="️")
            else: st.session_state.user_avatar_bytes = uploaded_avatar_file.getvalue(); st.toast("Avatar güncellendi!", icon="🖼️"); st.rerun()
        st.caption("Avatar sadece bu oturumda saklanır.")

        st.divider()
        st.subheader("🤖 Yapay Zeka ve Arayüz")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
             engine_ready = globals().get('tts_engine') is not None
             st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not engine_ready, help="AI yanıtlarını sesli oku.")
        with tcol2:
             st.session_state.gemini_stream_enabled = st.toggle("Yanıt Akışı (Stream)", value=st.session_state.gemini_stream_enabled, help="Yanıtları kelime kelime al.")

        # --- Hanogt AI (Gemini) Gelişmiş Yapılandırma ---
        st.markdown("---")
        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.session_state.gemini_model_name = st.selectbox("AI Modeli:", ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'], index=0 if st.session_state.gemini_model_name == 'gemini-1.5-flash-latest' else 1, key="gemini_model_selector_main", help="Model yetenekleri/maliyetleri farklıdır.")
            st.session_state.gemini_temperature = st.slider("Sıcaklık:", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="gemini_temp_slider_main", help="Yaratıcılık (0=Kesin, 1=Yaratıcı)")
            st.session_state.gemini_max_tokens = st.slider("Maks Token:", 256, 8192, st.session_state.gemini_max_tokens, 128, key="gemini_max_tokens_slider_main", help="Max yanıt uzunluğu")
        with gcol2:
            st.session_state.gemini_top_k = st.slider("Top K:", 1, 100, st.session_state.gemini_top_k, 1, key="gemini_top_k_slider_main", help="Kelime Seçim Çeşitliliği")
            st.session_state.gemini_top_p = st.slider("Top P:", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="gemini_top_p_slider_main", help="Kelime Seçim Odaklılığı")
            if st.button("⚙️ AI Ayarlarını Uygula", key="reload_gemini_settings_main_btn", use_container_width=True, type="primary", help="Seçili AI modelini ve parametreleri yeniden yükler."):
                global gemini_model
                with st.spinner("AI modeli yeniden başlatılıyor..."): gemini_model = initialize_gemini_model()
                if not gemini_model: st.error("AI modeli yüklenemedi.")
                st.rerun()

        # --- Geçmiş Yönetimi ---
        st.divider()
        st.subheader("🧼 Geçmiş Yönetimi")
        if st.button("🧹 TÜM Sohbet Geçmişini Sil", use_container_width=True, type="secondary", key="clear_all_history_main_btn", help="Dikkat! Kaydedilmiş tüm sohbetleri siler."):
            if st.session_state.all_chats:
                st.session_state.all_chats = {}; st.session_state.active_chat_id = None
                save_all_chats({}) # Dosyayı da boşalt
                st.toast("TÜM sohbet geçmişi silindi!", icon="🗑️"); st.rerun()
            else: st.toast("Sohbet geçmişi zaten boş.", icon="ℹ️")

def display_chat_list_and_about(left_column):
    """Sol kolonda sohbet listesini ve Hakkında bölümünü gösterir."""
    with left_column:
        st.markdown("#### Sohbetler")
        if st.button("➕ Yeni Sohbet", use_container_width=True, key="new_chat_button"):
            st.session_state.next_chat_id_counter += 1; ts = int(time.time())
            new_id = f"chat_{st.session_state.next_chat_id_counter}_{ts}"
            st.session_state.all_chats[new_id] = []
            st.session_state.active_chat_id = new_id
            save_all_chats(st.session_state.all_chats); st.rerun()

        st.markdown("---")
        chat_list_container = st.container(height=400, border=False)
        with chat_list_container:
            chats = st.session_state.all_chats
            sorted_ids = sorted(chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
            if not sorted_ids: st.caption("Henüz bir sohbet yok.")
            else:
                active_id = st.session_state.get('active_chat_id')
                for chat_id in sorted_ids:
                    history = chats.get(chat_id, [])
                    first_msg = next((m.get('parts','') for m in history if m.get('role')=='user'), None)
                    title = f"Sohbet {chat_id.split('_')[1]}"
                    if first_msg: title = first_msg[:35] + ("..." if len(first_msg)>35 else "")
                    elif history: title = "Başlıksız Sohbet"

                    lcol, rcol = st.columns([0.8, 0.2])
                    btn_type = "primary" if active_id == chat_id else "secondary"
                    if lcol.button(title, key=f"select_{chat_id}", use_container_width=True, type=btn_type, help=f"'{title}' aç"):
                        if active_id != chat_id: st.session_state.active_chat_id = chat_id; st.rerun()
                    if rcol.button("❌", key=f"delete_{chat_id}", use_container_width=True, help=f"'{title}' sil", type="secondary"):
                         if chat_id in chats:
                             del chats[chat_id]
                             if active_id == chat_id:
                                 remaining = sorted(chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
                                 st.session_state.active_chat_id = remaining[0] if remaining else None
                             save_all_chats(chats); st.toast(f"'{title}' silindi.", icon="🗑️"); st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Uygulama Hakkında", expanded=False):
             st.markdown(f"**{APP_NAME} v{APP_VERSION}**\n\nYapay zeka destekli kişisel asistan.\n\nGeliştirici: **Hanogt**\n\nTeknolojiler: Streamlit, Gemini API, Python...\n\n© 2024-{CURRENT_YEAR}")
             st.caption(f"Oturum: {_get_session_id()[:8]}...")

def display_chat_message_with_feedback(msg_data, msg_idx, chat_id):
    """Tek bir sohbet mesajını formatlar ve gösterir."""
    role = msg_data.get('role', 'model')
    content = msg_data.get('parts', '')
    sender = msg_data.get('sender_display', APP_NAME if role == 'model' else st.session_state.user_name)
    is_user = (role == 'user')
    avatar = "🧑"
    if is_user:
        if st.session_state.user_avatar_bytes:
            try: avatar = Image.open(BytesIO(st.session_state.user_avatar_bytes))
            except: pass
    else: # AI Avatar
        if "Gemini" in sender: avatar = "✨"
        elif "Web" in sender or "Wikipedia" in sender: avatar = "🌐"
        elif "Bilgi Tabanı" in sender or "Fonksiyonel" in sender: avatar = "📚"
        else: avatar = "🤖"

    with st.chat_message(role, avatar=avatar):
        if "```" in content: # Kod bloğu formatlama
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    lang_match = re.match(r"(\w+)\n", part)
                    lang = lang_match.group(1) if lang_match else None
                    code = part[len(lang)+1:] if lang and part.startswith(lang+"\n") else part
                    st.code(code, language=lang)
                    if st.button("📋", key=f"copy_{chat_id}_{msg_idx}_{i}", help="Kodu kopyala"):
                        st.write_to_clipboard(code); st.toast("Kod kopyalandı!", icon="✅")
                elif part.strip(): st.markdown(part, unsafe_allow_html=True)
        elif content.strip(): st.markdown(content, unsafe_allow_html=True)
        else: st.caption("[Boş Mesaj]")

        if not is_user and content.strip(): # AI mesajı eylemleri
             st.write("") # Boşluk
             bcols = st.columns([0.85, 0.075, 0.075])
             with bcols[1]: # TTS
                 if st.session_state.tts_enabled and globals().get('tts_engine'):
                     if st.button("🔊", key=f"tts_{chat_id}_{msg_idx}", help="Oku", use_container_width=True): speak(content)
             with bcols[2]: # Feedback
                 if st.button("✍️", key=f"fb_{chat_id}_{msg_idx}", help="Geri Bildirim", use_container_width=True):
                     st.session_state.current_message_id_for_feedback = f"{chat_id}_{msg_idx}"
                     prev_prompt = "[İstem bulunamadı]"
                     if msg_idx > 0 and st.session_state.all_chats[chat_id][msg_idx-1]['role'] == 'user':
                          prev_prompt = st.session_state.all_chats[chat_id][msg_idx-1]['parts']
                     st.session_state.last_user_prompt_for_feedback = prev_prompt
                     st.session_state.last_ai_response_for_feedback = content
                     st.session_state.show_feedback_comment_form = True
                     st.session_state.feedback_comment_input = ""
                     st.rerun()

def display_feedback_form_if_active():
    """Aktifse geri bildirim formunu gösterir."""
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        fkey = f"fb_form_{st.session_state.current_message_id_for_feedback}"
        with st.form(key=fkey):
            st.markdown("#### Yanıt Geri Bildirimi")
            st.caption(f"**İstem:** `{st.session_state.last_user_prompt_for_feedback[:80]}...`")
            st.caption(f"**Yanıt:** `{st.session_state.last_ai_response_for_feedback[:80]}...`")
            fb_type = st.radio("Değerlendirme:", ["👍 Beğendim", "👎 Beğenmedim"], horizontal=True, key=f"type_{fkey}", index=0 if st.session_state.last_feedback_type=='positive' else 1)
            comment = st.text_area("Yorum (isteğe bağlı):", value=st.session_state.feedback_comment_input, key=f"cmt_{fkey}", height=100, placeholder="Neden?")
            st.session_state.feedback_comment_input = comment
            scol, ccol = st.columns(2)
            submitted = scol.form_submit_button("✅ Gönder", use_container_width=True, type="primary")
            cancelled = ccol.form_submit_button("❌ Vazgeç", use_container_width=True)

            if submitted:
                parsed_type = "positive" if fb_type == "👍 Beğendim" else "negative"
                st.session_state.last_feedback_type = parsed_type
                log_feedback(st.session_state.current_message_id_for_feedback, st.session_state.last_user_prompt_for_feedback, st.session_state.last_ai_response_for_feedback, parsed_type, comment)
                st.session_state.show_feedback_comment_form = False; st.session_state.current_message_id_for_feedback = None; st.session_state.feedback_comment_input = ""; st.rerun()
            elif cancelled:
                st.session_state.show_feedback_comment_form = False; st.session_state.current_message_id_for_feedback = None; st.session_state.feedback_comment_input = ""; st.rerun()
        st.markdown("---")

def display_chat_interface_main(main_col_ref):
    """Ana sohbet arayüzünü sağ kolonda yönetir."""
    with main_col_ref:
        active_chat_id = st.session_state.get('active_chat_id')
        if active_chat_id is None:
            st.info("💬 Başlamak için **'➕ Yeni Sohbet'** butonuna tıklayın veya listeden bir sohbet seçin.", icon="👈"); return

        current_history = st.session_state.all_chats.get(active_chat_id, [])
        chat_container = st.container(height=550, border=False)
        with chat_container:
            if not current_history: st.info(f"Merhaba {st.session_state.user_name}! Yeni sohbetinize hoş geldiniz.", icon="👋")
            for i, msg in enumerate(current_history): display_chat_message_with_feedback(msg, i, active_chat_id)

        display_feedback_form_if_active() # Formu konteyner dışına taşıdık

        user_prompt = st.chat_input(f"{st.session_state.user_name}, ne sormak istersin?", key=f"input_{active_chat_id}")
        if user_prompt:
            user_msg = {'role': 'user', 'parts': user_prompt}
            st.session_state.all_chats[active_chat_id].append(user_msg)
            save_all_chats(st.session_state.all_chats)

            msg_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter += 1
            history_limit = 20 # Son N mesajı gönder
            history_for_model = st.session_state.all_chats[active_chat_id][-history_limit:-1] # Yeni eklenen hariç

            with st.chat_message("assistant", avatar="⏳"): placeholder = st.empty(); placeholder.markdown("🧠 _Düşünüyorum..._")

            ai_response, sender_name = get_hanogt_response_orchestrator(user_prompt, history_for_model, msg_id, active_chat_id, use_stream=st.session_state.gemini_stream_enabled)

            final_ai_msg = ""
            if st.session_state.gemini_stream_enabled and "Stream" in sender_name:
                 stream_container = placeholder; streamed_text = ""
                 try:
                     for chunk in ai_response:
                         if chunk.parts: text = "".join(p.text for p in chunk.parts if hasattr(p,'text')); streamed_text+=text; stream_container.markdown(streamed_text+"▌"); time.sleep(0.01)
                     stream_container.markdown(streamed_text); final_ai_msg = streamed_text
                     log_interaction(user_prompt, final_ai_msg, "Gemini Stream", msg_id, active_chat_id) # Stream bitince logla
                 except Exception as e: error = f"Stream hatası: {e}"; stream_container.error(error); final_ai_msg=error; sender_name=f"{APP_NAME} (Stream Hatası)"; log_interaction(user_prompt, final_ai_msg, "Stream Hatası", msg_id, active_chat_id)
            else: placeholder.empty(); final_ai_msg = str(ai_response) # Loglama orkestratörde yapıldı

            ai_msg_data = {'role': 'model', 'parts': final_ai_msg, 'sender_display': sender_name}
            st.session_state.all_chats[active_chat_id].append(ai_msg_data)
            save_all_chats(st.session_state.all_chats)
            if st.session_state.tts_enabled and globals().get('tts_engine') and isinstance(final_ai_msg, str) and "Stream" not in sender_name: speak(final_ai_msg)
            st.rerun()


# --- UYGULAMA ANA AKIŞI ---
# Başlık ve Alt Başlık
st.markdown(f"<h1 style='text-align: center; color: #0078D4;'>{APP_NAME} {APP_VERSION}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-style: italic; color: #555;'>Yapay zeka destekli kişisel asistanınız</p>", unsafe_allow_html=True)

# Hata Mesajları
if gemini_init_error: st.error(gemini_init_error, icon="🛑")
if supabase_error: st.warning(supabase_error, icon="🧱")
if tts_init_error and st.session_state.tts_enabled: st.toast(tts_init_error, icon="🔇")

# --- Giriş Ekranı ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce...")
    lcols = st.columns([0.2, 0.6, 0.2])
    with lcols[1]:
        with st.form("login"):
            name = st.text_input("Size nasıl hitap edelim?", placeholder="İsminiz...", key="login_name")
            if st.form_submit_button("✨ Başla", use_container_width=True, type="primary"):
                if name and name.strip():
                    st.session_state.user_name = name.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False
                    load_knowledge_from_file.clear()
                    if not st.session_state.active_chat_id and st.session_state.all_chats:
                         st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[-1]
                    st.rerun()
                else: st.error("Lütfen geçerli bir isim girin.")
else:
    # --- Ana Uygulama ---
    if not st.session_state.greeting_message_shown:
        st.success(f"Hoş geldiniz {st.session_state.user_name}!", icon="🎉")
        st.session_state.greeting_message_shown = True

    left_col, main_col = st.columns([1, 3]) # Ana Layout

    display_chat_list_and_about(left_col) # Sol Kolon

    with main_col: # Sağ Kolon
        display_settings_section() # Ayarlar

        # Mod Seçimi
        st.markdown("#### Uygulama Modu")
        modes = { "Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya)": "🎤", "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️" }
        keys = list(modes.keys()); idx = keys.index(st.session_state.app_mode) if st.session_state.app_mode in keys else 0
        selected = st.radio("Mod:", options=keys, index=idx, format_func=lambda k: f"{modes[k]} {k}", horizontal=True, label_visibility="collapsed", key="mode_radio")
        if selected != st.session_state.app_mode: st.session_state.app_mode = selected; st.rerun()
        st.markdown("<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

        # Mod İçeriği
        mode = st.session_state.app_mode
        if mode == "Yazılı Sohbet": display_chat_interface_main(main_col)
        elif mode == "Sesli Sohbet (Dosya)":
            # --- Sesli Sohbet Modu ---
            st.info("Yanıtlamamı istediğiniz ses dosyasını yükleyin.", icon="📢")
            a_file = st.file_uploader("Ses Dosyası:", type=['wav','mp3','ogg','flac','m4a'], label_visibility="collapsed", key="audio_up")
            if a_file:
                st.audio(a_file, format=a_file.type)
                active_id = st.session_state.get('active_chat_id')
                if not active_id: st.warning("Önce bir sohbet seçin/başlatın.", icon="⚠️")
                else:
                    audio_txt = None
                    with st.spinner(f"🔊 '{a_file.name}' işleniyor..."):
                        rec = sr.Recognizer()
                        try: # BytesIO ile dene
                            with sr.AudioFile(BytesIO(a_file.getvalue())) as src: audio_d = rec.record(src)
                            audio_txt = rec.recognize_google(audio_d, language="tr-TR")
                            st.success(f"**🎙️ Algılanan:**\n> {audio_txt}")
                        except Exception as e: st.error(f"Ses işleme hatası: {e}"); print(f"ERROR: Audio proc failed: {e}")
                    if audio_txt: # Yanıt al ve ekle
                        u_msg={'role':'user','parts':f"(Ses: {a_file.name}) {audio_txt}"}; st.session_state.all_chats[active_id].append(u_msg)
                        msg_id=f"audio_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter+=1
                        hist=st.session_state.all_chats[active_id][-20:-1] # Limit history
                        with st.spinner("🤖 Yanıt..."): ai_resp, sender = get_hanogt_response_orchestrator(audio_txt, hist, msg_id, active_id, False)
                        st.markdown(f"#### {sender} Yanıtı:"); st.markdown(str(ai_resp))
                        if st.session_state.tts_enabled and globals().get('tts_engine'):
                            if st.button("🔊 Oku", key="spk_aud_resp"): speak(str(ai_resp))
                        ai_msg={'role':'model','parts':str(ai_resp),'sender_display':sender}; st.session_state.all_chats[active_id].append(ai_msg)
                        save_all_chats(st.session_state.all_chats); st.success("✅ Yanıt sohbete eklendi!")
        elif mode == "Yaratıcı Stüdyo":
             # --- Yaratıcı Stüdyo Modu ---
            st.markdown("💡 Fikir verin, AI yaratıcı metin üretsin!")
            c_prompt = st.text_area("Yaratıcılık Tohumu:", key="cr_prompt", placeholder="Örn: 'Uzaydaki kütüphane'", height=100)
            cc1, cc2 = st.columns(2)
            len_p = cc1.selectbox("Uzunluk:", ["kısa", "orta", "uzun"], index=1, key="cr_len")
            sty_p = cc2.selectbox("Stil:", ["genel", "şiirsel", "hikaye"], index=0, key="cr_sty")
            if st.button("✨ Üret!", key="cr_gen_btn", type="primary", use_container_width=True):
                if c_prompt and c_prompt.strip():
                    active_id=st.session_state.get('active_chat_id','creative_no_chat'); msg_id=f"cr_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter+=1
                    final_resp, sender = None, f"{APP_NAME} (Yaratıcı)"
                    if globals().get('gemini_model'): # Gemini dene
                        with st.spinner("✨ Gemini ilham arıyor..."):
                            sys_p=f"Yaratıcı asistansın. İstem: '{c_prompt}'. Stil: '{sty_p}', Uzunluk: '{len_p}'. Özgün metin oluştur."; gem_resp=get_gemini_response_cached(sys_p,[],False)
                            if isinstance(gem_resp,str) and not gem_resp.startswith(GEMINI_ERROR_PREFIX): final_resp,sender=gem_resp,f"{APP_NAME} (Gemini Yaratıcı)"
                            else: st.toast("Gemini yaratıcı yanıtı alınamadı.",icon="ℹ️")
                    if not final_resp: # Yerel üretici
                        with st.spinner("✨ Hayal gücü..."): final_resp=creative_response_generator(c_prompt,len_p,sty_p); new_w=advanced_word_generator(c_prompt.split()[0] if c_prompt else "kelime"); final_resp+=f"\n\n---\n🔮 **Kelimatör:** {new_w}"; sender=f"{APP_NAME} (Yerel Yaratıcı)"
                    st.markdown(f"#### {sender} İlhamı:"); st.markdown(final_resp)
                    if st.session_state.tts_enabled and globals().get('tts_engine'):
                         if st.button("🔊 Dinle", key="spk_cr_resp"): speak(final_resp.split("🔮 **Kelimatör:**")[0].strip())
                    log_interaction(c_prompt, final_resp, sender, msg_id, active_id); st.success("✨ Yanıt oluşturuldu!")
                    # if active_id != 'creative_no_chat': ... # Sohbete ekle?
                else: st.warning("Lütfen bir metin girin.", icon="✍️")
        elif mode == "Görsel Oluşturucu":
            # --- Görsel Oluşturucu Modu ---
            st.markdown("🎨 Hayalinizi tarif edin, AI (basitçe) çizsin!"); st.info("ℹ️ Not: Sembolik çizimler yapar.", icon="💡")
            img_prompt = st.text_input("Görsel Tarifi:", key="img_prompt", placeholder="Örn: 'Deniz kenarında gün batımı'")
            if st.button("🖼️ Oluştur!", key="gen_img_btn", type="primary", use_container_width=True):
                if img_prompt and img_prompt.strip():
                    with st.spinner("🖌️ Çiziliyor..."): img = generate_prompt_influenced_image(img_prompt)
                    st.image(img, caption=f"'{img_prompt[:60]}' yorumu", use_container_width=True)
                    try: # İndirme
                        buf=BytesIO(); img.save(buf,format="PNG"); img_bytes=buf.getvalue(); fname_p=re.sub(r'[^\w\s-]','',img_prompt.lower())[:30].replace(' ','_'); fname=f"hanogt_{fname_p or 'gorsel'}_{int(time.time())}.png"
                        st.download_button("🖼️ İndir (PNG)", data=img_bytes, file_name=fname, mime="image/png", use_container_width=True)
                        active_id=st.session_state.get('active_chat_id') # Sohbete ekle
                        if active_id and active_id in st.session_state.all_chats:
                            u_msg={'role':'user','parts':f"(Görsel: {img_prompt})"}; ai_msg={'role':'model','parts':"(Görsel oluşturuldu - İndirme mevcut.)",'sender_display':f"{APP_NAME} (Görsel)"}
                            st.session_state.all_chats[active_id].extend([u_msg,ai_msg]); save_all_chats(st.session_state.all_chats); st.info("İstem sohbete eklendi.",icon="💾")
                    except Exception as e: st.error(f"İndirme hatası: {e}")
                else: st.warning("Lütfen bir tarif girin.", icon="✍️")

        # --- Alt Bilgi (Footer) ---
        st.markdown("<hr style='margin-top: 1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        fcols = st.columns(3)
        with fcols[0]: st.caption(f"Kullanıcı: {st.session_state.get('user_name', 'N/A')}")
        with fcols[1]: st.caption(f"{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}")
        with fcols[2]:
             ai_stat="Aktif" if globals().get('gemini_model') else "Devre Dışı"; log_stat="Aktif" if globals().get('supabase') else "Devre Dışı"
             st.caption(f"AI: {ai_stat} | Log: {log_stat}", help=f"Model: {st.session_state.gemini_model_name}")

