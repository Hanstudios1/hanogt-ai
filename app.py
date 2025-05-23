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
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.1.4 Pro+ Enhanced (Button Debug)" # Sürüm güncellendi
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
            print(f"INFO: {error_message}") # Konsola bilgi mesajı yazdırılır
            return default_knowledge, error_message
    except json.JSONDecodeError as e:
        error_message = f"Bilgi tabanı ({filename}) hatalı (JSONDecodeError: {e}). Varsayılan kullanılıyor."
        print(f"ERROR: {error_message}")
        return default_knowledge, error_message
    except Exception as e:
        error_message = f"Bilgi tabanı yüklenirken genel hata: {e}. Varsayılan kullanılıyor."
        print(f"ERROR: {error_message}")
        return default_knowledge, error_message

KNOWLEDGE_BASE = {} # Başlangıçta boş, başlatma bloğunda dolacak

def kb_chatbot_response(query, knowledge_base_dict):
    query_lower = query.lower().strip()
    if query_lower in DYNAMIC_FUNCTIONS_MAP:
        try:
            return DYNAMIC_FUNCTIONS_MAP[query_lower]()
        except Exception as e:
            st.error(f"Fonksiyon hatası ({query_lower}): {e}")
            return DEFAULT_ERROR_MESSAGE
    
    # knowledge_base_dict boş veya None ise hata vermemesi için kontrol
    if not knowledge_base_dict:
        return None

    if query_lower in knowledge_base_dict:
        resp = knowledge_base_dict[query_lower]
        return random.choice(resp) if isinstance(resp, list) else resp
    
    partial_matches = [
        resp 
        for key, resp_list in knowledge_base_dict.items() 
        if key in query_lower 
        for resp in (resp_list if isinstance(resp_list, list) else [resp_list])
    ]
    if partial_matches:
        return random.choice(list(set(partial_matches)))
    
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_score, best_responses = 0, []
    for key, resp_list in knowledge_base_dict.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words:
            continue
        score = len(query_words.intersection(key_words)) / len(key_words) if key_words else 0
        if score > 0.6: # Eşleşme eşiği
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
    global gemini_init_error_global # Global değişkeni değiştirmek için
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
        gemini_init_error_global = None # Başarılıysa hatayı temizle
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
    if not create_client: # Kütüphane yüklenememişse
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
        return client, None # Başarılıysa None (hata yok) dön
    except Exception as e:
        error_msg = f"Supabase bağlantısı sırasında kritik hata: {e}. Loglama ve geri bildirim devre dışı."
        print(f"CRITICAL_ERROR: Supabase connection failed: {e}")
        import traceback
        print(traceback.format_exc())
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
    try:
        engine = pyttsx3.init()
        # Motoru test etmek için kısa bir şey söyletebiliriz (opsiyonel, logda görülebilir)
        # engine.say("TTS hazır") 
        # engine.runAndWait()
        print("INFO: TTS motoru başarıyla başlatıldı.")
        return engine, None
    except Exception as e:
        error_message = f"TTS motoru başlatılamadı: {e}. Sesli yanıtlar devre dışı."
        print(f"ERROR: TTS Init Failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None, error_message

def speak(text):
    engine = globals().get('tts_engine')
    if not engine:
        st.toast("TTS motoru aktif değil, sesli yanıt verilemiyor.", icon="🔇")
        return
    if not st.session_state.get('tts_enabled', True):
        return # Ayarlardan kapalıysa sessizce geç
    try:
        # Metni temizle (istenmeyen karakterler TTS'i bozabilir)
        cleaned_text = re.sub(r'[^\w\s.,!?-]', '', text)
        engine.say(cleaned_text)
        engine.runAndWait()
    except RuntimeError as e: # pyttsx3'e özgü çalışma zamanı hatası
        st.warning(f"TTS çalışma zamanı sorunu (tekrar deneyin): {e}.", icon="🔊")
        print(f"WARNING: TTS runtime error: {e}")
    except Exception as e:
        st.error(f"TTS ile seslendirme sırasında hata: {e}", icon="🔥")
        print(f"ERROR: TTS Speak Failed: {e}")

# --- Metin Temizleme ---
def _clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n\s*\n', '\n\n', text) # Çoklu boş satırları tek boş satıra indir
    return text

# --- Web Kazıma (Cache'li)---
@st.cache_data(ttl=600) # 10 dakika cache
def scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS):
    print(f"INFO: Web içeriği kazınıyor: {url}")
    messages_to_show_outside = [] # UI mesajları için liste
    try:
        parsed_url = urlparse(url)
        headers = {
            'User-Agent': USER_AGENT,
            'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8', # Türkçe içeriğe öncelik ver
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'DNT': '1', # Do Not Track
            'Upgrade-Insecure-Requests': '1'
        }
        # URL geçerlilik kontrolü
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']:
            messages_to_show_outside.append({'type': 'warning', 'text': f"Geçersiz veya desteklenmeyen URL şeması: {url}", 'icon': "🔗"})
            return None, messages_to_show_outside
        
        # Stream ile büyük dosyaları parça parça al
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status() # HTTP hatalarını yakala (4xx, 5xx)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            messages_to_show_outside.append({'type': 'info', 'text': f"URL HTML içeriği değil ('{content_type}'). Kazıma atlanıyor: {url}", 'icon': "📄"})
            response.close()
            return None, messages_to_show_outside

        html_content = ""
        current_size_bytes = 0
        # Maksimum byte boyutu (çok büyük HTML'leri erken kesmek için, max_chars'tan bağımsız)
        # Karakter başına ortalama 2-4 byte UTF-8 varsayımıyla.
        absolute_max_html_bytes = max_chars * 6 
        
        try:
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True, errors='ignore'): # decode_unicode ile string chunk'lar
                if chunk:
                    html_content += chunk
                    current_size_bytes += len(chunk.encode('utf-8', 'ignore')) # Byte boyutunu kontrol et
                if current_size_bytes > absolute_max_html_bytes:
                    messages_to_show_outside.append({'type': 'warning', 'text': f"HTML içeriği çok büyük ({current_size_bytes // 1024}KB), tamamı işlenmeden kesiliyor: {url}", 'icon': "✂️"})
                    break
        finally:
            response.close() # Bağlantıyı kapat

        if not html_content.strip():
            messages_to_show_outside.append({'type': 'warning', 'text': f"URL'den boş HTML içeriği alındı: {url}", 'icon': "📄"})
            return None, messages_to_show_outside

        soup = BeautifulSoup(html_content, 'lxml') # lxml parser daha hızlı olabilir
        
        # Kaldırılacak gereksiz etiketler (daha kapsamlı liste)
        tags_to_remove = ["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio", "figure", "input", "select", "textarea", "path", "canvas", "figcaption", "details", "summary", "template", "dialog"]
        for tag in soup.find_all(tags_to_remove):
            tag.decompose()
        
        content_parts = []
        # Ana içerik alanlarını bulmak için daha fazla CSS seçici
        selectors = [
            'article[class*="content"]', 'article[class*="post"]', 'article[itemprop="articleBody"]',
            'main[id*="content"]', 'main[role="main"]', 'main',
            'div[class*="post-body"]', 'div[itemprop="articleBody"]', 'div[class*="article-content"]',
            'div[class*="main-content"]', 'div[class*="content-area"]', 'div[id="content"]',
            '.content', '#content', '.entry-content', '.article-body'
        ]
        main_container = next((found[0] for sel in selectors if (found := soup.select(sel, limit=1))), None)
        
        min_paragraph_len = 50 # Paragraf için minimum karakter uzunluğu
        min_meaningful_chars_in_paragraph = 3 # Nokta, virgül vb.

        target_element = main_container if main_container else soup.body # Ana konteyner yoksa body'yi kullan
        
        if target_element:
            # Paragrafları (p), bölümleri (div), liste elemanlarını (li) ve başlıkları (h1-h6) al
            for tag in target_element.find_all(['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'td'], limit=100): # limit eklendi
                text = _clean_text(tag.get_text(separator=' ', strip=True))
                # Çok kısa veya anlamsız metinleri atla
                if len(text) > min_paragraph_len and (text.count('.') + text.count(',') + text.count('!') + text.count('?')) >= min_meaningful_chars_in_paragraph:
                    content_parts.append(text)
                elif tag.name.startswith('h') and len(text) > 10: # Başlıklar daha kısa olabilir
                    content_parts.append(text)


        if not content_parts or len(" ".join(content_parts)) < 150: # Yeterli içerik yoksa
             # Eğer body'den de bir şey çıkmazsa, tüm görünür metni almayı dene (son çare)
            all_visible_text = _clean_text(soup.get_text(separator='\n', strip=True))
            if all_visible_text and len(all_visible_text) > 200: # Eğer tüm metin anlamlıysa
                messages_to_show_outside.append({'type': 'info', 'text': f"Sayfadan genel görünür metin kullanıldı (düşük özgüllük): {url}", 'icon': "ℹ️"})
                # Bu metni paragraflara bölmeye çalış
                content_parts = [p.strip() for p in all_visible_text.split('\n') if len(p.strip()) > min_paragraph_len][:70]
            else:
                messages_to_show_outside.append({'type': 'info', 'text': f"Sayfadan anlamlı metin çıkarılamadı: {url}", 'icon': "📄"})
                return None, messages_to_show_outside

        # Tekrarlanan içerik parçalarını kaldır (sırayı koruyarak)
        unique_content_parts = list(dict.fromkeys(content_parts))
        final_cleaned_content = _clean_text("\n\n".join(unique_content_parts))

        if not final_cleaned_content:
            messages_to_show_outside.append({'type': 'info', 'text': f"Kazıma sonucu anlamlı içerik bulunamadı: {url}", 'icon': "📄"})
            return None, messages_to_show_outside
        
        # Son olarak karakter limitine göre kırp
        truncated_content = final_cleaned_content[:max_chars] + ("..." if len(final_cleaned_content) > max_chars else "")
        
        messages_to_show_outside.append({'type': 'toast', 'text': f"'{urlparse(url).netloc}' sitesinden içerik başarıyla alındı.", 'icon': "✅"})
        return truncated_content, messages_to_show_outside

    except requests.exceptions.Timeout:
        messages_to_show_outside.append({'type': 'toast', 'text': f"⏳ İstek zaman aşımına uğradı: {url}", 'icon': '🌐'})
        print(f"ERROR: Timeout scraping '{url}'")
        return None, messages_to_show_outside
    except requests.exceptions.RequestException as e: # Diğer ağ hataları
        error_short = str(e).split('\n')[0][:100] # Hata mesajını kısalt
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Ağ hatası oluştu: {url} - {error_short}", 'icon': '🌐'})
        print(f"ERROR: Network error scraping '{url}': {e}")
        return None, messages_to_show_outside
    except Exception as e: # Diğer beklenmedik kazıma hataları
        error_short = str(e)[:100]
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Kazıma sırasında beklenmedik bir hata: {error_short}", 'icon': '🔥'})
        print(f"CRITICAL_ERROR: Scraping '{url}' failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None, messages_to_show_outside

# --- Web Arama (Cache'li) ---
@st.cache_data(ttl=600)
def search_web(query):
    print(f"INFO: Web'de '{query}' için arama yapılıyor...")
    messages_to_show_outside = [] # UI mesajları için
    wikipedia.set_lang("tr") # Wikipedia dilini Türkçe yap
    search_result_text = None # Nihai metin sonucunu tutacak

    # 1. Wikipedia Araması
    try:
        # Önce arama yapıp en iyi sonucu bul, sonra o sayfanın özetini al
        wp_search_titles = wikipedia.search(query, results=1) 
        if wp_search_titles:
            page_title = wp_search_titles[0]
            wp_page = wikipedia.page(page_title, auto_suggest=False, redirect=True) # Sayfayı al
            wp_summary = wikipedia.summary(page_title, sentences=4, auto_suggest=False, redirect=True) # Daha kısa özet
            
            search_result_text = f"**Wikipedia ({wp_page.title}):**\n\n{_clean_text(wp_summary)}\n\nKaynak: {wp_page.url}"
            messages_to_show_outside.append({'type': 'toast', 'text': f"✅ Wikipedia'dan '{wp_page.title}' bulundu.", 'icon': "📚"})
    except wikipedia.exceptions.PageError: # Sayfa bulunamazsa
        messages_to_show_outside.append({'type': 'info', 'text': f"ℹ️ Wikipedia'da '{query}' için direkt sayfa bulunamadı.", 'icon': "🤷"})
    except wikipedia.exceptions.DisambiguationError as e: # Çok anlamlılık durumu
        options = e.options[:3] # İlk 3 seçeneği göster
        search_result_text = (f"**Wikipedia Çok Anlamlı ({query}):**\n"
                              f"'{query}' için birden fazla anlam bulundu. Olası başlıklar: {', '.join(options)}...\n"
                              f"Daha spesifik bir arama yapmayı deneyin.")
        messages_to_show_outside.append({'type': 'toast', 'text': f"ℹ️ Wikipedia'da '{query}' için birden fazla sonuç var.", 'icon': "📚"})
    except Exception as e: # Diğer Wikipedia hataları
        error_short = str(e)[:100]
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ Wikipedia araması sırasında bir hata oluştu: {error_short}", 'icon': "🔥"})
        print(f"ERROR: Wikipedia search error for '{query}': {e}")

    # 2. DuckDuckGo Araması (Snippet ve kazınacak URL için)
    ddg_url_to_scrape = None
    ddg_snippet_used = False
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs:
            ddg_results = list(ddgs.text(query, region='tr-tr', safesearch='Strict', max_results=3)) # Daha güvenli arama
            if ddg_results:
                best_ddg_result = ddg_results[0] # Genellikle ilk sonuç en iyisidir
                snippet_text = best_ddg_result.get('body')
                href_url = best_ddg_result.get('href')

                if href_url: # Geçerli bir URL varsa
                    ddg_url_to_scrape = unquote(href_url) # URL'yi kazımak için sakla
                    domain_name = urlparse(ddg_url_to_scrape).netloc
                    
                    # Eğer Wikipedia sonucu yoksa veya çok kısaysa, DDG snippet'ini kullan
                    if snippet_text and (not search_result_text or len(search_result_text) < 150):
                        search_result_text = (f"**Web Özeti (DuckDuckGo - {domain_name}):**\n\n"
                                              f"{_clean_text(snippet_text)}\n\n"
                                              f"Kaynak: {ddg_url_to_scrape}")
                        ddg_snippet_used = True
                        messages_to_show_outside.append({'type': 'toast', 'text': f"ℹ️ DuckDuckGo'dan web özeti bulundu.", 'icon': "🦆"})

    except Exception as e:
        error_short = str(e)[:100]
        messages_to_show_outside.append({'type': 'toast', 'text': f"⚠️ DuckDuckGo araması sırasında bir hata: {error_short}", 'icon': "🔥"})
        print(f"ERROR: DDG search error for '{query}': {e}")

    # 3. Web Sayfası Kazıma (Eğer DDG'den URL alındıysa)
    if ddg_url_to_scrape:
        scraped_content, scrape_messages = scrape_url_content(ddg_url_to_scrape) # URL'yi kazı
        messages_to_show_outside.extend(scrape_messages) # Kazıma mesajlarını ekle
        
        if scraped_content:
            domain_name = urlparse(ddg_url_to_scrape).netloc
            scraped_text_prefix = f"**Web Sayfası İçeriği ({domain_name}):**\n\n"
            full_scraped_text = f"{scraped_text_prefix}{scraped_content}\n\nKaynak: {ddg_url_to_scrape}"
            
            # Eğer DDG snippet'i zaten kullanılmadıysa ve Wikipedia sonucu varsa, ona ekle
            # Ya da Wikipedia sonucu yoksa, kazınan içeriği ana sonuç yap
            if not ddg_snippet_used and search_result_text and "Wikipedia" in search_result_text and len(search_result_text) > 200:
                 search_result_text += f"\n\n---\n\n{full_scraped_text}" # Wikipedia'ya ekle
            elif not search_result_text or ddg_snippet_used: # Wikipedia sonucu yoksa veya DDG snippet zaten ana sonuçsa
                search_result_text = full_scraped_text # Kazınan içeriği ana sonuç yap
            # scrape_url_content zaten kendi başarılı kazıma mesajını ekliyor (eklendi)

    # Sonuç yoksa bildirim
    if not search_result_text:
        messages_to_show_outside.append({'type': 'toast', 'text': f"'{query}' için web'de anlamlı bir sonuç bulunamadı. Farklı anahtar kelimelerle deneyin.", 'icon': "❌"})
        return None, messages_to_show_outside # Sonuç yoksa None dön
        
    return search_result_text, messages_to_show_outside


# --- Sohbet Geçmişi Yönetimi ---
@st.cache_data(ttl=86400)
def load_all_chats_cached(file_path=CHAT_HISTORY_FILE):
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
                    err_msg = f"Sohbet geçmişi dosyası ({file_path}) beklenmedik formatta. Dosya yeniden adlandırılıyor."
                    print(f"WARNING: {err_msg}")
                    error_messages_for_outside.append({'type': 'warning', 'text': err_msg, 'icon': "⚠️"})
                    timestamp = int(time.time())
                    err_file_name = f"{os.path.splitext(file_path)[0]}.err_format_{timestamp}{os.path.splitext(file_path)[1]}"
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
                return {}, None 
        except json.JSONDecodeError as json_e:
            err_msg = f"Sohbet geçmişi dosyası ({file_path}) JSON olarak çözümlenemedi: {json_e}. Dosya yeniden adlandırılıyor."
            print(f"ERROR: {err_msg}")
            error_messages_for_outside.append({'type': 'error', 'text': err_msg, 'icon': "🔥"})
            timestamp = int(time.time())
            err_file_name = f"{os.path.splitext(file_path)[0]}.err_json_{timestamp}{os.path.splitext(file_path)[1]}"
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
            err_msg = f"Sohbet geçmişi ({file_path}) yüklenirken genel bir hata oluştu: {e}. Dosya yeniden adlandırılıyor."
            print(f"ERROR: {err_msg}")
            error_messages_for_outside.append({'type': 'error', 'text': err_msg, 'icon': "🔥"})
            timestamp = int(time.time())
            err_file_name = f"{os.path.splitext(file_path)[0]}.err_generic_{timestamp}{os.path.splitext(file_path)[1]}"
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
    return {}, None

def save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chats_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Sohbet geçmişi dosyaya kaydedilemedi: {e}", icon="🔥")
        print(f"ERROR: Save chats failed: {e}")

# --- Gemini Yanıt Alma ---
def get_gemini_response(prompt_text, history_list, stream_output=False):
    model_instance = globals().get('gemini_model')
    if not model_instance:
        return f"{GEMINI_ERROR_PREFIX} Gemini modeli aktif değil veya yüklenemedi."

    valid_history_for_api = []
    for msg in history_list:
        role = msg.get('role')
        parts_content = msg.get('parts')
        if role in ['user', 'model'] and isinstance(parts_content, str) and parts_content.strip():
            valid_history_for_api.append({'role': role, 'parts': [parts_content]})
        elif role in ['user', 'model'] and isinstance(parts_content, list):
             valid_history_for_api.append({'role': role, 'parts': parts_content})
    try:
        chat_session = model_instance.start_chat(history=valid_history_for_api)
        response = chat_session.send_message(prompt_text, stream=stream_output)

        if stream_output:
            return response
        else:
            if response.parts:
                full_response_text = "".join(p.text for p in response.parts if hasattr(p, 'text'))
                return full_response_text
            else:
                block_reason_obj = getattr(response.prompt_feedback, 'block_reason', None)
                block_reason = block_reason_obj.name if hasattr(block_reason_obj, 'name') else str(block_reason_obj)
                
                finish_reason_str = "Bilinmiyor"
                if response.candidates: # Bazen candidate boş olabilir
                    finish_reason_obj = getattr(response.candidates[0], 'finish_reason', None)
                    finish_reason_str = finish_reason_obj.name if hasattr(finish_reason_obj, 'name') else str(finish_reason_obj)
                
                error_message_detail = f"Yanıt engellendi (Neden: {block_reason})." if block_reason not in [None, "None", "OTHER", "SAFETY_REASON_UNSPECIFIED"] else f"Yanıt tamamlanamadı (Neden: {finish_reason_str})."
                
                st.warning(f"Gemini'den anlamlı bir yanıt alınamadı: {error_message_detail}", icon="🛡️" if block_reason not in [None, "None", "OTHER"] else "⚠️")
                return f"{GEMINI_ERROR_PREFIX} {error_message_detail}"
                
    except Exception as e:
        st.error(f"Gemini API ile iletişimde kritik bir hata oluştu: {e}", icon="🔥")
        print(f"CRITICAL_ERROR: Gemini API communication failed for prompt '{prompt_text[:50]}...': {e}")
        import traceback
        print(traceback.format_exc())
        return f"{GEMINI_ERROR_PREFIX} API hatası: {e}"

# --- Supabase Loglama ---
def log_to_supabase(table_name, data_dict):
    client = globals().get('supabase')
    # SupabaseAPIError globalde tanımlı (Exception'a fallback yapabilir)
    # specific_supabase_error = globals().get('SupabaseAPIError', Exception)

    if not client:
        print(f"INFO: Supabase client not available. Skipping log to table: {table_name}")
        return False
    try:
        default_data = {
            'user_name': st.session_state.get('user_name', 'Bilinmiyor'),
            'session_id': _get_session_id(),
            'app_version': APP_VERSION,
            'chat_id': st.session_state.get('active_chat_id', 'N/A')
        }
        final_data_to_log = {**default_data, **data_dict}
        response = client.table(table_name).insert(final_data_to_log).execute()
        
        # Modern Supabase client (v1 üzeri) hata durumunda exception fırlatır.
        # Bu nedenle, 'response.error' kontrolü genellikle gereksizdir.
        # Başarılı olup olmadığını 'response.data' varlığıyla kontrol edebiliriz.
        if response.data: # Veri döndüyse başarılıdır
            print(f"INFO: Successfully logged to Supabase table: {table_name}, data: {response.data}")
            return True
        else: # Veri yoksa veya beklenmedik bir durumsa
            print(f"WARNING: Supabase log to table {table_name} did not return data, response: {response}")
            # st.toast(f"⚠️ Supabase loglama yanıtı beklenmedik ({table_name}).", icon="💾") # Çok fazla toast olabilir
            return False

    except SupabaseAPIError as api_err: # Tanımladığımız SupabaseAPIError'u yakala
        st.toast(f"⚠️ Supabase API hatası ({table_name}): {str(api_err)[:150]}", icon="💾")
        print(f"ERROR: Supabase API error on table {table_name}: {api_err}")
        return False
    except Exception as e: # Diğer tüm hatalar
        st.toast(f"⚠️ Supabase loglama sırasında genel bir hata oluştu ({table_name}): {str(e)[:150]}", icon="💾")
        print(f"ERROR: Supabase log ({table_name}) general exception: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def log_interaction(prompt, ai_response, source, message_id, chat_id_val):
    MAX_LOG_LENGTH = 10000 # Supabase'deki alan boyutuna göre ayarla
    return log_to_supabase(SUPABASE_TABLE_LOGS, {
        "user_prompt": str(prompt)[:MAX_LOG_LENGTH],
        "ai_response": str(ai_response)[:MAX_LOG_LENGTH],
        "response_source": source,
        "message_id": message_id,
        # "chat_id": chat_id_val # Bu zaten default_data'da var
    })

def log_feedback(message_id, user_prompt, ai_response, feedback_type, comment=""):
    MAX_LOG_LENGTH = 10000
    data = {
        "message_id": message_id,
        "user_prompt": str(user_prompt)[:MAX_LOG_LENGTH],
        "ai_response": str(ai_response)[:MAX_LOG_LENGTH],
        "feedback_type": feedback_type,
        "comment": str(comment)[:MAX_LOG_LENGTH]
    }
    success = log_to_supabase(SUPABASE_TABLE_FEEDBACK, data)
    st.toast("Geri bildiriminiz için teşekkür ederiz!" if success else "Geri bildirim gönderilirken bir sorunla karşılaşıldı.", icon="💌" if success else "😔")
    return success

# --- Yanıt Orkestrasyonu ---
def get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id_val, use_stream=False):
    response_text, source_display_name = None, "Bilinmiyor"
    
    kb_response = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
    if kb_response:
        source_type = "Dinamik Fonksiyon" if prompt.lower() in DYNAMIC_FUNCTIONS_MAP else "Bilgi Tabanı"
        log_interaction(prompt, kb_response, source_type, msg_id, chat_id_val)
        return kb_response, f"{APP_NAME} ({source_type})"

    if globals().get('gemini_model'):
        gemini_response = get_gemini_response(prompt, history, stream=use_stream)
        if gemini_response:
            if use_stream and hasattr(gemini_response, '__iter__') and not isinstance(gemini_response, str):
                return gemini_response, f"{APP_NAME} (Gemini Stream)" 
            elif isinstance(gemini_response, str) and not gemini_response.startswith(GEMINI_ERROR_PREFIX):
                log_interaction(prompt, gemini_response, "Gemini", msg_id, chat_id_val)
                return gemini_response, f"{APP_NAME} (Gemini)"
            elif isinstance(gemini_response, str) and gemini_response.startswith(GEMINI_ERROR_PREFIX):
                print(f"INFO: Gemini returned an error message to orchestrator: {gemini_response}")
                response_text = gemini_response # Hata mesajını sakla, belki sonra kullanılır
    
    is_question_like = "?" in prompt or \
                       any(keyword in prompt.lower() for keyword in ["nedir", "kimdir", "nasıl", "bilgi", "araştır", "haber", "anlamı", "tanımı", "açıkla", "bul", "söyle"])
    
    if response_text is None and is_question_like and len(prompt.split()) >= 2 :
        web_search_result_text, web_messages = search_web(prompt)
        for msg_info in web_messages:
            if msg_info['type'] == 'toast': st.toast(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'warning': st.warning(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'info': st.info(msg_info['text'], icon=msg_info.get('icon'))

        if web_search_result_text:
            log_interaction(prompt, web_search_result_text, "Web Arama", msg_id, chat_id_val)
            return web_search_result_text, f"{APP_NAME} (Web Arama)"

    user_name_for_default = st.session_state.get('user_name', 'dostum')
    default_responses = [
        f"Üzgünüm {user_name_for_default}, bu konuda şu anda sana yardımcı olamıyorum.",
        "Bu soruyu tam olarak anlayamadım, farklı bir şekilde ifade edebilir misin?",
        "Bu konuda henüz bir bilgim yok ama öğrenmeye çalışıyorum!",
        "Hmm, bu ilginç bir soru. Biraz daha düşünmem gerekebilir."
    ]
    
    # Eğer Gemini'den bir hata mesajı geldiyse ve başka yanıt bulunamadıysa onu kullanabiliriz.
    # Ancak genellikle daha kullanıcı dostu bir varsayılan yanıt daha iyi olabilir.
    # Şimdilik, response_text (Gemini hatası olabilir) hala None ise varsayılanı seç.
    if response_text and response_text.startswith(GEMINI_ERROR_PREFIX) and not web_search_result_text: # Eğer Gemini hatası varsa ve web araması da boşsa
        final_default_response = response_text # Gemini'den gelen hata mesajını göster
        source_display_name = f"{APP_NAME} (Hata)"
    else:
        final_default_response = random.choice(default_responses)
        source_display_name = f"{APP_NAME} (Varsayılan)"
    
    log_interaction(prompt, final_default_response, source_display_name.split('(')[-1].replace(')','').strip(), msg_id, chat_id_val)
    return final_default_response, source_display_name

# --- Yaratıcı Modüller (İçerik önceki gibi) ---
def creative_response_generator(prompt_text, length_mode="orta", style_mode="genel"):
    templates = {
        "genel": ["İşte bir fikir: {}", "Şöyle bir düşünce: {}", "Belki de: {}"],
        "şiirsel": ["Kalbimden dökülenler: {}", "Mısralarla: {}", "İlham perisi fısıldadı: {}"],
        "hikaye": ["Bir varmış bir yokmuş, {}...", "Hikayemiz başlar: {}...", "Ve sonra olanlar oldu: {}..."],
        "bilgilendirici": ["Konuyla ilgili bazı notlar: {}", "Özetle: {}", "Detaylı olarak bakarsak: {}"],
        "esprili": ["Bak şimdi ne geldi aklıma: {}", "Hazır mısın? İşte geliyor: {}", "Biraz da gülelim: {}"]
    }
    creative_idea = generate_new_idea_creative(prompt_text, style_mode)
    sentences = [s.strip() for s in creative_idea.split('.') if s.strip()]
    num_sentences = len(sentences)

    if length_mode == "kısa":
        final_idea = ". ".join(sentences[:max(1, num_sentences // 3)]) + "." if num_sentences > 0 else creative_idea
    elif length_mode == "uzun":
        additional_idea = generate_new_idea_creative(prompt_text[::-1] + " farklı", style_mode) 
        final_idea = creative_idea + f"\n\nAyrıca, bir de şöyle bir bakış açısı var:\n{additional_idea}"
    else: # orta
        final_idea = creative_idea
    
    selected_template = random.choice(templates.get(style_mode, templates["genel"]))
    return selected_template.format(final_idea)

def generate_new_idea_creative(seed_text, style="genel"):
    elements = ["zamanın dokusu", "kayıp orman", "kırık bir rüya", "kuantum dalgaları", "gölgelerin dansı", "yıldız tozu", "sessizliğin şarkısı", "unutulmuş kehanetler", "dijital okyanuslar", "kristal şehirler"]
    actions = ["gizemi çözer", "sınırları yeniden çizer", "unutulmuş şarkıları fısıldar", "kaderi yeniden yazar", "sessizliği boyar", "gerçeği aralar", "umudu yeşertir", "yeni dünyalar keşfeder", "eski sırları açığa çıkarır"]
    objects = ["evrenin kalbi", "saklı bir gerçek", "sonsuzluğun melodisi", "kayıp bir hatıra", "umudun ışığı", "kristal bir küre", "eski bir günlük", "sihirli bir anahtar", "konuşan bir hayvan"]
    
    words_from_seed = re.findall(r'\b\w{4,}\b', seed_text.lower())
    chosen_seed_word = random.choice(words_from_seed) if words_from_seed else random.choice(["gizem", "macera", "umut", "hayal"])
    
    e1, a1, o1 = random.choice(elements), random.choice(actions), random.choice(objects)
    e2, a2 = random.choice(elements), random.choice(actions)

    # Stil parametresine göre farklı yapılar eklenebilir (gelecekte)
    if style == "şiirsel":
        structures = [
            f"{chosen_seed_word.capitalize()} rüzgarında, {e1} fısıldar,\n{o1} {a1} usulca.",
            f"Bir zamanlar {e1} diyarında, {chosen_seed_word} adında bir {o1},\n{a2} durmadan.",
        ]
    elif style == "hikaye":
         structures = [
            f"{chosen_seed_word.capitalize()}, {e1} derinliklerinde kaybolmuştu. Amacı {o1}'nı bulmak ve onunla {a1} idi.",
            f"Her şey {chosen_seed_word} ile başladı. {e1} diyarında, {o1} adında gizemli bir varlık {a2} ve dünyayı değiştirdi.",
        ]
    else: # genel, bilgilendirici, esprili için benzer yapılar
        structures = [
            f"{chosen_seed_word.capitalize()} konusunu ele alırsak, {e1} genellikle {a1} ve bu durum {o1} ile sonuçlanır.",
            f"Eğer {chosen_seed_word} ve {e1} bir araya gelirse, {o1}'nın {a1} olasılığı yüksektir.",
            f"Bir {APP_NAME} olarak şunu söyleyebilirim: {chosen_seed_word}, {e2}'yi {a2} zaman, {o1} genellikle ilginç bir şekilde ortaya çıkar."
        ]
    return random.choice(structures)

def advanced_word_generator(base_word):
    base = base_word or "kelime"
    cleaned_base = "".join(filter(str.isalpha, base.lower()))
    vowels = "aeıioöuü"
    consonants = "bcçdfgğhjklmnprsştvyz"
    prefixes = ["bio", "krono", "neo", "mega", "poli", "meta", "xeno", "astro", "hidro", "ludo", "psiko", "tekno", "ante", "post", "eko", "geo"]
    suffixes = ["genez", "sfer", "loji", "tronik", "morf", "matik", "skop", "nomi", "tek", "vers", "dinamik", "kurgu", "izm", "ist", "grafi", "fobi"]
    core_part = ""
    if len(cleaned_base) > 2 and random.random() < 0.7:
        start_index = random.randint(0, max(0, len(cleaned_base) - 3))
        core_part = cleaned_base[start_index : start_index + random.randint(2,4)] # 2-4 harflik çekirdek
    else:
        core_part = "".join(random.choice(consonants if i % 2 else vowels) for i in range(random.randint(3,5))) # 3-5 harflik rastgele çekirdek
    
    new_word = core_part
    has_prefix = False
    if random.random() > 0.4: # Prefix ekleme olasılığı
        new_word = random.choice(prefixes) + new_word
        has_prefix = True
    
    # Suffix ekleme (eğer prefix yoksa veya rastgele)
    if random.random() > 0.4 or not has_prefix: 
        new_word += random.choice(suffixes)
        
    return new_word.capitalize() if len(new_word) > 2 else f"{base_word.capitalize()}atron"


# --- Görsel Oluşturucu (İçerik önceki gibi) ---
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
            outline_color = (0,0,0,60) if len(shape_color) == 4 and shape_color[3] < 250 else None
            center_x, center_y = 0,0
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
            print(f"DEBUG: Shape drawing error for shape {shape_info.get('t', 'unknown')}: {e}")
            continue
            
    if themes_applied_count == 0:
        for _ in range(random.randint(4, 7)):
            x_pos, y_pos = random.randint(0, width), random.randint(0, height)
            clr = tuple(random.randint(50, 250) for _ in range(3)) + (random.randint(150, 220),)
            radius = random.randint(width // 20, width // 8)
            if random.random() > 0.5: draw.ellipse((x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius), fill=clr)
            else: draw.rectangle((x_pos - radius // 2, y_pos - radius // 2, x_pos + radius // 2, y_pos + radius // 2), fill=clr)
    try:
        font = ImageFont.load_default()
        text_to_draw = prompt[:70]
        font_path_to_check = FONT_FILE if os.path.exists(FONT_FILE) else None
        if font_path_to_check:
            try:
                font_size = max(12, min(26, int(width / (len(text_to_draw) * 0.35 + 10) if len(text_to_draw) > 0 else width / 12)))
                font = ImageFont.truetype(font_path_to_check, font_size)
            except (IOError, ZeroDivisionError) as font_e:
                print(f"INFO: Özel font ({FONT_FILE}) yüklenemedi ({font_e}), varsayılan kullanılıyor.")
                font = ImageFont.load_default()
        
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text_to_draw, font=font, anchor="lt")
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text_to_draw, font=font)

        pos_x = (width - text_width) / 2
        pos_y = height * 0.96 - text_height
        draw.text((pos_x + 1, pos_y + 1), text_to_draw, font=font, fill=(0, 0, 0, 128))
        draw.text((pos_x, pos_y), text_to_draw, font=font, fill=(255, 255, 255, 230))
    except Exception as e:
        st.toast(f"Görsel üzerine metin yazılamadı: {e}", icon="📝")
        print(f"ERROR: Could not write text on image: {e}")
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
if not st.session_state.models_initialized:
    print("INFO: Uygulama kaynakları ilk kez başlatılıyor...")
    
    gemini_model, gemini_init_error_global = initialize_gemini_model()
    if gemini_model: 
        st.toast(f"✨ Gemini modeli ({st.session_state.gemini_model_name}) başarıyla yüklendi!", icon="🤖")

    supabase, supabase_init_error_global = init_supabase_client_cached()
    if supabase: 
        st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱")

    tts_engine, tts_init_error_global = init_tts_engine_cached()
    if tts_engine: 
        st.toast("🔊 TTS motoru hazır.", icon="🗣️")

    all_chats_data, chat_load_errors = load_all_chats_cached()
    st.session_state.all_chats = all_chats_data
    if chat_load_errors:
        for msg_info in chat_load_errors:
            if msg_info['type'] == 'toast': st.toast(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'warning': st.warning(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'info': st.info(msg_info['text'], icon=msg_info.get('icon'))
            elif msg_info['type'] == 'error': st.error(msg_info['text'], icon=msg_info.get('icon'))

    if not st.session_state.active_chat_id and st.session_state.all_chats:
        try:
            valid_chat_ids = [cid for cid in st.session_state.all_chats.keys() if cid.startswith("chat_") and len(cid.split('_')) > 1]
            if valid_chat_ids:
                st.session_state.active_chat_id = sorted(
                    valid_chat_ids, 
                    key=lambda x: int(x.split('_')[-1]),
                    reverse=True
                )[0]
        except (IndexError, ValueError, TypeError) as e:
             print(f"WARNING: Aktif sohbet ID'si belirlenirken sorun: {e}. İlk geçerli ID'ye veya None'a ayarlanacak.")
             st.session_state.active_chat_id = next((cid for cid in st.session_state.all_chats.keys() if cid.startswith("chat_")), None)


    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    # KNOWLEDGE_BASE global değişkenini burada tanımla ve ata
    kb_data, kb_error = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    globals()['KNOWLEDGE_BASE'] = kb_data
    globals()['knowledge_base_load_error_global'] = kb_error
    
    st.session_state.models_initialized = True
    print("INFO: Uygulama kaynaklarının ilk başlatılması tamamlandı.")
else:
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    current_kb, kb_load_err_rerun = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)
    if kb_load_err_rerun and kb_load_err_rerun != knowledge_base_load_error_global:
        globals()['knowledge_base_load_error_global'] = kb_load_err_rerun
    elif not kb_load_err_rerun and knowledge_base_load_error_global:
        globals()['knowledge_base_load_error_global'] = None
        st.toast("Bilgi tabanı başarıyla güncellendi/yüklendi.", icon="📚")
    globals()['KNOWLEDGE_BASE'] = current_kb


# --- ARAYÜZ FONKSİYONLARI ---
def display_settings_section():
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**")
        new_user_name = st.text_input(
            "Adınız:", 
            value=st.session_state.user_name, 
            key="change_user_name_input", 
            label_visibility="collapsed",
            placeholder="Görünür adınız..."
        )
        if new_user_name != st.session_state.user_name and new_user_name.strip():
            st.session_state.user_name = new_user_name.strip()
            load_knowledge_from_file.clear()
            st.toast("Adınız güncellendi!", icon="✏️")
            st.rerun()

        avatar_col1, avatar_col2 = st.columns([0.8, 0.2])
        with avatar_col1:
            uploaded_avatar_file = st.file_uploader(
                "Avatar yükle (PNG, JPG - maks 2MB):", 
                type=["png", "jpg", "jpeg"], 
                key="upload_avatar_file",
                label_visibility="collapsed"
            )
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
            st.session_state.tts_enabled = st.toggle(
                "Metin Okuma (TTS)", 
                value=st.session_state.tts_enabled, 
                disabled=not is_tts_engine_ok,
                help="Yanıtları sesli olarak oku (TTS motoru aktifse)."
            )
        with stream_toggle_col:
            st.session_state.gemini_stream_enabled = st.toggle(
                "Yanıt Akışı (Streaming)", 
                value=st.session_state.gemini_stream_enabled, 
                help="Yanıtları kelime kelime alarak daha hızlı gösterim sağla (destekleyen modeller için)."
            )
        
        st.session_state.gemini_system_prompt = st.text_area(
            "AI Sistem Talimatı (Opsiyonel):",
            value=st.session_state.get('gemini_system_prompt', ""),
            key="system_prompt_input_area",
            height=100,
            placeholder="Yapay zekanın genel davranışını veya rolünü tanımlayın...",
            help="Modelin yanıtlarını etkilemek için genel bir talimat girin."
        )
        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma")
        gemini_config_col1, gemini_config_col2 = st.columns(2)
        available_gemini_models = ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']
        
        with gemini_config_col1:
            try:
                current_model_index = available_gemini_models.index(st.session_state.gemini_model_name)
            except ValueError:
                current_model_index = 0
                st.session_state.gemini_model_name = available_gemini_models[0]

            st.session_state.gemini_model_name = st.selectbox(
                "AI Modeli:", 
                available_gemini_models, 
                index=current_model_index, 
                key="select_gemini_model", 
                help="Kullanılacak Gemini modelini seçin."
            )
            st.session_state.gemini_temperature = st.slider(
                "Sıcaklık (Temperature):", 0.0, 1.0, 
                st.session_state.gemini_temperature, 0.05, 
                key="temperature_slider", 
                help="Yaratıcılık seviyesi (0=Daha kesin, 1=Daha yaratıcı)."
            )
            st.session_state.gemini_max_tokens = st.slider(
                "Maksimum Yanıt Token:", 256, 8192,
                st.session_state.gemini_max_tokens, 128, 
                key="max_tokens_slider", 
                help="Bir yanıtta üretilecek maksimum token sayısı."
            )
        with gemini_config_col2:
            st.session_state.gemini_top_k = st.slider(
                "Top K:", 1, 100,
                st.session_state.gemini_top_k, 1, 
                key="top_k_slider", 
                help="Kelime seçim çeşitliliği."
            )
            st.session_state.gemini_top_p = st.slider(
                "Top P:", 0.0, 1.0, 
                st.session_state.gemini_top_p, 0.05, 
                key="top_p_slider", 
                help="Kelime seçim odaklılığı."
            )
            if st.button("⚙️ AI Ayarlarını Uygula & Modeli Yeniden Başlat", key="reload_ai_model_button", use_container_width=True, type="primary", help="Seçili AI modelini ve parametreleri yeniden yükler."):
                with st.spinner("AI modeli yeni ayarlarla yeniden başlatılıyor..."):
                    new_model, new_error = initialize_gemini_model()
                    globals()['gemini_model'] = new_model
                    globals()['gemini_init_error_global'] = new_error
                
                if not globals()['gemini_model']:
                    st.error(f"AI modeli yüklenemedi: {globals()['gemini_init_error_global']}")
                else:
                    st.success("AI ayarları başarıyla uygulandı ve model yeniden başlatıldı!", icon="⚙️")
                st.rerun()
        st.divider()

        st.subheader("🧼 Geçmiş Yönetimi")
        clear_current_col, clear_all_col = st.columns(2)
        with clear_current_col:
            active_chat_id_for_clear = st.session_state.get('active_chat_id')
            is_clear_current_disabled = not bool(active_chat_id_for_clear and st.session_state.all_chats.get(active_chat_id_for_clear))
            
            if st.button("🧹 Aktif Sohbetin İçeriğini Temizle", use_container_width=True, type="secondary", key="clear_current_chat_button", help="Sadece açık olan sohbetin içeriğini temizler.", disabled=is_clear_current_disabled):
                if active_chat_id_for_clear and active_chat_id_for_clear in st.session_state.all_chats:
                    st.session_state.all_chats[active_chat_id_for_clear] = []
                    save_all_chats(st.session_state.all_chats)
                    st.toast("Aktif sohbetin içeriği temizlendi!", icon="🧹")
                    st.rerun()
        with clear_all_col:
            is_clear_all_disabled = not bool(st.session_state.all_chats)
            
            # --- HATA AYIKLAMA İÇİN BUTON BÖLÜMÜ ---
            st.markdown("--- *Buton Hata Ayıklama Alanı* ---")
            st.write(f"DEBUG: `is_clear_all_disabled` = {is_clear_all_disabled} (Tip: {type(is_clear_all_disabled)})")
            st.write(f"DEBUG: `st.session_state.all_chats` boş mu? = {not bool(st.session_state.all_chats)} (Tip: {type(st.session_state.all_chats)})")

            # Önce en basit haliyle butonu deneyin:
            st.write("Aşağıdaki BASİT TEST butonu çalışıyorsa, sorun orijinal butonun parametrelerindedir.")
            if st.button("🗑️ TÜM Geçmişi Sil (Basit Test)", key="clear_all_chats_button_simple_test_v2", disabled=is_clear_all_disabled):
                st.session_state.all_chats = {}
                st.session_state.active_chat_id = None
                save_all_chats({})
                st.toast("TÜM sohbet geçmişi silindi! (Basit Test ile)", icon="🗑️")
                st.rerun()
            
            st.write("Eğer yukarıdaki Basit Test butonu StreamlitAPIException veriyorsa, sorun daha derindedir (key, disabled durumu veya Streamlit iç hatası).")
            st.write("Eğer Basit Test butonu ÇALIŞIYORSA, aşağıdaki orijinal butonu YORUMDAN ÇIKARIP deneyin. Hata verirse, parametreleri (type, use_container_width, help) tek tek kaldırarak test edin.")

            # Orijinal buton (şimdilik yorumda bırakılabilir veya test için açılabilir):
            # if st.button("🗑️ TÜM Sohbet Geçmişini Kalıcı Olarak Sil", 
            #                use_container_width=True, 
            #                type="danger", 
            #                key="clear_all_chats_button_original",  # Key'i değiştirdim (çakışma olmasın diye)
            #                help="Dikkat! Tüm sohbet geçmişini kalıcı olarak siler.", 
            #                disabled=is_clear_all_disabled):
            #     st.session_state.all_chats = {}
            #     st.session_state.active_chat_id = None 
            #     save_all_chats({}) 
            #     st.toast("TÜM sohbet geçmişi kalıcı olarak silindi! (Orijinal Buton ile)", icon="🗑️")
            #     st.rerun()
            st.markdown("--- *Buton Hata Ayıklama Alanı Sonu* ---")
            # --- HATA AYIKLAMA BİTİŞ ---


# Diğer fonksiyonlar (display_chat_list_and_about, display_chat_message_with_feedback, vb.)
# önceki yanıttaki gibi devam eder. Uzunluğu nedeniyle buraya tekrar eklemiyorum.
# Lütfen bu fonksiyonları bir önceki yanıttaki güncellenmiş halleriyle kullanın.
# Sadece display_settings_section fonksiyonu yukarıdaki gibi güncellenmiştir.

# --- display_chat_list_and_about (Önceki yanıttan) ---
def display_chat_list_and_about(left_column_ref):
    with left_column_ref:
        st.markdown("#### Sohbetler")
        if st.button("➕ Yeni Sohbet Oluştur", use_container_width=True, key="new_chat_button"):
            st.session_state.next_chat_id_counter = st.session_state.get('next_chat_id_counter', 0) + 1
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
            try:
                valid_chat_ids = [cid for cid in current_chats.keys() if cid.startswith("chat_") and len(cid.split('_')) > 1]
                sorted_chat_ids = sorted(
                    valid_chat_ids, 
                    key=lambda x: int(x.split('_')[-1]), 
                    reverse=True
                )
            except (ValueError, TypeError):
                sorted_chat_ids = sorted([cid for cid in current_chats.keys() if cid.startswith("chat_")], reverse=True)

            if not sorted_chat_ids:
                st.caption("Henüz bir sohbet başlatılmamış.")
            else:
                active_chat_id_display = st.session_state.get('active_chat_id')
                for chat_id_item in sorted_chat_ids:
                    chat_history = current_chats.get(chat_id_item, [])
                    first_user_message = next((msg.get('parts', '') for msg in chat_history if msg.get('role') == 'user'), None)
                    
                    chat_title_prefix = f"Sohbet {chat_id_item.split('_')[1]}" if len(chat_id_item.split('_')) > 1 and chat_id_item.split('_')[1].isdigit() else chat_id_item.replace("chat_","Sohbet ")
                    if first_user_message:
                        chat_display_title = first_user_message[:30] + ("..." if len(first_user_message) > 30 else "")
                    elif chat_history:
                        chat_display_title = chat_title_prefix
                    else:
                        chat_display_title = f"{chat_title_prefix} (Boş)"
                        
                    select_col, download_col, delete_col = st.columns([0.7, 0.15, 0.15])
                    button_style_type = "primary" if active_chat_id_display == chat_id_item else "secondary"
                    
                    if select_col.button(chat_display_title, key=f"select_chat_{chat_id_item}", use_container_width=True, type=button_style_type, help=f"'{chat_display_title}' adlı sohbeti aç"):
                        if active_chat_id_display != chat_id_item:
                            st.session_state.active_chat_id = chat_id_item
                            st.rerun()
                    
                    chat_content_for_download = ""
                    for message_item in chat_history:
                        sender_name = st.session_state.get('user_name', 'Kullanıcı') if message_item.get('role') == 'user' else message_item.get('sender_display', APP_NAME)
                        chat_content_for_download += f"{sender_name}: {message_item.get('parts', '')}\n--------------------------------\n"
                    
                    download_col.download_button(
                        "⬇️", 
                        data=chat_content_for_download.encode('utf-8'), 
                        file_name=f"{chat_display_title.replace(' ', '_').replace('(', '').replace(')', '')}_{chat_id_item}.txt", 
                        mime="text/plain", 
                        key=f"download_chat_{chat_id_item}", 
                        help=f"'{chat_display_title}' sohbetini indir (.txt)", 
                        use_container_width=True,
                        disabled=not chat_history
                    )
                    
                    if delete_col.button("🗑️", key=f"delete_chat_{chat_id_item}", use_container_width=True, help=f"'{chat_display_title}' adlı sohbeti sil", type="secondary"):
                        if chat_id_item in current_chats:
                            del current_chats[chat_id_item]
                            if active_chat_id_display == chat_id_item:
                                remaining_valid_ids = [cid for cid in current_chats.keys() if cid.startswith("chat_") and len(cid.split('_')) > 1]
                                if remaining_valid_ids:
                                    st.session_state.active_chat_id = sorted(
                                        remaining_valid_ids, 
                                        key=lambda x: int(x.split('_')[-1]), 
                                        reverse=True
                                    )[0]
                                else:
                                    st.session_state.active_chat_id = None
                            save_all_chats(current_chats)
                            st.toast(f"'{chat_display_title}' sohbeti silindi.", icon="🗑️")
                            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Uygulama Hakkında", expanded=False):
            st.markdown(f"""
            **{APP_NAME} v{APP_VERSION}**
            AI Destekli Kişisel Asistanınız.
            Geliştirici: **Hanogt**
            © 2024-{CURRENT_YEAR} {APP_NAME} Projesi
            """)
            st.caption(f"Aktif Oturum ID: `{_get_session_id()[:12]}...`")

# --- display_chat_message_with_feedback (Önceki yanıttan) ---
def display_chat_message_with_feedback(message_data, message_index, current_chat_id):
    role = message_data.get('role', 'model')
    content_text = str(message_data.get('parts', ''))
    is_user_message = (role == 'user')
    if is_user_message:
        sender_display_name = st.session_state.get('user_name', 'Kullanıcı')
        avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes)) if st.session_state.user_avatar_bytes else "🧑"
    else:
        sender_display_name = message_data.get('sender_display', APP_NAME)
        if "Gemini" in sender_display_name: avatar_icon = "✨"
        elif any(w in sender_display_name.lower() for w in ["web", "wiki", "arama", "ddg"]): avatar_icon = "🌐"
        elif any(w in sender_display_name.lower() for w in ["bilgi", "fonksiyon", "taban"]): avatar_icon = "📚"
        elif "Yaratıcı" in sender_display_name: avatar_icon = "🎨"
        elif "Görsel" in sender_display_name: avatar_icon = "🖼️"
        else: avatar_icon = "🤖"

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
            if tiktoken_encoder:
                try:
                    token_count = len(tiktoken_encoder.encode(content_text))
                    token_count_display_str = f" (~{token_count} token)"
                except Exception: pass

            source_col, tts_col, feedback_col = st.columns([0.75, 0.1, 0.15])
            with source_col:
                source_name_only = sender_display_name.split('(')[-1].replace(')', '').strip() if '(' in sender_display_name else sender_display_name
                st.caption(f"Kaynak: {source_name_only}{token_count_display_str}")
            
            with tts_col:
                if st.session_state.tts_enabled and globals().get('tts_engine'):
                    if st.button("🔊", key=f"tts_button_{current_chat_id}_{message_index}", help="Yanıtı sesli oku", use_container_width=True):
                        speak(content_text)
            
            with feedback_col:
                if st.button("✍️", key=f"feedback_button_{current_chat_id}_{message_index}", help="Yanıt hakkında geri bildirim ver", use_container_width=True):
                    st.session_state.current_message_id_for_feedback = f"{current_chat_id}_{message_index}"
                    previous_user_prompt = "[Kullanıcı istemi bulunamadı]"
                    if message_index > 0:
                         prev_msg = st.session_state.all_chats[current_chat_id][message_index - 1]
                         if prev_msg['role'] == 'user':
                            previous_user_prompt = prev_msg['parts']
                    
                    st.session_state.last_user_prompt_for_feedback = previous_user_prompt
                    st.session_state.last_ai_response_for_feedback = content_text
                    st.session_state.show_feedback_comment_form = True
                    st.session_state.feedback_comment_input = ""
                    st.rerun()

# --- display_feedback_form_if_active (Önceki yanıttan) ---
def display_feedback_form_if_active():
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        form_unique_key = f"feedback_form_{st.session_state.current_message_id_for_feedback.replace('.', '_')}"
        
        with st.form(key=form_unique_key):
            st.markdown("#### Yanıt Geri Bildirimi")
            st.caption(f"**İstem:** `{str(st.session_state.last_user_prompt_for_feedback)[:70]}...`")
            st.caption(f"**AI Yanıtı:** `{str(st.session_state.last_ai_response_for_feedback)[:70]}...`")
            
            feedback_rating_type = st.radio(
                "Bu yanıtı nasıl buldunuz?",
                ["👍 Beğendim", "👎 Beğenmedim"],
                horizontal=True,
                key=f"rating_type_{form_unique_key}",
                index=0 if st.session_state.last_feedback_type == 'positive' else 1
            )
            feedback_user_comment = st.text_area(
                "Ek yorumunuz (isteğe bağlı):",
                value=st.session_state.feedback_comment_input,
                key=f"comment_input_{form_unique_key}",
                height=100,
                placeholder="Yanıtla ilgili düşüncelerinizi paylaşın..."
            )
            st.session_state.feedback_comment_input = feedback_user_comment # Anlık güncelle
            
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

# --- display_chat_interface_main (Önceki yanıttan) ---
def display_chat_interface_main(main_column_container_ref=None): # Parametre opsiyonel yapıldı
    active_chat_id_main = st.session_state.get('active_chat_id')
    if active_chat_id_main is None:
        st.info("💬 Başlamak için sol menüden **'➕ Yeni Sohbet Oluştur'** butonuna tıklayın veya var olan bir sohbeti seçin.", icon="👈")
        return

    current_chat_history = st.session_state.all_chats.get(active_chat_id_main, [])
    chat_messages_container = st.container(height=600, border=False)
    with chat_messages_container:
        if not current_chat_history:
            st.info(f"Merhaba {st.session_state.user_name}! Bu yeni sohbetinize hoş geldiniz. Size nasıl yardımcı olabilirim?", icon="👋")
        for idx, message in enumerate(current_chat_history):
            display_chat_message_with_feedback(message, idx, active_chat_id_main)
    
    display_feedback_form_if_active()

    user_chat_prompt = st.chat_input(
        f"{st.session_state.user_name}, ne sormak istersin? (Enter ile gönder)",
        key=f"chat_input_{active_chat_id_main}"
    )

    if user_chat_prompt:
        user_message_data = {'role': 'user', 'parts': user_chat_prompt}
        st.session_state.all_chats[active_chat_id_main].append(user_message_data)
        save_all_chats(st.session_state.all_chats)
        
        message_unique_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"
        st.session_state.message_id_counter += 1
        history_for_model_request = st.session_state.all_chats[active_chat_id_main][-21:-1] 
        
        with st.chat_message("assistant", avatar="⏳"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 _Yanıtınız itinayla hazırlanıyor... Lütfen bekleyiniz..._")
        
        ai_response_content, ai_sender_name = get_hanogt_response_orchestrator(
            user_chat_prompt,
            history_for_model_request,
            message_unique_id,
            active_chat_id_main,
            use_stream=st.session_state.gemini_stream_enabled
        )
        
        final_ai_response_text = ""
        if st.session_state.gemini_stream_enabled and "Stream" in ai_sender_name and hasattr(ai_response_content, '__iter__') and not isinstance(ai_response_content, str):
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
                error_message_stream = f"Stream yanıtı işlenirken hata oluştu: {e}"
                stream_display_container.error(error_message_stream)
                final_ai_response_text = error_message_stream
                ai_sender_name = f"{APP_NAME} (Stream Hatası)"
                log_interaction(user_chat_prompt, final_ai_response_text, "Stream Hatası", message_unique_id, active_chat_id_main)
        else:
            thinking_placeholder.empty()
            final_ai_response_text = str(ai_response_content)

        if final_ai_response_text.strip() or "Stream" not in ai_sender_name :
            ai_message_data = {'role': 'model', 'parts': final_ai_response_text, 'sender_display': ai_sender_name}
            st.session_state.all_chats[active_chat_id_main].append(ai_message_data)
            save_all_chats(st.session_state.all_chats)
        
        if st.session_state.tts_enabled and globals().get('tts_engine') and isinstance(final_ai_response_text, str) and "Stream" not in ai_sender_name:
            speak(final_ai_response_text)
        st.rerun()

# --- UYGULAMA ANA AKIŞI ---
st.markdown(f"<h1 style='text-align:center;color:#0078D4;'>{APP_NAME} <sup style='font-size:0.6em;color:#555;'>v{APP_VERSION}</sup></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-style:italic;color:#555;'>Yapay Zeka Destekli Kişisel Asistanınız</p>", unsafe_allow_html=True)
st.markdown("---")

if gemini_init_error_global: st.warning(gemini_init_error_global, icon="🗝️")
if supabase_init_error_global: st.warning(supabase_init_error_global, icon="🧱")
if tts_init_error_global: st.warning(tts_init_error_global, icon="🔇")
if knowledge_base_load_error_global: st.warning(knowledge_base_load_error_global, icon="📚")

if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce Sizi Tanıyalım")
    login_cols = st.columns([0.2, 0.6, 0.2])
    with login_cols[1]:
        with st.form("user_login_form"):
            user_entered_name = st.text_input(
                "Size nasıl hitap etmemizi istersiniz?", 
                placeholder="İsminiz veya takma adınız...", 
                key="login_name_input",
                value=st.session_state.get('user_name', '')
            )
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
    if not st.session_state.greeting_message_shown:
        st.success(f"Tekrar hoş geldiniz, **{st.session_state.user_name}**! Size nasıl yardımcı olabilirim?", icon="🎉")
        st.session_state.greeting_message_shown = True
        
    app_left_column, app_main_column = st.columns([1, 3])
    display_chat_list_and_about(app_left_column)
    
    with app_main_column:
        display_settings_section() # Ayarlar bölümünü göster
        
        st.markdown("#### Uygulama Modu")
        app_modes = {
            "Yazılı Sohbet": "💬",
            "Sesli Sohbet (Dosya Yükle)": "🎤",
            "Yaratıcı Stüdyo": "🎨",
            "Görsel Oluşturucu": "🖼️"
        }
        mode_options_keys = list(app_modes.keys())
        try:
            current_mode_index = mode_options_keys.index(st.session_state.app_mode)
        except ValueError:
            current_mode_index = 0
            st.session_state.app_mode = mode_options_keys[0]

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
            display_chat_interface_main() 
        
        elif current_app_mode == "Sesli Sohbet (Dosya Yükle)":
            st.info("Yanıt almak istediğiniz **Türkçe** bir ses dosyasını yükleyin (WAV, MP3, OGG, FLAC, M4A).", icon="📢")
            audio_file_uploaded = st.file_uploader(
                "Ses Dosyası:", 
                type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], 
                label_visibility="collapsed", 
                key="audio_file_uploader"
            )
            if audio_file_uploaded:
                st.audio(audio_file_uploaded, format=audio_file_uploaded.type)
                active_chat_id_for_audio = st.session_state.get('active_chat_id')
                if not active_chat_id_for_audio:
                    st.warning("Lütfen önce bir sohbet seçin veya yeni bir sohbet başlatın.", icon="⚠️")
                else:
                    transcribed_text = None 
                    with st.spinner(f"🔊 '{audio_file_uploaded.name}' ses dosyası işleniyor..."):
                        recognizer_instance = sr.Recognizer()
                        try:
                            audio_bytes = BytesIO(audio_file_uploaded.getvalue())
                            with sr.AudioFile(audio_bytes) as audio_source:
                                audio_data = recognizer_instance.record(audio_source)
                            transcribed_text = recognizer_instance.recognize_google(audio_data, language="tr-TR")
                            st.success(f"**🎙️ Algılanan Metin:**\n\n> {transcribed_text}")
                        except sr.UnknownValueError:
                            st.error("Ses anlaşılamadı veya boş.", icon="🔇")
                        except sr.RequestError as e:
                            st.error(f"Google Speech Recognition servisine ulaşılamadı; {e}.", icon="🌐")
                        except Exception as e:
                            st.error(f"Ses işleme sırasında hata: {e}")
                            print(f"ERROR: Audio processing failed: {e}")

                    if transcribed_text:
                        user_msg_audio = {'role': 'user', 'parts': f"(Yüklenen Ses: '{audio_file_uploaded.name}')\n{transcribed_text}"}
                        st.session_state.all_chats[active_chat_id_for_audio].append(user_msg_audio)
                        
                        audio_msg_id = f"audio_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                        st.session_state.message_id_counter += 1
                        history_for_audio_prompt = st.session_state.all_chats[active_chat_id_for_audio][-21:-1]
                        
                        with st.spinner("🤖 AI yanıtı hazırlanıyor..."):
                            ai_response_audio, sender_name_audio = get_hanogt_response_orchestrator(transcribed_text, history_for_audio_prompt, audio_msg_id, active_chat_id_for_audio, False)
                        
                        st.markdown(f"#### {sender_name_audio} Yanıtı:")
                        st.markdown(str(ai_response_audio))
                        
                        ai_msg_audio = {'role': 'model', 'parts': str(ai_response_audio), 'sender_display': sender_name_audio}
                        st.session_state.all_chats[active_chat_id_for_audio].append(ai_msg_audio)
                        save_all_chats(st.session_state.all_chats)
                        st.success("✅ Sesli istem ve AI yanıtı sohbete eklendi!")
                        
                        if st.session_state.tts_enabled and globals().get('tts_engine'): 
                            speak(str(ai_response_audio))

        elif current_app_mode == "Yaratıcı Stüdyo":
            st.markdown("💡 Bir fikir verin, AI sizin için ilham verici metinler üretsin!")
            creative_prompt_input = st.text_area(
                "Yaratıcı Metin Tohumu:", 
                key="creative_prompt_area", 
                placeholder="Örn: 'Geceleri parlayan çiçek ve sırrı'", 
                height=100
            )
            col_len, col_style = st.columns(2)
            length_selection = col_len.selectbox("Metin Uzunluğu:", ["kısa", "orta", "uzun"], index=1, key="creative_length_select")
            style_selection = col_style.selectbox("Metin Stili:", ["genel", "şiirsel", "hikaye", "bilgilendirici", "esprili"], index=0, key="creative_style_select")
            
            if st.button("✨ Yaratıcı Metin Üret!", key="generate_creative_text_button", type="primary", use_container_width=True):
                if creative_prompt_input and creative_prompt_input.strip():
                    active_chat_id_creative = st.session_state.get('active_chat_id', 'creative_mode_no_chat')
                    creative_msg_id = f"creative_{st.session_state.message_id_counter}_{int(time.time())}"
                    st.session_state.message_id_counter += 1
                    generated_response, response_sender_name = None, f"{APP_NAME} (Yaratıcı Modül)"
                    
                    if globals().get('gemini_model'):
                        with st.spinner("✨ Gemini ilham perilerini çağırıyor..."):
                            gemini_system_instruction = f"Sen yaratıcı bir metin yazarısın. Kullanıcının verdiği '{creative_prompt_input}' tohumundan yola çıkarak '{style_selection}' stilinde ve '{length_selection}' uzunluğunda bir metin üret."
                            gemini_creative_response = get_gemini_response(gemini_system_instruction, [], False) 
                            
                            if isinstance(gemini_creative_response, str) and not gemini_creative_response.startswith(GEMINI_ERROR_PREFIX):
                                generated_response = gemini_creative_response
                                response_sender_name = f"{APP_NAME} (Gemini Yaratıcı)"
                            else:
                                st.toast("Gemini yaratıcı yanıtı alınamadı, yerel üretici denenecek.", icon="ℹ️")
                                print(f"INFO: Gemini creative response failed: {gemini_creative_response}")
                    
                    if not generated_response:
                        with st.spinner("✨ Hayal gücü motoru çalışıyor..."):
                            generated_response = creative_response_generator(creative_prompt_input, length_selection, style_selection)
                            first_word_of_prompt = creative_prompt_input.split()[0] if creative_prompt_input else "yaratıcı"
                            new_generated_word = advanced_word_generator(first_word_of_prompt)
                            generated_response += f"\n\n---\n🔮 **Kelimatör Önerisi:** _{new_generated_word}_"
                            response_sender_name = f"{APP_NAME} (Yerel Yaratıcı)"
                            
                    st.markdown(f"#### {response_sender_name} İlhamı:")
                    st.markdown(generated_response)
                    log_interaction(f"Yaratıcı Stüdyo: '{creative_prompt_input}' (Stil: {style_selection}, Uzunluk: {length_selection})", generated_response, response_sender_name, creative_msg_id, active_chat_id_creative)
                    st.success("✨ Yaratıcı metin başarıyla oluşturuldu!")
                    if st.session_state.tts_enabled and globals().get('tts_engine'): speak(generated_response)
                else:
                    st.warning("Lütfen yaratıcı bir metin tohumu girin.", icon="✍️")

        elif current_app_mode == "Görsel Oluşturucu":
            st.markdown("🎨 Hayalinizi kelimelerle tarif edin, AI sizin için (basit) bir görsel çizsin!")
            st.info("ℹ️ Not: Bu mod sembolik ve basit çizimler üretir.", icon="💡")
            image_prompt_input = st.text_input(
                "Görsel Tarifi (Anahtar kelimeler: örn: 'karlı dağ, gün batımı'):", 
                key="image_generation_prompt_input", 
                placeholder="Örn: 'Mor gün batımında uçan kuşlar'"
            )
            if st.button("🖼️ Görsel Oluştur!", key="generate_image_button", type="primary", use_container_width=True):
                if image_prompt_input and image_prompt_input.strip():
                    with st.spinner("🖌️ Sanatçı fırçaları çalışıyor..."):
                        generated_image = generate_prompt_influenced_image(image_prompt_input)
                        st.image(generated_image, caption=f"'{image_prompt_input[:60]}' isteminin yorumu", use_container_width=True)
                    try:
                        image_buffer = BytesIO()
                        generated_image.save(image_buffer, format="PNG")
                        image_bytes = image_buffer.getvalue()
                        safe_filename_prompt_part = re.sub(r'[^\w\s-]', '', image_prompt_input.lower())[:25].strip().replace(' ', '_')
                        image_file_name = f"hanogt_gorsel_{safe_filename_prompt_part or 'tarif'}_{int(time.time())}.png"
                        st.download_button(
                            "🖼️ Oluşturulan Görseli İndir", 
                            data=image_bytes, 
                            file_name=image_file_name, 
                            mime="image/png", 
                            use_container_width=True
                        )
                        active_chat_id_image = st.session_state.get('active_chat_id')
                        if active_chat_id_image and active_chat_id_image in st.session_state.all_chats:
                            user_msg_image = {'role': 'user', 'parts': f"(Görsel Oluşturma İstemi: {image_prompt_input})"}
                            ai_msg_image = {'role': 'model', 'parts': f"'{image_prompt_input}' istemi için yukarıdaki görsel oluşturuldu.", 'sender_display': f"{APP_NAME} (Görsel Oluşturucu)"}
                            st.session_state.all_chats[active_chat_id_image].extend([user_msg_image, ai_msg_image])
                            save_all_chats(st.session_state.all_chats)
                            st.info("Görsel oluşturma bilgisi aktif sohbete eklendi.", icon="💾")
                    except Exception as e:
                        st.error(f"Görsel indirme/kaydetme hatası: {e}")
                else:
                    st.warning("Lütfen bir görsel tarifi girin.", icon="✍️")

        st.markdown("<hr style='margin-top:1rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        with footer_cols[0]:
            st.caption(f"Kullanıcı: **{st.session_state.get('user_name', 'Tanımlanmamış')}**")
        with footer_cols[1]:
            st.caption(f"<div style='text-align:center;'>{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}</div>", unsafe_allow_html=True)
        with footer_cols[2]:
            ai_model_name_display = st.session_state.gemini_model_name.split('/')[-1]
            ai_status_text = "Aktif" if globals().get('gemini_model') else "Devre Dışı"
            logging_status_text = "Aktif" if globals().get('supabase') else "Devre Dışı"
            st.caption(f"<div style='text-align:right;'>AI: {ai_status_text} ({ai_model_name_display}) | Log: {logging_status_text}</div>", unsafe_allow_html=True)

