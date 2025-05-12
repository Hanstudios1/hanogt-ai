# app.py

# --- Gerekli Kütüphaneler ---
import streamlit as st
import requests
from bs4 import BeautifulSoup # pip install beautifulsoup4 lxml
import wikipedia # pip install wikipedia
import speech_recognition as sr # pip install SpeechRecognition pydub
# Gerekirse: sudo apt-get install ffmpeg veya brew install ffmpeg
import pyttsx3 # pip install pyttsx3
# Linux için: sudo apt-get update && sudo apt-get install espeak ffmpeg libespeak1
import random
import re
import os
import json
from PIL import Image, ImageDraw, ImageFont # pip install Pillow
import time
from io import BytesIO
from duckduckgo_search import DDGS # pip install -U duckduckgo_search
from urllib.parse import urlparse, unquote
import google.generativeai as genai # pip install google-generativeai
from datetime import datetime
import uuid # Daha benzersiz ID'ler için

# Supabase (isteğe bağlı, loglama/feedback için)
try:
    from supabase import create_client, Client # pip install supabase
    from postgrest import APIError as SupabaseAPIError # Supabase özel hataları için
except ImportError:
    st.toast(
        "Supabase kütüphanesi bulunamadı. Loglama ve geri bildirim özellikleri çalışmayabilir.",
        icon="ℹ️"
    )
    create_client = None
    Client = None
    SupabaseAPIError = None # Tanımlı değilse None yapalım

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed" # Sidebar kullanılmıyor
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.0.0 Pro+ Enhanced" # Sürüm güncellendi (İyileştirmeler)
CURRENT_YEAR = datetime.now().year
CHAT_HISTORY_FILE = "chat_history_v2.json" # Tüm sohbetleri içeren dosya
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen biraz sonra tekrar deneyin."
REQUEST_TIMEOUT = 20 # Biraz artırıldı
SCRAPE_MAX_CHARS = 3500 # Biraz artırıldı
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 {APP_NAME}/{APP_VERSION}" # User agent güncel tutulabilir
SUPABASE_TABLE_LOGS = "chat_logs"
SUPABASE_TABLE_FEEDBACK = "user_feedback"
FONT_FILE = "arial.ttf" # Varsa kullanılacak font dosyası

# --- Dinamik Fonksiyonlar (Global) ---
DYNAMIC_FUNCTIONS_MAP = {
    "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
    "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}",
    "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}"
}

# --- Bilgi Tabanı ---
knowledge_base_load_error = None

@st.cache_data(ttl=3600) # Bilgi tabanını 1 saat cache'le
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE, user_name_for_greeting="kullanıcı"):
    """Bilgi tabanını dosyadan yükler veya varsayılanı kullanır."""
    global knowledge_base_load_error
    # Kullanıcı adını içeren dinamik varsayılanlar
    default_knowledge = {
        "merhaba": [f"Merhaba {user_name_for_greeting}!", "Selam!", "Hoş geldin!", f"Size nasıl yardımcı olabilirim?"],
        "selam": ["Merhaba!", "Selam sana da!", "Nasıl gidiyor?"],
        "nasılsın": ["İyiyim, teşekkürler! Siz nasılsınız?", "Harika hissediyorum, yardımcı olmak için buradayım!", "Her şey yolunda, sizin için ne yapabilirim?"],
        "hanogt kimdir": [f"Ben {APP_NAME} ({APP_VERSION}), Streamlit ve Python ile geliştirilmiş bir yapay zeka asistanıyım.", f"{APP_NAME} ({APP_VERSION}), sorularınızı yanıtlamak, metinler üretmek ve hatta basit görseller oluşturmak için tasarlandı."],
        "teşekkür ederim": ["Rica ederim!", "Ne demek!", "Yardımcı olabildiğime sevindim.", "Her zaman!"],
        "görüşürüz": ["Görüşmek üzere!", "Hoşça kal!", "İyi günler dilerim!", "Tekrar beklerim!"],
        "adın ne": [f"Ben {APP_NAME}, versiyon {APP_VERSION}.", f"Bana {APP_NAME} diyebilirsiniz."],
        "ne yapabilirsin": ["Sorularınızı yanıtlayabilir, metin özetleyebilir, web'de arama yapabilir, yaratıcı metinler üretebilir ve basit görseller çizebilirim.", "Size çeşitli konularda yardımcı olabilirim. Ne merak ediyorsunuz?"],
        # Dinamik fonksiyon çağrıları için placeholder'lar
        "saat kaç": ["Saat bilgisini sizin için alıyorum."],
        "bugün ayın kaçı": ["Tarih bilgisini sizin için alıyorum."],
        "tarih ne": ["Tarih bilgisini sizin için alıyorum."],
        "hava durumu": ["Üzgünüm, şu an için güncel hava durumu bilgisi sağlayamıyorum. Bunun için özel bir hava durumu servisine göz atabilirsiniz.", "Hava durumu servisim henüz aktif değil, ancak bu konuda bir geliştirme yapmayı planlıyorum!"]
    }

    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                loaded_kb = json.load(f)
            # Varsayılanları yüklenenlerle birleştir (varsayılanlar üzerine yazılır)
            merged_kb = {**default_knowledge, **loaded_kb}
            knowledge_base_load_error = None
            return merged_kb
        else:
            knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) bulunamadı. Varsayılan kullanılıyor."
            st.toast(knowledge_base_load_error, icon="ℹ️")
            return default_knowledge
    except json.JSONDecodeError:
        knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) hatalı formatta. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="⚠️")
        return default_knowledge
    except Exception as e:
        knowledge_base_load_error = f"Bilgi tabanı yüklenirken bilinmeyen bir hata oluştu: {e}. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="🔥")
        return default_knowledge


def kb_chatbot_response(query, knowledge_base_dict):
    """Bilgi tabanından veya dinamik fonksiyonlardan yanıt döndürür."""
    query_lower = query.lower().strip()

    # 1. Dinamik Fonksiyon Kontrolü
    if query_lower in DYNAMIC_FUNCTIONS_MAP:
        try:
            return DYNAMIC_FUNCTIONS_MAP[query_lower]()
        except Exception as e_dyn:
            st.error(f"Dinamik fonksiyon '{query_lower}' çalıştırılırken hata: {e_dyn}")
            return DEFAULT_ERROR_MESSAGE

    # 2. Tam Eşleşme Kontrolü
    if query_lower in knowledge_base_dict:
        response_options = knowledge_base_dict[query_lower]
        return random.choice(response_options) if isinstance(response_options, list) else response_options

    # 3. Kısmi Eşleşme Kontrolü (Anahtar kelime içeriyor mu?)
    possible_partial_responses = []
    for key, responses_list in knowledge_base_dict.items():
        if key in query_lower: # Eğer sorgu, KB anahtarını içeriyorsa
            current_options = responses_list if isinstance(responses_list, list) else [responses_list]
            possible_partial_responses.extend(current_options)
    if possible_partial_responses:
        return random.choice(list(set(possible_partial_responses))) # Tekrarları kaldırıp rastgele seç

    # 4. Benzerlik Skoru Kontrolü (Kelime kesişimi)
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower)) # En az 3 harfli kelimeler
    best_match_score = 0
    best_match_responses = []
    for key, responses_list in knowledge_base_dict.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words: continue
        common_words = query_words.intersection(key_words)
        # Jaccard benzerliği veya basit oran kullanılabilir
        score = len(common_words) / len(query_words.union(key_words)) if query_words.union(key_words) else 0
        # score = len(common_words) / len(key_words) if len(key_words) > 0 else 0 # Alternatif skorlama

        similarity_threshold = 0.5 # Eşik değeri ayarlanabilir
        if score >= similarity_threshold:
            current_options = responses_list if isinstance(responses_list, list) else [responses_list]
            if score > best_match_score:
                best_match_score = score
                best_match_responses = current_options
            elif score == best_match_score:
                best_match_responses.extend(current_options) # Eşit skorda ekle

    if best_match_responses:
        return random.choice(list(set(best_match_responses))) # Tekrarları kaldırıp rastgele seç

    # Hiçbir eşleşme bulunamadı
    return None

# --- API Anahtarı ve Gemini Yapılandırması ---
gemini_model = None
gemini_init_error_global = None

def initialize_gemini_model():
    """Google Generative AI modelini session state'deki ayarlarla başlatır."""
    global gemini_init_error_global
    api_key_local = st.secrets.get("GOOGLE_API_KEY")

    if not api_key_local:
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta (st.secrets['GOOGLE_API_KEY']) bulunamadı! Lütfen ekleyin."
        return None
    try:
        genai.configure(api_key=api_key_local)
        # Güvenlik ayarları (İsteğe bağlı olarak değiştirilebilir)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Session state'den alınan yapılandırma değerleri
        model_name = st.session_state.get('gemini_model_name', 'gemini-1.5-flash-latest')
        temperature = st.session_state.get('gemini_temperature', 0.7)
        top_p = st.session_state.get('gemini_top_p', 0.95)
        top_k = st.session_state.get('gemini_top_k', 40)
        max_output_tokens = st.session_state.get('gemini_max_tokens', 4096)

        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens
            )
            # system_instruction eklenebilir (model destekliyorsa)
            # system_instruction="Sen yardımsever bir asistansın.",
        )
        gemini_init_error_global = None
        st.toast(f"✨ Gemini modeli ({model_name}) başarıyla yüklendi!", icon="🤖")
        return model
    except Exception as e:
        gemini_init_error_global = f"🛑 Gemini yapılandırma hatası: {e}. API anahtarını, ayarları ve internet bağlantısını kontrol edin."
        print(f"ERROR: Gemini Initialization Failed: {e}") # Loglama
        return None

# --- Supabase İstemcisini Başlatma ---
supabase = None
supabase_error_global = None

@st.cache_resource(ttl=3600) # Supabase client'ı 1 saat cache'le
def init_supabase_client_cached():
    """Supabase istemcisini başlatır ve cache'ler."""
    global supabase_error_global
    if not create_client:
        supabase_error_global = "Supabase kütüphanesi yüklenemedi. Loglama/Feedback devre dışı."
        return None

    supabase_url_local = st.secrets.get("SUPABASE_URL")
    supabase_key_local = st.secrets.get("SUPABASE_SERVICE_KEY")

    if not supabase_url_local or not supabase_key_local:
        supabase_error_global = "Supabase URL veya Service Key Secrets'ta bulunamadı! Loglama/Feedback devre dışı."
        return None

    try:
        client: Client = create_client(supabase_url_local, supabase_key_local)
        # Bağlantıyı test etmek için basit bir sorgu (opsiyonel)
        # client.table(SUPABASE_TABLE_LOGS).select("id", head=True).limit(1).execute()
        supabase_error_global = None
        st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱")
        return client
    except Exception as e:
        error_msg_supabase = f"Supabase bağlantı hatası: {e}. Loglama/Feedback yapılamayacak."
        if "invalid url" in str(e).lower():
            error_msg_supabase += " URL formatını kontrol edin (örn: https://xyz.supabase.co)."
        elif "invalid key" in str(e).lower():
            error_msg_supabase += " Service Key'inizi kontrol edin."
        supabase_error_global = error_msg_supabase
        print(f"ERROR: Supabase Connection Failed: {e}") # Loglama
        return None

# --- YARDIMCI FONKSİYONLAR ---

def _get_session_id():
    """Mevcut oturum ID'sini alır veya yeni bir tane oluşturur."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

tts_engine = None
tts_init_error_global = None

@st.cache_resource
def init_tts_engine_cached():
    """Metin okuma (TTS) motorunu başlatır ve cache'ler."""
    global tts_init_error_global
    try:
        engine = pyttsx3.init()
        # İsteğe bağlı: Hızı veya sesi ayarlama
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate-50)
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[1].id) # Farklı bir ses deneyin (varsa)
        tts_init_error_global = None
        st.toast("🔊 Metin okuma motoru (TTS) hazır.", icon="🗣️")
        return engine
    except Exception as e:
        tts_init_error_global = f"⚠️ Metin okuma (TTS) motoru başlatılamadı: {e}. Bu özellik kullanılamayacak."
        print(f"ERROR: TTS Initialization Failed: {e}") # Loglama
        return None

def speak(text_to_speak):
    """Verilen metni sesli olarak okur."""
    current_tts_engine = globals().get('tts_engine')
    if not current_tts_engine:
        st.toast("Metin okuma motoru aktif değil veya başlatılamadı.", icon="🔇")
        return
    if not st.session_state.get('tts_enabled', True):
        st.toast("Metin okuma özelliği ayarlardan kapatılmış.", icon="🔇")
        return

    try:
        # Metni temizleme (emoji vb. sorun çıkarabilir)
        cleaned_text = re.sub(r'[^\w\s.,!?-]', '', text_to_speak) # Basit temizleme
        if not cleaned_text.strip():
             st.toast("Okunacak anlamlı bir metin bulunamadı.", icon="ℹ️")
             return

        current_tts_engine.say(cleaned_text)
        current_tts_engine.runAndWait()
    except RuntimeError as re_tts:
        # Bu hata genellikle motor meşgulken tekrar çağrıldığında olur
        st.warning(f"Konuşma motorunda bir çalışma zamanı sorunu: {re_tts}. Biraz bekleyip tekrar deneyin.", icon="🔊")
        # Gelişmiş: Motoru durdurup yeniden başlatmayı deneyebiliriz
        # try:
        #     current_tts_engine.stop()
        # except: pass
    except Exception as e_tts:
        st.error(f"Konuşma sırasında beklenmedik bir hata oluştu: {e_tts}", icon="🔥")
        print(f"ERROR: TTS Speak Failed: {e_tts}") # Loglama

def _clean_text(text: str) -> str:
    """Metindeki fazla boşlukları ve satırları temizler."""
    text = re.sub(r'\s+', ' ', text) # Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\n\s*\n', '\n\n', text) # Boş satırları temizle
    return text.strip() # Başındaki ve sonundaki boşlukları kaldır

def scrape_url_content(url: str, timeout: int = REQUEST_TIMEOUT, max_chars: int = SCRAPE_MAX_CHARS) -> str | None:
    """Verilen URL'den ana metin içeriğini kazır."""
    st.toast(f"🌐 '{urlparse(url).netloc}' sayfasından içerik alınıyor...", icon="⏳")
    try:
        # URL geçerliliğini kontrol et
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']:
            st.warning(f"Geçersiz URL formatı, kazıma atlanıyor: {url}", icon="🔗")
            return None

        headers = {
            'User-Agent': USER_AGENT,
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'DNT': '1', # Do Not Track
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status() # HTTP hatalarını kontrol et (4xx, 5xx)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            st.info(f"URL HTML içeriği değil ('{content_type}' tipinde). Kazıma atlanıyor: {url}", icon="📄")
            response.close()
            return None

        # İçeriği parça parça oku (büyük dosyalar için)
        html_content = ""
        content_length_processed = 0
        # İşlenecek maksimum HTML boyutunu sınırlayalım (kazınacak metin boyutunun ~10 katı)
        max_html_size_to_process = max_chars * 10
        try:
            for chunk in response.iter_content(chunk_size=16384, decode_unicode=True, errors='ignore'):
                if chunk: # None chunk gelme ihtimaline karşı
                    html_content += chunk
                    content_length_processed += len(chunk.encode('utf-8', 'ignore')) # Yaklaşık byte boyutu
                    if content_length_processed > max_html_size_to_process:
                        st.warning(f"HTML içeriği çok büyük ({content_length_processed / 1024:.0f}KB+), ilk kısmı işlenecek.", icon="✂️")
                        break
        finally:
            response.close() # Her durumda bağlantıyı kapat

        if not html_content:
             st.warning("URL'den boş içerik alındı.", icon="📄")
             return None

        # BeautifulSoup ile parse et
        soup = BeautifulSoup(html_content, 'lxml')

        # İstenmeyen etiketleri kaldır
        tags_to_remove = ["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio", "figure", "figcaption", "input", "textarea", "select"]
        for element_to_remove in soup.find_all(tags_to_remove):
            element_to_remove.decompose()

        # Ana içerik alanlarını bulmaya çalış (daha fazla seçici eklenebilir)
        potential_content_parts = []
        content_selectors = [
            'article[class*="content"]', 'article[class*="post"]', 'article[class*="entry"]',
            'main[class*="content"]', 'main[id*="content"]', 'main',
            'div[class*="post-body"]', 'div[class*="article-body"]', 'div[class*="entry-content"]',
            'div[itemprop="articleBody"]', 'div[role="main"]',
            'article', '.content', '.post-content', '.entry-content', 'section[role="main"]',
            '#content', '#main', '#article' # ID seçicileri
        ]
        content_found_flag = False
        min_meaningful_text_len = 100 # Anlamlı paragraf için minimum karakter sayısı
        min_sentence_indicators = 1 # Anlamlı paragraf için minimum cümle sonu işareti

        for selector in content_selectors:
            elements_found = soup.select(selector, limit=1) # İlk eşleşeni al
            if elements_found:
                container_element = elements_found[0]
                # Konteyner içindeki paragrafları veya anlamlı div'leri topla
                # recursive=False ile sadece doğrudan alt elemanları almayı deneyebiliriz
                # paragraphs_and_divs = container_element.find_all(['p', 'div'], recursive=False, limit=35)
                # Veya tüm metni alıp temizlemeyi deneyebiliriz
                temp_content_list = []
                # Sadece <p> etiketlerini almayı deneyelim
                paragraphs = container_element.find_all('p', limit=50)
                for p_tag in paragraphs:
                    text_from_element = _clean_text(p_tag.get_text(separator=' ', strip=True))
                    # Anlamlı metin kontrolü
                    if len(text_from_element) > min_meaningful_text_len and \
                       (text_from_element.count('.') + text_from_element.count('!') + text_from_element.count('?')) >= min_sentence_indicators:
                        temp_content_list.append(text_from_element)

                # Yeterince içerik bulunduysa döngüden çık
                if len(" ".join(temp_content_list)) > 500: # Toplam karakter kontrolü
                    potential_content_parts = temp_content_list
                    content_found_flag = True
                    break # İlk başarılı seçiciden sonra dur

        # Eğer özel içerik alanları bulunamadıysa, body'nin genel metnini kullan
        if not content_found_flag:
            body_element = soup.body
            if body_element:
                body_text_content = _clean_text(body_element.get_text(separator='\n', strip=True))
                # Body metnini anlamlı parçalara ayır
                body_parts = [part.strip() for part in body_text_content.split('\n') if len(part.strip()) > min_meaningful_text_len]
                if len(" ".join(body_parts)) > 300:
                    st.toast("Özel içerik alanları bulunamadı, sayfanın genel metni kullanıldı.", icon="ℹ️")
                    potential_content_parts = body_parts[:30] # Çok uzamasın diye sınırla
                else:
                    st.toast("Sayfada anlamlı metin içeriği bulunamadı.", icon="📄")
                    return None
            else:
                 st.toast("Sayfada body etiketi veya anlamlı içerik bulunamadı.", icon="📄")
                 return None

        # Toplanan metin parçalarını birleştir ve temizle
        full_text_content = "\n\n".join(potential_content_parts)
        cleaned_text_content = _clean_text(full_text_content)

        if not cleaned_text_content:
            st.toast("Kazıma sonrası boş içerik elde edildi.", icon="📄")
            return None

        # Maksimum karakter sınırını uygula
        final_text_output = cleaned_text_content[:max_chars]
        if len(cleaned_text_content) > max_chars:
            final_text_output += "..." # Kesildiğini belirt

        st.toast(f"'{urlparse(url).netloc}' sayfasının içeriği başarıyla alındı.", icon="✅")
        return final_text_output

    except requests.exceptions.HTTPError as e_http:
        st.toast(f"⚠️ Sayfa alınırken HTTP hatası ({e_http.response.status_code}): {url}", icon='🌐')
    except requests.exceptions.Timeout:
        st.toast(f"⚠️ Sayfa alınırken zaman aşımı oluştu ({timeout}sn): {url}", icon='⏳')
    except requests.exceptions.ConnectionError:
        st.toast(f"⚠️ Sayfa bağlantı hatası (siteye ulaşılamıyor olabilir): {url}", icon='🔌')
    except requests.exceptions.RequestException as e_req:
        st.toast(f"⚠️ Sayfa alınırken genel bir ağ hatası: {e_req}", icon='🌐')
    except Exception as e_scrape:
        st.toast(f"⚠️ Sayfa içeriği işlenirken beklenmedik bir hata: {e_scrape}", icon='🔥')
        print(f"ERROR: Scraping URL '{url}' failed: {e_scrape}") # Loglama
    return None


def search_web(query: str) -> str | None:
    """Web'de (Wikipedia, DuckDuckGo) arama yapar ve kazınmış içerik döndürür."""
    st.toast(f"🔍 '{query}' için web'de arama yapılıyor...", icon="⏳")
    wikipedia.set_lang("tr")
    final_result = None

    # 1. Wikipedia Araması
    try:
        # Önce direkt eşleşme ara, sonra öneri kullan
        wp_page = wikipedia.page(query, auto_suggest=False, redirect=True)
        summary = wikipedia.summary(query, sentences=5, auto_suggest=False, redirect=True)
        final_result = f"**Wikipedia'dan ({wp_page.title}):**\n\n{_clean_text(summary)}\n\nKaynak: {wp_page.url}"
        st.toast(f"ℹ️ '{wp_page.title}' için Wikipedia'dan bilgi bulundu.", icon="✅")
        return final_result # Wikipedia sonucu yeterliyse direkt döndür
    except wikipedia.exceptions.PageError:
        st.toast(f"ℹ️ '{query}' için Wikipedia'da doğrudan sayfa bulunamadı, öneriler aranıyor...", icon="🤷")
        try:
            search_results = wikipedia.search(query, results=1)
            if search_results:
                suggested_title = search_results[0]
                wp_page = wikipedia.page(suggested_title, auto_suggest=False) # Önerilen başlığı kullan
                summary = wikipedia.summary(suggested_title, sentences=5, auto_suggest=False)
                final_result = f"**Wikipedia'dan (Öneri: {wp_page.title}):**\n\n{_clean_text(summary)}\n\nKaynak: {wp_page.url}"
                st.toast(f"ℹ️ Önerilen '{wp_page.title}' başlığı için Wikipedia'dan bilgi bulundu.", icon="✅")
                return final_result # Öneri sonucu yeterliyse döndür
            else:
                 st.toast(f"ℹ️ '{query}' için Wikipedia'da öneri de bulunamadı.", icon="🤷")
        except wikipedia.exceptions.PageError:
             st.toast(f"ℹ️ Önerilen Wikipedia sayfası yüklenemedi.", icon="🤷")
        except wikipedia.exceptions.DisambiguationError as e_disamb:
             options_text = "\n\nWikipedia'da olası başlıklar (ilk 3):\n" + "\n".join([f"- {opt}" for opt in e_disamb.options[:3]])
             final_result = f"**Wikipedia'da Birden Fazla Anlam Bulundu ({query}):**\n{str(e_disamb).splitlines()[0]}{options_text}"
             st.toast(f"Wikipedia'da '{query}' için birden fazla anlam bulundu. Daha spesifik arayın.", icon="📚")
             # Çok anlamlılık durumunda DDG aramasına devam et
        except Exception as e_wiki:
             st.toast(f"⚠️ Wikipedia araması sırasında hata: {e_wiki}", icon="🔥")
             print(f"ERROR: Wikipedia search failed for '{query}': {e_wiki}")

    # 2. DuckDuckGo Araması (Wikipedia sonucu yoksa veya yetersizse)
    ddg_url_to_scrape = None
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs_search:
            # Daha fazla sonuç alıp en iyisini seçmeyi deneyebiliriz
            results = list(ddgs_search.text(query, region='tr-tr', safesearch='moderate', max_results=5))
            if results:
                # En alakalı görünen sonucu seç (basitçe ilkini alabiliriz)
                best_res = results[0]
                snippet_text = best_res.get('body')
                temp_source_url = best_res.get('href')
                if snippet_text and temp_source_url:
                    decoded_url_source = unquote(temp_source_url)
                    ddg_source_domain = urlparse(decoded_url_source).netloc
                    st.toast(f"ℹ️ DuckDuckGo'dan '{ddg_source_domain}' için özet bulundu.", icon="🦆")
                    # Önce özeti sonuç olarak ayarla, sonra kazımayı dene
                    final_result = f"**Web Özeti (DuckDuckGo - {ddg_source_domain}):**\n\n{_clean_text(snippet_text)}\n\nKaynak: {decoded_url_source}"
                    ddg_url_to_scrape = decoded_url_source
    except Exception as e_ddg:
        st.toast(f"⚠️ DuckDuckGo araması sırasında hata: {e_ddg}", icon="🔥")
        print(f"ERROR: DuckDuckGo search failed for '{query}': {e_ddg}")

    # 3. DDG'den bulunan URL'yi Kazıma
    if ddg_url_to_scrape:
        scraped_content = scrape_url_content(ddg_url_to_scrape)
        if scraped_content:
            scraped_source_domain = urlparse(ddg_url_to_scrape).netloc
            # Kazınan içeriği, DDG özetinin yerine veya ek olarak döndür
            final_result = f"**Web Sayfasından ({scraped_source_domain}):**\n\n{scraped_content}\n\nKaynak: {ddg_url_to_scrape}"
            st.toast(f"✅ Web sayfası içeriği başarıyla alındı: {scraped_source_domain}", icon="📄")
        elif final_result: # Kazıma başarısız oldu ama DDG özeti vardı
             st.toast("ℹ️ Sayfa içeriği kazınamadı, DuckDuckGo özeti kullanılıyor.", icon="📝")
        else: # Kazıma başarısız ve DDG özeti de yoktu (nadiren olmalı)
             final_result = f"Detaylı bilgi için şu adresi ziyaret edebilirsiniz: {ddg_url_to_scrape}"


    if not final_result:
        st.toast(f"'{query}' için web'de anlamlı bir sonuç bulunamadı.", icon="❌")
        return None

    return final_result

# --- Sohbet Geçmişi Yönetimi ---

@st.cache_data(ttl=86400) # Cache süresi ayarlanabilir
def load_all_chats_cached(file_path: str = CHAT_HISTORY_FILE) -> dict:
    """Tüm sohbet geçmişlerini içeren sözlüğü dosyadan yükler."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content_read = f.read()
            if content_read and content_read.strip():
                data = json.loads(content_read)
                # Verinin beklenen formatta (dict) olduğundan emin ol
                if isinstance(data, dict):
                    # Anahtarların string olduğundan emin ol
                    return {str(k): v for k, v in data.items()}
                else:
                    st.warning(f"Geçersiz sohbet geçmişi formatı ({file_path}). Yeni yapıya geçiliyor. Eski sohbetler kaybolmuş olabilir.", icon="⚠️")
                    # Eski dosyayı yedekle
                    backup_path = f"{file_path}.backup_{int(time.time())}"
                    try: os.rename(file_path, backup_path)
                    except OSError: pass
                    return {} # Boş başlat
            else: return {} # Boş dosya
        except json.JSONDecodeError:
            st.error(f"Sohbet geçmişi dosyası ({file_path}) bozuk. Yeni bir geçmiş başlatılıyor.")
            backup_path = f"{file_path}.corrupt_{int(time.time())}"
            try: os.rename(file_path, backup_path)
            except OSError: pass
            return {}
        except Exception as e_load_hist:
            st.error(f"Sohbet geçmişi dosyası ({file_path}) yüklenirken bir hata oluştu: {e_load_hist}")
            return {}
    return {} # Dosya yoksa boş sözlük döndür

def save_all_chats(all_chats_dict: dict, file_path: str = CHAT_HISTORY_FILE):
    """Tüm sohbet geçmişlerini içeren sözlüğü dosyaya kaydeder."""
    try:
        with open(file_path, "w", encoding="utf-8") as f_save:
            json.dump(all_chats_dict, f_save, ensure_ascii=False, indent=2)
    except Exception as e_save_hist:
        st.error(f"Sohbet geçmişi kaydedilemedi: {e_save_hist}")
        print(f"ERROR: Failed to save chat history to {file_path}: {e_save_hist}")


# --- Gemini Yanıt Alma ---

def get_gemini_response_cached(prompt_text: str, chat_history_for_api: list[dict], stream_output: bool = False) -> str | object:
    """Gemini API'den yanıt alır (cache'leme burada yapılmaz, orchestrator'da yapılır)."""
    current_gemini_model = globals().get('gemini_model')
    if not current_gemini_model:
        return f"{GEMINI_ERROR_PREFIX} Gemini modeli aktif değil veya başlatılamadı."

    # API'ye gönderilecek history formatını doğrula/düzelt
    validated_history = []
    for msg in chat_history_for_api:
        role = msg.get('role')
        parts = msg.get('parts')
        # Rol ve içeriğin geçerli olduğundan emin ol
        if role in ['user', 'model'] and isinstance(parts, str) and parts.strip():
             validated_history.append({'role': role, 'parts': [parts]}) # API 'parts'ı liste bekler
        elif role in ['user', 'model'] and isinstance(parts, list) and parts and isinstance(parts[0], str):
             # Zaten doğru formatta ise doğrudan ekle
             validated_history.append(msg)
        # else: Geçersiz formatlı mesajları atla veya logla

    try:
        # Start chat ile oturum başlatma (history ile)
        chat_session = current_gemini_model.start_chat(history=validated_history)
        # Yeni mesajı gönder
        response_from_gemini = chat_session.send_message(prompt_text, stream=stream_output)

        if stream_output:
            return response_from_gemini # Stream objesini döndür
        else:
            # Yanıtın içeriğini kontrol et
            if response_from_gemini.parts:
                 return "".join(part.text for part in response_from_gemini.parts if hasattr(part, 'text'))
            else:
                # Yanıt neden boş geldi? Güvenlik, uzunluk vs.
                block_reason = getattr(response_from_gemini.prompt_feedback, 'block_reason', None)
                finish_reason = getattr(response_from_gemini.candidates[0], 'finish_reason', None) if response_from_gemini.candidates else None

                if block_reason:
                    block_message = getattr(response_from_gemini.prompt_feedback, 'block_reason_message', "Detay yok.")
                    warning_msg = f"Yanıt güvenlik nedeniyle engellendi ({block_reason}). Detay: {block_message}"
                    st.warning(warning_msg, icon="🛡️")
                    return f"{GEMINI_ERROR_PREFIX} {warning_msg}"
                elif finish_reason and finish_reason != 'STOP':
                    st.warning(f"Yanıt tam oluşturulamadı. Sebep: {finish_reason}", icon="⚠️")
                    return f"{GEMINI_ERROR_PREFIX} Yanıt tam değil ({finish_reason})."
                else:
                    st.warning(f"Gemini'dan boş veya beklenmedik bir yanıt alındı.", icon="⁉️")
                    print(f"DEBUG: Empty Gemini Response: {response_from_gemini}") # Loglama
                    return f"{GEMINI_ERROR_PREFIX} Boş veya anlaşılamayan yanıt."

    # Gemini API özel hataları
    except genai.types.BlockedPromptException as bpe_gemini:
        st.error(f"Gemini İstem Engelleme Hatası: İstem güvenlik filtrelerini tetikledi. Detay: {bpe_gemini}", icon="🛡️")
        return f"{GEMINI_ERROR_PREFIX} İstem güvenlik nedeniyle engellendi."
    except genai.types.StopCandidateException as sce_gemini:
        st.error(f"Gemini Yanıt Kesintisi: Yanıt oluşturulurken durdu. Detay: {sce_gemini}", icon="🛑")
        return f"{GEMINI_ERROR_PREFIX} Yanıt oluşturulurken kesildi."
    # Genel ağ ve API hataları
    except requests.exceptions.Timeout:
        st.error("Gemini API isteği zaman aşımına uğradı. İnternet bağlantınızı kontrol edin.", icon="⏳")
        return f"{GEMINI_ERROR_PREFIX} API zaman aşımı."
    except requests.exceptions.RequestException as req_err:
         st.error(f"Gemini API ağı hatası: {req_err}", icon="📡")
         return f"{GEMINI_ERROR_PREFIX} API ağı hatası: {req_err}"
    except Exception as e_gemini_api:
        error_message_gemini = f"Gemini API ile iletişimde hata: {e_gemini_api}"
        st.error(error_message_gemini, icon="🔥")
        print(f"ERROR: Gemini API communication failed: {type(e_gemini_api).__name__} - {e_gemini_api}") # Loglama
        # API Anahtarı hatasını spesifik olarak kontrol et
        if "API key not valid" in str(e_gemini_api):
            return f"{GEMINI_ERROR_PREFIX} Google API Anahtarı geçersiz."
        elif "RateLimitExceeded" in str(e_gemini_api) or "429" in str(e_gemini_api):
             return f"{GEMINI_ERROR_PREFIX} API kullanım limiti aşıldı. Biraz bekleyip tekrar deneyin."
        elif "Deadline Exceeded" in str(e_gemini_api) or "504" in str(e_gemini_api):
            return f"{GEMINI_ERROR_PREFIX} API isteği zaman aşımına uğradı (Sunucu tarafı)."
        return f"{GEMINI_ERROR_PREFIX} API Hatası: {str(e_gemini_api)[:150]}"


# --- Supabase Loglama ---

def log_to_supabase(table_name: str, data_to_log: dict):
    """Verilen veriyi belirtilen Supabase tablosuna loglar."""
    current_supabase_client = globals().get('supabase')
    if not current_supabase_client:
        print(f"INFO: Supabase client not available, skipping log to '{table_name}'.")
        return False # Loglama yapılamadı

    try:
        # Eksik olabilecek anahtarlar için varsayılan değerler ekle
        data_to_log.setdefault('user_name', st.session_state.get('user_name', 'Bilinmiyor'))
        data_to_log.setdefault('session_id', _get_session_id())
        data_to_log.setdefault('app_version', APP_VERSION)
        data_to_log.setdefault('chat_id', st.session_state.get('active_chat_id', 'N/A'))

        insert_result = current_supabase_client.table(table_name).insert(data_to_log).execute()

        # Supabase API v2 sonrası 'data' yerine direkt sonuç dönebilir veya hata fırlatabilir.
        # Hata kontrolünü exception handling ile yapmak daha güvenilir.
        print(f"DEBUG: Supabase insert result to '{table_name}': {insert_result}") # Başarılı loglamayı da görelim
        return True # Başarılı loglama

    except SupabaseAPIError as e_supabase_api:
        error_message = f"Supabase API hatası ('{table_name}'): {e_supabase_api.message} (Kod: {e_supabase_api.code}, Detay: {e_supabase_api.details})"
        st.toast(f"⚠️ Loglama hatası: {e_supabase_api.message}", icon="💾")
        print(f"ERROR: {error_message}")
        return False
    except Exception as e_supabase_log:
        error_message = f"Supabase '{table_name}' tablosuna loglama sırasında kritik hata: {type(e_supabase_log).__name__}: {e_supabase_log}"
        st.error(f"Loglama sırasında kritik hata oluştu! Detaylar loglarda.")
        print(f"ERROR: {error_message}")
        return False

def log_interaction(user_prompt: str, ai_response: str, response_source: str, message_id: str, chat_id: str):
    """Kullanıcı-AI etkileşimini Supabase'e loglar."""
    log_data = {
        "user_prompt": user_prompt,
        "ai_response": ai_response,
        "response_source": response_source,
        "message_id": message_id,
        "chat_id": chat_id # Hangi sohbetle ilgili olduğunu ekledik
        # Diğer bilgiler (user_name, session_id vs.) log_to_supabase içinde eklenecek
    }
    return log_to_supabase(SUPABASE_TABLE_LOGS, log_data)

def log_feedback(message_id: str, user_prompt: str, ai_response: str, feedback_type: str, comment: str = ""):
    """Kullanıcı geri bildirimini Supabase'e loglar."""
    log_data = {
        "message_id": message_id, # Hangi mesaja geri bildirim verildiği
        "user_prompt": user_prompt,
        "ai_response": ai_response,
        "feedback_type": feedback_type, # 'positive' veya 'negative'
        "comment": comment,
        # chat_id log_to_supabase içinde eklenecek
    }
    if log_to_supabase(SUPABASE_TABLE_FEEDBACK, log_data):
        st.toast("Geri bildiriminiz için teşekkür ederiz!", icon="💌")
        return True
    else:
        st.toast("Üzgünüz, geri bildiriminiz gönderilemedi.", icon="😔")
        return False


# --- Yanıt Orkestrasyonu ---

def get_hanogt_response_orchestrator(user_prompt: str, chat_history: list[dict], message_id: str, chat_id: str, use_stream:bool = False) -> tuple[str | object, str]:
    """Farklı kaynaklardan (KB, Gemini, Web) yanıt alır ve loglar."""
    ai_response = None
    response_source = "Bilinmiyor" # Yanıtın kaynağı

    # 1. Bilgi Tabanı ve Dinamik Fonksiyonlar
    kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
    if kb_resp:
        response_source = "Fonksiyonel" if user_prompt.lower() in DYNAMIC_FUNCTIONS_MAP else "Bilgi Tabanı"
        log_interaction(user_prompt, kb_resp, response_source, message_id, chat_id)
        return kb_resp, f"{APP_NAME} ({response_source})"

    # 2. Gemini Modeli
    if globals().get('gemini_model'):
        # Gemini'ye gönderilecek geçmişi hazırla (role, parts formatında)
        gemini_history = chat_history # Zaten doğru formatta olmalı

        gemini_resp = get_gemini_response_cached(user_prompt, gemini_history, stream_output=use_stream)

        if gemini_resp:
            if use_stream:
                 # Stream objesini döndür, loglama stream bittikten sonra yapılır
                 return gemini_resp, f"{APP_NAME} (Gemini Stream)"
            elif isinstance(gemini_resp, str) and not gemini_resp.startswith(GEMINI_ERROR_PREFIX):
                 # Başarılı metin yanıtı
                 response_source = "Gemini"
                 log_interaction(user_prompt, gemini_resp, response_source, message_id, chat_id)
                 return gemini_resp, f"{APP_NAME} ({response_source})"
            # else: Hata durumu veya boş yanıt, diğer yöntemlere geç
    # Gemini yoksa veya hata verdiyse buraya gelinir

    # 3. Web Araması (Gemini yanıt vermediyse veya bilgi gerektiren bir soruysa)
    # Soru formatını veya anahtar kelimeleri kontrol ederek web aramasına karar ver
    is_question = "?" in user_prompt or \
                  any(kw in user_prompt.lower() for kw in ["nedir", "kimdir", "nasıl", "ne zaman", "nerede", "bilgi", "araştır", "haber", "son durum", "açıkla"])
    if not ai_response and is_question and len(user_prompt.split()) > 2:
        web_resp = search_web(user_prompt)
        if web_resp:
            # Web yanıtının kaynağını belirle (Wikipedia, DDG, Kazıma)
            if "Wikipedia" in web_resp: response_source = "Wikipedia"
            elif "DuckDuckGo" in web_resp: response_source = "Web Özeti (DDG)"
            elif "Web Sayfasından" in web_resp: response_source = "Web Kazıma"
            else: response_source = "Web Arama"

            log_interaction(user_prompt, web_resp, response_source, message_id, chat_id)
            return web_resp, f"{APP_NAME} ({response_source})"
    elif not ai_response and is_question:
         st.toast("ℹ️ Web araması için sorgu çok kısa veya belirsiz.", icon="⏩")


    # 4. Hiçbir yerden yanıt alınamadıysa varsayılan yanıt
    if not ai_response:
        st.toast("🤔 Bu isteğiniz için uygun bir yanıt bulamadım.", icon="🤷")
        default_responses = [
            f"Üzgünüm {st.session_state.get('user_name', 'dostum')}, bu konuda yardımcı olamıyorum. Farklı sorabilir misiniz?",
            "Sorunuzu tam anlayamadım. Daha basit ifade eder misiniz?",
            "Bu konuda bilgim yok maalesef. Başka bir sorunuz var mı?",
            "Yanıt veremiyorum ama öğrenmeye devam ediyorum!",
        ]
        ai_response = random.choice(default_responses)
        response_source = "Varsayılan"
        log_interaction(user_prompt, ai_response, response_source, message_id, chat_id)
        return ai_response, f"{APP_NAME} ({response_source})"

    # Bu noktaya gelinmemeli ama fallback
    return ai_response or DEFAULT_ERROR_MESSAGE, f"{APP_NAME} ({response_source})"


# --- Yaratıcı Modüller ---

def creative_response_generator(prompt: str, length: str = "orta", style: str = "genel") -> str:
    """Yerel olarak basit yaratıcı metinler üretir."""
    templates = {
        "genel": ["Farklı bir bakış açısıyla: {}", "Hayal gücümüzle: {}", "Aklıma gelen: {}"],
        "şiirsel": ["Kalbimden dökülenler: {}", "Sözcüklerin dansı: {}", "Duyguların ritmi: {}"],
        "hikaye": ["Bir zamanlar: {}", "Perde aralanır: {}", "O gün başladı: {}"]
    }
    chosen_template = random.choice(templates.get(style, templates["genel"]))
    base_idea = generate_new_idea_creative(prompt, style) # Temel fikri üret

    # Uzunluk ayarı (çok basit)
    sentences = [s.strip() for s in base_idea.split('.') if s.strip()]
    num_sentences = len(sentences)
    if length == "kısa" and num_sentences > 1:
        base_idea = ". ".join(sentences[:max(1, num_sentences // 3)]) + "."
    elif length == "uzun" and num_sentences > 0:
        # Daha fazla detay ekle (basit tekrar veya ek fikir)
        additional_idea = generate_new_idea_creative(prompt[::-1], style) # Tersten prompt ile ek fikir
        base_idea += f"\n\nDahası, {additional_idea}"

    return chosen_template.format(base_idea)

def generate_new_idea_creative(seed_prompt: str, style:str = "genel") -> str:
    """Basitçe rastgele kelimelerle yeni fikirler üretir."""
    elements = ["zaman kristalleri", "psişik ormanlar", "rüya mimarisi", "kuantum köpüğü", "gölge enerjisi", "yankılanan anılar", "yıldız haritaları", "fraktal düşünce", "kozmik senfoni", "unutulmuş kehanetler", "eterik varlıklar"]
    actions = ["dokur", "çözer", "yansıtır", "inşa eder", "fısıldar", "dönüştürür", "keşfeder", "haritalar", "bağlantı kurar", "çağırır", "şekillendirir"]
    outcomes = ["kaderin ipliklerini", "varoluşun kodunu", "bilincin sınırlarını", "kadim sırları", "evrenin melodisini", "gerçekliğin dokusunu", "sonsuz potansiyelleri", "yeni bir çağın şafağını", "ruhun yolculuğunu"]

    prompt_words = re.findall(r'\b\w{4,}\b', seed_prompt.lower()) # 4+ harfli kelimeler
    seed_elements = random.sample(prompt_words, k=min(len(prompt_words), 2)) if prompt_words else ["gizemli", "ışık"]

    e1, a1, o1 = random.choice(elements), random.choice(actions), random.choice(outcomes)
    e2 = random.choice(elements)

    if style == "şiirsel":
        return f"{e1.capitalize()} arasında süzülürken, {seed_elements[0]} {a1}, {o1}."
    elif style == "hikaye":
        return f"{' '.join(seed_elements).capitalize()} {a1} ve {e1} kullanarak, sonunda {e2} aracılığıyla {o1} keşfeder."
    # Genel stil
    return f"{' '.join(seed_elements).capitalize()} {a1}, {e1} aracılığıyla {o1}."


def advanced_word_generator(base_word: str) -> str:
    """Verilen kelimeden veya rastgele yeni 'teknik' kelimeler türetir."""
    if not base_word or len(base_word) < 2: return "KelimatörProMax"
    vowels = "aeıioöuü"; consonants = "bcçdfgğhjklmnprsştvyz"
    # Temizleme: Sadece harfleri al
    cleaned_base = "".join(filter(str.isalpha, base_word.lower()))
    if not cleaned_base: return "SözcükMimar"

    prefixes = ["bio", "krono", "psiko", "tera", "neo", "mega", "nano", "astro", "poli", "eko", "meta", "trans", "ultra", "omni", "xeno", "kripto", "holo", "quantum", "neuro"]
    suffixes = ["genez", "sfer", "nomi", "tek", "loji", "tronik", "morf", "vers", "dinamik", "matik", "kinezis", "skop", "grafi", "mant", "krom", "faz", "sentez", "nium", "oid"]

    # Kelimenin bir kısmını veya rastgele harfleri kullan
    if len(cleaned_base) > 3 and random.random() < 0.7:
        start = random.randint(0, max(0, len(cleaned_base) - 3))
        core = cleaned_base[start : start + random.randint(2, 4)]
    else: # Rastgele harflerle çekirdek oluştur
        core_len = random.randint(3, 5)
        core_chars = [random.choice(consonants if i % 2 == (random.random() > 0.5) else vowels) for i in range(core_len)]
        core = "".join(core_chars)

    # Ön ek ve son ek ekle (rastgele)
    new_word = core
    if random.random() > 0.3: new_word = random.choice(prefixes) + new_word
    if random.random() > 0.3: new_word += random.choice(suffixes)

    # Çok kısaysa veya aynı kaldıysa sonuna rastgele ek yap
    if len(new_word) < 5 or new_word == core:
        new_word += random.choice(suffixes) if random.random() > 0.5 else random.choice(vowels)

    return new_word.capitalize()

# --- Görsel Oluşturucu ---

def generate_prompt_influenced_image(prompt: str) -> Image.Image:
    """Prompt'taki anahtar kelimelere göre basit, kural tabanlı bir görsel oluşturur."""
    width, height = 512, 512
    prompt_lower = prompt.lower()

    # Tema tanımları: Anahtar kelime -> {arka plan renkleri, şekiller}
    themes = {
        "güneş": {"bg": [(255, 230, 150), (255, 160, 0)], "shapes": [{"type": "circle", "color": (255, 255, 0, 220), "pos": (0.25, 0.25), "size": 0.2}]},
        "ay": {"bg": [(10, 10, 50), (40, 40, 100)], "shapes": [{"type": "circle", "color": (240, 240, 240, 200), "pos": (0.75, 0.2), "size": 0.15}]},
        "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []},
        "bulut": {"bg": None, "shapes": [{"type": "ellipse", "color": (255, 255, 255, 180), "pos": (random.uniform(0.2, 0.8), random.uniform(0.1, 0.4)), "size_wh": (random.uniform(0.15, 0.35), random.uniform(0.08, 0.15))} for _ in range(random.randint(2, 4))]},
        "deniz": {"bg": [(0, 105, 148), (0, 0, 100)], "shapes": [{"type": "rectangle", "color": (60,120,180, 150), "pos": (0.5, 0.75), "size_wh": (1.0, 0.5)}]},
        "orman": {"bg": [(34, 139, 34), (0, 100, 0)], "shapes": [{"type": "triangle", "color": (random.randint(0,30),random.randint(70,100),random.randint(0,30),200), "pos": (random.uniform(0.1,0.9), random.uniform(0.55, 0.85)), "size": random.uniform(0.08, 0.25)} for _ in range(random.randint(7, 12))]},
        "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [{"type": "rectangle", "color": (139, 69, 19, 255), "pos": (random.uniform(0.2, 0.8), 0.75), "size_wh": (0.06, 0.4)}, {"type": "ellipse", "color": (34, 139, 34, 200), "pos": (lambda x: x[0])(st.session_state.get('last_tree_pos', (0.5,0.75))), 0.45), "size_wh": (0.3, 0.25)}]}, # Lambda ile pozisyonu sakla/kullan? Veya basitçe rastgele
        "dağ": {"bg": [(200,200,200), (100,100,100)], "shapes": [{"type": "polygon", "color": (random.randint(130,170),random.randint(130,170),random.randint(130,170),230), "points": [(random.uniform(0.1,0.4),0.85),(0.5,random.uniform(0.1,0.4)),(random.uniform(0.6,0.9),0.85)]} for _ in range(random.randint(1,3))]},
        "şehir": {"bg": [(100,100,120), (50,50,70)], "shapes": [{"type":"rectangle", "color":(random.randint(60,100),random.randint(60,100),random.randint(70,110),random.randint(180,220)), "pos":(random.uniform(0.1,0.9), random.uniform(0.4, 0.85)), "size_wh": (random.uniform(0.04,0.15), random.uniform(0.15,0.65))} for _ in range(random.randint(8,15))]},
        "kar": {"bg": None, "shapes": [{"type": "circle", "color": (255, 255, 255, 150), "pos": (random.random(), random.random()), "size": 0.005} for _ in range(100)]},
        "yıldız": {"bg": None, "shapes": [{"type": "circle", "color": (255, 255, 200, 200), "pos": (random.random(), random.uniform(0, 0.5)), "size": 0.003} for _ in range(70)]},

    }
    # Varsayılan arka plan
    bg_color1 = (random.randint(30, 120), random.randint(30, 120), random.randint(30, 120))
    bg_color2 = (random.randint(120, 220), random.randint(120, 220), random.randint(120, 220))
    shapes_to_draw = []
    themes_applied_count = 0

    # Prompt'taki anahtar kelimelere göre temaları uygula
    for keyword, theme in themes.items():
        if keyword in prompt_lower:
            if theme["bg"] and themes_applied_count == 0: # Sadece ilk eşleşen temanın BG'sini al
                bg_color1, bg_color2 = theme["bg"]
            shapes_to_draw.extend(theme["shapes"])
            themes_applied_count += 1

    # Görsel tuvalini oluştur
    image = Image.new('RGBA', (width, height), (0,0,0,0)) # Şeffaf başla
    draw = ImageDraw.Draw(image)

    # Arka planı çiz (gradient)
    for y in range(height):
        ratio = y / height
        r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio)
        g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio)
        b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255)) # Alpha 255 (opak)

    # Şekilleri çiz
    for shape in shapes_to_draw:
        try: # Şekil çiziminde hata olursa atla
            s_type = shape["type"]
            s_color = shape["color"]
            s_pos_ratio = shape.get("pos") # Pos bazı şekillerde olmayabilir (polygon)
            s_outline = (0,0,0,50) if len(s_color) == 4 and s_color[3] < 250 else None # Hafif outline

            if s_pos_ratio:
                 cx = int(s_pos_ratio[0] * width)
                 cy = int(s_pos_ratio[1] * height)

            if s_type == "circle":
                radius = int(shape["size"] * min(width, height) / 2)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=s_color, outline=s_outline)
            elif s_type == "rectangle":
                w_ratio, h_ratio = shape["size_wh"]
                w_px = int(w_ratio * width)
                h_px = int(h_ratio * height)
                draw.rectangle((cx - w_px // 2, cy - h_px // 2, cx + w_px // 2, cy + h_px // 2), fill=s_color, outline=s_outline)
            elif s_type == "ellipse":
                 w_ratio, h_ratio = shape["size_wh"]
                 w_px = int(w_ratio * width)
                 h_px = int(h_ratio * height)
                 draw.ellipse((cx - w_px // 2, cy - h_px // 2, cx + w_px // 2, cy + h_px // 2), fill=s_color, outline=s_outline)
            elif s_type == "triangle": # Basit eşkenar üçgen
                 size_px = int(shape["size"] * min(width, height))
                 # Tepe noktası yukarıda olacak şekilde
                 p1 = (cx, cy - int(size_px * 0.577)) # Üst
                 p2 = (cx - size_px // 2, cy + int(size_px * 0.288)) # Sol alt
                 p3 = (cx + size_px // 2, cy + int(size_px * 0.288)) # Sağ alt
                 draw.polygon([p1, p2, p3], fill=s_color, outline=s_outline)
            elif s_type == "polygon": # Noktaları verilen poligon
                 points_ratio = shape["points"]
                 points_px = [(int(p[0]*width), int(p[1]*height)) for p in points_ratio]
                 draw.polygon(points_px, fill=s_color, outline=s_outline)
        except Exception as e_draw_shape:
             print(f"DEBUG: Error drawing shape {shape}: {e_draw_shape}")
             continue # Hatalı şekli atla

    # Eğer hiç tema uygulanmadıysa rastgele şekiller çiz
    if themes_applied_count == 0:
        for _ in range(random.randint(4, 7)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            shape_color = (random.randint(50, 250), random.randint(50, 250), random.randint(50, 250), random.randint(150, 220))
            if random.random() > 0.5:
                radius = random.randint(20, 70)
                draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=shape_color)
            else:
                rw, rh = random.randint(30, 100), random.randint(30, 100)
                draw.rectangle((x1 - rw // 2, y1 - rh // 2, x1 + rw // 2, y1 + rh // 2), fill=shape_color)

    # Görselin altına prompt metnini yazdır (opsiyonel)
    try:
        font_size = max(14, min(28, int(width / (len(prompt) * 0.3 + 10))))
        font = None
        if os.path.exists(FONT_FILE):
            try: font = ImageFont.truetype(FONT_FILE, font_size)
            except IOError: st.toast(f"Font dosyası '{FONT_FILE}' yüklenemedi.", icon="⚠️")
        if not font: font = ImageFont.load_default() # Varsayılan font (boyut ayarlanamaz)

        text_to_write = prompt[:80] # Metni kısalt
        # Metin boyutunu hesapla (yeni Pillow versiyonları için)
        if hasattr(draw, 'textbbox'):
             bbox = draw.textbbox((0, 0), text_to_write, font=font, anchor="lt")
             text_width = bbox[2] - bbox[0]
             text_height = bbox[3] - bbox[1]
        else: # Eski versiyonlar için fallback
             text_width, text_height = draw.textsize(text_to_write, font=font)

        # Metin pozisyonu (alt ortaya yakın)
        text_x = (width - text_width) / 2
        text_y = height * 0.95 - text_height # Biraz daha aşağıda
        # Gölge efekti için önce siyah sonra beyaz yazdır
        draw.text((text_x + 1, text_y + 1), text_to_write, font=font, fill=(0,0,0,150))
        draw.text((text_x, text_y), text_to_write, font=font, fill=(255,255,255,230))
    except Exception as e_font:
        st.toast(f"Görsel üzerine metin yazdırılamadı: {e_font}", icon="📝")

    # RGBA'dan RGB'ye çevir (çoğu format için gerekli)
    return image.convert("RGB")


# --- Session State Başlatma ---
def initialize_session_state():
    """Session State için varsayılan değerleri ayarlar."""
    defaults = {
        'all_chats': {}, # Tüm sohbetler: {chat_id: [{'role': 'user'/'model', 'parts': '...', 'sender_display': '...'}, ...], ...}
        'active_chat_id': None, # Aktif sohbetin ID'si
        'next_chat_id_counter': 0, # Yeni sohbet ID'leri için sayaç
        'app_mode': "Yazılı Sohbet",
        'user_name': None,
        'user_avatar_bytes': None,
        'show_main_app': False, # Giriş yapılıp yapılmadığı
        'greeting_message_shown': False, # Karşılama mesajı gösterildi mi?
        'tts_enabled': True, # Metin okuma açık mı?
        'gemini_stream_enabled': True, # Gemini stream açık mı?
        'gemini_temperature': 0.7,
        'gemini_top_p': 0.95,
        'gemini_top_k': 40,
        'gemini_max_tokens': 4096,
        'gemini_model_name': 'gemini-1.5-flash-latest',
        'message_id_counter': 0, # Genel mesaj sayacı (loglama için)
        'last_ai_response_for_feedback': None, # Feedback formu için AI yanıtı
        'last_user_prompt_for_feedback': None, # Feedback formu için kullanıcı istemi
        'current_message_id_for_feedback': None, # Feedback verilen mesajın unique ID'si
        'feedback_comment_input': "", # Feedback yorumu
        'show_feedback_comment_form': False, # Feedback formu görünür mü?
        'session_id': str(uuid.uuid4()), # Tarayıcı oturum ID'si
        'last_feedback_type': 'positive', # Son feedback seçimi (UI için)
        'models_initialized': False # Modeller yüklendi mi?
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state() # Session state'i başlat

# --- Modelleri ve İstemcileri Başlatma (Sadece ilk çalıştırmada) ---
if not st.session_state.models_initialized:
    print("INFO: Initializing models and clients...")
    # Modelleri global değişkenlere ata
    gemini_model = initialize_gemini_model()
    supabase = init_supabase_client_cached()
    tts_engine = init_tts_engine_cached()

    # Sohbet geçmişini yükle
    st.session_state.all_chats = load_all_chats_cached()
    # Başlangıçta aktif sohbeti belirle
    if not st.session_state.active_chat_id and st.session_state.all_chats:
        # En son (veya ilk) sohbeti aktif yap
        st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[-1] # En sonuncuyu aktif yapalım
    elif not st.session_state.all_chats:
         # Hiç sohbet yoksa aktif ID None kalsın
         st.session_state.active_chat_id = None

    # KB'yi kullanıcı adıyla yükle (eğer kullanıcı adı varsa)
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)

    st.session_state.models_initialized = True
    print("INFO: Models and clients initialized.")
else:
    # Modeller zaten yüklendiyse global değişkenlerden al (rerun sonrası)
    gemini_model = globals().get('gemini_model')
    supabase = globals().get('supabase')
    tts_engine = globals().get('tts_engine')
    # KB'nin güncel olduğundan emin ol (kullanıcı adı değişirse diye)
    user_greeting_name = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting_name)


# Hata mesajlarını globalden al (varsa)
gemini_init_error = globals().get('gemini_init_error_global')
supabase_error = globals().get('supabase_error_global')
tts_init_error = globals().get('tts_init_error_global')

# Kullanıcı giriş yapmışsa ana uygulamayı göster
if st.session_state.user_name and not st.session_state.show_main_app:
    st.session_state.show_main_app = True

# --- ARAYÜZ BÖLÜMLERİ ---

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
                # KNOWLEDGE_BASE globalde yeniden yüklenecek
                st.toast("Adınız güncellendi!", icon="✏️"); st.rerun()
        with col2:
            if st.session_state.user_avatar_bytes:
                st.image(st.session_state.user_avatar_bytes, width=60, use_column_width='auto')
                if st.button("🗑️", key="remove_avatar_main_button", help="Avatarı kaldır", use_container_width=True):
                    st.session_state.user_avatar_bytes = None
                    st.toast("Avatar kaldırıldı.", icon="🗑️"); st.rerun()
            else:
                 st.caption("Avatar Yok")

        uploaded_avatar_file = st.file_uploader("Yeni Avatar Yükle (PNG, JPG - Maks 2MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader_main_file", label_visibility="collapsed")
        if uploaded_avatar_file:
            if uploaded_avatar_file.size > 2 * 1024 * 1024:
                st.error("Dosya boyutu 2MB'den büyük!", icon=" oversized_file:")
            else:
                st.session_state.user_avatar_bytes = uploaded_avatar_file.getvalue()
                st.toast("Avatar güncellendi!", icon="🖼️"); st.rerun()
        st.caption("Avatar sadece bu oturumda saklanır.")

        st.divider()
        st.subheader("🤖 Yapay Zeka ve Arayüz")
        # Toggle butonları yan yana koyalım
        tcol1, tcol2 = st.columns(2)
        with tcol1:
             current_tts_engine_settings = globals().get('tts_engine')
             st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not current_tts_engine_settings, help="AI yanıtlarının sesli okunmasını aç/kapat.")
        with tcol2:
             st.session_state.gemini_stream_enabled = st.toggle("Yanıt Akışı (Stream)", value=st.session_state.gemini_stream_enabled, help="Yanıtların kelime kelime gelmesini aç/kapat.")

        # --- Hanogt AI (Gemini) Gelişmiş Yapılandırma ---
        st.markdown("---")
        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma")
        # Selectbox ve sliderlar için kolonlar
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.session_state.gemini_model_name = st.selectbox(
                "AI Modeli:",
                ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'],
                index=0 if st.session_state.gemini_model_name == 'gemini-1.5-flash-latest' else 1,
                key="gemini_model_selector_main",
                help="Farklı modellerin yetenekleri/maliyetleri değişebilir."
            )
            st.session_state.gemini_temperature = st.slider("Sıcaklık:", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="gemini_temp_slider_main", help="Yaratıcılık (0=Kesin, 1=Yaratıcı)")
            st.session_state.gemini_max_tokens = st.slider("Maks Token:", 256, 8192, st.session_state.gemini_max_tokens, 128, key="gemini_max_tokens_slider_main", help="Max yanıt uzunluğu")

        with gcol2:
            st.session_state.gemini_top_k = st.slider("Top K:", 1, 100, st.session_state.gemini_top_k, 1, key="gemini_top_k_slider_main", help="Kelime Seçim Çeşitliliği")
            st.session_state.gemini_top_p = st.slider("Top P:", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="gemini_top_p_slider_main", help="Kelime Seçim Odaklılığı")

            # Ayarları Uygula Butonu (sağ alt köşede)
            if st.button("⚙️ AI Ayarlarını Uygula", key="reload_gemini_settings_main_btn", use_container_width=True, type="primary", help="Seçili AI modelini ve parametreleri yeniden yükler."):
                global gemini_model
                st.spinner("AI modeli yeniden başlatılıyor...")
                gemini_model = initialize_gemini_model() # Ayarlarla modeli yeniden başlat
                if not gemini_model:
                     st.error("AI modeli yüklenemedi. Lütfen API anahtarınızı ve ayarları kontrol edin.")
                # Başarılı toast initialize_gemini_model içinde veriliyor
                st.rerun() # Arayüzü yenile

        # --- Geçmiş Yönetimi ---
        st.divider()
        st.subheader("🧼 Geçmiş Yönetimi")
        if st.button("🧹 TÜM Sohbet Geçmişini Sil", use_container_width=True, type="secondary", key="clear_all_history_main_btn", help="Dikkat! Kaydedilmiş tüm sohbetleri siler."):
            if st.session_state.all_chats:
                # Kullanıcıdan onay isteyelim (daha güvenli)
                # confirmed = st.confirm("Emin misiniz? Bu işlem geri alınamaz!", key="confirm_delete_all")
                # if confirmed: ... (confirm widget'ı expander içinde sorun çıkarabilir)
                st.session_state.all_chats = {}
                st.session_state.active_chat_id = None
                save_all_chats({}) # Dosyayı boşalt
                st.toast("TÜM sohbet geçmişi silindi!", icon="🗑️"); st.rerun()
            else:
                st.toast("Sohbet geçmişi zaten boş.", icon="ℹ️")


def display_chat_list_and_about(left_column):
    """Sol kolonda sohbet listesini ve Hakkında bölümünü gösterir."""
    with left_column:
        st.markdown("#### Sohbetler")

        if st.button("➕ Yeni Sohbet", use_container_width=True, key="new_chat_button"):
            st.session_state.next_chat_id_counter += 1
            # Daha okunaklı ID: Sayaç + Zaman damgası
            ts = int(time.time())
            new_chat_id = f"chat_{st.session_state.next_chat_id_counter}_{ts}"
            st.session_state.all_chats[new_chat_id] = [] # Yeni boş sohbet listesi ekle
            st.session_state.active_chat_id = new_chat_id # Yeni sohbeti aktif yap
            save_all_chats(st.session_state.all_chats) # Değişikliği kaydet
            st.rerun()

        st.markdown("---")

        # Sohbet Listesi için scrollable container
        chat_list_container = st.container(height=400, border=False)
        with chat_list_container:
            # Sohbetleri listele (en yeniden en eskiye doğru)
            chat_ids_sorted = sorted(st.session_state.all_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)

            if not chat_ids_sorted:
                st.caption("Henüz bir sohbet yok.")
            else:
                active_chat_id = st.session_state.get('active_chat_id')
                for chat_id in chat_ids_sorted:
                    chat_history = st.session_state.all_chats.get(chat_id, [])
                    # Başlık oluşturma
                    first_user_msg_obj = next((msg for msg in chat_history if msg.get('role') == 'user'), None)
                    chat_title = f"Sohbet {chat_id.split('_')[1]}" # Varsayılan başlık
                    if first_user_msg_obj:
                         first_message = first_user_msg_obj.get('parts', '')
                         chat_title = first_message[:35] + ("..." if len(first_message) > 35 else "")
                    elif chat_history : # Sohbet var ama kullanıcı mesajı yoksa (nadiren)
                        chat_title = "Başlıksız Sohbet"


                    list_cols = st.columns([0.8, 0.2])
                    button_type = "primary" if active_chat_id == chat_id else "secondary"
                    # Sohbet seçme butonu
                    if list_cols[0].button(chat_title, key=f"select_chat_{chat_id}", use_container_width=True, type=button_type, help=f"'{chat_title}' sohbetini aç"):
                        if active_chat_id != chat_id:
                            st.session_state.active_chat_id = chat_id
                            st.rerun()
                    # Sohbet silme butonu
                    if list_cols[1].button("❌", key=f"delete_chat_{chat_id}", use_container_width=True, help=f"'{chat_title}' sohbetini sil", type="secondary"):
                         if chat_id in st.session_state.all_chats:
                             del st.session_state.all_chats[chat_id]
                             if active_chat_id == chat_id:
                                 # Silinen sohbet aktifse, başka birini aktif yap
                                 remaining_chats = sorted(st.session_state.all_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
                                 st.session_state.active_chat_id = remaining_chats[0] if remaining_chats else None
                             save_all_chats(st.session_state.all_chats)
                             st.toast(f"'{chat_title}' sohbeti silindi.", icon="🗑️")
                             st.rerun()

        # Hakkında Bölümü (Scrollable alanın dışında)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Uygulama Hakkında", expanded=False): # Başlangıçta kapalı
            st.markdown(f"""
            **{APP_NAME} v{APP_VERSION}**
            Yapay zeka destekli kişisel sohbet asistanınız.
            Geliştirici: **Hanogt** ([GitHub](https://github.com/Hanogt))

            Kullanılan Teknolojiler: Streamlit, Google Gemini API, Python Kütüphaneleri (Requests, BS4, Wikipedia, DDGS, TTS, SR vb.)
            Loglama/Geri Bildirim: Supabase (isteğe bağlı)
            © 2024-{CURRENT_YEAR}
            """)
            st.caption(f"Oturum ID: {_get_session_id()[:8]}...")


def display_chat_message_with_feedback(message_data: dict, message_index: int, chat_id: str):
    """Tek bir sohbet mesajını formatlar ve gösterir, AI için feedback butonu ekler."""
    role = message_data.get('role', 'model') # user veya model
    message_content = message_data.get('parts', '')
    # AI mesajları için özel gönderici adı (eğer history'de varsa)
    sender_display_name = message_data.get('sender_display', APP_NAME if role == 'model' else st.session_state.user_name)
    is_user = (role == 'user')

    # Avatar belirleme
    avatar_icon = "🧑" # Varsayılan kullanıcı
    if is_user:
        if st.session_state.user_avatar_bytes:
            try: avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes))
            except Exception: pass # Hata olursa varsayılan kalsın
    else: # AI mesajı ikonları
        if "Gemini" in sender_display_name: avatar_icon = "✨"
        elif "Web" in sender_display_name or "Wikipedia" in sender_display_name: avatar_icon = "🌐"
        elif "Bilgi Tabanı" in sender_display_name or "Fonksiyonel" in sender_display_name: avatar_icon = "📚"
        else: avatar_icon = "🤖" # Varsayılan AI

    with st.chat_message(role, avatar=avatar_icon):
        # Mesaj içeriğini formatlı göster (Markdown, Kod Blokları)
        if "```" in message_content:
            parts = message_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1: # Kod bloğu
                    lang_match = re.match(r"(\w+)\n", part)
                    language = lang_match.group(1) if lang_match else None
                    code = part[len(language)+1:] if language and part.startswith(language+"\n") else part
                    st.code(code, language=language)
                    # Kod kopyalama butonu
                    code_copy_key = f"copy_code_{chat_id}_{message_index}_{i}"
                    if st.button("📋 Kopyala", key=code_copy_key, help="Kodu kopyala"):
                        st.write_to_clipboard(code)
                        st.toast("Kod kopyalandı!", icon="✅")
                elif part.strip(): # Boş metin kısımlarını gösterme
                    st.markdown(part, unsafe_allow_html=True)
        elif message_content.strip(): # Boş mesajları gösterme
            st.markdown(message_content, unsafe_allow_html=True)
        else:
             st.caption("[Boş mesaj]") # İçerik yoksa belirt

        # Sadece AI mesajları için eylem butonları (sağ alt)
        if not is_user and message_content.strip():
             # Butonları sağa yaslamak için kolon kullanabiliriz veya direkt ekleyebiliriz
             st.write("") # Butonların altına biraz boşluk
             cols = st.columns([0.85, 0.075, 0.075]) # Alan ayarı
             with cols[1]: # Seslendir butonu
                 tts_key = f"tts_{chat_id}_{message_index}"
                 current_tts_engine = globals().get('tts_engine')
                 if st.session_state.tts_enabled and current_tts_engine:
                     if st.button("🔊", key=tts_key, help="Mesajı oku", use_container_width=True):
                         speak(message_content)
             with cols[2]: # Feedback butonu
                 feedback_key = f"feedback_{chat_id}_{message_index}"
                 if st.button("✍️", key=feedback_key, help="Geri bildirim ver", use_container_width=True):
                     st.session_state.current_message_id_for_feedback = f"{chat_id}_{message_index}"
                     # Önceki mesajı bul (varsa ve kullanıcıysa)
                     if message_index > 0 and st.session_state.all_chats[chat_id][message_index-1]['role'] == 'user':
                          st.session_state.last_user_prompt_for_feedback = st.session_state.all_chats[chat_id][message_index-1]['parts']
                     else:
                          st.session_state.last_user_prompt_for_feedback = "[Önceki istem bulunamadı]"
                     st.session_state.last_ai_response_for_feedback = message_content
                     st.session_state.show_feedback_comment_form = True # Formu aç
                     st.session_state.feedback_comment_input = "" # Yorumu sıfırla
                     st.rerun() # Formu göstermek için


def display_feedback_form_if_active():
    """Aktifse geri bildirim formunu gösterir."""
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        form_key = f"feedback_form_{st.session_state.current_message_id_for_feedback}"
        with st.form(key=form_key):
            st.markdown(f"#### Yanıt Hakkında Geri Bildirim")
            st.caption(f"**İstem:** `{st.session_state.last_user_prompt_for_feedback[:80]}...`")
            st.caption(f"**Yanıt:** `{st.session_state.last_ai_response_for_feedback[:80]}...`")

            feedback_type = st.radio(
                "Değerlendirme:",
                ["👍 Beğendim", "👎 Beğenmedim"],
                horizontal=True, key=f"feedback_type_{form_key}",
                index=0 if st.session_state.get('last_feedback_type', 'positive') == 'positive' else 1
            )
            comment = st.text_area(
                "Yorum (isteğe bağlı):",
                value=st.session_state.get('feedback_comment_input', ""),
                key=f"feedback_comment_{form_key}", height=100,
                placeholder="Yanıt neden iyi veya kötüydü?"
            )
            st.session_state.feedback_comment_input = comment # Değeri anlık sakla

            # Form butonları yan yana
            s_col, c_col = st.columns(2)
            with s_col:
                 submitted = st.form_submit_button("✅ Gönder", use_container_width=True, type="primary")
            with c_col:
                 cancelled = st.form_submit_button("❌ Vazgeç", use_container_width=True)

            if submitted:
                parsed_feedback = "positive" if feedback_type == "👍 Beğendim" else "negative"
                st.session_state.last_feedback_type = parsed_feedback
                log_feedback(
                    st.session_state.current_message_id_for_feedback,
                    st.session_state.last_user_prompt_for_feedback,
                    st.session_state.last_ai_response_for_feedback,
                    parsed_feedback,
                    comment
                )
                # Formu kapat ve state'i sıfırla
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.feedback_comment_input = ""
                st.rerun()
            elif cancelled:
                # Formu kapat ve state'i sıfırla
                st.session_state.show_feedback_comment_form = False
                st.session_state.current_message_id_for_feedback = None
                st.session_state.feedback_comment_input = ""
                st.rerun()
        st.markdown("---")


def display_chat_interface_main(main_column_ref):
    """Ana sohbet arayüzünü sağ kolonda yönetir."""
    with main_column_ref:
        active_chat_id = st.session_state.get('active_chat_id')

        if active_chat_id is None:
            st.info("💬 Başlamak için sol menüden **'➕ Yeni Sohbet'** butonuna tıklayın veya mevcut bir sohbeti seçin.", icon="👈")
            return

        # Aktif sohbet geçmişini al (varsa)
        current_chat_history = st.session_state.all_chats.get(active_chat_id, [])

        # Mesajları göstermek için konteyner (scrollable)
        chat_container = st.container(height=550, border=False) # Yüksekliği ayarla
        with chat_container:
            if not current_chat_history:
                st.info(f"Merhaba {st.session_state.user_name}! Yeni sohbetinize hoş geldiniz.", icon="👋")

            # Mesajları döngüyle göster
            for i, msg_data in enumerate(current_chat_history):
                 display_chat_message_with_feedback(msg_data, i, active_chat_id)

        # Geri bildirim formu aktifse göster (konteyner dışında)
        display_feedback_form_if_active()

        # Sohbet giriş alanı
        prompt_placeholder = f"{st.session_state.user_name}, ne düşünüyorsun?"
        user_prompt = st.chat_input(prompt_placeholder, key=f"chat_input_{active_chat_id}")

        if user_prompt:
            # Kullanıcı mesajını aktif sohbete ekle (yeni format)
            user_message_data = {'role': 'user', 'parts': user_prompt}
            st.session_state.all_chats[active_chat_id].append(user_message_data)
            save_all_chats(st.session_state.all_chats) # Kaydet

            # AI yanıtı için hazırlan
            message_unique_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"
            st.session_state.message_id_counter += 1

            # Gemini'ye gönderilecek geçmişi al (son N mesaj)
            history_limit = 20 # Son 20 mesajı (10 çift) gönderelim
            history_for_model = st.session_state.all_chats[active_chat_id][-history_limit:-1] # Yeni eklenen hariç

            # Yanıt alınırken yer tutucu göster
            with st.chat_message("assistant", avatar="⏳"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("🧠 _Düşünüyorum..._")

            # Orkestratör ile AI yanıtını al
            ai_response, ai_sender_name = get_hanogt_response_orchestrator(
                user_prompt,
                history_for_model,
                message_unique_id,
                active_chat_id,
                use_stream=st.session_state.gemini_stream_enabled
            )

            # Yanıtı işle ve ekle
            final_ai_message = ""
            if st.session_state.gemini_stream_enabled and "Stream" in ai_sender_name:
                # Stream yanıtını işle
                stream_container = thinking_placeholder # Yer tutucuda göster
                streamed_text = ""
                try:
                    for chunk in ai_response: # ai_response stream objesi olmalı
                        if chunk.parts:
                             text_part = "".join(p.text for p in chunk.parts if hasattr(p, 'text'))
                             streamed_text += text_part
                             stream_container.markdown(streamed_text + "▌")
                             time.sleep(0.01) # Çok hızlı olmasın
                    stream_container.markdown(streamed_text) # Son halini göster
                    final_ai_message = streamed_text
                    # Stream bittikten sonra logla
                    log_interaction(user_prompt, final_ai_message, "Gemini Stream", message_unique_id, active_chat_id)
                except Exception as e_stream:
                    error_msg = f"Stream hatası: {e_stream}"
                    stream_container.error(error_msg)
                    final_ai_message = error_msg
                    ai_sender_name = f"{APP_NAME} (Stream Hatası)"
                    log_interaction(user_prompt, final_ai_message, "Stream Hatası", message_unique_id, active_chat_id)
            else:
                # Normal (stream olmayan) yanıt
                thinking_placeholder.empty() # Yer tutucuyu kaldır
                final_ai_message = str(ai_response)
                # Loglama zaten orchestrator içinde yapıldı

            # AI yanıtını aktif sohbete ekle (yeni format)
            ai_message_data = {
                'role': 'model',
                'parts': final_ai_message,
                'sender_display': ai_sender_name # Kaynağı da saklayalım
            }
            st.session_state.all_chats[active_chat_id].append(ai_message_data)
            save_all_chats(st.session_state.all_chats) # Kaydet

            # TTS (Stream olmayanlar için)
            if st.session_state.tts_enabled and globals().get('tts_engine') and \
               isinstance(final_ai_message, str) and "Stream" not in ai_sender_name:
                speak(final_ai_message)

            # Sayfayı yenileyerek yeni mesajları göster
            st.rerun()


# --- UYGULAMA ANA AKIŞI ---

# Başlık
st.markdown(f"<h1 style='text-align: center; color: #0078D4;'>{APP_NAME} {APP_VERSION}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-style: italic; color: #555;'>Yapay zeka destekli kişisel asistanınız</p>", unsafe_allow_html=True)

# Başlatma Hatalarını Göster
if gemini_init_error: st.error(gemini_init_error, icon="🛑")
if supabase_error: st.warning(supabase_error, icon="🧱") # Warning olarak gösterelim
if tts_init_error and st.session_state.tts_enabled: st.toast(tts_init_error, icon="🔇")

# --- Kullanıcı Giriş Ekranı ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce...")
    login_cols = st.columns([0.2, 0.6, 0.2])
    with login_cols[1]:
        with st.form("login_form"):
            user_name = st.text_input("Size nasıl hitap edelim?", placeholder="İsminiz veya takma adınız...", key="login_name_input")
            submitted = st.form_submit_button("✨ Başla", use_container_width=True, type="primary")
            if submitted:
                if user_name and user_name.strip():
                    st.session_state.user_name = user_name.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False # Karşılama gösterilsin
                    # Kullanıcı adı değişti, KB'yi yeniden yükle/cache'i temizle
                    load_knowledge_from_file.clear()
                    # Aktif sohbeti belirle (varsa)
                    if not st.session_state.active_chat_id and st.session_state.all_chats:
                         st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[-1]
                    st.rerun()
                else:
                    st.error("Lütfen geçerli bir isim girin.")
else:
    # --- Ana Uygulama Alanı ---
    # Karşılama (sadece bir kere)
    if not st.session_state.greeting_message_shown:
        st.success(f"Hoş geldiniz {st.session_state.user_name}! Size nasıl yardımcı olabilirim?", icon="🎉")
        st.session_state.greeting_message_shown = True

    # Ana Layout (Sol: Sohbet Listesi+Hakkında, Sağ: Ayarlar+Modlar+İçerik)
    left_column, main_column = st.columns([1, 3]) # Oran ayarlanabilir

    # Sol Kolon
    display_chat_list_and_about(left_column)

    # Sağ (Ana) Kolon
    with main_column:
        # Ayarlar Bölümü (Expander)
        display_settings_section()

        # Mod Seçimi
        st.markdown("#### Uygulama Modu")
        modes = { "Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya)": "🎤", "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️" }
        mode_keys = list(modes.keys())
        try: # Aktif modun index'ini bul, bulamazsa 0 kullan
             current_mode_index = mode_keys.index(st.session_state.app_mode)
        except ValueError:
             current_mode_index = 0
             st.session_state.app_mode = mode_keys[0] # Hata varsa varsayılana dön

        selected_mode = st.radio(
            "Mod Seçin:", options=mode_keys, index=current_mode_index,
            format_func=lambda k: f"{modes[k]} {k}",
            horizontal=True, label_visibility="collapsed", key="app_mode_radio"
        )
        if selected_mode != st.session_state.app_mode:
            st.session_state.app_mode = selected_mode
            st.rerun() # Mod değişince yeniden çalıştır
        st.markdown("<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

        # --- Seçilen Modun İçeriği ---
        current_mode = st.session_state.app_mode

        if current_mode == "Yazılı Sohbet":
            display_chat_interface_main(main_column)

        elif current_mode == "Sesli Sohbet (Dosya)":
            st.info("Yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası (WAV, MP3, OGG, FLAC vb.) yükleyin.", icon="📢")
            audio_file = st.file_uploader("Ses Dosyası:", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed", key="audio_uploader")
            if audio_file:
                st.audio(audio_file, format=audio_file.type)
                active_chat_id = st.session_state.get('active_chat_id')
                if not active_chat_id:
                     st.warning("Lütfen önce sol menüden bir sohbet seçin veya yeni bir sohbet başlatın.", icon="⚠️")
                else:
                     audio_text = None
                     with st.spinner(f"🔊 '{audio_file.name}' işleniyor..."):
                         r = sr.Recognizer()
                         # Dosyayı geçici olarak kaydetmek yerine BytesIO kullanmayı deneyelim
                         try:
                              audio_bytes = BytesIO(audio_file.getvalue())
                              # Dosya formatına göre AudioFile kullan
                              # Pydub gerekebilir: pip install pydub
                              # from pydub import AudioSegment
                              # if audio_file.type != 'audio/wav':
                              #     sound = AudioSegment.from_file(audio_bytes)
                              #     wav_bytes = BytesIO()
                              #     sound.export(wav_bytes, format="wav")
                              #     wav_bytes.seek(0)
                              #     audio_source_file = wav_bytes
                              # else:
                              #     audio_source_file = audio_bytes

                              # Doğrudan AudioFile ile deneyelim (birçok formatı destekler)
                              with sr.AudioFile(audio_bytes) as source:
                                  audio_data = r.record(source)
                              audio_text = r.recognize_google(audio_data, language="tr-TR")
                              st.success(f"**🎙️ Algılanan Metin:**\n> {audio_text}")
                         except sr.UnknownValueError:
                             st.error("🔇 Ses anlaşılamadı. Daha net bir dosya deneyin.")
                         except sr.RequestError as e:
                             st.error(f"🤖 Ses tanıma servisine ulaşılamadı: {e}")
                         except Exception as e_audio:
                             st.error(f"Ses dosyası işlenirken hata: {e_audio}")
                             print(f"ERROR: Audio processing failed: {e_audio}")

                     if audio_text:
                         # Kullanıcı mesajı olarak ekle
                         user_msg_data = {'role': 'user', 'parts': f"(Ses Dosyası: {audio_file.name}) {audio_text}"}
                         st.session_state.all_chats[active_chat_id].append(user_msg_data)
                         # AI yanıtı için hazırlan ve al
                         message_id = f"audio_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                         st.session_state.message_id_counter += 1
                         history_limit = 20
                         history = st.session_state.all_chats[active_chat_id][-history_limit:-1]
                         with st.spinner("🤖 Yanıt hazırlanıyor..."):
                             ai_response, sender_name = get_hanogt_response_orchestrator(audio_text, history, message_id, active_chat_id, use_stream=False)
                         # Yanıtı göster ve ekle
                         st.markdown(f"#### {sender_name} Yanıtı:")
                         st.markdown(str(ai_response))
                         if st.session_state.tts_enabled and globals().get('tts_engine'):
                             if st.button("🔊 Yanıtı Oku", key="speak_audio_resp"): speak(str(ai_response))
                         ai_msg_data = {'role': 'model', 'parts': str(ai_response), 'sender_display': sender_name}
                         st.session_state.all_chats[active_chat_id].append(ai_msg_data)
                         save_all_chats(st.session_state.all_chats)
                         st.success("✅ Yanıt oluşturuldu ve sohbete eklendi!")

        elif current_mode == "Yaratıcı Stüdyo":
            st.markdown("💡 Bir fikir, kelime veya cümle yazın, AI yaratıcı bir metin oluştursun!")
            creative_prompt = st.text_area("Yaratıcılık Tohumu:", key="creative_prompt_input", placeholder="Örn: 'Güneşin batışını izleyen yalnız bir robot'", height=100)
            ccol1, ccol2 = st.columns(2)
            with ccol1: length_pref = st.selectbox("Uzunluk:", ["kısa", "orta", "uzun"], index=1, key="creative_length")
            with ccol2: style_pref = st.selectbox("Stil:", ["genel", "şiirsel", "hikaye"], index=0, key="creative_style")

            if st.button("✨ Üret!", key="generate_creative_btn", type="primary", use_container_width=True):
                if creative_prompt and creative_prompt.strip():
                    active_chat_id = st.session_state.get('active_chat_id', 'creative_mode_no_chat')
                    message_id = f"creative_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                    st.session_state.message_id_counter += 1
                    final_response = None
                    sender_name = f"{APP_NAME} (Yaratıcı)"

                    if globals().get('gemini_model'): # Gemini kullanmayı dene
                         with st.spinner("✨ Gemini ilham arıyor..."):
                             sys_prompt = f"Çok yaratıcı bir asistansın. Şu isteme '{creative_prompt}' dayanarak, '{style_pref}' stilinde ve '{length_pref}' uzunlukta özgün ve sanatsal bir metin oluştur."
                             gemini_creative_resp = get_gemini_response_cached(sys_prompt, [], stream_output=False)
                             if isinstance(gemini_creative_resp, str) and not gemini_creative_resp.startswith(GEMINI_ERROR_PREFIX):
                                  final_response = gemini_creative_resp
                                  sender_name = f"{APP_NAME} (Gemini Yaratıcı)"
                             else:
                                  st.toast("Gemini yaratıcı yanıtı alınamadı, yerel modül kullanılıyor.", icon="ℹ️")

                    if not final_response: # Gemini başarısızsa veya yoksa yerel üretici
                        with st.spinner("✨ Hayal gücümü kullanıyorum..."):
                            final_response = creative_response_generator(creative_prompt, length=length_pref, style=style_pref)
                            # Türetilen kelimeyi ekle
                            new_word = advanced_word_generator(creative_prompt.split()[0] if creative_prompt else "kelime")
                            final_response += f"\n\n---\n🔮 **Kelimatör Sözcüğü:** {new_word}"
                            sender_name = f"{APP_NAME} (Yerel Yaratıcı)"

                    st.markdown(f"#### {sender_name} İlhamı:")
                    st.markdown(final_response)
                    if st.session_state.tts_enabled and globals().get('tts_engine'):
                         speak_text = final_response.split("🔮 **Kelimatör Sözcüğü:**")[0].strip()
                         if st.button("🔊 İlhamı Dinle", key="speak_creative_resp"): speak(speak_text)
                    log_interaction(creative_prompt, final_response, sender_name, message_id, active_chat_id)
                    st.success("✨ Yaratıcı yanıt oluşturuldu!")
                    # İsteğe bağlı: Aktif sohbete ekle
                    # if active_chat_id != 'creative_mode_no_chat': ...

                else:
                    st.warning("Lütfen yaratıcılık tohumu olarak bir metin girin.", icon="✍️")


        elif current_mode == "Görsel Oluşturucu":
            st.markdown("🎨 Hayalinizdeki görseli tarif edin, AI (basitçe) çizsin!")
            st.info("ℹ️ **Not:** Bu mod, metindeki anahtar kelimelere göre sembolik çizimler yapar. Fotogerçekçi sonuçlar beklemeyin.", icon="💡")
            image_prompt = st.text_input("Görsel Tarifi:", key="image_prompt_input", placeholder="Örn: 'Karlı dağların üzerinde parlayan ay ve çam ağaçları'")

            if st.button("🖼️ Oluştur!", key="generate_image_btn", type="primary", use_container_width=True):
                if image_prompt and image_prompt.strip():
                    with st.spinner("🖌️ Çiziliyor..."):
                        generated_image = generate_prompt_influenced_image(image_prompt)
                    st.image(generated_image, caption=f"'{image_prompt[:60]}' yorumu", use_container_width=True)

                    # İndirme Butonu
                    try:
                        img_buffer = BytesIO()
                        generated_image.save(img_buffer, format="PNG")
                        img_bytes = img_buffer.getvalue()
                        file_name_prompt = re.sub(r'[^\w\s-]', '', image_prompt.lower())[:30].replace(' ','_')
                        file_name = f"hanogt_cizim_{file_name_prompt or 'gorsel'}_{int(time.time())}.png"
                        st.download_button("🖼️ İndir (PNG)", data=img_bytes, file_name=file_name, mime="image/png", use_container_width=True)

                        # İsteğe bağlı: Aktif sohbete ekle
                        active_chat_id = st.session_state.get('active_chat_id')
                        if active_chat_id and active_chat_id in st.session_state.all_chats:
                            user_msg = {'role': 'user', 'parts': f"(Görsel Oluşturucu: {image_prompt})"}
                            ai_msg = {'role': 'model', 'parts': "(Yukarıdaki istemle bir görsel oluşturuldu - İndirme butonu mevcut.)", 'sender_display': f"{APP_NAME} (Görsel)"}
                            st.session_state.all_chats[active_chat_id].extend([user_msg, ai_msg])
                            save_all_chats(st.session_state.all_chats)
                            st.info("Görsel istemi aktif sohbete eklendi.", icon="💾")
                    except Exception as e_img_dl:
                        st.error(f"Görsel indirilemedi: {e_img_dl}")
                else:
                    st.warning("Lütfen görsel için bir tarif girin.", icon="✍️")


        # --- Alt Bilgi (Footer) ---
        st.markdown("<hr style='margin-top: 1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        with footer_cols[0]:
             st.caption(f"Kullanıcı: {st.session_state.get('user_name', 'Misafir')}")
        with footer_cols[1]:
             st.caption(f"{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}")
        with footer_cols[2]:
             ai_status = "Aktif" if globals().get('gemini_model') else "Devre Dışı"
             log_status = "Aktif" if globals().get('supabase') else "Devre Dışı"
             st.caption(f"AI: {ai_status} | Log: {log_status}", help=f"AI Modeli: {st.session_state.gemini_model_name}")

