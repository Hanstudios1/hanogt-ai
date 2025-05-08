# app.py

# --- Gerekli Kütüphaneler ---
import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
import random
import re
import os
import json
from PIL import Image, ImageDraw, ImageFont
import time
from io import BytesIO
from duckduckgo_search import DDGS
from urllib.parse import urlparse, unquote
import google.generativeai as genai
from datetime import datetime
import uuid # Daha benzersiz ID'ler için

# tiktoken kütüphanesi (isteğe bağlı, token sayımı için)
# try:
#     import tiktoken
#     tiktoken_encoder = tiktoken.get_encoding("cl100k_base") # Örnek bir encoder
# except ImportError:
#     tiktoken = None
#     tiktoken_encoder = None
#     # st.toast("tiktoken kütüphanesi bulunamadı. Token sayımı yaklaşık olacaktır.", icon="⚠️")

try:
    from supabase import create_client, Client
    from postgrest import APIError as SupabaseAPIError # Supabase özel hataları için
except ImportError:
    st.warning(
        "Supabase veya postgrest kütüphanesi bulunamadı. Loglama ve bazı Supabase özellikleri kısıtlı olabilir. "
        "`requirements.txt` dosyanızı kontrol edin: `supabase`, `psycopg2-binary` (veya eşdeğeri) ve `postgrest` ekli olmalı.",
        icon="ℹ️"
    )
    create_client = None
    Client = None
    SupabaseAPIError = None

# --- Sayfa Yapılandırması (İLK STREAMLIT KOMUTU OLMALI!) ---
st.set_page_config(
    page_title="Hanogt AI Pro+",
    page_icon="🚀", # Yeni ikon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "4.8 Pro+"
CURRENT_YEAR = datetime.now().year
CHAT_HISTORY_FILE = "chat_history.json"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen biraz sonra tekrar deneyin."
REQUEST_TIMEOUT = 18 # Biraz daha artırıldı
SCRAPE_MAX_CHARS = 3000 # Daha fazla içerik
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 {APP_NAME}/{APP_VERSION}"
SUPABASE_TABLE_LOGS = "chat_logs"
SUPABASE_TABLE_FEEDBACK = "user_feedback"
FONT_FILE = "arial.ttf" # Görsel oluşturucu için font

# --- Tema Ayarları (Basit) ---
# Tam dinamik tema için CSS enjeksiyonu daha iyi olur. Bu, sadece başlangıç renklerini ayarlar.
# st.config.set_option('theme.primaryColor', '#00A1E0') # Örnek

# --- Bilgi Tabanı ---
knowledge_base_load_error = None

@st.cache_data(ttl=3600) # Bilgi tabanını 1 saat cache'le
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE):
    global knowledge_base_load_error
    default_knowledge = {
        "merhaba": ["Merhaba!", "Selam!", "Hoş geldin!", f"Size nasıl yardımcı olabilirim, {st.session_state.get('user_name', 'kullanıcı')}?"],
        "selam": ["Merhaba!", "Selam sana da!", "Nasıl gidiyor?"],
        "nasılsın": ["İyiyim, teşekkürler! Siz nasılsınız?", "Harika hissediyorum, yardımcı olmak için buradayım!", "Her şey yolunda, sizin için ne yapabilirim?"],
        "hanogt kimdir": [f"Ben {APP_NAME} ({APP_VERSION}), Streamlit ve Python ile geliştirilmiş bir yapay zeka asistanıyım.", f"{APP_NAME} ({APP_VERSION}), sorularınızı yanıtlamak, metinler üretmek ve hatta basit görseller oluşturmak için tasarlandı."],
        "teşekkür ederim": ["Rica ederim!", "Ne demek!", "Yardımcı olabildiğime sevindim.", "Her zaman!"],
        "görüşürüz": ["Görüşmek üzere!", "Hoşça kal!", "İyi günler dilerim!", "Tekrar beklerim!"],
        "adın ne": [f"Ben {APP_NAME}, versiyon {APP_VERSION}.", f"Bana {APP_NAME} diyebilirsiniz."],
        "ne yapabilirsin": ["Sorularınızı yanıtlayabilir, metin özetleyebilir, web'de arama yapabilir, yaratıcı metinler üretebilir ve basit görseller çizebilirim.", "Size çeşitli konularda yardımcı olabilirim. Ne merak ediyorsunuz?"],
        "saat kaç": ["<function>"],
        "bugün ayın kaçı": ["<function>"],
        "tarih ne": ["<function>"], # "bugün ayın kaçı" ile aynı lambda'yı kullanabilir
        "hava durumu": ["Üzgünüm, şu an için güncel hava durumu bilgisi sağlayamıyorum. Bunun için özel bir hava durumu servisine göz atabilirsiniz.", "Hava durumu servisim henüz aktif değil, ancak bu konuda bir geliştirme yapmayı planlıyorum!"]
    }
    dynamic_functions = {
        "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
        "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}",
        "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}"
    }

    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                loaded_kb = json.load(f)
                # Fonksiyon işaretçilerini gerçek lambda'larla değiştir
                for key, value_list in loaded_kb.items():
                    if isinstance(value_list, list):
                        for i, val in enumerate(value_list):
                            if val == "<function>" and key in dynamic_functions:
                                loaded_kb[key][i] = dynamic_functions[key]
                return loaded_kb
        else:
            knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) bulunamadı. Varsayılan kullanılıyor."
            return default_knowledge
    except json.JSONDecodeError:
        knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) hatalı formatta. Varsayılan kullanılıyor."
        return default_knowledge
    except Exception as e:
        knowledge_base_load_error = f"Bilgi tabanı yüklenirken hata: {e}. Varsayılan kullanılıyor."
        return default_knowledge

KNOWLEDGE_BASE = load_knowledge_from_file()
if knowledge_base_load_error: st.toast(knowledge_base_load_error, icon="⚠️")


def kb_chatbot_response(query, knowledge):
    query_lower = query.lower().strip()
    # Önce fonksiyonel anahtar kelimeleri kontrol et
    for key, func in knowledge.get("_functions_", {}).items(): # Özel bir _functions_ anahtarı varsayımıyla
        if key in query_lower: return func()

    if query_lower in knowledge:
        response_options = knowledge[query_lower]
        chosen_response = random.choice(response_options)
        return chosen_response() if callable(chosen_response) else chosen_response

    # Geri kalan mantık (kısmi eşleşme vs.)
    possible_responses = []
    for key, responses in knowledge.items():
        if key in query_lower and not key.startswith("_"): # Fonksiyon olmayanlar
            for resp_opt in responses:
                possible_responses.append(resp_opt() if callable(resp_opt) else resp_opt)
    if possible_responses: return random.choice(list(set(possible_responses))) # Tekrarları engelle

    # Kelime bazlı eşleştirme (daha az öncelikli)
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_match_score = 0; best_responses_options = []
    for key, responses in knowledge.items():
        if key.startswith("_"): continue # Fonksiyonları atla
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words: continue
        common_words = query_words.intersection(key_words)
        score = len(common_words) / len(key_words) # Oransal skor
        if score > 0.6: # Eşleşme oranı eşiği
            if score > best_match_score:
                best_match_score = score; best_responses_options = responses
            elif score == best_match_score: best_responses_options.extend(responses)
    if best_responses_options:
        chosen_response = random.choice(list(set(best_responses_options)))
        return chosen_response() if callable(chosen_response) else chosen_response
    return None

# --- API Anahtarı ve Gemini Yapılandırması ---
# @st.cache_resource(ttl=3600) # Modeli 1 saat cache'le (parametreler değişince yenilenmeli)
def initialize_gemini_model():
    global gemini_init_error # Global değişkeni güncellemek için
    api_key_local = st.secrets.get("GOOGLE_API_KEY")
    if not api_key_local:
        gemini_init_error = "🛑 Google API Anahtarı Secrets'ta (st.secrets) bulunamadı! Gemini özellikleri kısıtlı olacak."
        return None
    try:
        genai.configure(api_key=api_key_local)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name=st.session_state.get('gemini_model_name', 'gemini-1.5-flash-latest'),
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=st.session_state.get('gemini_temperature', 0.7),
                top_p=st.session_state.get('gemini_top_p', 0.95), # Biraz daha odaklı
                top_k=st.session_state.get('gemini_top_k', 40),  # Biraz daha çeşitlilik
                max_output_tokens=st.session_state.get('gemini_max_tokens', 4096) # Artırıldı
            )
        )
        gemini_init_error = None # Başarılıysa hatayı temizle
        # st.toast("Gemini modeli başarıyla yapılandırıldı/güncellendi!", icon="✨")
        return model
    except Exception as e:
        gemini_init_error = f"🛑 Gemini yapılandırma hatası: {e}. Lütfen API anahtarınızı ve internet bağlantınızı kontrol edin."
        return None

gemini_model = initialize_gemini_model() # İlk yüklemede çalıştır
gemini_init_error = globals().get('gemini_init_error') # Fonksiyondan gelen hatayı al

# --- Supabase İstemcisini Başlatma ---
@st.cache_resource(ttl=3600) # Bağlantıyı 1 saat cache'le
def init_supabase_client_cached():
    global supabase_error # Global değişkeni güncellemek için
    supabase_url_local = st.secrets.get("SUPABASE_URL")
    supabase_key_local = st.secrets.get("SUPABASE_SERVICE_KEY")
    if not create_client:
        supabase_error = "Supabase kütüphanesi yüklenemedi. Loglama çalışmayacak."
        return None
    if not supabase_url_local or not supabase_key_local:
        supabase_error = "Supabase URL veya Service Key Secrets'ta bulunamadı! Loglama devre dışı."
        return None
    try:
        client = create_client(supabase_url_local, supabase_key_local)
        # Basit bir test (opsiyonel, tablo var mı diye kontrol etmeyebilir)
        # client.table(SUPABASE_TABLE_LOGS).select("id", count="exact").limit(0).execute()
        supabase_error = None # Başarılıysa hatayı temizle
        return client
    except Exception as e:
        error_msg = f"Supabase bağlantı hatası: {e}. Loglama yapılamayacak."
        if "failed to parse" in str(e).lower() or "invalid url" in str(e).lower():
            error_msg += " Lütfen Supabase URL'inizin doğru formatta olduğundan emin olun (örn: https://xyz.supabase.co)."
        elif "invalid key" in str(e).lower():
            error_msg += " Lütfen Supabase Service Key'inizin doğru olduğundan emin olun."
        supabase_error = error_msg
        return None

supabase = init_supabase_client_cached()
supabase_error = globals().get('supabase_error') # Fonksiyondan gelen hatayı al


# --- YARDIMCI FONKSİYONLAR ---
def _get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4()) # Daha benzersiz ID
    return st.session_state.session_id

@st.cache_resource # TTS motorunu cache'le
def init_tts_engine():
    global tts_init_error
    try:
        engine = pyttsx3.init()
        # İsteğe bağlı: Hız, ses vb. ayarlar
        # engine.setProperty('rate', 160)
        tts_init_error = None
        return engine
    except Exception as e:
        tts_init_error = f"⚠️ Metin okuma (TTS) motoru başlatılamadı: {e}."
        return None

tts_engine = init_tts_engine()
tts_init_error = globals().get('tts_init_error')

def speak(text):
    if not tts_engine or not st.session_state.get('tts_enabled', True):
        if tts_engine: st.toast("Metin okuma kapalı.", icon="🔇")
        else: st.toast("Metin okuma motoru aktif değil.", icon="🔇")
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except RuntimeError as re_tts: st.warning(f"Konuşma motorunda bir sorun: {re_tts}", icon="🔊")
    except Exception as e_tts: st.error(f"Konuşma sırasında hata: {e_tts}", icon="🔊")

def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text) # Birden fazla boş satırı tek boş satıra indir
    return text.strip()

def scrape_url_content(url: str, timeout: int = REQUEST_TIMEOUT, max_chars: int = SCRAPE_MAX_CHARS) -> str | None:
    st.toast(f"🌐 '{urlparse(url).netloc}' sayfasından içerik alınıyor...", icon="⏳")
    try:
        parsed_url = urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']:
            st.warning(f"Geçersiz URL: {url}", icon="🔗"); return None
        headers = {'User-Agent': USER_AGENT, 'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            st.info(f"URL HTML değil ('{content_type}'). Kazıma atlanıyor: {url}", icon="📄"); return None

        html_content = ""; content_length = 0
        max_html_size = max_chars * 10 # HTML boyutu için bir üst sınır (çıkarılan metinden daha büyük olabilir)
        for chunk in response.iter_content(chunk_size=16384, decode_unicode=True, errors='ignore'): # 16KB, hatalı karakterleri yok say
            html_content += chunk; content_length += len(chunk)
            if content_length > max_html_size:
                st.warning(f"HTML içeriği çok büyük ({content_length} byte), ilk kısmı işlenecek.", icon="✂️"); break
        response.close()

        soup = BeautifulSoup(html_content, 'lxml') # lxml daha hızlı ve toleranslı olabilir
        for element in soup(["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio"]):
            element.decompose()

        potential_content_parts = []
        selectors = ['article[class*="content"]', 'main[class*="content"]', 'div[class*="post-body"]', 'div[class*="article-body"]', 'article', 'main', '.content', '.post-content', '.entry-content', 'section']
        content_found = False
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                container = elements[0]
                paragraphs = container.find_all(['p', 'div'], recursive=False, limit=30) # Sadece ilk seviye p ve div'ler
                temp_content = []
                for p_or_div in paragraphs:
                    text = _clean_text(p_or_div.get_text(separator=' ', strip=True))
                    if len(text) > 100 and (text.count('.') + text.count('!') + text.count('?')) >= 1: # Anlamlı cümleler
                        temp_content.append(text)
                if len(" ".join(temp_content)) > 500: # Yeterli içerik varsa
                    potential_content_parts = temp_content; content_found = True; break
        if not content_found:
            body_text = _clean_text(soup.body.get_text(separator=' ', strip=True) if soup.body else "")
            if len(body_text) > 300:
                st.toast("Özel alan bulunamadı, genel sayfa metni kullanıldı.", icon="ℹ️")
                potential_content_parts = [body_text] # Tek parça olarak ekle
            else:
                st.toast("Sayfada anlamlı metin içeriği bulunamadı.", icon="📄"); return None

        full_text = "\n\n".join(potential_content_parts) # Paragrafları boş satırla ayır
        cleaned_text = _clean_text(full_text)
        if not cleaned_text: return None
        final_text = cleaned_text[:max_chars]
        if len(cleaned_text) > max_chars: final_text += "..."
        st.toast(f"'{urlparse(url).netloc}' içeriği başarıyla alındı.", icon="✅")
        return final_text
    except requests.exceptions.HTTPError as e: st.toast(f"⚠️ Sayfa HTTP hatası ({e.response.status_code}): {url}", icon='🌐')
    except requests.exceptions.Timeout: st.toast(f"⚠️ Sayfa zaman aşımı: {url}", icon='⏳')
    except requests.exceptions.ConnectionError: st.toast(f"⚠️ Sayfa bağlantı hatası: {url}", icon='🔌')
    except requests.exceptions.RequestException as e: st.toast(f"⚠️ Sayfa genel hata: {e}", icon='🌐')
    except Exception as e: st.toast(f"⚠️ Sayfa işleme hatası: {e}", icon='⚙️')
    return None

def search_web(query: str) -> str | None:
    # ... (Önceki search_web mantığı büyük ölçüde korunabilir, scrape_url_content güncellendiği için dolaylı olarak iyileşir)
    # Wikipedia ve DuckDuckGo kısımları aynı kalabilir.
    st.toast(f"🔍 '{query}' web'de aranıyor...", icon="⏳")
    wikipedia.set_lang("tr")
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True, redirect=True)
        st.toast("ℹ️ Wikipedia'dan bilgi bulundu.", icon="✅")
        return f"**Wikipedia'dan ({query}):**\n\n{_clean_text(summary)}"
    except wikipedia.exceptions.PageError: st.toast(f"ℹ️ '{query}' için Wikipedia'da sayfa bulunamadı.", icon="🤷")
    except wikipedia.exceptions.DisambiguationError as e:
        options_text = "\n\nWikipedia'da olası başlıklar:\n" + "\n".join([f"- {opt}" for opt in e.options[:3]])
        st.toast(f"Wikipedia'da '{query}' için birden fazla anlam bulundu.", icon="📚")
        return f"**Wikipedia'da Birden Fazla Anlam Var ({query}):**\n{str(e).splitlines()[0]}{options_text}"
    except Exception: pass

    ddg_result_text = None; ddg_url = None
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs:
            results = list(ddgs.text(query, region='tr-tr', safesearch='moderate', max_results=3))
            if results:
                for res in results:
                    snippet = res.get('body'); temp_url = res.get('href')
                    if snippet and temp_url:
                        decoded_url = unquote(temp_url)
                        st.toast(f"ℹ️ DuckDuckGo'dan özet bulundu: {urlparse(decoded_url).netloc}", icon="🦆")
                        ddg_result_text = f"**Web Özeti (DuckDuckGo - {urlparse(decoded_url).netloc}):**\n\n{_clean_text(snippet)}\n\nKaynak: {decoded_url}"
                        ddg_url = decoded_url; break
    except Exception: pass

    if ddg_url:
        scraped_content = scrape_url_content(ddg_url)
        if scraped_content: return f"**Web Sayfasından Detay ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nTam İçerik İçin Kaynak: {ddg_url}"
        elif ddg_result_text: return ddg_result_text
        else: return f"Detaylı bilgi için: {ddg_url}"
    if ddg_result_text: return ddg_result_text
    st.toast(f"'{query}' için web'de kapsamlı yanıt bulunamadı.", icon="❌"); return None


@st.cache_data(ttl=86400) # Günlük cache
def load_chat_history_cached(file_path: str = CHAT_HISTORY_FILE) -> list:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            if content and content.strip(): return json.loads(content)
            else: return []
        except json.JSONDecodeError:
            st.error(f"Geçmiş dosyası ({file_path}) bozuk. Yeni bir geçmiş başlatılıyor.")
            try: os.rename(file_path, f"{file_path}.backup_{int(time.time())}")
            except OSError: pass
            return []
        except Exception as e: st.error(f"Geçmiş dosyası ({file_path}) yüklenirken hata: {e}"); return []
    return []

def save_chat_history(history: list, file_path: str = CHAT_HISTORY_FILE):
    try:
        with open(file_path, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=2) # Daha kompakt indent
    except Exception as e: st.error(f"Sohbet geçmişi kaydedilemedi: {e}")

# @st.cache_data(ttl=300, show_spinner=False, persist="disk") # Prompt ve geçmiş aynıysa cache'le (stream için uygun değil)
def get_gemini_response_cached(prompt: str, chat_history_for_gemini: list[dict], stream: bool = False) -> str | object:
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Gemini modeli aktif değil."
    # Token sayımı ve geçmiş kırpma (tiktoken ile daha iyi)
    # approx_tokens = sum(len(msg['parts'][0].split()) for msg in chat_history_for_gemini) + len(prompt.split())
    # if tiktoken_encoder:
    #     history_tokens = sum(len(tiktoken_encoder.encode(msg['parts'][0])) for msg in chat_history_for_gemini)
    #     prompt_tokens = len(tiktoken_encoder.encode(prompt))
    #     total_tokens = history_tokens + prompt_tokens
    #     # st.sidebar.caption(f"Yaklaşık Token: {total_tokens}")
    #     # Modelin token limitine göre (örn: Flash için 32k, Pro için daha fazla) kırpma yapılabilir.
    #     # if total_tokens > 30000:
    #     #     st.warning("Sohbet geçmişi çok uzun, son kısmı kullanılıyor (token optimizasyonu).")
    #     #     # Daha akıllıca bir kırpma/özetleme yapılabilir
    #     #     # Şimdilik en son N mesajı al (kaba bir yaklaşım)
    #     #     # chat_history_for_gemini = chat_history_for_gemini[-10:] # Son 10 etkileşim
    # else: # tiktoken yoksa kelime bazlı kaba tahmin
    #     if approx_tokens > 15000: # Daha düşük bir eşik
    #         chat_history_for_gemini = chat_history_for_gemini[-10:]


    try:
        chat = gemini_model.start_chat(history=chat_history_for_gemini)
        response = chat.send_message(prompt, stream=stream)
        if stream: return response
        else:
            if not response.parts:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason
                    msg = f"Yanıt güvenlik nedeniyle engellendi: {reason}. ({response.prompt_feedback.block_reason_message or 'Ek detay yok.'})"
                    st.warning(msg, icon="🛡️"); return f"{GEMINI_ERROR_PREFIX} {msg}"
                elif response.candidates and hasattr(response.candidates[0], 'finish_reason') and response.candidates[0].finish_reason != 'STOP':
                    finish_reason = response.candidates[0].finish_reason
                    st.warning(f"Gemini yanıtı tam değil. Sebep: {finish_reason}", icon="⚠️"); return f"{GEMINI_ERROR_PREFIX} Yanıt tam değil. Sebep: {finish_reason}."
                else: st.warning(f"Gemini'dan boş yanıt: {response}", icon="⁉️"); return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
    except genai.types.BlockedPromptException as bpe: st.error(f"Gemini İstem Engelleme Hatası: {bpe}", icon="🛡️"); return f"{GEMINI_ERROR_PREFIX} İstem güvenlik nedeniyle engellendi."
    except genai.types.StopCandidateException as sce: st.error(f"Gemini Yanıt Kesintisi: {sce}", icon="🛑"); return f"{GEMINI_ERROR_PREFIX} Yanıt kesildi."
    except requests.exceptions.ReadTimeout: st.error("Gemini API isteği zaman aşımına uğradı (ReadTimeout). Lütfen tekrar deneyin.", icon="⏳"); return f"{GEMINI_ERROR_PREFIX} API okuma zaman aşımı."
    except Exception as e:
        error_message = f"Gemini API hatası: {e}"
        st.error(error_message, icon="📡")
        if "API key not valid" in str(e): return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        elif "Deadline Exceeded" in str(e) or "504" in str(e): return f"{GEMINI_ERROR_PREFIX} API zaman aşımı."
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı: {str(e)[:150]}..."

def log_to_supabase(table_name: str, data: dict):
    if not supabase: return False
    try:
        insert_result = supabase.table(table_name).insert(data).execute()
        if hasattr(insert_result, 'data') and not insert_result.data and hasattr(insert_result, 'error') and insert_result.error:
            error_info = insert_result.error; err_msg = str(error_info)
            if SupabaseAPIError and isinstance(error_info, SupabaseAPIError): # postgrest.exceptions.APIError
               err_msg = f"Kod: {error_info.code}, Mesaj: {error_info.message}, Detay: {error_info.details}, İpucu: {error_info.hint}"
            st.toast(f"⚠️ '{table_name}' logu Supabase'e kaydedilemedi: {err_msg}", icon="💾")
            print(f"WARN: Supabase '{table_name}' insert hatası. Error: {err_msg}")
            return False
        elif not insert_result.data and not hasattr(insert_result, 'error'): # Hata yok ama veri de yok
             st.toast(f"⚠️ '{table_name}' logu Supabase'e kaydedildi ancak yanıt verisi boş.", icon="💾")
             return True
        return True
    except SupabaseAPIError as e_api: # Eğer SupabaseAPIError import edilebildiyse
        print(f"ERROR: Supabase API hatası '{table_name}': {e_api.message} (Kod: {e_api.code})")
        st.error(f"Supabase loglama API hatası: {e_api.message}")
        return False
    except Exception as e:
        print(f"ERROR: Supabase '{table_name}' loglama kritik hata: {e}")
        st.error(f"Supabase '{table_name}' loglama kritik hata! Detay: {type(e).__name__}: {e}")
        return False

def log_interaction(prompt: str, response: str, source: str, message_id: str):
    log_data = {
        "user_prompt": prompt, "ai_response": response, "response_source": source,
        "user_name": st.session_state.get('user_name', 'Bilinmiyor'),
        "session_id": _get_session_id(), "app_version": APP_VERSION, "message_id": message_id
    }
    return log_to_supabase(SUPABASE_TABLE_LOGS, log_data)

def log_feedback(message_id: str, prompt: str, response: str, feedback_type: str, comment: str = ""):
    feedback_data = {
        "message_id": message_id, "user_prompt": prompt, "ai_response": response,
        "feedback_type": feedback_type, "comment": comment,
        "user_name": st.session_state.get('user_name', 'Bilinmiyor'),
        "session_id": _get_session_id(), "app_version": APP_VERSION
    }
    if log_to_supabase(SUPABASE_TABLE_FEEDBACK, feedback_data):
        st.toast(f"Geri bildiriminiz için teşekkürler!", icon="💌"); return True
    else: st.toast(f"Geri bildiriminiz gönderilemedi.", icon="😔"); return False


def get_hanogt_response_orchestrator(user_prompt: str, chat_history_for_model: list[dict], current_message_id: str, stream_enabled:bool = False) -> tuple[str | object, str]:
    response = None; ai_sender = APP_NAME

    # Öncelik 1: Fonksiyonel Bilgi Tabanı Komutları
    kb_resp_check = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
    if kb_resp_check and callable(kb_resp_check): # Eğer çağrılabilir bir fonksiyon döndüyse
        try:
            response = kb_resp_check()
            ai_sender = f"{APP_NAME} (Fonksiyonel)"
            log_interaction(user_prompt, response, ai_sender, current_message_id)
            return response, ai_sender
        except Exception as e_func:
            st.error(f"Fonksiyonel yanıt işlenirken hata: {e_func}")
            response = None # Devam et


    # Öncelik 2: Gemini
    if gemini_model:
        response = get_gemini_response_cached(user_prompt, chat_history_for_model, stream=stream_enabled)
        if response:
            if stream_enabled: return response, f"{APP_NAME} (Gemini Stream)"
            elif not isinstance(response, str) or not response.startswith(GEMINI_ERROR_PREFIX):
                ai_sender = f"{APP_NAME} (Gemini)"
                log_interaction(user_prompt, str(response), ai_sender, current_message_id)
                return str(response), ai_sender
            response = None # Hata varsa veya stream objesi değilse (ve stream kapalıysa)

    # Öncelik 3: Statik Bilgi Tabanı
    if not response:
        st.toast("📚 Bilgi tabanı kontrol ediliyor...", icon="🗂️")
        kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE) # Tekrar çağır (non-callable için)
        if kb_resp and not callable(kb_resp):
            response = kb_resp; ai_sender = f"{APP_NAME} (Bilgi Tabanı)"
            log_interaction(user_prompt, response, ai_sender, current_message_id)
            return response, ai_sender

    # Öncelik 4: Web Arama
    if not response:
        if len(user_prompt.split()) > 2 and ("?" in user_prompt or any(kw in user_prompt.lower() for kw in ["nedir", "kimdir", "nasıl", "ne zaman", "nerede", "anlamı", "hakkında bilgi", "araştır"])):
            st.toast("🌐 Web'de arama yapılıyor...", icon="🔍")
            web_resp = search_web(user_prompt)
            if web_resp:
                response = web_resp; ai_sender = f"{APP_NAME} (Web Arama)"
                log_interaction(user_prompt, response, ai_sender, current_message_id)
                return response, ai_sender
        else: st.toast("ℹ️ Kısa/genel istem için web araması atlandı.", icon="⏩")

    # Öncelik 5: Varsayılan Yanıt
    if not response:
        st.toast("🤔 Uygun bir yanıt bulunamadı.", icon="🤷")
        default_responses = [
            f"Üzgünüm {st.session_state.get('user_name', '')}, bu konuda size şu an yardımcı olamıyorum.",
            "Bu soruyu tam olarak anlayamadım. Farklı bir şekilde ifade edebilir misiniz?",
            "Bu konuda bir fikrim yok. Başka bir şey sormak ister misiniz?",
            "Yanıt veremiyorum ama her geçen gün öğrenmeye devam ediyorum!",
        ]
        response = random.choice(default_responses); ai_sender = f"{APP_NAME} (Varsayılan)"
        log_interaction(user_prompt, response, ai_sender, current_message_id)
    return response, ai_sender


# --- Yaratıcı/Görsel Fonksiyonlar ---
def creative_response_generator(prompt: str, length_preference: str = "orta", style_preference: str = "genel") -> str:
    styles = {
        "genel": ["Farklı bir bakış açısıyla: {}", "Hayal gücümüzü kullanalım: {}", "Aklıma şöyle bir fikir geldi: {}"],
        "şiirsel": ["Kalbimden dökülen mısralar: {}", "Sözcüklerin dansıyla: {}", "Duyguların ritmiyle: {}"],
        "hikaye": ["Bir zamanlar, uzak diyarlarda başlar hikayemiz: {}", "Perde açılır ve sahne senindir: {}", "Her şey o gün başladı: {}"]
    }
    selected_style_templates = styles.get(style_preference, styles["genel"])
    base_idea = generate_new_idea_creative(prompt, style_preference)

    # Uzunluk ayarı (basitçe kelime/cümle sayısı ile yapılabilir)
    if length_preference == "kısa":
        base_idea = ". ".join(base_idea.split(".")[:max(1, len(base_idea.split(".")) // 3)]) + "." if "." in base_idea else base_idea
    elif length_preference == "uzun":
        base_idea += f"\n\nBu konuyu daha da derinleştirecek olursak, {generate_new_idea_creative(prompt[::-1], style_preference)} diyebiliriz."

    return random.choice(selected_style_templates).format(base_idea)

def generate_new_idea_creative(seed_prompt: str, style:str = "genel") -> str:
    elements = ["zaman kristalleri", "psişik ormanlar", "rüya mimarisi eserleri", "kuantum köpüğü okyanusları", "gölge enerjisi", "yankılanan anılar", "kayıp yıldız haritaları", "fraktal düşünce kalıpları", "kozmik yankılar", "unutulmuş kehanetler"]
    actions = ["dokur", "çözer", "yansıtır", "inşa eder", "fısıldar", "dönüştürür", "keşfeder", "haritalar", "bağlantı kurar", "çağırır"]
    outcomes = ["kaderin gizli ipliklerini", "varoluşun unutulmuş kodunu", "bilincin en derin sınırlarını", "kayıp uygarlıkların sırlarını", "evrenin melodisini", "gerçekliğin dokusunu", "saklı potansiyelleri", "yeni bir çağın başlangıcını"]
    words = re.findall(r'\b\w{3,}\b', seed_prompt.lower()); seed_elements = random.sample(words, k=min(len(words), 2)) if words else ["gizemli", "bir şey"]
    if style == "şiirsel": return f"{random.choice(elements).capitalize()} arasında, {seed_elements[0]} fısıldar {random.choice(outcomes)}."
    return f"{' '.join(seed_elements).capitalize()} {random.choice(actions)} ve {random.choice(elements)} kullanarak {random.choice(outcomes)}."

def advanced_word_generator(base_word: str) -> str:
    # ... (Öncekiyle aynı, yeterince iyi)
    if not base_word or len(base_word) < 2: return "KelimatörPro+"
    vowels = "aeiouüöıAEIOUÜÖI"; consonants = "bcçdfgğhjklmnprsştvyzBCÇDFGĞHJKLMNPRSŞTVYZ"
    cleaned_base = "".join(filter(str.isalpha, base_word))
    if not cleaned_base: return "SözcükMimar"
    prefixes = ["bio", "krono", "psiko", "tera", "neo", "mega", "nano", "astro", "poli", "eko", "meta", "trans", "ultra", "omni", "xeno"]
    suffixes = ["genez", "sfer", "nomi", "tek", "loji", "tronik", "morf", "vers", "dinamik", "matik", "kinezis", "skop", "grafi", "mant", "krom"]
    if len(cleaned_base) > 3 and random.random() < 0.7:
        start_index = random.randint(0, max(0, len(cleaned_base) - 3))
        core = cleaned_base[start_index : start_index + random.randint(2,4)]
    else:
        core_len = random.randint(3, 5); core_chars = [random.choice(consonants if random.random() > 0.4 else vowels) for _ in range(core_len)]; core = "".join(core_chars)
    new_word = core
    if random.random() > 0.45: new_word = random.choice(prefixes) + new_word
    if random.random() > 0.45: new_word += random.choice(suffixes)
    return new_word.capitalize() if len(new_word) > 1 else new_word

def generate_prompt_influenced_image(prompt: str) -> Image.Image:
    # ... (Öncekiyle büyük ölçüde aynı, font dosyası için FONT_FILE kullanıldı)
    width, height = 512, 512; prompt_lower = prompt.lower()
    keyword_themes = { "güneş": {"bg": [(255, 230, 150), (255, 160, 0)], "shapes": [{"type": "circle", "color": (255, 255, 0, 220), "pos": (0.25, 0.25), "size": 0.2}]}, "ay": {"bg": [(10, 10, 50), (40, 40, 100)], "shapes": [{"type": "circle", "color": (240, 240, 240, 200), "pos": (0.75, 0.2), "size": 0.15}]}, "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []}, "deniz": {"bg": [(0, 105, 148), (0, 0, 100)], "shapes": [{"type": "rectangle", "color": (60,120,180, 150), "pos": (0.5, 0.75), "size": (1.0, 0.5)}]}, "orman": {"bg": [(34, 139, 34), (0, 100, 0)], "shapes": [{"type": "triangle", "color": (0,80,0,200), "pos": (random.uniform(0.2,0.8), random.uniform(0.4,0.7)), "size": random.uniform(0.1,0.25)} for _ in range(random.randint(4,7)) ]}, "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [ {"type": "rectangle", "color": (139, 69, 19, 255), "pos": (0.5, 0.75), "size": (0.08, 0.4)}, {"type": "ellipse", "color": (34, 139, 34, 200), "pos": (0.5, 0.45), "size_wh": (0.3, 0.25)} ]}, "dağ": {"bg": [(200,200,200), (100,100,100)], "shapes": [{"type": "triangle", "color": (150,150,150,230), "pos": (0.5,0.6), "size":0.4, "points": [(random.uniform(0.1,0.3),0.8),(0.5,random.uniform(0.1,0.3)),(random.uniform(0.7,0.9),0.8)] }]}, "şehir": {"bg": [(100,100,120), (50,50,70)], "shapes": [{"type":"rectangle", "color":(random.randint(60,90),random.randint(60,90),random.randint(70,100),200), "pos":(random.uniform(0.1,0.9), random.uniform(0.5,0.8)), "size": (random.uniform(0.05,0.12), random.uniform(0.2,0.6))} for _ in range(random.randint(6,10))]} }
    bg_color1 = (random.randint(30, 120), random.randint(30, 120), random.randint(30, 120)); bg_color2 = (random.randint(120, 220), random.randint(120, 220), random.randint(120, 220)); shapes_to_draw = []; theme_applied_count = 0
    for keyword, theme in keyword_themes.items():
        if keyword in prompt_lower:
            if theme_applied_count == 0: bg_color1, bg_color2 = theme["bg"] # İlk tema arka planı belirlesin
            shapes_to_draw.extend(theme["shapes"]); theme_applied_count +=1
    img = Image.new('RGBA', (width, height), (0,0,0,0)); draw = ImageDraw.Draw(img)
    for y in range(height): ratio = y / height; r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio); g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio); b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio); draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
    for shape_info in shapes_to_draw:
        s_type = shape_info["type"]; s_color = shape_info["color"]; s_pos_x_ratio, s_pos_y_ratio = shape_info["pos"]; s_center_x = int(s_pos_x_ratio * width); s_center_y = int(s_pos_y_ratio * height)
        if s_type == "circle": s_radius = int(shape_info["size"] * min(width, height) / 2); draw.ellipse((s_center_x - s_radius, s_center_y - s_radius, s_center_x + s_radius, s_center_y + s_radius), fill=s_color, outline=(0,0,0,50) if len(s_color) == 4 and s_color[3] < 255 else None)
        elif s_type == "rectangle": s_w_ratio, s_h_ratio = shape_info["size"]; rect_w = int(s_w_ratio * width); rect_h = int(s_h_ratio * height); draw.rectangle((s_center_x - rect_w // 2, s_center_y - rect_h // 2, s_center_x + rect_w // 2, s_center_y + rect_h // 2), fill=s_color, outline=(0,0,0,50) if len(s_color) == 4 and s_color[3] < 255 else None)
        elif s_type == "triangle" and "points" in shape_info: abs_points = [(int(p[0]*width), int(p[1]*height)) for p in shape_info["points"]]; draw.polygon(abs_points, fill=s_color, outline=(0,0,0,50) if len(s_color) == 4 and s_color[3] < 255 else None)
        elif s_type == "triangle": s_base = int(shape_info["size"] * min(width, height)); p1 = (s_center_x, s_center_y - int(s_base * 0.577)); p2 = (s_center_x - s_base // 2, s_center_y + int(s_base * 0.288)); p3 = (s_center_x + s_base // 2, s_center_y + int(s_base * 0.288)); draw.polygon([p1, p2, p3], fill=s_color, outline=(0,0,0,50) if len(s_color) == 4 and s_color[3] < 255 else None)
        elif s_type == "ellipse": size_wh = shape_info["size_wh"]; el_w = int(size_wh[0] * width); el_h = int(size_wh[1] * height); draw.ellipse((s_center_x - el_w // 2, s_center_y - el_h // 2, s_center_x + el_w // 2, s_center_y + el_h // 2), fill=s_color, outline=(0,0,0,50) if len(s_color) == 4 and s_color[3] < 255 else None)
    if theme_applied_count == 0: # Tema yoksa rastgele
        for _ in range(random.randint(3, 6)): x1 = random.randint(0, width); y1 = random.randint(0, height); shape_fill = (random.randint(50, 250), random.randint(50, 250), random.randint(50, 250), random.randint(150, 220))
            if random.random() > 0.5: radius = random.randint(25, 85); draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=shape_fill)
            else: w_rect, h_rect = random.randint(35, 125), random.randint(35, 125); draw.rectangle((x1 - w_rect // 2, y1 - h_rect // 2, x1 + w_rect // 2, y1 + h_rect // 2), fill=shape_fill)
    try:
        font_path = FONT_FILE if os.path.exists(FONT_FILE) else None
        font_size = max(16, min(30, int(width / (len(prompt) * 0.35 + 10))))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default() # Pillow 10+ size arg yok
        if not font_path and hasattr(ImageFont, 'load_default') and 'size' in ImageFont.load_default.__code__.co_varnames : # Eski Pillow
             font = ImageFont.load_default(size=font_size)


        text_to_display = prompt[:70] # Kısalt
        if hasattr(draw, 'textbbox'): bbox = draw.textbbox((0,0), text_to_display, font=font, anchor="lt"); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
        else: text_width, text_height = draw.textsize(text_to_display, font=font)
        text_x = (width - text_width) / 2; text_y = height * 0.93 - text_height # Biraz daha aşağıda
        draw.text((text_x + 1, text_y + 1), text_to_display, font=font, fill=(0,0,0,150)); draw.text((text_x, text_y), text_to_display, font=font, fill=(255,255,255,230))
    except Exception as e_font: st.toast(f"Görsel metni yazılamadı: {e_font}", icon="📝")
    return img.convert("RGB")


# --- Session State Başlatma ---
DEFAULT_SESSION_STATE = {
    'chat_history': [], 'app_mode': "Yazılı Sohbet", 'user_name': None,
    'user_avatar_bytes': None, 'show_main_app': False, 'greeting_message_shown': False,
    'tts_enabled': True, 'theme': "Light", # Tema henüz tam entegre değil
    'gemini_stream_enabled': True, 'gemini_temperature': 0.7, 'gemini_top_p': 0.95,
    'gemini_top_k': 40, 'gemini_max_tokens': 4096, 'gemini_model_name': 'gemini-1.5-flash-latest',
    'message_id_counter': 0, 'last_ai_response_for_feedback': None,
    'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
    'feedback_comment_input': "", 'show_feedback_comment_form': False,
    'session_id': str(uuid.uuid4()) # Her oturum için benzersiz ID
}
for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

if not st.session_state.chat_history: st.session_state.chat_history = load_chat_history_cached()
if st.session_state.user_name and not st.session_state.show_main_app: st.session_state.show_main_app = True


# --- ARAYÜZ BÖLÜMLERİ İÇİN FONKSİYONLAR ---
def display_sidebar_content():
    with st.sidebar:
        st.markdown(f"### Hoş Geldin, {st.session_state.user_name}!")
        if st.session_state.user_avatar_bytes: st.image(st.session_state.user_avatar_bytes, width=100, use_column_width='auto')
        else: st.caption("🖼️ _Avatar yüklemediniz._")
        st.markdown("---")
        st.subheader("⚙️ Ayarlar")

        # İsim ve Avatar
        with st.expander("👤 Profil Ayarları", expanded=False):
            new_name = st.text_input("Adını Değiştir:", value=st.session_state.user_name, key="change_name_sidebar_exp")
            if new_name != st.session_state.user_name and new_name.strip():
                st.session_state.user_name = new_name.strip(); st.toast("Adın güncellendi!", icon="✏️"); st.rerun()
            uploaded_avatar = st.file_uploader("Yeni Avatar Yükle (Maks 2MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader_sidebar_exp")
            if uploaded_avatar:
                if uploaded_avatar.size > 2 * 1024 * 1024: st.error("Dosya boyutu 2MB'den büyük olamaz!")
                else: st.session_state.user_avatar_bytes = uploaded_avatar.getvalue(); st.toast("Avatar güncellendi!", icon="🖼️"); st.rerun()
            if st.session_state.user_avatar_bytes and st.button("🗑️ Avatarı Kaldır", use_container_width=True, key="remove_avatar_sidebar_exp"):
                st.session_state.user_avatar_bytes = None; st.toast("Avatar kaldırıldı.", icon="🗑️"); st.rerun()

        # Genel Ayarlar
        st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS)", value=st.session_state.tts_enabled, disabled=not tts_engine, help="AI yanıtlarını sesli okur.")
        st.session_state.gemini_stream_enabled = st.toggle("Gemini Yanıt Akışı", value=st.session_state.gemini_stream_enabled, help="Yanıtlar kelime kelime gelir.")

        # Gemini Gelişmiş Ayarları
        with st.expander("🤖 Gemini Gelişmiş Ayarları", expanded=False):
            st.session_state.gemini_model_name = st.selectbox("Gemini Modeli:", ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'], index=0 if st.session_state.gemini_model_name == 'gemini-1.5-flash-latest' else 1, key="gemini_model_selector")
            st.session_state.gemini_temperature = st.slider("Sıcaklık (Yaratıcılık):", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="gemini_temp_slider")
            st.session_state.gemini_top_p = st.slider("Top P (Odaklanma):", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="gemini_top_p_slider")
            st.session_state.gemini_top_k = st.slider("Top K (Çeşitlilik):", 1, 100, st.session_state.gemini_top_k, 1, key="gemini_top_k_slider")
            st.session_state.gemini_max_tokens = st.slider("Maksimum Çıktı Token:", 256, 8192, st.session_state.gemini_max_tokens, 128, key="gemini_max_tokens_slider")
            if st.button("Ayarları Uygula ve Modeli Yeniden Yükle", key="reload_gemini_settings_btn", use_container_width=True):
                global gemini_model # Global modeli güncellemek için
                gemini_model = initialize_gemini_model() # Ayarlarla yeniden yükle
                if gemini_model: st.toast("Gemini ayarları güncellendi ve model yeniden yüklendi!", icon="✨")
                else: st.error("Gemini modeli yüklenirken hata oluştu. Lütfen API anahtarınızı ve ayarları kontrol edin.")

        st.divider()
        if st.button("🧹 Sohbet Geçmişini Temizle", use_container_width=True, type="secondary", key="clear_history_sidebar_btn"):
            if st.session_state.chat_history:
                st.session_state.chat_history = []; save_chat_history([])
                st.toast("Sohbet geçmişi temizlendi!", icon="🧹"); st.rerun()
            else: st.toast("Geçmiş zaten boş.", icon="ℹ️")

        with st.expander("ℹ️ Hakkında", expanded=False):
            st.markdown(f"""
            **{APP_NAME} v{APP_VERSION}**
            Yapay zeka destekli sohbet asistanınız.
            Geliştirici: [Hanogt (GitHub)](https://github.com/Hanogt)

            Bu uygulama Streamlit, Google Gemini API ve çeşitli açık kaynak kütüphaneler kullanılarak oluşturulmuştur.
            Supabase loglama ve geri bildirim için kullanılmaktadır.
            © {CURRENT_YEAR}
            """)
        st.caption(f"{APP_NAME} v{APP_VERSION} - Oturum ID: {st.session_state.session_id[:8]}...")


def display_chat_message_with_feedback(sender: str, message_content: str, message_index: int, is_user: bool):
    avatar_icon = "🧑"
    if is_user:
        if st.session_state.user_avatar_bytes: avatar_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes))
    else: # AI mesajı
        if "Gemini" in sender: avatar_icon = "✨"
        elif "Web" in sender: avatar_icon = "🌐"
        elif "Bilgi Tabanı" in sender or "Fonksiyonel" in sender: avatar_icon = "📚"
        else: avatar_icon = "🤖"

    with st.chat_message("user" if is_user else "assistant", avatar=avatar_icon):
        # Kopyala butonu için (kod blokları)
        if "```" in message_content:
            parts = message_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1: # Kod bloğu
                    lang_match = re.match(r"(\w+)\n", part)
                    lang = lang_match.group(1) if lang_match else None
                    code_content = part[len(lang)+1:] if lang else part
                    st.code(code_content, language=lang)
                    if st.button("📋 Kopyala", key=f"copy_code_{message_index}_{i}", help="Kodu kopyala"):
                        st.write_to_clipboard(code_content); st.toast("Kod kopyalandı!", icon="✅")
                else:
                    st.markdown(part, unsafe_allow_html=True) # HTML'e izin ver (dikkatli kullanılmalı)
        else:
            st.markdown(message_content, unsafe_allow_html=True)

        if not is_user: # AI mesajları için işlemler
            source_name = sender.split('(')[-1].replace(')','').strip() if '(' in sender else sender.replace(f'{APP_NAME} ','')
            cols_ai_actions = st.columns([0.6, 0.2, 0.2]) # Kaynak, Oku, Geri bildirim butonu
            with cols_ai_actions[0]:
                st.caption(f"Kaynak: {source_name}")
            with cols_ai_actions[1]:
                if st.session_state.tts_enabled and tts_engine and message_content:
                    if st.button("🔊", key=f"speak_msg_chat_{message_index}", help="Mesajı sesli oku", use_container_width=True):
                        speak(message_content) # Sadece text kısmını oku (markdown'sız)
            with cols_ai_actions[2]:
                 # Basit bir butonla geri bildirim formunu aç/kapat
                if st.button("✍️", key=f"toggle_feedback_{message_index}", help="Bu yanıt hakkında geri bildirim ver", use_container_width=True):
                    st.session_state.current_message_id_for_feedback = f"chat_{message_index}"
                    st.session_state.last_user_prompt_for_feedback = st.session_state.chat_history[message_index-1][1] if message_index > 0 else "N/A"
                    st.session_state.last_ai_response_for_feedback = message_content
                    st.session_state.show_feedback_comment_form = not st.session_state.get('show_feedback_comment_form', False)
                    st.rerun()


def display_feedback_form():
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        with st.form(key=f"feedback_form_{st.session_state.current_message_id_for_feedback}"):
            st.markdown(f"**'{st.session_state.last_user_prompt_for_feedback[:50]}...' istemine verilen yanıt için geri bildirim:**")
            feedback_type = st.radio("Değerlendirme:", ["👍 Beğendim", "👎 Beğenmedim"], horizontal=True, key="feedback_type_radio")
            comment = st.text_area("Yorumunuz (isteğe bağlı):", value=st.session_state.get('feedback_comment_input', ""), key="feedback_comment_text_area")
            submit_feedback = st.form_submit_button("✅ Geri Bildirimi Gönder")

            if submit_feedback:
                parsed_feedback_type = "positive" if feedback_type == "👍 Beğendim" else "negative"
                if log_feedback(
                    st.session_state.current_message_id_for_feedback,
                    st.session_state.last_user_prompt_for_feedback,
                    st.session_state.last_ai_response_for_feedback,
                    parsed_feedback_type,
                    comment
                ):
                    st.session_state.show_feedback_comment_form = False
                    st.session_state.feedback_comment_input = "" # Formu temizle
                    st.session_state.current_message_id_for_feedback = None # ID'yi temizle
                    st.rerun()
        st.markdown("---")

def display_chat_interface_main():
    chat_display_container = st.container() # height=600 eklenebilir
    with chat_display_container:
        if not st.session_state.chat_history:
            st.info(f"Merhaba {st.session_state.user_name}! Nasıl yardımcı olabilirim? Aşağıdan mesajınızı yazabilirsiniz.", icon="👋")
        for i, (sender, message_content) in enumerate(st.session_state.chat_history):
            display_chat_message_with_feedback(sender, message_content, i, sender.startswith("Sen"))

    display_feedback_form() # Geri bildirim formunu göster (eğer aktifse)

    if prompt := st.chat_input(f"{st.session_state.user_name} olarak mesajınızı yazın...", key="main_chat_input_field"):
        current_message_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"
        st.session_state.message_id_counter += 1
        st.session_state.chat_history.append(("Sen", prompt))

        # Gemini'ye gönderilecek geçmişi hazırla (son N mesaj veya token bazlı)
        # Şimdilik son 10 etkileşim (kullanıcı + model = 1 etkileşim)
        history_for_gemini_raw = st.session_state.chat_history[-21:-1] # En fazla son 10 çift (20 mesaj) + son kullanıcı mesajı hariç
        gemini_history_formatted = [{'role': ("user" if s.startswith("Sen") else "model"), 'parts': [m]} for s, m in history_for_gemini_raw]

        with st.chat_message("assistant", avatar="⏳"):
            placeholder = st.empty(); placeholder.markdown("🧠 _Düşünüyorum..._")
            time.sleep(0.1) # Efekt için

        response_content, ai_sender = get_hanogt_response_orchestrator(prompt, gemini_history_formatted, current_message_id, stream_enabled=st.session_state.gemini_stream_enabled)

        if st.session_state.gemini_stream_enabled and ai_sender == f"{APP_NAME} (Gemini Stream)":
            full_response_text = ""
            try:
                for chunk_index, chunk in enumerate(response_content): # response_content burada stream objesi
                    if chunk.parts:
                        chunk_text = "".join(part.text for part in chunk.parts if hasattr(part, 'text'))
                        full_response_text += chunk_text
                        placeholder.markdown(full_response_text + "▌")
                        if chunk_index % 5 == 0: time.sleep(0.01) # Çok hızlı akışı biraz yavaşlat
                placeholder.markdown(full_response_text)
                log_interaction(prompt, full_response_text, ai_sender, current_message_id) # Stream bitince logla
                st.session_state.chat_history.append((ai_sender, full_response_text))
            except Exception as e_stream:
                error_text = f"Stream sırasında hata: {e_stream}"
                placeholder.error(error_text); st.session_state.chat_history.append((f"{APP_NAME} (Hata)", error_text))
        else:
            placeholder.empty() # "Düşünüyorum" mesajını kaldır
            # Loglama zaten get_hanogt_response_orchestrator içinde yapılıyor (stream olmayan durumlar için)
            st.session_state.chat_history.append((ai_sender, str(response_content)))

        save_chat_history(st.session_state.chat_history)
        if st.session_state.tts_enabled and tts_engine and isinstance(response_content, str) and not (st.session_state.gemini_stream_enabled and ai_sender == f"{APP_NAME} (Gemini Stream)"):
            # Stream ise, full_response_text okunabilir, ama kullanıcı deneyimi için belki stream sonrası butonla okunmalı.
            # Şimdilik sadece stream olmayan yanıtları otomatik oku.
            speak(response_content)
        st.rerun()


# --- UYGULAMA ANA AKIŞI ---
st.markdown(f"<h1 style='text-align: center; color: #0078D4;'>🚀 {APP_NAME} {APP_VERSION} 🚀</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-style: italic;'>Yapay zeka destekli süper asistanınız!</p>", unsafe_allow_html=True)

# Başlangıç Hataları
if gemini_init_error: st.error(gemini_init_error, icon="🛑")
if supabase_error: st.error(supabase_error, icon="🧱")
if tts_init_error and st.session_state.tts_enabled: st.toast(tts_init_error, icon="🔇")


if not st.session_state.show_main_app: # Kullanıcı Giriş Ekranı
    st.subheader("👋 Merhaba! Başlamadan önce tanışalım...")
    form_cols = st.columns([0.1, 0.8, 0.1])
    with form_cols[1]:
        with st.form("user_details_form_main"):
            name_input = st.text_input("Size nasıl hitap etmeliyim?", placeholder="İsminiz veya takma adınız...", value=st.session_state.get('user_name_temp', ''), key="name_input_login")
            submitted = st.form_submit_button("✨ Başlayalım!", use_container_width=True, type="primary")
            if submitted:
                if name_input and name_input.strip():
                    st.session_state.user_name = name_input.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False
                    st.rerun()
                else: st.error("Lütfen bir isim girin.")
else: # Ana Uygulama
    if not st.session_state.greeting_message_shown and st.session_state.user_name:
        greeting = random.choice([f"Tekrar hoş geldin, {st.session_state.user_name}!", f"Merhaba {st.session_state.user_name}! Senin için hazırım.", f"Harika bir gün, {st.session_state.user_name}! Neler yapıyoruz?"])
        st.success(greeting, icon="🎉"); st.session_state.greeting_message_shown = True
        st.balloons()

    display_sidebar_content() # Kenar çubuğunu göster

    # Mod Seçimi
    mode_options = {
        "Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya)": "🎤",
        "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️"
    }
    selected_mode_key = st.radio(
        "Uygulama Modu Seçin:", options=list(mode_options.keys()),
        index=list(mode_options.keys()).index(st.session_state.app_mode),
        format_func=lambda x: f"{mode_options[x]} {x}",
        horizontal=True, label_visibility="collapsed", key="app_mode_radio"
    )
    if selected_mode_key != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode_key; st.rerun()
    app_mode = st.session_state.app_mode
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)


    # Mod Arayüzleri
    if app_mode == "Yazılı Sohbet":
        display_chat_interface_main()
    elif app_mode == "Sesli Sohbet (Dosya)":
        st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası (WAV, MP3, OGG, FLAC, M4A) yükleyin.", icon="📢")
        uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed", key="audio_uploader_page")
        if uploaded_file:
            st.audio(uploaded_file, format=uploaded_file.type)
            user_prompt_audio = None; file_name = uploaded_file.name
            temp_audio_path = f"temp_audio_{st.session_state.session_id}_{file_name[:10]}.wav" # Daha spesifik dosya adı
            with st.spinner(f"🔊 '{file_name}' işleniyor... Lütfen bekleyin."):
                recognizer = sr.Recognizer();
                try: # Ses işleme
                    with open(temp_audio_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    with sr.AudioFile(temp_audio_path) as source:
                        # recognizer.adjust_for_ambient_noise(source, duration=0.3) # Gürültü azaltma (opsiyonel)
                        audio_data = recognizer.record(source)
                    user_prompt_audio = recognizer.recognize_google(audio_data, language="tr-TR")
                    st.success(f"**🎙️ Algılanan Metin:**\n\n> {user_prompt_audio}")
                except sr.UnknownValueError: st.error("🔇 Ses anlaşılamadı. Lütfen daha net bir dosya deneyin.")
                except sr.RequestError as e_sr: st.error(f"🤖 Ses tanıma servisine ulaşılamadı: {e_sr}.")
                except Exception as e_audio: st.error(f"Ses dosyası işlenirken beklenmedik bir hata: {e_audio}")
                finally:
                    if os.path.exists(temp_audio_path): os.remove(temp_audio_path)

            if user_prompt_audio: # Yanıt oluşturma
                current_message_id_audio = f"audio_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                st.session_state.message_id_counter += 1
                st.session_state.chat_history.append(("Sen (Ses Dosyası)", user_prompt_audio))
                history_for_gemini_audio = [{'role': ("user" if s.startswith("Sen") else "model"), 'parts': [m]} for s, m in st.session_state.chat_history[-21:-1]]
                with st.spinner("🤖 Yanıtınız hazırlanıyor..."):
                    response_audio, ai_sender_audio = get_hanogt_response_orchestrator(user_prompt_audio, history_for_gemini_audio, current_message_id_audio, stream_enabled=False) # Stream'siz
                st.markdown(f"#### {ai_sender_audio} Yanıtı:"); st.markdown(response_audio)
                if st.session_state.tts_enabled and tts_engine and response_audio:
                    if st.button("🔊 Yanıtı Seslendir", key="speak_audio_response_btn_page"): speak(response_audio)
                st.session_state.chat_history.append((ai_sender_audio, response_audio)); save_chat_history(st.session_state.chat_history)
                st.success("✅ Yanıt başarıyla oluşturuldu ve sohbete eklendi!")

    elif app_mode == "Yaratıcı Stüdyo":
        st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Hanogt AI size ilham verici ve yaratıcı bir yanıt oluştursun!", icon="💡")
        creative_prompt_text = st.text_area("Yaratıcılık Tohumunuz:", key="creative_input_studio_page", placeholder="Örneğin: 'Geceleri parlayan bir çöl çiçeği hakkında kısa bir hikaye'", height=120)
        cols_creative_options = st.columns(2)
        with cols_creative_options[0]:
            length_pref = st.selectbox("Yanıt Uzunluğu:", ["kısa", "orta", "uzun"], index=1, key="creative_length_pref_page", help="İstenen yanıtın yaklaşık uzunluğu.")
        with cols_creative_options[1]:
            style_pref = st.selectbox("Yaratıcılık Stili:", ["genel", "şiirsel", "hikaye"], index=0, key="creative_style_pref_page", help="Yanıtın üslubu.")

        if st.button("✨ Fikir Üret!", key="generate_creative_btn_page", type="primary", use_container_width=True):
            if creative_prompt_text and creative_prompt_text.strip():
                final_response_creative = None; ai_sender_creative = f"{APP_NAME} (Yaratıcı)"
                current_message_id_creative = f"creative_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                st.session_state.message_id_counter += 1

                if gemini_model: # Gemini'yi dene
                    with st.spinner("✨ Gemini ilham perileriyle fısıldaşıyor..."):
                        gemini_creative_system_prompt = (
                            f"Sen çok yaratıcı ve hayal gücü geniş bir asistansın. Sana verilen '{creative_prompt_text}' istemine dayanarak, "
                            f"'{style_pref}' stilinde ve yaklaşık '{length_pref}' uzunlukta özgün, ilginç ve sanatsal bir metin oluştur. "
                            "Sıradanlıktan kaçın, okuyucuyu etkileyecek bir dil kullan."
                        )
                        gemini_resp = get_gemini_response_cached(gemini_creative_system_prompt, [], stream=False) # Stream'siz, geçmişsiz
                        if gemini_resp and not (isinstance(gemini_resp, str) and gemini_resp.startswith(GEMINI_ERROR_PREFIX)):
                            final_response_creative = str(gemini_resp); ai_sender_creative = f"{APP_NAME} (Gemini Yaratıcı)"
                        else: st.warning(f"Gemini yaratıcı yanıtı alınamadı. Yerel modül kullanılacak. (Hata: {gemini_resp if isinstance(gemini_resp, str) else 'Bilinmeyen'})", icon="⚠️")

                if not final_response_creative: # Yerel modül
                    with st.spinner("✨ Kendi fikirlerimi demliyorum... Hayal gücüm çalışıyor..."):
                        time.sleep(0.2) # Efekt
                        local_creative_text = creative_response_generator(creative_prompt_text, length_preference=length_pref, style_preference=style_pref)
                        new_word = advanced_word_generator(creative_prompt_text.split()[0] if creative_prompt_text else "kelime")
                        final_response_creative = f"{local_creative_text}\n\n---\n🔮 **Kelimatörden Türetilen Sözcük:** {new_word}"
                        ai_sender_creative = f"{APP_NAME} (Yerel Yaratıcı)"

                st.markdown(f"#### {ai_sender_creative} İlhamı:"); st.markdown(final_response_creative)
                if st.session_state.tts_enabled and tts_engine and final_response_creative:
                    if st.button("🔊 İlhamı Dinle", key="speak_creative_response_btn_page"):
                        speak(final_response_creative.split("🔮 **Kelimatörden Türetilen Sözcük:**")[0].strip()) # Sadece ana metni oku
                log_interaction(creative_prompt_text, final_response_creative, ai_sender_creative, current_message_id_creative)
                st.success("✨ Yaratıcı yanıtınız hazır!")
            else: st.error("Lütfen yaratıcılığınızı ateşleyecek bir şeyler yazın!", icon="✍️")

    elif app_mode == "Görsel Oluşturucu":
        st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!", icon="🎨")
        st.info("ℹ️ Not: Bu mod, girilen anahtar kelimelere (örn: güneş, deniz, ağaç, ay, gökyüzü, orman, dağ, şehir) göre basit, kural tabanlı çizimler yapar. Gerçekçi görüntüler beklemeyin, bu daha çok sembolik bir yorumlayıcıdır.", icon="💡")
        image_prompt_text = st.text_input("Ne çizmemi istersiniz? (Anahtar kelimeler kullanın)", key="image_input_generator_page", placeholder="Örn: 'Gece vakti karlı dağların üzerinde parlayan bir dolunay'")
        if st.button("🖼️ Görseli Oluştur!", key="generate_rule_image_btn_page", type="primary", use_container_width=True):
            if image_prompt_text and image_prompt_text.strip():
                with st.spinner("🖌️ Fırçalarım ve renklerim hazırlanıyor... Hayaliniz çiziliyor..."):
                    time.sleep(0.3) # Efekt
                    generated_image_obj = generate_prompt_influenced_image(image_prompt_text)
                st.image(generated_image_obj, caption=f"{APP_NAME}'ın '{image_prompt_text[:60]}' yorumu (Kural Tabanlı)", use_container_width=True)
                try: # İndirme butonu
                    buf = BytesIO(); generated_image_obj.save(buf, format="PNG"); byte_im = buf.getvalue()
                    file_name_prompt_clean = re.sub(r'[^\w\s-]', '', image_prompt_text.lower())
                    file_name_prompt_clean = re.sub(r'\s+', '_', file_name_prompt_clean).strip('_')[:35]
                    download_file_name = f"hanogt_ai_cizim_{file_name_prompt_clean or 'gorsel'}_{int(time.time())}.png"
                    st.download_button(label="🖼️ Görseli İndir (PNG)", data=byte_im, file_name=download_file_name, mime="image/png", use_container_width=True)
                except Exception as e_download: st.error(f"Görsel indirilirken bir hata oluştu: {e_download}", icon="⚠️")
            else: st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!", icon="✍️")

    # --- Alt Bilgi ---
    st.markdown("<hr style='margin-top: 1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center; font-size: 0.8rem; color: #888;'>"
        f"{APP_NAME} v{APP_VERSION} - {st.session_state.get('user_name', 'Misafir')} için çalışıyor - © 2024-{CURRENT_YEAR}"
        f"<br>Gemini Modeli: {'Aktif' if gemini_model else 'Devre Dışı'} ({st.session_state.gemini_model_name if gemini_model else 'N/A'}) | "
        f"Supabase Loglama: {'Aktif' if supabase else 'Devre Dışı'}"
        f"</p>", unsafe_allow_html=True
    )
