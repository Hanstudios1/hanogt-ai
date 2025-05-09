# app.py

# --- Gerekli Kütüphaneler ---
import streamlit as st
import requests
from bs4 import BeautifulSoup # lxml parser için pip install lxml de gerekebilir
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
#     # tiktoken_encoder = tiktoken.get_encoding("cl100k_base") # Örnek bir encoder
#     # tiktoken_encoder = tiktoken.encoding_for_model("gemini-1.5-flash-latest") # Modele göre encoder
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
    SupabaseAPIError = None # Tanımlı değilse None yapalım

# --- Sayfa Yapılandırması (İLK STREAMLIT KOMUTU OLMALI!) ---
st.set_page_config(
    page_title="Hanogt AI Pro+",
    page_icon="🚀",
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
REQUEST_TIMEOUT = 18
SCRAPE_MAX_CHARS = 3000
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 {APP_NAME}/{APP_VERSION}"
SUPABASE_TABLE_LOGS = "chat_logs"
SUPABASE_TABLE_FEEDBACK = "user_feedback"
FONT_FILE = "arial.ttf" # Görsel oluşturucu için kullanılacak font dosyası adı

# --- Bilgi Tabanı ---
knowledge_base_load_error = None

@st.cache_data(ttl=3600) # Bilgi tabanını 1 saat cache'le
def load_knowledge_from_file(filename=KNOWLEDGE_BASE_FILE):
    global knowledge_base_load_error
    dynamic_functions = {
        "saat kaç": lambda: f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}",
        "bugün ayın kaçı": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}",
        "tarih ne": lambda: f"Bugün {datetime.now().strftime('%d %B %Y, %A')}"
    }
    default_knowledge = {
        "merhaba": ["Merhaba!", "Selam!", "Hoş geldin!", f"Size nasıl yardımcı olabilirim, {st.session_state.get('user_name', 'kullanıcı')}?"],
        "selam": ["Merhaba!", "Selam sana da!", "Nasıl gidiyor?"],
        "nasılsın": ["İyiyim, teşekkürler! Siz nasılsınız?", "Harika hissediyorum, yardımcı olmak için buradayım!", "Her şey yolunda, sizin için ne yapabilirim?"],
        "hanogt kimdir": [f"Ben {APP_NAME} ({APP_VERSION}), Streamlit ve Python ile geliştirilmiş bir yapay zeka asistanıyım.", f"{APP_NAME} ({APP_VERSION}), sorularınızı yanıtlamak, metinler üretmek ve hatta basit görseller oluşturmak için tasarlandı."],
        "teşekkür ederim": ["Rica ederim!", "Ne demek!", "Yardımcı olabildiğime sevindim.", "Her zaman!"],
        "görüşürüz": ["Görüşmek üzere!", "Hoşça kal!", "İyi günler dilerim!", "Tekrar beklerim!"],
        "adın ne": [f"Ben {APP_NAME}, versiyon {APP_VERSION}.", f"Bana {APP_NAME} diyebilirsiniz."],
        "ne yapabilirsin": ["Sorularınızı yanıtlayabilir, metin özetleyebilir, web'de arama yapabilir, yaratıcı metinler üretebilir ve basit görseller çizebilirim.", "Size çeşitli konularda yardımcı olabilirim. Ne merak ediyorsunuz?"],
        "saat kaç": [dynamic_functions["saat kaç"]],
        "bugün ayın kaçı": [dynamic_functions["bugün ayın kaçı"]],
        "tarih ne": [dynamic_functions["tarih ne"]],
        "hava durumu": ["Üzgünüm, şu an için güncel hava durumu bilgisi sağlayamıyorum. Bunun için özel bir hava durumu servisine göz atabilirsiniz.", "Hava durumu servisim henüz aktif değil, ancak bu konuda bir geliştirme yapmayı planlıyorum!"]
    }

    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                loaded_kb = json.load(f)
            for key, value_list in loaded_kb.items():
                if isinstance(value_list, list):
                    for i, val_str in enumerate(value_list):
                        if val_str == "<function>" and key in dynamic_functions:
                            loaded_kb[key][i] = dynamic_functions[key]
            knowledge_base_load_error = None
            return loaded_kb
        else:
            knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) bulunamadı. Varsayılan kullanılıyor."
            return default_knowledge
    except json.JSONDecodeError:
        knowledge_base_load_error = f"Bilgi tabanı dosyası ({filename}) hatalı formatta. Varsayılan kullanılıyor."
        return default_knowledge
    except Exception as e:
        knowledge_base_load_error = f"Bilgi tabanı yüklenirken bilinmeyen bir hata oluştu: {e}. Varsayılan kullanılıyor."
        return default_knowledge

KNOWLEDGE_BASE = load_knowledge_from_file()
if knowledge_base_load_error: st.toast(knowledge_base_load_error, icon="⚠️")

def kb_chatbot_response(query, knowledge):
    query_lower = query.lower().strip()
    if query_lower in knowledge:
        response_options = knowledge[query_lower]
        chosen_response = random.choice(response_options)
        return chosen_response() if callable(chosen_response) else chosen_response

    possible_responses = []
    for key, responses in knowledge.items():
        if key in query_lower:
            for resp_opt in responses:
                possible_responses.append(resp_opt() if callable(resp_opt) else resp_opt)
    if possible_responses: return random.choice(list(set(possible_responses)))

    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_match_score = 0; best_responses_options = []
    for key, responses in knowledge.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words: continue
        common_words = query_words.intersection(key_words)
        score = len(common_words) / len(key_words) if len(key_words) > 0 else 0
        if score > 0.6:
            if score > best_match_score:
                best_match_score = score; best_responses_options = responses
            elif score == best_match_score: best_responses_options.extend(responses)
    if best_responses_options:
        chosen_response = random.choice(list(set(best_responses_options)))
        return chosen_response() if callable(chosen_response) else chosen_response
    return None

# --- API Anahtarı ve Gemini Yapılandırması ---
gemini_model = None
gemini_init_error_global = None

def initialize_gemini_model():
    global gemini_init_error_global
    api_key_local = st.secrets.get("GOOGLE_API_KEY")
    if not api_key_local:
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta (st.secrets) bulunamadı! Gemini özellikleri kısıtlı olacak."
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
                top_p=st.session_state.get('gemini_top_p', 0.95),
                top_k=st.session_state.get('gemini_top_k', 40),
                max_output_tokens=st.session_state.get('gemini_max_tokens', 4096)
            )
        )
        gemini_init_error_global = None
        return model
    except Exception as e:
        gemini_init_error_global = f"🛑 Gemini yapılandırma hatası: {e}. Lütfen API anahtarınızı ve internet bağlantınızı kontrol edin."
        return None

# --- Supabase İstemcisini Başlatma ---
supabase = None
supabase_error_global = None

@st.cache_resource(ttl=3600)
def init_supabase_client_cached():
    global supabase_error_global
    supabase_url_local = st.secrets.get("SUPABASE_URL")
    supabase_key_local = st.secrets.get("SUPABASE_SERVICE_KEY")
    if not create_client:
        supabase_error_global = "Supabase kütüphanesi yüklenemedi. Loglama çalışmayacak."
        return None
    if not supabase_url_local or not supabase_key_local:
        supabase_error_global = "Supabase URL veya Service Key Secrets'ta bulunamadı! Loglama özelliği devre dışı kalacak."
        return None
    try:
        client = create_client(supabase_url_local, supabase_key_local)
        supabase_error_global = None
        return client
    except Exception as e:
        error_msg_supabase = f"Supabase bağlantı hatası: {e}. Loglama yapılamayacak."
        if "failed to parse" in str(e).lower() or "invalid url" in str(e).lower():
            error_msg_supabase += " Lütfen Supabase URL'inizin doğru formatta olduğundan emin olun (örn: https://xyz.supabase.co)."
        elif "invalid key" in str(e).lower():
            error_msg_supabase += " Lütfen Supabase Service Key'inizin doğru olduğundan emin olun."
        supabase_error_global = error_msg_supabase
        return None

# --- YARDIMCI FONKSİYONLAR ---
def _get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

tts_engine = None
tts_init_error_global = None

@st.cache_resource
def init_tts_engine_cached():
    global tts_init_error_global
    try:
        engine = pyttsx3.init()
        tts_init_error_global = None
        return engine
    except Exception as e:
        tts_init_error_global = f"⚠️ Metin okuma (TTS) motoru başlatılamadı: {e}. Bu özellik kullanılamayacak."
        return None

def speak(text_to_speak):
    current_tts_engine = globals().get('tts_engine')
    if not current_tts_engine or not st.session_state.get('tts_enabled', True):
        if current_tts_engine: st.toast("Metin okuma özelliği kapalı.", icon="🔇")
        else: st.toast("Metin okuma motoru aktif değil veya başlatılamadı.", icon="🔇")
        return
    try:
        current_tts_engine.say(text_to_speak)
        current_tts_engine.runAndWait()
    except RuntimeError as re_tts:
        st.warning(f"Konuşma motorunda bir çalışma zamanı sorunu oluştu: {re_tts}", icon="🔊")
    except Exception as e_tts:
        st.error(f"Konuşma sırasında beklenmedik bir hata oluştu: {e_tts}", icon="🔊")

def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def scrape_url_content(url: str, timeout: int = REQUEST_TIMEOUT, max_chars: int = SCRAPE_MAX_CHARS) -> str | None:
    st.toast(f"🌐 '{urlparse(url).netloc}' sayfasından içerik alınıyor...", icon="⏳")
    try:
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']:
            st.warning(f"Geçersiz URL formatı: {url}", icon="🔗"); return None
        headers = {
            'User-Agent': USER_AGENT,
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            st.info(f"URL HTML içeriği değil ('{content_type}' tipinde). Kazıma atlanıyor: {url}", icon="📄"); return None

        html_content = ""; content_length_processed = 0
        max_html_size_to_process = max_chars * 10
        for chunk in response.iter_content(chunk_size=16384, decode_unicode=True, errors='ignore'):
            html_content += chunk
            content_length_processed += len(chunk.encode('utf-8', 'ignore'))
            if content_length_processed > max_html_size_to_process:
                st.warning(f"HTML içeriği çok büyük ({content_length_processed / 1024:.0f}KB), ilk kısmı işlenecek.", icon="✂️"); break
        response.close()

        soup = BeautifulSoup(html_content, 'lxml')
        for element_to_remove in soup(["script", "style", "nav", "footer", "aside", "form", "button", "iframe", "header", "noscript", "link", "meta", "img", "svg", "video", "audio", "figure", "figcaption"]):
            element_to_remove.decompose()

        potential_content_parts = []
        content_selectors = [
            'article[class*="content"]', 'article[class*="post"]', 'article[class*="entry"]',
            'main[class*="content"]', 'main[id*="content"]',
            'div[class*="post-body"]', 'div[class*="article-body"]', 'div[class*="entry-content"]',
            'div[itemprop="articleBody"]',
            'article', 'main', '.content', '.post-content', '.entry-content', 'section[role="main"]'
        ]
        content_found_flag = False
        for selector in content_selectors:
            elements_found = soup.select(selector)
            if elements_found:
                container_element = elements_found[0]
                paragraphs_and_divs = container_element.find_all(['p', 'div'], recursive=False, limit=35)
                temp_content_list = []
                for p_or_div in paragraphs_and_divs:
                    text_from_element = _clean_text(p_or_div.get_text(separator=' ', strip=True))
                    if len(text_from_element) > 100 and (text_from_element.count('.') + text_from_element.count('!') + text_from_element.count('?')) >= 1:
                        temp_content_list.append(text_from_element)
                if len(" ".join(temp_content_list)) > 500:
                    potential_content_parts = temp_content_list
                    content_found_flag = True; break
        
        if not content_found_flag:
            body_text_content = _clean_text(soup.body.get_text(separator=' ', strip=True) if soup.body else "")
            if len(body_text_content) > 300:
                st.toast("Özel içerik alanları bulunamadı, sayfanın genel metni kullanıldı.", icon="ℹ️")
                potential_content_parts = [body_text_content]
            else:
                st.toast("Sayfada anlamlı metin içeriği bulunamadı.", icon="📄"); return None

        full_text_content = "\n\n".join(potential_content_parts)
        cleaned_text_content = _clean_text(full_text_content)
        if not cleaned_text_content: return None

        final_text_output = cleaned_text_content[:max_chars]
        if len(cleaned_text_content) > max_chars: final_text_output += "..."
        st.toast(f"'{urlparse(url).netloc}' sayfasının içeriği başarıyla alındı.", icon="✅")
        return final_text_output

    except requests.exceptions.HTTPError as e_http: st.toast(f"⚠️ Sayfa alınırken HTTP hatası ({e_http.response.status_code}): {url}", icon='🌐')
    except requests.exceptions.Timeout: st.toast(f"⚠️ Sayfa alınırken zaman aşımı oluştu: {url}", icon='⏳')
    except requests.exceptions.ConnectionError: st.toast(f"⚠️ Sayfa bağlantı hatası (siteye ulaşılamıyor olabilir): {url}", icon='🔌')
    except requests.exceptions.RequestException as e_req: st.toast(f"⚠️ Sayfa alınırken genel bir ağ hatası: {e_req}", icon='🌐')
    except Exception as e_scrape: st.toast(f"⚠️ Sayfa içeriği işlenirken beklenmedik bir hata: {e_scrape}", icon='⚙️')
    return None

def search_web(query: str) -> str | None:
    st.toast(f"🔍 '{query}' için web'de arama yapılıyor...", icon="⏳")
    wikipedia.set_lang("tr")
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True, redirect=True)
        st.toast(f"ℹ️ '{query}' için Wikipedia'dan bilgi bulundu.", icon="✅")
        return f"**Wikipedia'dan ({query}):**\n\n{_clean_text(summary)}"
    except wikipedia.exceptions.PageError:
        st.toast(f"ℹ️ '{query}' için Wikipedia'da doğrudan eşleşen bir sayfa bulunamadı.", icon="🤷")
    except wikipedia.exceptions.DisambiguationError as e_disamb:
        options_text = "\n\nWikipedia'da olası başlıklar (ilk 3):\n" + "\n".join([f"- {opt}" for opt in e_disamb.options[:3]])
        st.toast(f"Wikipedia'da '{query}' için birden fazla anlam bulundu. Daha spesifik bir arama yapabilirsiniz.", icon="📚")
        return f"**Wikipedia'da Birden Fazla Anlam Bulundu ({query}):**\n{str(e_disamb).splitlines()[0]}{options_text}"
    except Exception: pass

    ddg_result_text = None; ddg_url_source = None
    try:
        with DDGS(headers={'User-Agent': USER_AGENT}, timeout=REQUEST_TIMEOUT) as ddgs_search:
            results = list(ddgs_search.text(query, region='tr-tr', safesearch='moderate', max_results=3))
            if results:
                for res_item in results:
                    snippet_text = res_item.get('body')
                    temp_source_url = res_item.get('href')
                    if snippet_text and temp_source_url:
                        decoded_url_source = unquote(temp_source_url)
                        st.toast(f"ℹ️ DuckDuckGo'dan '{urlparse(decoded_url_source).netloc}' için özet bulundu.", icon="🦆")
                        ddg_result_text = f"**Web Özeti (DuckDuckGo - {urlparse(decoded_url_source).netloc}):**\n\n{_clean_text(snippet_text)}\n\nKaynak: {decoded_url_source}"
                        ddg_url_source = decoded_url_source; break
    except Exception: pass

    if ddg_url_source:
        scraped_content_from_url = scrape_url_content(ddg_url_source)
        if scraped_content_from_url:
            return f"**Web Sayfasından Detaylı Bilgi ({urlparse(ddg_url_source).netloc}):**\n\n{scraped_content_from_url}\n\nTam İçerik İçin Kaynak Adres: {ddg_url_source}"
        elif ddg_result_text:
             st.toast("ℹ️ Sayfa içeriği kazınamadı, DuckDuckGo özeti kullanılıyor.", icon="📝")
             return ddg_result_text
        else:
            return f"Detaylı bilgi için şu adresi ziyaret edebilirsiniz: {ddg_url_source}"

    if ddg_result_text:
        return ddg_result_text

    st.toast(f"'{query}' için web'de kapsamlı bir yanıt bulunamadı.", icon="❌"); return None

@st.cache_data(ttl=86400)
def load_chat_history_cached(file_path: str = CHAT_HISTORY_FILE) -> list:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f: content_read = f.read()
            if content_read and content_read.strip(): return json.loads(content_read)
            else: return []
        except json.JSONDecodeError:
            st.error(f"Sohbet geçmişi dosyası ({file_path}) bozuk veya hatalı formatta. Yeni bir geçmiş başlatılıyor.")
            try: os.rename(file_path, f"{file_path}.backup_{int(time.time())}")
            except OSError: pass
            return []
        except Exception as e_load_hist:
            st.error(f"Sohbet geçmişi dosyası ({file_path}) yüklenirken bir hata oluştu: {e_load_hist}"); return []
    return []

def save_chat_history(history_to_save: list, file_path: str = CHAT_HISTORY_FILE):
    try:
        with open(file_path, "w", encoding="utf-8") as f_save:
            json.dump(history_to_save, f_save, ensure_ascii=False, indent=2)
    except Exception as e_save_hist: st.error(f"Sohbet geçmişi kaydedilemedi: {e_save_hist}")

def get_gemini_response_cached(prompt_text: str, chat_history_for_gemini_model: list[dict], stream_output: bool = False) -> str | object:
    current_gemini_model = globals().get('gemini_model')
    if not current_gemini_model: return f"{GEMINI_ERROR_PREFIX} Gemini modeli aktif değil veya başlatılamadı."
    try:
        chat_session = current_gemini_model.start_chat(history=chat_history_for_gemini_model)
        response_from_gemini = chat_session.send_message(prompt_text, stream=stream_output)

        if stream_output: return response_from_gemini
        else:
            if not response_from_gemini.parts:
                if hasattr(response_from_gemini, 'prompt_feedback') and response_from_gemini.prompt_feedback.block_reason:
                    block_reason = response_from_gemini.prompt_feedback.block_reason
                    block_message = response_from_gemini.prompt_feedback.block_reason_message or "Ek detay verilmedi."
                    warning_msg = f"Gemini yanıtı güvenlik nedeniyle engellendi: {block_reason}. Detay: {block_message}"
                    st.warning(warning_msg, icon="🛡️"); return f"{GEMINI_ERROR_PREFIX} {warning_msg}"
                elif response_from_gemini.candidates and hasattr(response_from_gemini.candidates[0], 'finish_reason') and response_from_gemini.candidates[0].finish_reason != 'STOP':
                    finish_reason_gemini = response_from_gemini.candidates[0].finish_reason
                    st.warning(f"Gemini yanıtı tam olarak oluşturulamadı. Sebep: {finish_reason_gemini}", icon="⚠️"); return f"{GEMINI_ERROR_PREFIX} Yanıt tam değil. Sebep: {finish_reason_gemini}."
                else:
                    st.warning(f"Gemini'dan boş veya beklenmedik bir yanıt alındı: {response_from_gemini}", icon="⁉️"); return f"{GEMINI_ERROR_PREFIX} Boş veya anlaşılamayan yanıt."
            return "".join(part.text for part in response_from_gemini.parts if hasattr(part, 'text'))

    except genai.types.BlockedPromptException as bpe_gemini:
        st.error(f"Gemini İstem Engelleme Hatası: Gönderdiğiniz istem güvenlik filtrelerini tetikledi. Lütfen isteminizi gözden geçirin. Detay: {bpe_gemini}", icon="🛡️")
        return f"{GEMINI_ERROR_PREFIX} İstem güvenlik nedeniyle engellendi."
    except genai.types.StopCandidateException as sce_gemini:
        st.error(f"Gemini Yanıt Kesintisi: Yanıt oluşturulurken beklenmedik bir durma yaşandı. Detay: {sce_gemini}", icon="🛑")
        return f"{GEMINI_ERROR_PREFIX} Yanıt oluşturulurken kesildi."
    except requests.exceptions.ReadTimeout:
        st.error("Gemini API isteği zaman aşımına uğradı (ReadTimeout). Lütfen internet bağlantınızı kontrol edip tekrar deneyin.", icon="⏳")
        return f"{GEMINI_ERROR_PREFIX} API okuma zaman aşımı."
    except Exception as e_gemini_api:
        error_message_gemini = f"Gemini API ile iletişimde bir hata oluştu: {e_gemini_api}"
        st.error(error_message_gemini, icon="📡")
        if "API key not valid" in str(e_gemini_api).lower(): return f"{GEMINI_ERROR_PREFIX} Google API Anahtarı geçersiz veya hatalı."
        elif "Deadline Exceeded" in str(e_gemini_api).lower() or "504" in str(e_gemini_api).lower() or "timeout" in str(e_gemini_api).lower():
            return f"{GEMINI_ERROR_PREFIX} API isteği zaman aşımına uğradı. Lütfen tekrar deneyin."
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı: {str(e_gemini_api)[:150]}..."

def log_to_supabase(table_name: str, data_to_log: dict):
    current_supabase_client = globals().get('supabase')
    if not current_supabase_client:
        print(f"INFO: Supabase istemcisi None, '{table_name}' tablosuna loglama atlanıyor.")
        return False
    try:
        insert_result_supabase = current_supabase_client.table(table_name).insert(data_to_log).execute()
        if hasattr(insert_result_supabase, 'data') and not insert_result_supabase.data and hasattr(insert_result_supabase, 'error') and insert_result_supabase.error:
            error_info_supabase = insert_result_supabase.error; error_message_log = str(error_info_supabase)
            if SupabaseAPIError and isinstance(error_info_supabase, SupabaseAPIError):
               error_message_log = f"Supabase API Hatası - Kod: {error_info_supabase.code}, Mesaj: {error_info_supabase.message}, Detay: {error_info_supabase.details}, İpucu: {error_info_supabase.hint}"
            st.toast(f"⚠️ '{table_name}' logu Supabase'e kaydedilemedi: {error_message_log}", icon="💾")
            print(f"WARN: Supabase '{table_name}' tablosuna insert işlemi başarısız. Hata: {error_message_log}")
            return False
        elif not insert_result_supabase.data and not hasattr(insert_result_supabase, 'error'):
             st.toast(f"⚠️ '{table_name}' logu Supabase'e kaydedildi ancak sunucudan boş yanıt alındı.", icon="💾")
             print(f"WARN: Supabase '{table_name}' insert başarılı ancak dönen data boş. Result: {insert_result_supabase}")
             return True
        return True
    except SupabaseAPIError as e_supabase_api:
        print(f"ERROR: Supabase API hatası ('{table_name}'): {e_supabase_api.message} (Kod: {e_supabase_api.code})")
        st.error(f"Supabase loglama sırasında API hatası: {e_supabase_api.message}")
        return False
    except Exception as e_supabase_log:
        print(f"ERROR: Supabase '{table_name}' tablosuna loglama sırasında kritik bir hata: {e_supabase_log}")
        st.error(f"Supabase '{table_name}' tablosuna loglama sırasında kritik bir hata oluştu! Detay: {type(e_supabase_log).__name__}: {e_supabase_log}")
        return False

def log_interaction(user_prompt_text: str, ai_response_text: str, response_source_info: str, message_unique_id: str):
    interaction_log_data = {
        "user_prompt": user_prompt_text, "ai_response": ai_response_text, "response_source": response_source_info,
        "user_name": st.session_state.get('user_name', 'Bilinmiyor'),
        "session_id": _get_session_id(), "app_version": APP_VERSION, "message_id": message_unique_id
    }
    return log_to_supabase(SUPABASE_TABLE_LOGS, interaction_log_data)

def log_feedback(message_unique_id: str, user_prompt_text: str, ai_response_text: str, feedback_category: str, user_comment: str = ""):
    feedback_log_data = {
        "message_id": message_unique_id, "user_prompt": user_prompt_text, "ai_response": ai_response_text,
        "feedback_type": feedback_category, "comment": user_comment,
        "user_name": st.session_state.get('user_name', 'Bilinmiyor'),
        "session_id": _get_session_id(), "app_version": APP_VERSION
    }
    if log_to_supabase(SUPABASE_TABLE_FEEDBACK, feedback_log_data):
        st.toast(f"Geri bildiriminiz için teşekkür ederiz!", icon="💌"); return True
    else: st.toast(f"Üzgünüz, geri bildiriminiz gönderilemedi. Lütfen daha sonra tekrar deneyin.", icon="😔"); return False

def get_hanogt_response_orchestrator(user_prompt_text: str, chat_history_for_model_processing: list[dict], current_message_unique_id: str, use_stream_output:bool = False) -> tuple[str | object, str]:
    ai_response_content = None; ai_sender_name = APP_NAME

    kb_functional_response = kb_chatbot_response(user_prompt_text, KNOWLEDGE_BASE)
    if kb_functional_response and callable(kb_functional_response):
        try:
            ai_response_content = kb_functional_response()
            ai_sender_name = f"{APP_NAME} (Fonksiyonel)"
            log_interaction(user_prompt_text, ai_response_content, ai_sender_name, current_message_unique_id)
            return ai_response_content, ai_sender_name
        except Exception as e_kb_func:
            st.error(f"Bilgi tabanı fonksiyonel yanıtı işlenirken bir hata oluştu: {e_kb_func}")
            ai_response_content = None

    current_gemini_model = globals().get('gemini_model')
    if current_gemini_model:
        ai_response_content = get_gemini_response_cached(user_prompt_text, chat_history_for_model_processing, stream=use_stream_output)
        if ai_response_content:
            if use_stream_output: return ai_response_content, f"{APP_NAME} (Gemini Stream)"
            elif not (isinstance(ai_response_content, str) and ai_response_content.startswith(GEMINI_ERROR_PREFIX)):
                ai_sender_name = f"{APP_NAME} (Gemini)"
                log_interaction(user_prompt_text, str(ai_response_content), ai_sender_name, current_message_unique_id)
                return str(ai_response_content), ai_sender_name
            ai_response_content = None

    if not ai_response_content:
        st.toast("📚 Bilgi tabanı kontrol ediliyor...", icon="🗂️")
        kb_static_response = kb_chatbot_response(user_prompt_text, KNOWLEDGE_BASE)
        if kb_static_response and not callable(kb_static_response):
            ai_response_content = kb_static_response; ai_sender_name = f"{APP_NAME} (Bilgi Tabanı)"
            log_interaction(user_prompt_text, ai_response_content, ai_sender_name, current_message_unique_id)
            return ai_response_content, ai_sender_name

    if not ai_response_content:
        if len(user_prompt_text.split()) > 2 and \
           ("?" in user_prompt_text or \
            any(keyword in user_prompt_text.lower() for keyword in ["nedir", "kimdir", "nasıl", "ne zaman", "nerede", "anlamı", "hakkında bilgi", "araştır", "son durum"])):
            st.toast("🌐 Web'de arama yapılıyor...", icon="🔍")
            web_search_response = search_web(user_prompt_text)
            if web_search_response:
                ai_response_content = web_search_response; ai_sender_name = f"{APP_NAME} (Web Arama)"
                log_interaction(user_prompt_text, ai_response_content, ai_sender_name, current_message_unique_id)
                return ai_response_content, ai_sender_name
        else:
            st.toast("ℹ️ Kısa veya genel bir istem olduğu için web araması atlandı.", icon="⏩")

    if not ai_response_content:
        st.toast("🤔 Üzgünüm, bu isteğiniz için uygun bir yanıt bulamadım.", icon="🤷")
        default_responses_list = [
            f"Üzgünüm {st.session_state.get('user_name', 'dostum')}, bu konuda size şu an yardımcı olamıyorum. Farklı bir şekilde sorabilir misiniz?",
            "Bu soruyu tam olarak anlayamadım. Daha basit veya farklı kelimelerle tekrar ifade edebilir misiniz?",
            "Bu konuda şu anda bir fikrim yok maalesef. Başka bir konuda yardımcı olabilirim belki?",
            "Yanıt veremiyorum ama her geçen gün yeni şeyler öğrenmeye devam ediyorum! Başka bir sorunuz var mı?",
        ]
        ai_response_content = random.choice(default_responses_list); ai_sender_name = f"{APP_NAME} (Varsayılan)"
        log_interaction(user_prompt_text, ai_response_content, ai_sender_name, current_message_unique_id)

    return ai_response_content, ai_sender_name

def creative_response_generator(user_prompt_text: str, length_preference: str = "orta", style_preference: str = "genel") -> str:
    style_templates_map = {
        "genel": ["Farklı bir bakış açısıyla ele alırsak: {}", "Hayal gücümüzü serbest bırakalım: {}", "Aklıma şöyle bir fikir geldi: {}"],
        "şiirsel": ["Kalbimden dökülen mısralar şöyle fısıldar: {}", "Sözcüklerin dansıyla, bir şiir doğar: {}", "Duyguların ritmiyle, mısralar canlanır: {}"],
        "hikaye": ["Bir zamanlar, uzak diyarlarda başlayan bir hikaye bu: {}", "Perde aralanır ve sahne sizin hayal gücünüzündür: {}", "Her şey o büyülü günde başladı: {}"]
    }
    selected_templates = style_templates_map.get(style_preference, style_templates_map["genel"])
    base_creative_idea = generate_new_idea_creative(user_prompt_text, style_preference)

    if length_preference == "kısa":
        sentences = base_creative_idea.split('.')
        base_creative_idea = ". ".join(sentences[:max(1, len(sentences) // 3)]).strip()
        if base_creative_idea and not base_creative_idea.endswith('.'): base_creative_idea += "."
    elif length_preference == "uzun":
        additional_idea = generate_new_idea_creative(user_prompt_text[::-1] + " devamı", style_preference)
        base_creative_idea += f"\n\nBu konuyu daha da derinleştirecek olursak, belki de {additional_idea} diyebiliriz. Hayal gücünün sınırı yoktur!"

    return random.choice(selected_templates).format(base_creative_idea)

def generate_new_idea_creative(seed_prompt_text: str, style:str = "genel") -> str:
    elements_list = ["zaman kristalleri", "psişik ormanlar", "rüya mimarisi eserleri", "kuantum köpüğü okyanusları", "gölge enerjisi dansı", "yankılanan anıların fısıltısı", "kayıp yıldız haritalarının rehberliği", "fraktal düşünce kalıpları", "kozmik senfoninin yankıları", "unutulmuş kehanetlerin gizemi", "eterik varlıkların şarkıları"]
    actions_list = ["dokur", "çözer", "yansıtır", "inşa eder", "fısıldar", "dönüştürür", "keşfeder", "haritalarını çizer", "ile bağlantı kurar", "çağırır", "şekillendirir"]
    outcomes_list = ["kaderin gizli ipliklerini", "varoluşun unutulmuş kodunu", "bilincin en derin sınırlarını", "kayıp uygarlıkların kadim sırlarını", "evrenin ebedi melodisini", "gerçekliğin çok boyutlu dokusunu", "saklı kalmış sonsuz potansiyelleri", "yepyeni bir çağın şafağını", "ruhun aydınlanma yolculuğunu"]
    
    prompt_words = re.findall(r'\b\w{3,}\b', seed_prompt_text.lower())
    seed_elements_for_idea = random.sample(prompt_words, k=min(len(prompt_words), 2)) if prompt_words else ["gizemli", "bir ışık"]
    
    if style == "şiirsel":
        return f"{random.choice(elements_list).capitalize()} arasında süzülürken, {seed_elements_for_idea[0]} fısıldar usulca, {random.choice(outcomes_list)}."
    elif style == "hikaye":
        return f"{' '.join(seed_elements_for_idea).capitalize()} {random.choice(actions_list)} ve {random.choice(elements_list)} kullanarak, sonunda {random.choice(outcomes_list)} keşfeder."
    return f"{' '.join(seed_elements_for_idea).capitalize()} {random.choice(actions_list)} ve {random.choice(elements_list)} aracılığıyla {random.choice(outcomes_list)}."

def advanced_word_generator(base_word_input: str) -> str:
    if not base_word_input or len(base_word_input) < 2: return "KelimatörProMax"
    vowels_set = "aeiouüöıAEIOUÜÖI"; consonants_set = "bcçdfgğhjklmnprsştvyzBCÇDFGĞHJKLMNPRSŞTVYZ"
    cleaned_base_word = "".join(filter(str.isalpha, base_word_input))
    if not cleaned_base_word: return "SözcükMimarUzmanı"
    prefixes_list = ["bio", "krono", "psiko", "tera", "neo", "mega", "nano", "astro", "poli", "eko", "meta", "trans", "ultra", "omni", "xeno", "kripto", "holo"]
    suffixes_list = ["genez", "sfer", "nomi", "tek", "loji", "tronik", "morf", "vers", "dinamik", "matik", "kinezis", "skop", "grafi", "mant", "krom", "faz", "sentez"]
    
    if len(cleaned_base_word) > 3 and random.random() < 0.75:
        start_index = random.randint(0, max(0, len(cleaned_base_word) - 3))
        core_word_part = cleaned_base_word[start_index : start_index + random.randint(2,4)]
    else:
        core_length = random.randint(3, 5)
        core_chars_list = [random.choice(consonants_set if random.random() > 0.4 else vowels_set) for _ in range(core_length)]
        core_word_part = "".join(core_chars_list)
    
    new_generated_word = core_word_part
    if random.random() > 0.4: new_generated_word = random.choice(prefixes_list) + new_generated_word
    if random.random() > 0.4: new_generated_word += random.choice(suffixes_list)
    
    return new_generated_word.capitalize() if len(new_generated_word) > 1 else new_generated_word

def generate_prompt_influenced_image(prompt_text: str) -> Image.Image:
    width, height = 512, 512
    prompt_lower_case = prompt_text.lower()
    keyword_themes_map = {
        "güneş": {"bg": [(255, 230, 150), (255, 160, 0)], "shapes": [{"type": "circle", "color": (255, 255, 0, 220), "pos": (random.uniform(0.2,0.35), random.uniform(0.2,0.35)), "size": random.uniform(0.18,0.25)}]},
        "ay": {"bg": [(10, 10, 50), (40, 40, 100)], "shapes": [{"type": "circle", "color": (240, 240, 240, 200), "pos": (random.uniform(0.65,0.8), random.uniform(0.15,0.3)), "size": random.uniform(0.12,0.18)}]},
        "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []},
        "deniz": {"bg": [(0, 105, 148), (0, 0, 100)], "shapes": [{"type": "rectangle", "color": (60,120,180, 150), "pos": (0.5, 0.75), "size": (1.0, 0.5)}]},
        "orman": {"bg": [(34, 139, 34), (0, 100, 0)], "shapes": [{"type": "triangle", "color": (random.randint(0,30),random.randint(70,100),random.randint(0,30),200), "pos": (random.uniform(0.1,0.9), random.uniform(0.4,0.75)), "size": random.uniform(0.08,0.28)} for _ in range(random.randint(5,10))]},
        "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [ {"type": "rectangle", "color": (139, 69, 19, 255), "pos": (0.5, 0.75), "size": (0.08, 0.4)}, {"type": "ellipse", "color": (34, 139, 34, 200), "pos": (0.5, 0.45), "size_wh": (0.3, 0.25)} ]},
        "dağ": {"bg": [(200,200,200), (100,100,100)], "shapes": [{"type": "triangle", "color": (random.randint(130,170),random.randint(130,170),random.randint(130,170),230), "pos": (0.5,0.6), "size":0.4, "points": [(random.uniform(0.05,0.35),0.85),(0.5,random.uniform(0.05,0.35)),(random.uniform(0.65,0.95),0.85)] } for _ in range(random.randint(1,3))]},
        "şehir": {"bg": [(100,100,120), (50,50,70)], "shapes": [{"type":"rectangle", "color":(random.randint(60,100),random.randint(60,100),random.randint(70,110),random.randint(180,220)), "pos":(random.uniform(0.1,0.9), random.uniform(0.4,0.85)), "size": (random.uniform(0.04,0.15), random.uniform(0.15,0.65))} for _ in range(random.randint(7,12))]}
    }
    bg_color1_tuple = (random.randint(30, 120), random.randint(30, 120), random.randint(30, 120))
    bg_color2_tuple = (random.randint(120, 220), random.randint(120, 220), random.randint(120, 220))
    shapes_to_draw_list = []
    num_themes_applied = 0

    for keyword_theme, theme_details in keyword_themes_map.items():
        if keyword_theme in prompt_lower_case:
            if num_themes_applied == 0:
                bg_color1_tuple, bg_color2_tuple = theme_details["bg"]
            shapes_to_draw_list.extend(theme_details["shapes"])
            num_themes_applied +=1

    image_canvas = Image.new('RGBA', (width, height), (0,0,0,0))
    draw_context = ImageDraw.Draw(image_canvas)

    for y_coordinate in range(height):
        blend_ratio = y_coordinate / height
        r_channel = int(bg_color1_tuple[0] * (1 - blend_ratio) + bg_color2_tuple[0] * blend_ratio)
        g_channel = int(bg_color1_tuple[1] * (1 - blend_ratio) + bg_color2_tuple[1] * blend_ratio)
        b_channel = int(bg_color1_tuple[2] * (1 - blend_ratio) + bg_color2_tuple[2] * blend_ratio)
        draw_context.line([(0, y_coordinate), (width, y_coordinate)], fill=(r_channel, g_channel, b_channel, 255))

    for shape_properties in shapes_to_draw_list:
        shape_type = shape_properties["type"]
        shape_color = shape_properties["color"]
        shape_pos_x_ratio, shape_pos_y_ratio = shape_properties["pos"]
        shape_center_x = int(shape_pos_x_ratio * width)
        shape_center_y = int(shape_pos_y_ratio * height)
        shape_outline_color = (0,0,0,50) if len(shape_color) == 4 and shape_color[3] < 250 else None

        if shape_type == "circle":
            shape_radius = int(shape_properties["size"] * min(width, height) / 2)
            draw_context.ellipse((shape_center_x - shape_radius, shape_center_y - shape_radius, shape_center_x + shape_radius, shape_center_y + shape_radius), fill=shape_color, outline=shape_outline_color)
        elif shape_type == "rectangle":
            shape_width_ratio, shape_height_ratio = shape_properties["size"]
            rect_width_pixels = int(shape_width_ratio * width)
            rect_height_pixels = int(shape_height_ratio * height)
            draw_context.rectangle((shape_center_x - rect_width_pixels // 2, shape_center_y - rect_height_pixels // 2, shape_center_x + rect_width_pixels // 2, shape_center_y + rect_height_pixels // 2), fill=shape_color, outline=shape_outline_color)
        elif shape_type == "triangle" and "points" in shape_properties:
            absolute_points_list = [(int(point[0]*width), int(point[1]*height)) for point in shape_properties["points"]]
            draw_context.polygon(absolute_points_list, fill=shape_color, outline=shape_outline_color)
        elif shape_type == "triangle":
            shape_base_size = int(shape_properties["size"] * min(width, height))
            point1 = (shape_center_x, shape_center_y - int(shape_base_size * 0.577))
            point2 = (shape_center_x - shape_base_size // 2, shape_center_y + int(shape_base_size * 0.288))
            point3 = (shape_center_x + shape_base_size // 2, shape_center_y + int(shape_base_size * 0.288))
            draw_context.polygon([point1, point2, point3], fill=shape_color, outline=shape_outline_color)
        elif shape_type == "ellipse":
            ellipse_size_wh_ratios = shape_properties["size_wh"]
            ellipse_width_pixels = int(ellipse_size_wh_ratios[0] * width)
            ellipse_height_pixels = int(ellipse_size_wh_ratios[1] * height)
            draw_context.ellipse((shape_center_x - ellipse_width_pixels // 2, shape_center_y - ellipse_height_pixels // 2, shape_center_x + ellipse_width_pixels // 2, shape_center_y + ellipse_height_pixels // 2), fill=shape_color, outline=shape_outline_color)

    if num_themes_applied == 0:
        for _ in range(random.randint(3, 6)):
            x1_coord = random.randint(0, width)
            y1_coord = random.randint(0, height)
            random_shape_fill_color = (random.randint(50, 250), random.randint(50, 250), random.randint(50, 250), random.randint(150, 220))
            
            # Bu if/else bloğunun girintisi 'for' döngüsüne göre doğru olmalı
            if random.random() > 0.5:
                random_radius = random.randint(25, 85)
                draw_context.ellipse((x1_coord - random_radius, y1_coord - random_radius, x1_coord + random_radius, y1_coord + random_radius), fill=random_shape_fill_color)
            else:
                random_rect_w, random_rect_h = random.randint(35, 125), random.randint(35, 125)
                draw_context.rectangle((x1_coord - random_rect_w // 2, y1_coord - random_rect_h // 2, x1_coord + random_rect_w // 2, y1_coord + random_rect_h // 2), fill=random_shape_fill_color)

    try:
        calculated_font_size = max(16, min(30, int(width / (len(prompt_text) * 0.35 + 10))))
        font_object_to_use = None
        if os.path.exists(FONT_FILE):
            try:
                font_object_to_use = ImageFont.truetype(FONT_FILE, calculated_font_size)
            except IOError:
                st.toast(f"Font dosyası '{FONT_FILE}' yüklenemedi. Varsayılan font kullanılacak.", icon="⚠️")
        
        if not font_object_to_use:
            font_object_to_use = ImageFont.load_default()
            # Varsayılan font kullanıldığında boyut farklı olabileceği için bir not düşülebilir
            # st.toast(f"Varsayılan font kullanılıyor, metin boyutu istenen '{calculated_font_size}px' olmayabilir.", icon="ℹ️")

        text_to_display_on_image = prompt_text[:70]
        if hasattr(draw_context, 'textbbox'):
            text_bounding_box = draw_context.textbbox((0,0), text_to_display_on_image, font=font_object_to_use, anchor="lt")
            text_render_width = text_bounding_box[2] - text_bounding_box[0]
            text_render_height = text_bounding_box[3] - text_bounding_box[1]
        else:
            text_render_width, text_render_height = draw_context.textsize(text_to_display_on_image, font=font_object_to_use)

        text_x_position = (width - text_render_width) / 2
        text_y_position = height * 0.93 - text_render_height
        draw_context.text((text_x_position + 1, text_y_position + 1), text_to_display_on_image, font=font_object_to_use, fill=(0,0,0,150))
        draw_context.text((text_x_position, text_y_position), text_to_display_on_image, font=font_object_to_use, fill=(255,255,255,230))
    except Exception as e_font_drawing:
        st.toast(f"Görsel üzerine metin yazdırılırken bir hata oluştu: {e_font_drawing}", icon="📝")

    return image_canvas.convert("RGB")

# --- Session State Başlatma ---
DEFAULT_SESSION_STATE_VALUES = {
    'chat_history': [], 'app_mode': "Yazılı Sohbet", 'user_name': None,
    'user_avatar_bytes': None, 'show_main_app': False, 'greeting_message_shown': False,
    'tts_enabled': True, 'theme': "Light",
    'gemini_stream_enabled': True, 'gemini_temperature': 0.7, 'gemini_top_p': 0.95,
    'gemini_top_k': 40, 'gemini_max_tokens': 4096, 'gemini_model_name': 'gemini-1.5-flash-latest',
    'message_id_counter': 0, 'last_ai_response_for_feedback': None,
    'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
    'feedback_comment_input': "", 'show_feedback_comment_form': False,
    'session_id': str(uuid.uuid4())
}
for key_ss, default_val_ss in DEFAULT_SESSION_STATE_VALUES.items():
    if key_ss not in st.session_state: st.session_state[key_ss] = default_val_ss

if 'models_initialized' not in st.session_state:
    gemini_model = initialize_gemini_model()
    supabase = init_supabase_client_cached()
    tts_engine = init_tts_engine_cached()
    st.session_state.models_initialized = True
else:
    gemini_model = globals().get('gemini_model')
    supabase = globals().get('supabase')
    tts_engine = globals().get('tts_engine')

gemini_init_error = globals().get('gemini_init_error_global')
supabase_error = globals().get('supabase_error_global')
tts_init_error = globals().get('tts_init_error_global')

if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history_cached()
if st.session_state.user_name and not st.session_state.show_main_app:
    st.session_state.show_main_app = True

# --- ARAYÜZ BÖLÜMLERİ İÇİN FONKSİYONLAR ---
def display_sidebar_content():
    with st.sidebar:
        st.markdown(f"### Hoş Geldin, {st.session_state.user_name}!")
        if st.session_state.user_avatar_bytes:
            st.image(st.session_state.user_avatar_bytes, width=100, use_column_width='auto', caption="Avatarınız")
        else:
            st.caption("🖼️ _Henüz bir avatar yüklemediniz._")
        st.markdown("---")
        st.subheader("⚙️ Ayarlar ve Kişiselleştirme")

        with st.expander("👤 Profil Ayarları", expanded=False):
            new_user_name = st.text_input("Adınızı Değiştirin:", value=st.session_state.user_name, key="change_name_sidebar_input")
            if new_user_name != st.session_state.user_name and new_user_name.strip():
                st.session_state.user_name = new_user_name.strip()
                st.toast("Adınız başarıyla güncellendi!", icon="✏️"); st.rerun()

            uploaded_avatar_file = st.file_uploader("Yeni Avatar Yükle (PNG, JPG - Maks 2MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader_sidebar_file")
            if uploaded_avatar_file:
                if uploaded_avatar_file.size > 2 * 1024 * 1024:
                    st.error("Dosya boyutu 2MB'den büyük olamaz! Lütfen daha küçük bir dosya seçin.", icon=" oversized_file:")
                else:
                    st.session_state.user_avatar_bytes = uploaded_avatar_file.getvalue()
                    st.toast("Avatarınız başarıyla güncellendi!", icon="🖼️"); st.rerun()
            
            if st.session_state.user_avatar_bytes and st.button("🗑️ Mevcut Avatarı Kaldır", use_container_width=True, key="remove_avatar_sidebar_button"):
                st.session_state.user_avatar_bytes = None
                st.toast("Avatarınız kaldırıldı.", icon="🗑️"); st.rerun()
            st.caption("Avatarınız sadece bu tarayıcı oturumunda saklanır.")

        current_tts_engine = globals().get('tts_engine')
        st.session_state.tts_enabled = st.toggle("Metin Okuma (TTS) Aktif", value=st.session_state.tts_enabled, disabled=not current_tts_engine, help="AI yanıtlarının sesli okunmasını açar veya kapatır.")
        st.session_state.gemini_stream_enabled = st.toggle("Gemini Yanıt Akışını Etkinleştir", value=st.session_state.gemini_stream_enabled, help="Yanıtların kelime kelime gelmesini sağlar (daha hızlı ilk tepki).")

        with st.expander("🤖 Gemini Gelişmiş Yapılandırma", expanded=False):
            st.session_state.gemini_model_name = st.selectbox(
                "Kullanılacak Gemini Modeli:",
                ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'],
                index=0 if st.session_state.gemini_model_name == 'gemini-1.5-flash-latest' else 1,
                key="gemini_model_selector_sidebar",
                help="Farklı modellerin yetenekleri ve maliyetleri değişebilir."
            )
            st.session_state.gemini_temperature = st.slider("Sıcaklık (Yaratıcılık Seviyesi):", 0.0, 1.0, st.session_state.gemini_temperature, 0.05, key="gemini_temp_slider_sidebar", help="Düşük değerler daha kesin, yüksek değerler daha yaratıcı yanıtlar üretir.")
            st.session_state.gemini_top_p = st.slider("Top P (Odaklanma Düzeyi):", 0.0, 1.0, st.session_state.gemini_top_p, 0.05, key="gemini_top_p_slider_sidebar", help="Yanıtların ne kadar odaklı olacağını belirler. Genellikle 1.0 veya 0.95 kullanılır.")
            st.session_state.gemini_top_k = st.slider("Top K (Çeşitlilik Filtresi):", 1, 100, st.session_state.gemini_top_k, 1, key="gemini_top_k_slider_sidebar", help="Yanıt oluşturulurken en olası K token arasından seçim yapılmasını sağlar.")
            st.session_state.gemini_max_tokens = st.slider("Maksimum Çıktı Token Sayısı:", 256, 8192, st.session_state.gemini_max_tokens, 128, key="gemini_max_tokens_slider_sidebar", help="Modelin üreteceği yanıtın maksimum uzunluğunu sınırlar.")
            
            if st.button("⚙️ Gemini Ayarlarını Uygula ve Modeli Yeniden Başlat", key="reload_gemini_settings_sidebar_btn", use_container_width=True, type="primary"):
                global gemini_model
                gemini_model = initialize_gemini_model()
                if gemini_model: st.toast("Gemini ayarları başarıyla güncellendi ve model yeniden yüklendi!", icon="✨")
                else: st.error("Gemini modeli güncellenirken bir hata oluştu. Lütfen API anahtarınızı ve yapılandırma ayarlarınızı kontrol edin.")

        st.divider()
        if st.button("🧹 Sohbet Geçmişini Temizle", use_container_width=True, type="secondary", key="clear_history_sidebar_main_btn"):
            if st.session_state.chat_history:
                st.session_state.chat_history = []
                save_chat_history([])
                st.toast("Sohbet geçmişi başarıyla temizlendi!", icon="🧹"); st.rerun()
            else:
                st.toast("Sohbet geçmişi zaten boş.", icon="ℹ️")

        with st.expander("ℹ️ Uygulama Hakkında", expanded=True): # Başlangıçta açık olsun
            st.markdown(f"""
            **{APP_NAME} v{APP_VERSION}**
            Yapay zeka destekli kişisel sohbet asistanınız.
            Geliştirici: **Hanogt** ([GitHub](https://github.com/Hanogt))

            Bu uygulama Streamlit, Google Gemini API ve çeşitli Python açık kaynak kütüphaneleri kullanılarak geliştirilmiştir.
            Kullanıcı etkileşimleri ve geri bildirimler, isteğe bağlı olarak Supabase üzerinde güvenli bir şekilde saklanabilir.
            Tüm hakları saklıdır © 2024-{CURRENT_YEAR}
            """)
        st.caption(f"{APP_NAME} v{APP_VERSION} - Oturum ID: {st.session_state.session_id[:8]}...")

def display_chat_message_with_feedback(sender_name: str, message_text_content: str, message_unique_index: int, is_user_message: bool):
    avatar_display_icon = "🧑"
    if is_user_message:
        if st.session_state.user_avatar_bytes:
            try: avatar_display_icon = Image.open(BytesIO(st.session_state.user_avatar_bytes))
            except Exception: pass
    else:
        if "Gemini" in sender_name: avatar_display_icon = "✨"
        elif "Web" in sender_name: avatar_display_icon = "🌐"
        elif "Bilgi Tabanı" in sender_name or "Fonksiyonel" in sender_name: avatar_display_icon = "📚"
        else: avatar_display_icon = "🤖"

    with st.chat_message("user" if is_user_message else "assistant", avatar=avatar_display_icon):
        if "```" in message_text_content:
            code_block_parts = message_text_content.split("```")
            for i, part_text in enumerate(code_block_parts):
                if i % 2 == 1:
                    language_match = re.match(r"(\w+)\n", part_text)
                    code_language = language_match.group(1) if language_match else None
                    actual_code_content = part_text[len(code_language)+1:] if code_language and part_text.startswith(code_language+"\n") else part_text
                    
                    st.code(actual_code_content, language=code_language)
                    if st.button("📋 Kopyala", key=f"copy_code_btn_{message_unique_index}_{i}", help="Bu kod bloğunu panoya kopyala"):
                        st.write_to_clipboard(actual_code_content)
                        st.toast("Kod başarıyla panoya kopyalandı!", icon="✅")
                else:
                    st.markdown(part_text, unsafe_allow_html=True)
        else:
            st.markdown(message_text_content, unsafe_allow_html=True)

        if not is_user_message:
            ai_response_source_name = sender_name.split('(')[-1].replace(')','').strip() if '(' in sender_name else sender_name.replace(f'{APP_NAME} ','')
            action_cols = st.columns([0.7, 0.15, 0.15])
            with action_cols[0]:
                st.caption(f"Kaynak: {ai_response_source_name}")
            with action_cols[1]:
                current_tts_engine = globals().get('tts_engine')
                if st.session_state.tts_enabled and current_tts_engine and message_text_content:
                    if st.button("🔊", key=f"speak_msg_btn_chat_{message_unique_index}", help="Bu mesajı sesli oku", use_container_width=True):
                        speak(message_text_content)
            with action_cols[2]:
                if st.button("✍️", key=f"toggle_feedback_btn_chat_{message_unique_index}", help="Bu yanıt hakkında geri bildirimde bulunun", use_container_width=True):
                    st.session_state.current_message_id_for_feedback = f"chat_{message_unique_index}"
                    st.session_state.last_user_prompt_for_feedback = st.session_state.chat_history[message_unique_index-1][1] if message_unique_index > 0 else "N/A (Prompt bulunamadı)"
                    st.session_state.last_ai_response_for_feedback = message_text_content
                    st.session_state.show_feedback_comment_form = not st.session_state.get('show_feedback_comment_form', False)
                    if not st.session_state.show_feedback_comment_form:
                        st.session_state.feedback_comment_input = ""
                    st.rerun()

def display_feedback_form_if_active():
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---")
        with st.form(key=f"feedback_submission_form_{st.session_state.current_message_id_for_feedback}"):
            st.markdown(f"#### Geri Bildiriminiz")
            st.caption(f"**İstem:** `{st.session_state.last_user_prompt_for_feedback[:70]}...`")
            st.caption(f"**Yanıt:** `{st.session_state.last_ai_response_for_feedback[:70]}...`")

            feedback_rating_type = st.radio(
                "Bu yanıtı nasıl değerlendirirsiniz?",
                ["👍 Beğendim", "👎 Beğenmedim"],
                horizontal=True, key="feedback_type_radio_form",
                index=0 if st.session_state.get('last_feedback_type', 'positive') == 'positive' else 1
            )
            user_feedback_comment = st.text_area(
                "Yorumunuz (isteğe bağlı, özellikle beğenmediyseniz nedenini belirtmeniz çok yardımcı olur):",
                value=st.session_state.get('feedback_comment_input', ""),
                key="feedback_comment_textarea_form", height=100
            )
            st.session_state.feedback_comment_input = user_feedback_comment

            submitted_feedback_button = st.form_submit_button("✅ Geri Bildirimi Gönder ve Formu Kapat", type="primary")

            if submitted_feedback_button:
                parsed_feedback_category = "positive" if feedback_rating_type == "👍 Beğendim" else "negative"
                st.session_state.last_feedback_type = parsed_feedback_category

                if log_feedback(
                    st.session_state.current_message_id_for_feedback,
                    st.session_state.last_user_prompt_for_feedback,
                    st.session_state.last_ai_response_for_feedback,
                    parsed_feedback_category,
                    user_feedback_comment
                ):
                    st.session_state.show_feedback_comment_form = False
                    st.session_state.feedback_comment_input = ""
                    st.session_state.current_message_id_for_feedback = None
                    st.rerun()
        st.markdown("---")

def display_chat_interface_main():
    chat_display_container = st.container()
    with chat_display_container:
        if not st.session_state.chat_history:
            st.info(f"Merhaba {st.session_state.user_name}! Size nasıl yardımcı olabilirim? Lütfen aşağıdan mesajınızı yazın.", icon="👋")
        
        for i, (sender_id_name, message_content_text) in enumerate(st.session_state.chat_history):
            display_chat_message_with_feedback(sender_id_name, message_content_text, i, sender_id_name.startswith("Sen"))

    display_feedback_form_if_active()

    if user_new_prompt := st.chat_input(f"{st.session_state.user_name} olarak mesajınızı yazın...", key="main_chat_input_field_bottom"):
        current_message_unique_id = f"msg_{st.session_state.message_id_counter}_{int(time.time())}"
        st.session_state.message_id_counter += 1
        st.session_state.chat_history.append(("Sen", user_new_prompt))

        raw_history_for_gemini = st.session_state.chat_history[-21:-1]
        gemini_formatted_chat_history = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]} for sender, message in raw_history_for_gemini]

        with st.chat_message("assistant", avatar="⏳"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 _Düşünüyorum... Lütfen bekleyin..._")
            time.sleep(0.05) # Placeholder'ın görünmesi için çok kısa bekleme

        ai_response_data, ai_sender_identity = get_hanogt_response_orchestrator(
            user_new_prompt,
            gemini_formatted_chat_history,
            current_message_unique_id,
            use_stream_output=st.session_state.gemini_stream_enabled
        )

        if st.session_state.gemini_stream_enabled and ai_sender_identity == f"{APP_NAME} (Gemini Stream)":
            streamed_full_response_text = ""
            try:
                for chunk_index, stream_chunk in enumerate(ai_response_data):
                    if stream_chunk.parts:
                        text_from_chunk = "".join(part.text for part in stream_chunk.parts if hasattr(part, 'text'))
                        streamed_full_response_text += text_from_chunk
                        thinking_placeholder.markdown(streamed_full_response_text + "▌")
                        if chunk_index % 5 == 0: time.sleep(0.005) 
                thinking_placeholder.markdown(streamed_full_response_text)
                log_interaction(user_new_prompt, streamed_full_response_text, ai_sender_identity, current_message_unique_id)
                st.session_state.chat_history.append((ai_sender_identity, streamed_full_response_text))
            except Exception as e_stream_processing:
                error_text_stream = f"Stream yanıtı işlenirken bir hata oluştu: {e_stream_processing}"
                thinking_placeholder.error(error_text_stream)
                st.session_state.chat_history.append((f"{APP_NAME} (Stream Hatası)", error_text_stream))
        else:
            thinking_placeholder.empty()
            st.session_state.chat_history.append((ai_sender_identity, str(ai_response_data)))

        save_chat_history(st.session_state.chat_history)
        
        if st.session_state.tts_enabled and globals().get('tts_engine') and \
           isinstance(ai_response_data, str) and not \
           (st.session_state.gemini_stream_enabled and ai_sender_identity == f"{APP_NAME} (Gemini Stream)"):
            speak(ai_response_data)
        
        st.rerun()

# --- UYGULAMA ANA AKIŞI (MAIN FLOW) ---
st.markdown(f"<h1 style='text-align: center; color: #0078D4;'>🚀 {APP_NAME} {APP_VERSION} 🚀</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-style: italic; color: #555;'>Yapay zeka destekli kişisel sohbet asistanınız!</p>", unsafe_allow_html=True)

if gemini_init_error: st.error(gemini_init_error, icon="🛑")
if supabase_error: st.error(supabase_error, icon="🧱")
if tts_init_error and st.session_state.tts_enabled:
    st.toast(tts_init_error, icon="🔇")

if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce Sizi Tanıyabilir Miyim?")
    login_form_cols = st.columns([0.15, 0.7, 0.15])
    with login_form_cols[1]:
        with st.form("user_details_login_form"):
            user_name_input = st.text_input(
                "Size nasıl hitap etmeliyim?",
                placeholder="İsminiz veya takma adınız...",
                value=st.session_state.get('user_name_temp', ''),
                key="user_name_login_input_field"
            )
            login_submitted_button = st.form_submit_button("✨ Başlayalım!", use_container_width=True, type="primary")
            
            if login_submitted_button:
                if user_name_input and user_name_input.strip():
                    st.session_state.user_name = user_name_input.strip()
                    st.session_state.show_main_app = True
                    st.session_state.greeting_message_shown = False
                    st.rerun()
                else:
                    st.error("Lütfen geçerli bir isim veya takma ad girin.")
else:
    if not st.session_state.greeting_message_shown and st.session_state.user_name:
        greeting_message = random.choice([
            f"Tekrar hoş geldiniz, Sayın {st.session_state.user_name}! Bugün size nasıl yardımcı olabilirim?",
            f"Merhaba {st.session_state.user_name}! Sizin için hazırım, ne merak ediyorsunuz?",
            f"Harika bir gün geçirmeniz dileğiyle, {st.session_state.user_name}! Ne yapmak istersiniz?"
        ])
        st.success(greeting_message, icon="🎉"); st.session_state.greeting_message_shown = True
        st.balloons()

    display_sidebar_content()

    app_mode_options_map = {
        "Yazılı Sohbet": "💬", "Sesli Sohbet (Dosya Yükle)": "🎤",
        "Yaratıcı Stüdyo": "🎨", "Görsel Oluşturucu": "🖼️"
    }
    selected_app_mode_key = st.radio(
        "Uygulama Modunu Seçin:",
        options=list(app_mode_options_map.keys()),
        index=list(app_mode_options_map.keys()).index(st.session_state.app_mode),
        format_func=lambda mode_key: f"{app_mode_options_map[mode_key]} {mode_key}",
        horizontal=True, label_visibility="collapsed", key="app_mode_selector_radio"
    )
    if selected_app_mode_key != st.session_state.app_mode:
        st.session_state.app_mode = selected_app_mode_key; st.rerun()
    current_app_mode = st.session_state.app_mode
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

    if current_app_mode == "Yazılı Sohbet":
        display_chat_interface_main()

    elif current_app_mode == "Sesli Sohbet (Dosya Yükle)":
        st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası (WAV, MP3, OGG, FLAC, M4A formatlarında) yükleyin.", icon="📢")
        uploaded_audio_file = st.file_uploader(
            "Ses Dosyası Seçin:", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            label_visibility="collapsed", key="audio_file_uploader_page_main"
        )
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
            user_prompt_from_audio = None; audio_file_name = uploaded_audio_file.name
            temp_audio_file_path = f"temp_audio_{st.session_state.session_id}_{re.sub(r'[^a-zA-Z0-9_.-]', '', audio_file_name)[:20]}.wav"

            with st.spinner(f"🔊 '{audio_file_name}' ses dosyası işleniyor... Lütfen bekleyin."):
                speech_recognizer = sr.Recognizer();
                try:
                    with open(temp_audio_file_path, "wb") as temp_f: temp_f.write(uploaded_audio_file.getbuffer())
                    with sr.AudioFile(temp_audio_file_path) as audio_source:
                        audio_data_recorded = speech_recognizer.record(audio_source)
                    user_prompt_from_audio = speech_recognizer.recognize_google(audio_data_recorded, language="tr-TR")
                    st.success(f"**🎙️ Algılanan Metin:**\n\n> {user_prompt_from_audio}")
                except sr.UnknownValueError:
                    st.error("🔇 Üzgünüm, ses anlaşılamadı. Lütfen daha net bir ses dosyası deneyin veya ses kalitesini kontrol edin.")
                except sr.RequestError as e_sr_request:
                    st.error(f"🤖 Ses tanıma servisine ulaşılamadı: {e_sr_request}. Lütfen internet bağlantınızı kontrol edin.")
                except Exception as e_audio_processing:
                    st.error(f"Ses dosyası işlenirken beklenmedik bir hata oluştu: {e_audio_processing}")
                finally:
                    if os.path.exists(temp_audio_file_path): os.remove(temp_audio_file_path)

            if user_prompt_from_audio:
                current_message_id_audio_mode = f"audio_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                st.session_state.message_id_counter += 1
                st.session_state.chat_history.append(("Sen (Ses Dosyasından)", user_prompt_from_audio))
                raw_history_for_gemini_audio = st.session_state.chat_history[-21:-1]
                gemini_formatted_history_audio = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]} for sender, message in raw_history_for_gemini_audio]
                
                with st.spinner("🤖 Yapay zeka yanıtınızı hazırlıyor..."):
                    ai_response_audio, ai_sender_audio_mode = get_hanogt_response_orchestrator(user_prompt_from_audio, gemini_formatted_history_audio, current_message_id_audio_mode, use_stream_output=False)
                
                st.markdown(f"#### {ai_sender_audio_mode} Yanıtı:")
                st.markdown(ai_response_audio)
                
                current_tts_engine_audio = globals().get('tts_engine')
                if st.session_state.tts_enabled and current_tts_engine_audio and ai_response_audio:
                    if st.button("🔊 AI Yanıtını Seslendir", key="speak_audio_response_button_page"):
                        speak(str(ai_response_audio))
                
                st.session_state.chat_history.append((ai_sender_audio_mode, str(ai_response_audio)))
                save_chat_history(st.session_state.chat_history)
                st.success("✅ Yanıt başarıyla oluşturuldu ve genel sohbet geçmişine eklendi!")

    elif current_app_mode == "Yaratıcı Stüdyo":
        st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Hanogt AI size ilham verici ve yaratıcı bir yanıt oluştursun!", icon="💡")
        user_creative_prompt_text = st.text_area(
            "Yaratıcılık Tohumunuzu Buraya Ekleyin:",
            key="creative_input_studio_main_page",
            placeholder="Örneğin: 'Ay ışığında dans eden bir tilkinin rüyası hakkında kısa bir şiirsel metin'",
            height=120
        )
        creative_options_cols = st.columns(2)
        with creative_options_cols[0]:
            response_length_preference = st.selectbox(
                "İstenen Yanıt Uzunluğu:",
                ["kısa", "orta", "uzun"], index=1,
                key="creative_length_preference_selector_page",
                help="Yapay zekanın üreteceği metnin yaklaşık uzunluğunu belirler."
            )
        with creative_options_cols[1]:
            response_style_preference = st.selectbox(
                "İstenen Yaratıcılık Stili:",
                ["genel", "şiirsel", "hikaye"], index=0,
                key="creative_style_preference_selector_page",
                help="Yapay zekanın kullanacağı yazım üslubunu seçin."
            )

        if st.button("✨ İlham Veren Fikri Üret!", key="generate_creative_response_button_page", type="primary", use_container_width=True):
            if user_creative_prompt_text and user_creative_prompt_text.strip():
                final_creative_response_text = None; ai_sender_creative_mode = f"{APP_NAME} (Yaratıcı)"
                current_message_id_creative_mode = f"creative_msg_{st.session_state.message_id_counter}_{int(time.time())}"
                st.session_state.message_id_counter += 1

                current_gemini_model_creative = globals().get('gemini_model')
                if current_gemini_model_creative:
                    with st.spinner("✨ Gemini ilham perileriyle fısıldaşıyor ve sizin için özel bir metin hazırlıyor..."):
                        gemini_creative_system_prompt_text = (
                            f"Sen çok yaratıcı, hayal gücü geniş ve edebi yönü kuvvetli bir asistansın. "
                            f"Sana verilen şu isteme: '{user_creative_prompt_text}' dayanarak, "
                            f"'{response_style_preference}' stilinde ve yaklaşık '{response_length_preference}' uzunlukta özgün, ilginç ve sanatsal bir metin oluştur. "
                            "Sıradanlıktan kaçın, okuyucuyu etkileyecek ve düşündürecek bir dil kullan. Eğer uygunsa, metaforlar ve benzetmeler de kullanabilirsin."
                        )
                        gemini_creative_response = get_gemini_response_cached(gemini_creative_system_prompt_text, [], stream_output=False)
                        
                        if gemini_creative_response and not (isinstance(gemini_creative_response, str) and gemini_creative_response.startswith(GEMINI_ERROR_PREFIX)):
                            final_creative_response_text = str(gemini_creative_response)
                            ai_sender_creative_mode = f"{APP_NAME} (Gemini Yaratıcı)"
                        else:
                            error_msg_creative = gemini_creative_response if isinstance(gemini_creative_response, str) else "Bilinmeyen bir sorun oluştu."
                            st.warning(f"Gemini yaratıcı yanıtı alınamadı. Yerel modül kullanılacak. (Detay: {error_msg_creative.replace(GEMINI_ERROR_PREFIX, '').strip()})", icon="⚠️")
                
                if not final_creative_response_text:
                    with st.spinner("✨ Kendi fikirlerimi demliyorum ve hayal gücümün sınırlarını zorluyorum..."):
                        time.sleep(0.2)
                        local_creative_generated_text = creative_response_generator(user_creative_prompt_text, length_preference=response_length_preference, style_preference=response_style_preference)
                        newly_generated_word = advanced_word_generator(user_creative_prompt_text.split()[0] if user_creative_prompt_text else "kelime")
                        final_creative_response_text = f"{local_creative_generated_text}\n\n---\n🔮 **Kelimatörden Türetilen Özel Sözcük:** {newly_generated_word}"
                        ai_sender_creative_mode = f"{APP_NAME} (Yerel Yaratıcı)"
                
                st.markdown(f"#### {ai_sender_creative_mode} İlhamı:")
                st.markdown(final_creative_response_text)
                
                current_tts_engine_creative = globals().get('tts_engine')
                if st.session_state.tts_enabled and current_tts_engine_creative and final_creative_response_text:
                    if st.button("🔊 Bu İlham Veren Metni Dinle", key="speak_creative_response_button_page"):
                        text_to_speak_creative = final_creative_response_text.split("🔮 **Kelimatörden Türetilen Özel Sözcük:**")[0].strip()
                        speak(text_to_speak_creative)
                
                log_interaction(user_creative_prompt_text, final_creative_response_text, ai_sender_creative_mode, current_message_id_creative_mode)
                st.success("✨ Yaratıcı yanıtınız başarıyla oluşturuldu!")
            else:
                st.error("Lütfen yaratıcılığınızı ateşleyecek bir fikir, kelime veya cümle yazın!", icon="✍️")

    elif current_app_mode == "Görsel Oluşturucu":
        st.markdown("Hayalinizdeki görseli tarif edin, Hanogt AI anahtar kelimelere göre sizin için (sembolik olarak) çizecektir!", icon="🎨")
        st.info("ℹ️ **Not:** Bu mod, girdiğiniz metindeki anahtar kelimelere (örneğin: güneş, deniz, ağaç, ay, gökyüzü, orman, dağ, şehir vb.) göre basit, kural tabanlı ve sembolik çizimler yapar. Lütfen fotogerçekçi veya karmaşık sanat eserleri beklemeyin; bu daha çok prompt'unuzun eğlenceli bir yorumlayıcısıdır.", icon="💡")
        
        user_image_prompt_text = st.text_input(
            "Ne tür bir görsel hayal ediyorsunuz? (Anahtar kelimelerle tarif edin)",
            key="image_prompt_input_generator_page",
            placeholder="Örnek: 'Gece vakti karlı dağların üzerinde parlayan bir dolunay ve birkaç çam ağacı'"
        )
        
        if st.button("🖼️ Hayalimdeki Görseli Oluştur!", key="generate_rule_based_image_button_page", type="primary", use_container_width=True):
            if user_image_prompt_text and user_image_prompt_text.strip():
                with st.spinner("🖌️ Fırçalarım ve renklerim hazırlanıyor... Hayaliniz tuvale aktarılıyor..."):
                    time.sleep(0.3)
                    generated_image_object = generate_prompt_influenced_image(user_image_prompt_text)
                
                st.image(generated_image_object, caption=f"{APP_NAME}'ın '{user_image_prompt_text[:60]}' yorumu (Kural Tabanlı Çizim)", use_container_width=True)
                
                try:
                    image_buffer = BytesIO()
                    generated_image_object.save(image_buffer, format="PNG")
                    image_bytes_for_download = image_buffer.getvalue()
                    
                    cleaned_prompt_for_filename = re.sub(r'[^\w\s-]', '', user_image_prompt_text.lower())
                    cleaned_prompt_for_filename = re.sub(r'\s+', '_', cleaned_prompt_for_filename).strip('_')[:35]
                    downloadable_file_name = f"hanogt_ai_cizim_{cleaned_prompt_for_filename or 'gorsel'}_{int(time.time())}.png"
                    
                    st.download_button(
                        label="🖼️ Oluşturulan Görseli İndir (PNG)",
                        data=image_bytes_for_download,
                        file_name=downloadable_file_name,
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e_image_download:
                    st.error(f"Görsel indirilirken bir hata oluştu: {e_image_download}", icon="⚠️")
            else:
                st.error("Lütfen ne tür bir görsel çizmemi istediğinizi açıklayan bir metin girin!", icon="✍️")

    # --- Alt Bilgi (Footer) ---
    st.markdown("<hr style='margin-top: 1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <p style='text-align: center; font-size: 0.8rem; color: #777;'>
            {APP_NAME} v{APP_VERSION} &nbsp;&nbsp;|&nbsp;&nbsp; 
            Kullanıcı: {st.session_state.get('user_name', 'Misafir')} &nbsp;&nbsp;|&nbsp;&nbsp;
            © 2024-{CURRENT_YEAR}
            <br>
            Gemini Modeli: <span style="color: {'green' if globals().get('gemini_model') else 'red'};">{st.session_state.gemini_model_name if globals().get('gemini_model') else 'Devre Dışı'}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            Supabase Loglama: <span style="color: {'green' if globals().get('supabase') else 'red'};">{'Aktif' if globals().get('supabase') else 'Devre Dışı'}</span>
        </p>
        """, unsafe_allow_html=True
    )
