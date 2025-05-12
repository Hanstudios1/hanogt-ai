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
    from postgrest import APIError as SupabaseAPIError
except ImportError:
    st.toast("Supabase kütüphanesi bulunamadı. Loglama/Feedback devre dışı.", icon="ℹ️")
    create_client = None
    Client = None
    SupabaseAPIError = None

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Hanogt AI Pro+ Enhanced",
    page_icon="✨", # İkon güncellendi
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sabitler ve Yapılandırma ---
APP_NAME = "Hanogt AI"
APP_VERSION = "5.1.0 Pro+ FeatureRich" # Sürüm güncellendi
CURRENT_YEAR = datetime.now().year
CHAT_HISTORY_FILE = "chat_history_v2.json"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir sorun oluştu. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 20
SCRAPE_MAX_CHARS = 3800 # Biraz daha artırıldı
GEMINI_ERROR_PREFIX = "GeminiError:"
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 {APP_NAME}/{APP_VERSION}"
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
            knowledge_base_load_error = None; return merged_kb
        else:
            knowledge_base_load_error = f"Bilgi tabanı ({filename}) bulunamadı. Varsayılan kullanılıyor."
            st.toast(knowledge_base_load_error, icon="ℹ️"); return default_knowledge
    except json.JSONDecodeError:
        knowledge_base_load_error = f"Bilgi tabanı ({filename}) hatalı. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="⚠️"); return default_knowledge
    except Exception as e:
        knowledge_base_load_error = f"Bilgi tabanı yüklenirken hata: {e}. Varsayılan kullanılıyor."
        st.toast(knowledge_base_load_error, icon="🔥"); return default_knowledge

def kb_chatbot_response(query, knowledge_base_dict):
    """Bilgi tabanından veya dinamik fonksiyonlardan yanıt döndürür."""
    query_lower = query.lower().strip()
    if query_lower in DYNAMIC_FUNCTIONS_MAP:
        try: return DYNAMIC_FUNCTIONS_MAP[query_lower]()
        except Exception as e: st.error(f"Fonksiyon hatası ({query_lower}): {e}"); return DEFAULT_ERROR_MESSAGE
    if query_lower in knowledge_base_dict:
        resp = knowledge_base_dict[query_lower]; return random.choice(resp) if isinstance(resp, list) else resp
    partial_matches = [resp for key, resp_list in knowledge_base_dict.items() if key in query_lower for resp in (resp_list if isinstance(resp_list, list) else [resp_list])]
    if partial_matches: return random.choice(list(set(partial_matches)))
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    best_score, best_responses = 0, []
    for key, resp_list in knowledge_base_dict.items():
        key_words = set(re.findall(r'\b\w{3,}\b', key.lower()))
        if not key_words: continue
        score = len(query_words.intersection(key_words)) / len(key_words) if key_words else 0
        if score > 0.6:
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
        gemini_init_error_global = "🛑 Google API Anahtarı Secrets'ta bulunamadı!"
        return None
    try:
        genai.configure(api_key=api_key)
        safety = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        model_name = st.session_state.get('gemini_model_name', 'gemini-1.5-flash-latest')
        system_prompt = st.session_state.get('gemini_system_prompt', None) # YENİ: Sistem talimatı
        config = genai.types.GenerationConfig(
            temperature=st.session_state.get('gemini_temperature', 0.7),
            top_p=st.session_state.get('gemini_top_p', 0.95), # Geri geldi
            top_k=st.session_state.get('gemini_top_k', 40),   # Geri geldi
            max_output_tokens=st.session_state.get('gemini_max_tokens', 4096)
        )
        model_args = {
            "model_name": model_name,
            "safety_settings": safety,
            "generation_config": config
        }
        # Sadece system_prompt varsa ve boş değilse ekle
        if system_prompt and system_prompt.strip():
            model_args["system_instruction"] = system_prompt.strip()

        model = genai.GenerativeModel(**model_args)
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
    """Supabase istemcisini başlatır ve cache'ler. (UI elemanı içermez)"""
    if not create_client: print("ERROR: Supabase library not loaded."); return None
    url, key = st.secrets.get("SUPABASE_URL"), st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key: print("ERROR: Supabase URL/Key not found in secrets."); return None
    try:
        client: Client = create_client(url, key)
        print("INFO: Supabase client created successfully via cache function.")
        return client
    except Exception as e: print(f"ERROR: Supabase connection failed during init: {e}"); return None

# --- YARDIMCI FONKSİYONLAR ---
def _get_session_id():
    if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# --- TTS Motoru ---
tts_engine = None
tts_init_error_global = None
@st.cache_resource
def init_tts_engine_cached():
    global tts_init_error_global
    try:
        engine = pyttsx3.init(); tts_init_error_global = None; st.toast("🔊 TTS motoru hazır.", icon="🗣️"); return engine
    except Exception as e: tts_init_error_global = f"⚠️ TTS motoru başlatılamadı: {e}."; print(f"ERROR: TTS Init Failed: {e}"); return None

def speak(text):
    engine = globals().get('tts_engine')
    if not engine: st.toast("TTS motoru aktif değil.", icon="🔇"); return
    if not st.session_state.get('tts_enabled', True): st.toast("TTS ayarlardan kapalı.", icon="🔇"); return
    try:
        cleaned = re.sub(r'[^\w\s.,!?-]', '', text); engine.say(cleaned); engine.runAndWait()
    except RuntimeError as e: st.warning(f"TTS çalışma zamanı sorunu: {e}.", icon="🔊")
    except Exception as e: st.error(f"TTS hatası: {e}", icon="🔥"); print(f"ERROR: TTS Speak Failed: {e}")

# --- Metin Temizleme ---
def _clean_text(text):
    text = re.sub(r'\s+', ' ', text); text = re.sub(r'\n\s*\n', '\n\n', text); return text.strip()

# --- Web Kazıma (Cache'li)---
@st.cache_data(ttl=600)
def scrape_url_content(url, timeout=REQUEST_TIMEOUT, max_chars=SCRAPE_MAX_CHARS):
    st.toast(f"🌐 '{urlparse(url).netloc}' alınıyor...", icon="⏳")
    try:
        parsed = urlparse(url); headers = {'User-Agent': USER_AGENT, 'Accept-Language': 'tr-TR,tr;q=0.9', 'Accept': 'text/html', 'DNT': '1'}
        if not all([parsed.scheme, parsed.netloc]) or parsed.scheme not in ['http', 'https']: st.warning(f"Geçersiz URL: {url}", icon="🔗"); return None
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True); resp.raise_for_status()
        ctype = resp.headers.get('content-type', '').lower()
        if 'html' not in ctype: st.info(f"HTML değil ('{ctype}'). Atlanıyor.", icon="📄"); resp.close(); return None
        html = ""; size = 0; max_size = max_chars * 12
        try:
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True, errors='ignore'):
                if chunk: html += chunk; size += len(chunk.encode('utf-8', 'ignore'))
                if size > max_size: st.warning(f"HTML > {max_size//1024}KB, kesiliyor.", icon="✂️"); break
        finally: resp.close()
        if not html: st.warning("Boş içerik.", icon="📄"); return None
        soup = BeautifulSoup(html, 'lxml'); tags_remove=["script","style","nav","footer","aside","form","button","iframe","header","noscript","link","meta","img","svg","video","audio","figure","input","select","textarea"]
        for tag in soup.find_all(tags_remove): tag.decompose()
        content = []; selectors=['article[class*="content"]','article[class*="post"]','main[id*="content"]','main','div[class*="post-body"]','div[itemprop="articleBody"]','article','.content','#content']
        container=next((found[0] for sel in selectors if (found:=soup.select(sel, limit=1))), None)
        min_len=80; min_ind=1
        if container:
            for p in container.find_all('p', limit=60): text=_clean_text(p.get_text(separator=' ',strip=True));  content.append(text) if len(text)>min_len and (text.count('.')+text.count('?')+text.count('!'))>=min_ind else None
        if not content or len(" ".join(content))<300:
             body = soup.body; parts=[p.strip() for p in _clean_text(body.get_text(separator='\n',strip=True)).split('\n') if len(p.strip())>min_len] if body else []
             if len(" ".join(parts))>200: st.toast("Genel metin kullanıldı.",icon="ℹ️"); content=parts[:40]
             else: st.toast("Anlamlı içerik yok.",icon="📄"); return None
        cleaned=_clean_text("\n\n".join(content))
        if not cleaned: st.toast("Kazıma sonucu boş.",icon="📄"); return None
        final=cleaned[:max_chars]+("..." if len(cleaned)>max_chars else ""); st.toast(f"'{urlparse(url).netloc}' alındı.",icon="✅"); return final
    except requests.exceptions.RequestException as e: st.toast(f"⚠️ Ağ hatası: {url}", icon='🌐')
    except Exception as e: st.toast(f"⚠️ Kazıma hatası: {e}", icon='🔥'); print(f"ERROR: Scraping '{url}' failed: {e}"); return None

# --- Web Arama (Cache'li) ---
@st.cache_data(ttl=600)
def search_web(query):
    st.toast(f"🔍 '{query}' aranıyor...", icon="⏳")
    wikipedia.set_lang("tr"); result = None
    try: wp_page=wikipedia.page(query,auto_suggest=False,redirect=True); summary=wikipedia.summary(query,sentences=6,auto_suggest=False,redirect=True); result=f"**Wikipedia ({wp_page.title}):**\n\n{_clean_text(summary)}\n\nKaynak: {wp_page.url}"; st.toast(f"✅ Wiki: '{wp_page.title}'", icon="📚"); return result
    except wikipedia.exceptions.PageError: st.toast(f"ℹ️ Wiki'de '{query}' yok.", icon="🤷")
    except wikipedia.exceptions.DisambiguationError as e: result = f"**Wiki Çok Anlamlı ({query}):**\n{e.options[:3]}..."
    except Exception as e: st.toast(f"⚠️ Wiki hatası: {e}", icon="🔥")
    ddg_url = None
    try:
        with DDGS(headers={'User-Agent':USER_AGENT},timeout=REQUEST_TIMEOUT) as ddgs:
            res=list(ddgs.text(query,region='tr-tr',safesearch='moderate',max_results=3))
            if res: snp,url=res[0].get('body'),res[0].get('href'); ddg_url=unquote(url) if url else None; domain=urlparse(ddg_url).netloc if ddg_url else ''; result=f"**Web Özeti (DDG - {domain}):**\n\n{_clean_text(snp)}\n\nKaynak: {ddg_url}" if snp and ddg_url else result; st.toast(f"ℹ️ DDG özeti bulundu.", icon="🦆") if result and not result.startswith("**Wiki") else None
    except Exception as e: st.toast(f"⚠️ DDG hatası: {e}", icon="🔥")
    if ddg_url: scraped=scrape_url_content(ddg_url); domain=urlparse(ddg_url).netloc; result=f"**Web Sayfası ({domain}):**\n\n{scraped}\n\nKaynak: {ddg_url}" if scraped else result; st.toast(f"✅ '{domain}' kazındı." if scraped else f"ℹ️ '{domain}' kazınamadı.", icon="📄" if scraped else "📝")
    if not result: st.toast(f"'{query}' için web sonucu yok.", icon="❌"); return None
    return result

# --- Sohbet Geçmişi Yönetimi ---
@st.cache_data(ttl=86400)
def load_all_chats_cached(file_path=CHAT_HISTORY_FILE):
    if os.path.exists(file_path):
        try:
            with open(file_path,"r",encoding="utf-8") as f: content=f.read()
            if content and content.strip(): data=json.loads(content); return {str(k):v for k,v in data.items()} if isinstance(data,dict) else {}
            else: return {}
        except (json.JSONDecodeError, Exception) as e: st.error(f"Sohbet yüklenemedi: {e}", icon="🔥"); try: os.rename(file_path,f"{file_path}.err_{int(time.time())}") except OSError: pass; return {}
    return {}
def save_all_chats(chats_dict, file_path=CHAT_HISTORY_FILE):
    try: with open(file_path,"w",encoding="utf-8") as f: json.dump(chats_dict,f,ensure_ascii=False,indent=2)
    except Exception as e: st.error(f"Sohbet kaydedilemedi: {e}", icon="🔥"); print(f"ERROR: Save chats failed: {e}")

# --- Gemini Yanıt Alma ---
def get_gemini_response_cached(prompt, history, stream=False):
    model = globals().get('gemini_model');
    if not model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    val_hist=[{'role':m['role'],'parts':[m['parts']]} for m in history if m.get('role') in ['user','model'] and isinstance(m.get('parts'),str) and m['parts'].strip()]
    try:
        chat=model.start_chat(history=val_hist); response=chat.send_message(prompt,stream=stream)
        if stream: return response
        else:
             if response.parts: return "".join(p.text for p in response.parts if hasattr(p,'text'))
             else: reason=getattr(response.prompt_feedback,'block_reason',None); msg=f"Engellendi ({reason})." if reason else f"Tamamlanmadı ({getattr(response.candidates[0],'finish_reason','?')})." if response.candidates else "Boş yanıt."; st.warning(msg,icon="🛡️" if reason else "⚠️"); return f"{GEMINI_ERROR_PREFIX} {msg}"
    except Exception as e: st.error(f"Gemini API Hatası: {e}",icon="🔥"); print(f"ERROR: Gemini API failed: {e}"); return f"{GEMINI_ERROR_PREFIX} {e}"

# --- Supabase Loglama ---
def log_to_supabase(table, data):
    client=globals().get('supabase');
    if not client: print(f"INFO: Supabase unavailable, skip log: {table}"); return False
    try: defaults={'user_name':st.session_state.get('user_name','N/A'),'session_id':_get_session_id(),'app_version':APP_VERSION,'chat_id':st.session_state.get('active_chat_id','N/A')}; client.table(table).insert({**defaults,**data}).execute(); return True
    except Exception as e: st.toast(f"⚠️ Loglama hatası: {table}",icon="💾"); print(f"ERROR: Supabase log ({table}): {e}"); return False
def log_interaction(p, r, s, mid, cid): return log_to_supabase(SUPABASE_TABLE_LOGS, {"user_prompt":p,"ai_response":r,"response_source":s,"message_id":mid,"chat_id":cid})
def log_feedback(mid, p, r, ft, c=""): data={"message_id":mid,"user_prompt":p,"ai_response":r,"feedback_type":ft,"comment":c}; success=log_to_supabase(SUPABASE_TABLE_FEEDBACK,data); st.toast("Geri bildiriminiz alındı!" if success else "Geri bildirim gönderilemedi.",icon="💌" if success else "😔"); return success

# --- Yanıt Orkestrasyonu ---
def get_hanogt_response_orchestrator(prompt, history, msg_id, chat_id, use_stream=False):
    response, source = None, "Bilinmiyor"
    kb_resp = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
    if kb_resp: source="Fonksiyonel" if prompt.lower() in DYNAMIC_FUNCTIONS_MAP else "Bilgi Tabanı"; log_interaction(prompt,kb_resp,source,msg_id,chat_id); return kb_resp, f"{APP_NAME} ({source})"
    if globals().get('gemini_model'):
        gem_resp=get_gemini_response_cached(prompt,history,stream=use_stream)
        if gem_resp:
            if use_stream: return gem_resp,f"{APP_NAME} (Gemini Stream)"
            elif isinstance(gem_resp,str) and not gem_resp.startswith(GEMINI_ERROR_PREFIX): source="Gemini"; log_interaction(prompt,gem_resp,source,msg_id,chat_id); return gem_resp,f"{APP_NAME} ({source})"
    is_q="?" in prompt or any(k in prompt.lower() for k in ["nedir","kimdir","nasıl","bilgi","araştır","haber"]);
    if not response and is_q and len(prompt.split())>2:
        web_resp=search_web(prompt)
        if web_resp: source="Web"; log_interaction(prompt,web_resp,source,msg_id,chat_id); return web_resp,f"{APP_NAME} ({source})" # Kaynağı daha detaylı ayrıştırabiliriz
    response=random.choice([f"Üzgünüm {st.session_state.get('user_name','')}, yardımcı olamıyorum.","Anlayamadım.","Bilgim yok."]); source="Varsayılan"
    log_interaction(prompt,response,source,msg_id,chat_id); return response,f"{APP_NAME} ({source})"

# --- Yaratıcı Modüller ---
def creative_response_generator(p,l="orta",s="genel"): tmpl={"genel":["Fikir: {}"],"şiirsel":["Kalbimden: {}"],"hikaye":["Bir varmış: {}"]}; idea=generate_new_idea_creative(p,s); sens=[s.strip() for s in idea.split('.') if s.strip()]; n=len(sens); idea=idea if n==0 else ". ".join(sens[:max(1,n//3)])+"." if l=="kısa" else idea+f"\n\nDahası, {generate_new_idea_creative(p[::-1],s)}" if l=="uzun" else idea; return random.choice(tmpl.get(s,tmpl["genel"])).format(idea)
def generate_new_idea_creative(sd,st="genel"): e=["zaman","orman","rüya","kuantum","gölge"]; a=["dokur","çözer","yansıtır","inşa eder","fısıldar"]; o=["kaderi","kodu","sınırları","sırları","melodiyi"]; w=re.findall(r'\b\w{4,}\b',sd.lower()); sds=random.sample(w,k=min(len(w),1))+["gizem"]; e1,a1,o1=random.choice(e),random.choice(a),random.choice(o); return f"{sds[0].capitalize()} {a1}, {e1} aracılığıyla {o1}."
def advanced_word_generator(b): base=b or "kelime"; cln="".join(filter(str.isalpha,base.lower())); v="aeıioöuü";c="bcçdfgğhjklmnprsştvyz"; pre=["bio","krono","neo","mega","poli","meta","xeno"]; suf=["genez","sfer","loji","tronik","morf","matik","skop"]; core=cln[random.randint(0,max(0,len(cln)-3)):][:2] if len(cln)>2 and random.random()<0.6 else "".join(random.choice(c if i%2 else v) for i in range(3)); w=core; w=random.choice(pre)+w if random.random()>0.4 else w; w+=random.choice(suf) if random.random()>0.4 else ""; return w.capitalize() if len(w)>1 else "KelimeX"

# --- Görsel Oluşturucu (Geliştirilmiş) ---
def generate_prompt_influenced_image(prompt):
    w,h=512,512; p_lower=prompt.lower()
    themes = { # Anahtar kelime -> {bg_colors, shapes_list} (kısaltmalar: t=type, c=color, p=pos(x,y), s=size(radius/base), swh=size(w,h), pts=points list, l=layer)
        "güneş": {"bg":[(255,230,150),(255,160,0)],"sh":[{"t":"circle","c":(255,255,0,220),"p":(0.25,0.25),"s":0.2,"l":1}]},
        "ay": {"bg":[(10,10,50),(40,40,100)],"sh":[{"t":"circle","c":(240,240,240,200),"p":(0.75,0.2),"s":0.15,"l":1}]},
        "gökyüzü": {"bg":[(135,206,250),(70,130,180)],"sh":[]},
        "bulut": {"bg":None,"sh":[{"t":"ellipse","c":(255,255,255,180),"p":(random.uniform(0.2,0.8),random.uniform(0.1,0.4)),"swh":(random.uniform(0.15,0.35),random.uniform(0.08,0.15)),"l":1} for _ in range(random.randint(2,4))]},
        "deniz": {"bg":[(0,105,148),(0,0,100)],"sh":[{"t":"rect","c":(60,120,180,150),"p":(0.5,0.75),"swh":(1.0,0.5),"l":0}]}, # Layer 0 (arka)
        "nehir": {"bg":None,"sh":[{"t":"line","c":(100,149,237,180),"pts":[(0,random.uniform(0.6,0.8)),(0.3,random.uniform(0.65,0.75)),(0.7,random.uniform(0.6,0.7)),(1,random.uniform(0.55,0.75))],"w":15,"l":0}]}, # Layer 0
        "orman": {"bg":[(34,139,34),(0,100,0)],"sh":[{"t":"tri","c":(random.randint(0,30),random.randint(70,100),random.randint(0,30),200),"p":(random.uniform(0.1,0.9),random.uniform(0.65,0.9)),"s":random.uniform(0.07,0.20),"l":2} for _ in range(random.randint(9,16))]}, # Layer 2 (orta-ön)
        "ağaç": {"bg":[(180,220,180),(140,190,140)],"sh":[{"t":"rect","c":(139,69,19,255),"p":(rx:=random.uniform(0.2,0.8),0.8),"swh":(0.05,0.3),"l":2},{"t":"ellipse","c":(34,139,34,200),"p":(rx,0.6),"swh":(0.25,0.2),"l":2}]}, # Aynı X'i kullan (rx:= walrus op.) Layer 2
        "ev": {"bg":None,"sh":[{"t":"rect","c":(200,180,150,240),"p":(ex:=random.uniform(0.2,0.8),0.8),"swh":(0.15,0.2),"l":2},{"t":"poly","c":(139,0,0,240),"pts":[(ex-0.075,0.7),(ex+0.075,0.7),(ex,0.6)],"l":2}]}, # Basit ev+çatı, Layer 2
        "dağ": {"bg":[(200,200,200),(100,100,100)],"sh":[{"t":"poly","c":(random.randint(100,160),)*3+(230,),"pts":[(random.uniform(0.1,0.4),0.85),(0.5,random.uniform(0.2,0.5)),(random.uniform(0.6,0.9),0.85)],"l":0} for _ in range(random.randint(1,2))]}, # Layer 0
        "şehir": {"bg":[(100,100,120),(50,50,70)],"sh":[{"t":"rect","c":(random.randint(60,100),)*3+(random.randint(190,230),),"p":(random.uniform(0.1,0.9),random.uniform(0.5,0.9)),"swh":(random.uniform(0.04,0.12),random.uniform(0.1,0.55)),"l":1} for _ in range(random.randint(10,18))]}, # Layer 1 (orta)
        "çiçek": {"bg":None,"sh":[{"t":"circle","c":(random.randint(200,255),random.randint(100,200),random.randint(150,255),210),"p":(random.uniform(0.1,0.9),random.uniform(0.8,0.95)),"s":0.015,"l":3} for _ in range(random.randint(5,10))]}, # Layer 3 (ön)
        "kar": {"bg":None,"sh":[{"t":"circle","c":(255,255,255,150),"p":(random.random(),random.random()),"s":0.005,"l":3}]}, # Layer 3
        "yıldız": {"bg":None,"sh":[{"t":"circle","c":(255,255,200,200),"p":(random.random(),random.uniform(0,0.5)),"s":0.003,"l":1}]}, # Layer 1
    }
    bg1,bg2=(random.randint(30,120),)*3,(random.randint(120,220),)*3
    all_shapes = []; themes_applied = 0
    for kw,th in themes.items():
        if kw in p_lower:
            if th["bg"] and themes_applied==0: bg1,bg2=th["bg"]
            all_shapes.extend(th["sh"]); themes_applied+=1

    img=Image.new('RGBA',(w,h),(0,0,0,0)); draw=ImageDraw.Draw(img)
    for y in range(h): r=y/h; R,G,B=[int(bg1[i]*(1-r)+bg2[i]*r) for i in range(3)]; draw.line([(0,y),(w,y)],fill=(R,G,B,255)) # BG gradient

    # Katmanlara göre sırala ve çiz
    all_shapes.sort(key=lambda s: s.get("l", 2)) # Default layer 2 (orta)
    for s in all_shapes:
        try: # Şekil çizimi
            st,sc,sp,out=s["t"],s["c"],s.get("p"),(0,0,0,40) if len(s["c"])==4 and s["c"][3]<250 else None
            if sp: cx,cy=int(sp[0]*w),int(sp[1]*h)
            if st=="circle": r=int(s["s"]*min(w,h)/2); draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=sc,outline=out)
            elif st=="rect" or st=="ellipse": wr,hr=s["swh"]; wp,hp=int(wr*w),int(hr*h); box=(cx-wp//2,cy-hp//2,cx+wp//2,cy+hp//2); draw.rectangle(box,fill=sc,outline=out) if st=="rect" else draw.ellipse(box,fill=sc,outline=out)
            elif st=="tri": sz=int(s["s"]*min(w,h)); pts=[(cx,cy-int(sz*0.58)),(cx-sz//2,cy+int(sz*0.3)),(cx+sz//2,cy+int(sz*0.3))]; draw.polygon(pts,fill=sc,outline=out)
            elif st=="poly": pts_px=[(int(p[0]*w),int(p[1]*h)) for p in s["pts"]]; draw.polygon(pts_px,fill=sc,outline=out)
            elif st=="line": pts_px=[(int(p[0]*w),int(p[1]*h)) for p in s["pts"]]; line_w=s.get("w",5); draw.line(pts_px,fill=sc,width=line_w,joint="curve") # Eğri çizgi
        except Exception as e: print(f"DEBUG: Shape draw error {s}: {e}"); continue
    if themes_applied==0: # Rastgele şekiller (fallback)
        for _ in range(random.randint(4,7)): x,y=random.randint(0,w),random.randint(0,h); clr=tuple(random.randint(50,250) for _ in range(3))+(random.randint(150,220),); r=random.randint(20,70); draw.ellipse((x-r,y-r,x+r,y+r),fill=clr) if random.random()>0.5 else draw.rectangle((x-r//2,y-r//2,x+r//2,y+r//2),fill=clr)
    # Metin
    try:
        font=ImageFont.load_default(); txt=prompt[:80]
        if os.path.exists(FONT_FILE): try: fsize=max(14,min(28,int(w/(len(txt)*0.3+10)))); font=ImageFont.truetype(FONT_FILE,fsize) except IOError: pass
        bb=draw.textbbox((0,0),txt,font=font,anchor="lt") if hasattr(draw,'textbbox') else draw.textsize(txt,font=font); tw,th=bb[2]-bb[0] if hasattr(draw,'textbbox') else bb[0], bb[3]-bb[1] if hasattr(draw,'textbbox') else bb[1]
        tx,ty=(w-tw)/2,h*0.95-th; draw.text((tx+1,ty+1),txt,font=font,fill=(0,0,0,150)); draw.text((tx,ty),txt,font=font,fill=(255,255,255,230))
    except Exception as e: st.toast(f"Metin yazılamadı: {e}",icon="📝")
    return img.convert("RGB")

# --- Session State Başlatma ---
def initialize_session_state():
    defaults = {
        'all_chats': {}, 'active_chat_id': None, 'next_chat_id_counter': 0,
        'app_mode': "Yazılı Sohbet", 'user_name': None, 'user_avatar_bytes': None,
        'show_main_app': False, 'greeting_message_shown': False,
        'tts_enabled': True, 'gemini_stream_enabled': True,
        'gemini_temperature': 0.7, 'gemini_top_p': 0.95, 'gemini_top_k': 40,
        'gemini_max_tokens': 4096, 'gemini_model_name': 'gemini-1.5-flash-latest',
        'gemini_system_prompt': "", # YENİ: Sistem talimatı
        'message_id_counter': 0, 'last_ai_response_for_feedback': None,
        'last_user_prompt_for_feedback': None, 'current_message_id_for_feedback': None,
        'feedback_comment_input': "", 'show_feedback_comment_form': False,
        'session_id': str(uuid.uuid4()), 'last_feedback_type': 'positive',
        'models_initialized': False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

initialize_session_state()

# --- Modelleri ve İstemcileri Başlatma ---
if not st.session_state.models_initialized:
    print("INFO: Initializing resources...")
    gemini_model = initialize_gemini_model() # Global modeli ayarlar
    supabase_client_instance = init_supabase_client_cached() # Cache'li fonksiyonu çağırır
    if supabase_client_instance: supabase = supabase_client_instance; st.toast("🔗 Supabase bağlantısı başarılı.", icon="🧱"); supabase_error_global = None
    else: supabase = None; supabase_error_global = "Supabase başlatılamadı. (Detaylar loglarda)"
    tts_engine = init_tts_engine_cached() # Global motoru ayarlar
    st.session_state.all_chats = load_all_chats_cached()
    if not st.session_state.active_chat_id and st.session_state.all_chats:
        try: st.session_state.active_chat_id = sorted(st.session_state.all_chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)[0]
        except: st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None
    user_greeting = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting)
    st.session_state.models_initialized = True
    print("INFO: Initialization complete.")
else: # Sonraki çalıştırmalar
    gemini_model = globals().get('gemini_model')
    supabase = globals().get('supabase')
    tts_engine = globals().get('tts_engine')
    user_greeting = st.session_state.get('user_name', "kullanıcı")
    KNOWLEDGE_BASE = load_knowledge_from_file(user_name_for_greeting=user_greeting)
    supabase_error_global = globals().get('supabase_error_global')
    gemini_init_error_global = globals().get('gemini_init_error_global')
    tts_init_error_global = globals().get('tts_init_error_global')

# --- ARAYÜZ FONKSİYONLARI ---
def display_settings_section():
    """Ayarlar ve Kişiselleştirme bölümünü gösterir."""
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        # Profil
        col1, col2 = st.columns([0.8, 0.2]);
        with col1: st.markdown(f"**Hoş Geldin, {st.session_state.user_name}!**"); new_name=st.text_input("Ad:",value=st.session_state.user_name,key="ch_name",label_visibility="collapsed");
        if new_name != st.session_state.user_name and new_name.strip(): st.session_state.user_name=new_name.strip(); load_knowledge_from_file.clear(); st.toast("Ad güncellendi!",icon="✏️"); st.rerun()
        with col2:
            if st.session_state.user_avatar_bytes: st.image(st.session_state.user_avatar_bytes,width=60);
            if st.button("🗑️", key="rm_av", help="Avatarı kaldır",use_container_width=True): st.session_state.user_avatar_bytes=None; st.toast("Avatar kaldırıldı.",icon="🗑️"); st.rerun()
        up_avatar=st.file_uploader("Avatar:",type=["png","jpg"],key="up_av",label_visibility="collapsed");
        if up_avatar:
             if up_avatar.size>2*1024*1024: st.error("Dosya > 2MB!",icon="️")
             else: st.session_state.user_avatar_bytes=up_avatar.getvalue(); st.toast("Avatar güncellendi!",icon="🖼️"); st.rerun()
        st.caption("Avatar sadece bu oturumda saklanır.")
        st.divider()
        # Arayüz ve AI
        st.subheader("🤖 Yapay Zeka ve Arayüz")
        tcol1,tcol2=st.columns(2); engine_ok=globals().get('tts_engine') is not None
        with tcol1: st.session_state.tts_enabled=st.toggle("Metin Okuma (TTS)",value=st.session_state.tts_enabled,disabled=not engine_ok,help="Yanıtları sesli oku.")
        with tcol2: st.session_state.gemini_stream_enabled=st.toggle("Yanıt Akışı",value=st.session_state.gemini_stream_enabled,help="Yanıtları kelime kelime al.")
        # Sistem Talimatı (Yeni)
        st.session_state.gemini_system_prompt = st.text_area(
            "AI Sistem Talimatı (Opsiyonel):",
            value=st.session_state.get('gemini_system_prompt', ""),
            key="system_prompt_input",
            height=100,
            placeholder="Yapay zekanın genel davranışını veya rolünü tanımlayın (örn: 'Sen esprili bir asistansın.', 'Kısa ve öz cevap ver.', 'Bir uzay kaşifi gibi konuş.')",
            help="Modelin yanıtlarını etkilemek için genel bir talimat girin. Modelin desteklemesi gerekir."
        )

        st.markdown("##### 🧠 Hanogt AI Gelişmiş Yapılandırma")
        gcol1,gcol2=st.columns(2)
        with gcol1:
            st.session_state.gemini_model_name=st.selectbox("AI Modeli:",['gemini-1.5-flash-latest','gemini-1.5-pro-latest'],index=0 if st.session_state.gemini_model_name=='gemini-1.5-flash-latest' else 1,key="sel_model",help="Model yetenekleri/maliyetleri farklıdır.")
            st.session_state.gemini_temperature=st.slider("Sıcaklık:",0.0,1.0,st.session_state.gemini_temperature,0.05,key="temp_slider",help="Yaratıcılık (0=Kesin, 1=Yaratıcı)")
            st.session_state.gemini_max_tokens=st.slider("Maks Token:",256,8192,st.session_state.gemini_max_tokens,128,key="max_tokens_slider",help="Max yanıt uzunluğu")
        with gcol2:
            st.session_state.gemini_top_k=st.slider("Top K:",1,100,st.session_state.gemini_top_k,1,key="topk_slider",help="Kelime Seçim Çeşitliliği") # Geri geldi
            st.session_state.gemini_top_p=st.slider("Top P:",0.0,1.0,st.session_state.gemini_top_p,0.05,key="topp_slider",help="Kelime Seçim Odaklılığı") # Geri geldi
            if st.button("⚙️ AI Ayarlarını Uygula",key="reload_ai_btn",use_container_width=True,type="primary",help="Seçili AI modelini ve parametreleri yeniden yükler."):
                global gemini_model; with st.spinner("AI modeli yeniden başlatılıyor..."): gemini_model=initialize_gemini_model()
                if not gemini_model: st.error("AI modeli yüklenemedi."); st.rerun() # Hata varsa rerun ile göster

        # Geçmiş Yönetimi
        st.divider(); st.subheader("🧼 Geçmiş Yönetimi")
        hcol1, hcol2 = st.columns(2)
        with hcol1: # Aktif Sohbeti Temizle (Yeni)
             active_chat_id = st.session_state.get('active_chat_id')
             disabled_clear_current = not bool(active_chat_id and st.session_state.all_chats.get(active_chat_id))
             if st.button("🧹 Aktif Sohbeti Temizle", use_container_width=True, type="secondary", key="clear_current_chat_btn", help="Sadece şu an açık olan sohbeti temizler.", disabled=disabled_clear_current):
                  if active_chat_id and active_chat_id in st.session_state.all_chats:
                       st.session_state.all_chats[active_chat_id] = [] # Aktif sohbeti boşalt
                       save_all_chats(st.session_state.all_chats)
                       st.toast("Aktif sohbet temizlendi!", icon="🧹"); st.rerun()
        with hcol2: # Tümünü Temizle
             if st.button("🗑️ TÜM Geçmişi Sil", use_container_width=True, type="danger", key="clear_all_chats_btn", help="Dikkat! Tüm sohbetleri kalıcı olarak siler.", disabled=not st.session_state.all_chats):
                  # !! Güvenlik için belki bir onay mekanizması daha eklenebilir !!
                  st.session_state.all_chats={}; st.session_state.active_chat_id=None; save_all_chats({})
                  st.toast("TÜM sohbet geçmişi silindi!", icon="🗑️"); st.rerun()

def display_chat_list_and_about(left_column):
    """Sol kolonda sohbet listesini, yönetimi ve Hakkında'yı gösterir."""
    with left_column:
        st.markdown("#### Sohbetler")
        if st.button("➕ Yeni Sohbet", use_container_width=True, key="new_chat"):
            st.session_state.next_chat_id_counter+=1; ts=int(time.time())
            new_id=f"chat_{st.session_state.next_chat_id_counter}_{ts}"; st.session_state.all_chats[new_id]=[]; st.session_state.active_chat_id=new_id; save_all_chats(st.session_state.all_chats); st.rerun()
        st.markdown("---")
        chat_list_cont=st.container(height=450, border=False) # Yükseklik ayarlandı
        with chat_list_cont:
            chats=st.session_state.all_chats; sorted_ids=sorted(chats.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)
            if not sorted_ids: st.caption("Henüz sohbet yok.")
            else:
                active_id=st.session_state.get('active_chat_id')
                for chat_id in sorted_ids:
                    history=chats.get(chat_id,[]); first_msg=next((m.get('parts','') for m in history if m.get('role')=='user'), None)
                    title=f"Sohbet {chat_id.split('_')[1]}" if not first_msg else first_msg[:30]+("..." if len(first_msg)>30 else "")
                    title=title if history else "Boş Sohbet"
                    lcol, dcol, rcol=st.columns([0.7,0.15,0.15]) # Select, Download, Delete
                    btn_type="primary" if active_id==chat_id else "secondary"
                    if lcol.button(title,key=f"sel_{chat_id}",use_container_width=True,type=btn_type,help=f"'{title}' aç"):
                        if active_id!=chat_id: st.session_state.active_chat_id=chat_id; st.rerun()
                    # İndirme Butonu (Yeni)
                    chat_content_str = ""
                    for msg in history: chat_content_str += f"{'Kullanıcı' if msg.get('role')=='user' else msg.get('sender_display', 'AI')}: {msg.get('parts', '')}\n\n"
                    dcol.download_button("⬇️", data=chat_content_str.encode('utf-8'), file_name=f"{title.replace(' ','_')}_{chat_id}.txt", mime="text/plain", key=f"dl_{chat_id}", help=f"'{title}' sohbetini indir (.txt)", use_container_width=True, disabled=not history)
                    # Silme Butonu
                    if rcol.button("🗑️", key=f"del_{chat_id}", use_container_width=True, help=f"'{title}' sil", type="secondary"):
                         if chat_id in chats:
                             del chats[chat_id];
                             if active_id==chat_id: remaining=sorted(chats.keys(), key=lambda x:int(x.split('_')[-1]),reverse=True); st.session_state.active_chat_id=remaining[0] if remaining else None
                             save_all_chats(chats); st.toast(f"'{title}' silindi.",icon="🗑️"); st.rerun()
        # Hakkında
        st.markdown("<br>",unsafe_allow_html=True)
        with st.expander("ℹ️ Uygulama Hakkında",expanded=False): st.markdown(f"**{APP_NAME} v{APP_VERSION}**\n\nAI Destekli Asistan\n\nGeliştirici: **Hanogt**\n\n© 2024-{CURRENT_YEAR}"); st.caption(f"Oturum: {_get_session_id()[:8]}...")

def display_chat_message_with_feedback(msg_data, msg_idx, chat_id):
    """Tek bir sohbet mesajını formatlar ve gösterir."""
    role=msg_data.get('role','model'); content=msg_data.get('parts',''); sender=msg_data.get('sender_display',APP_NAME if role=='model' else st.session_state.user_name); is_user=(role=='user')
    avatar="🧑";
    if is_user: avatar=Image.open(BytesIO(st.session_state.user_avatar_bytes)) if st.session_state.user_avatar_bytes else "🧑"
    else: avatar="✨" if "Gemini" in sender else "🌐" if any(w in sender for w in ["Web","Wiki"]) else "📚" if any(w in sender for w in ["Bilgi","Fonksiyon"]) else "🤖"
    with st.chat_message(role,avatar=avatar):
        if "```" in content: parts=content.split("```"); # Kod bloğu
            for i,part in enumerate(parts):
                if i%2==1: lang=re.match(r"(\w+)\n",part); code=part[len(lang.group(1))+1:] if lang else part; st.code(code,language=lang.group(1) if lang else None);
                if st.button("📋",key=f"copy_{chat_id}_{msg_idx}_{i}",help="Kopyala"): st.write_to_clipboard(code); st.toast("Kod kopyalandı!",icon="✅")
                elif part.strip(): st.markdown(part,unsafe_allow_html=True)
        elif content.strip(): st.markdown(content,unsafe_allow_html=True)
        else: st.caption("[Boş Mesaj]")
        # Token Sayımı (İsteğe Bağlı)
        token_count = None
        if tiktoken_encoder and content.strip():
             try: token_count = len(tiktoken_encoder.encode(content))
             except Exception: pass # Hata olursa sayma
        # Eylemler (AI Mesajları İçin)
        if not is_user and content.strip():
             action_cols=st.columns([0.8,0.1,0.1]) # Alan ayarı
             with action_cols[0]: st.caption(f"Kaynak: {sender.split('(')[-1].replace(')','')} {f'| ~{token_count} token' if token_count else ''}") # Kaynak ve Token
             with action_cols[1]: # TTS
                 if st.session_state.tts_enabled and globals().get('tts_engine'):
                     if st.button("🔊",key=f"tts_{chat_id}_{msg_idx}",help="Oku",use_container_width=True): speak(content)
             with action_cols[2]: # Feedback
                 if st.button("✍️",key=f"fb_{chat_id}_{msg_idx}",help="Geri Bildirim",use_container_width=True):
                     st.session_state.current_message_id_for_feedback=f"{chat_id}_{msg_idx}"; prev_p="[İstem yok]"
                     if msg_idx>0 and st.session_state.all_chats[chat_id][msg_idx-1]['role']=='user': prev_p=st.session_state.all_chats[chat_id][msg_idx-1]['parts']
                     st.session_state.last_user_prompt_for_feedback=prev_p; st.session_state.last_ai_response_for_feedback=content; st.session_state.show_feedback_comment_form=True; st.session_state.feedback_comment_input=""; st.rerun()

def display_feedback_form_if_active():
    """Aktifse geri bildirim formunu gösterir."""
    if st.session_state.get('show_feedback_comment_form') and st.session_state.current_message_id_for_feedback:
        st.markdown("---"); fkey=f"fb_form_{st.session_state.current_message_id_for_feedback}"
        with st.form(key=fkey):
            st.markdown("#### Yanıt Geri Bildirimi"); st.caption(f"**İstem:** `{st.session_state.last_user_prompt_for_feedback[:80]}...`"); st.caption(f"**Yanıt:** `{st.session_state.last_ai_response_for_feedback[:80]}...`")
            fb_type=st.radio("Değerlendirme:",["👍 Beğendim","👎 Beğenmedim"],horizontal=True,key=f"type_{fkey}",index=0 if st.session_state.last_feedback_type=='positive' else 1)
            comment=st.text_area("Yorum (isteğe bağlı):",value=st.session_state.feedback_comment_input,key=f"cmt_{fkey}",height=100,placeholder="Neden?")
            st.session_state.feedback_comment_input=comment; scol,ccol=st.columns(2); submitted=scol.form_submit_button("✅ Gönder",use_container_width=True,type="primary"); cancelled=ccol.form_submit_button("❌ Vazgeç",use_container_width=True)
            if submitted: parsed_type="positive" if fb_type=="👍 Beğendim" else "negative"; st.session_state.last_feedback_type=parsed_type; log_feedback(st.session_state.current_message_id_for_feedback,st.session_state.last_user_prompt_for_feedback,st.session_state.last_ai_response_for_feedback,parsed_type,comment); st.session_state.show_feedback_comment_form=False; st.session_state.current_message_id_for_feedback=None; st.session_state.feedback_comment_input=""; st.rerun()
            elif cancelled: st.session_state.show_feedback_comment_form=False; st.session_state.current_message_id_for_feedback=None; st.session_state.feedback_comment_input=""; st.rerun()
        st.markdown("---")

def display_chat_interface_main(main_col_ref):
    """Ana sohbet arayüzünü sağ kolonda yönetir."""
    with main_col_ref:
        active_chat_id=st.session_state.get('active_chat_id')
        if active_chat_id is None: st.info("💬 Başlamak için **'➕ Yeni Sohbet'** butonuna tıklayın veya listeden bir sohbet seçin.",icon="👈"); return
        current_history=st.session_state.all_chats.get(active_chat_id,[])
        chat_container=st.container(height=600,border=False) # Yükseklik artırıldı
        with chat_container:
            if not current_history: st.info(f"Merhaba {st.session_state.user_name}! Yeni sohbetinize hoş geldiniz.",icon="👋")
            for i,msg in enumerate(current_history): display_chat_message_with_feedback(msg,i,active_chat_id)
        display_feedback_form_if_active() # Konteyner dışı
        prompt_placeholder = f"{st.session_state.user_name}, ne merak ediyorsun?"
        # Token sayısını chat input'a ekle (opsiyonel)
        prompt_token_count = None
        if tiktoken_encoder:
             try: prompt_token_count = len(tiktoken_encoder.encode(st.session_state.get(f"input_{active_chat_id}_value",""))) # Chat input değerini al (bu doğrudan mümkün değil, callback lazım)
             except: pass # Şimdilik placeholder
        user_prompt=st.chat_input(prompt_placeholder, key=f"input_{active_chat_id}") # , help=f"Prompt token: ~{prompt_token_count}" if prompt_token_count else None)

        if user_prompt:
            user_msg={'role':'user','parts':user_prompt}; st.session_state.all_chats[active_chat_id].append(user_msg); save_all_chats(st.session_state.all_chats)
            msg_id=f"msg_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter+=1
            hist_limit=20; history_for_model=st.session_state.all_chats[active_chat_id][-hist_limit:-1]
            with st.chat_message("assistant",avatar="⏳"): placeholder=st.empty(); placeholder.markdown("🧠 _Düşünüyorum..._")
            ai_response,sender_name=get_hanogt_response_orchestrator(user_prompt,history_for_model,msg_id,active_chat_id,use_stream=st.session_state.gemini_stream_enabled)
            final_ai_msg="";
            if st.session_state.gemini_stream_enabled and "Stream" in sender_name:
                 stream_cont=placeholder; streamed_text=""
                 try:
                     for chunk in ai_response:
                         if chunk.parts: text="".join(p.text for p in chunk.parts if hasattr(p,'text')); streamed_text+=text; stream_cont.markdown(streamed_text+"▌"); time.sleep(0.005) # Daha akıcı
                     stream_cont.markdown(streamed_text); final_ai_msg=streamed_text; log_interaction(user_prompt,final_ai_msg,"Gemini Stream",msg_id,active_chat_id)
                 except Exception as e: error=f"Stream hatası: {e}"; stream_cont.error(error); final_ai_msg=error; sender_name=f"{APP_NAME} (Stream Hatası)"; log_interaction(user_prompt,final_ai_msg,"Stream Hatası",msg_id,active_chat_id)
            else: placeholder.empty(); final_ai_msg=str(ai_response) # Loglama orkestratörde yapıldı
            ai_msg_data={'role':'model','parts':final_ai_msg,'sender_display':sender_name}; st.session_state.all_chats[active_chat_id].append(ai_msg_data); save_all_chats(st.session_state.all_chats)
            if st.session_state.tts_enabled and globals().get('tts_engine') and isinstance(final_ai_msg,str) and "Stream" not in sender_name: speak(final_ai_msg)
            st.rerun()

# --- UYGULAMA ANA AKIŞI ---
st.markdown(f"<h1 style='text-align:center;color:#0078D4;'>{APP_NAME} {APP_VERSION}</h1>",unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;font-style:italic;color:#555;'>Yapay zeka destekli kişisel asistanınız</p>",unsafe_allow_html=True)

# Hatalar
init_errors = [gemini_init_error_global, supabase_error_global] # TTS hatası toast ile gösteriliyor
for error in init_errors:
    if error: st.error(error, icon="🛑") if "API Anahtarı" in error else st.warning(error, icon="🧱" if "Supabase" in error else "⚠️")

# --- Giriş ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Başlamadan Önce...")
    lcols=st.columns([0.2,0.6,0.2])
    with lcols[1]:
        with st.form("login"):
            name=st.text_input("Size nasıl hitap edelim?",placeholder="İsminiz...",key="login_name")
            if st.form_submit_button("✨ Başla",use_container_width=True,type="primary"):
                if name and name.strip(): st.session_state.user_name=name.strip(); st.session_state.show_main_app=True; st.session_state.greeting_message_shown=False; load_knowledge_from_file.clear(); st.rerun()
                else: st.error("Geçerli bir isim girin.")
else:
    # --- Ana Uygulama ---
    if not st.session_state.greeting_message_shown: st.success(f"Hoş geldiniz {st.session_state.user_name}!",icon="🎉"); st.session_state.greeting_message_shown=True
    left_col, main_col = st.columns([1, 3]) # Layout
    display_chat_list_and_about(left_col) # Sol Kolon
    with main_col: # Sağ Kolon
        display_settings_section() # Ayarlar
        # Mod Seçimi
        st.markdown("#### Uygulama Modu")
        modes={"Yazılı Sohbet":"💬","Sesli Sohbet (Dosya)":"🎤","Yaratıcı Stüdyo":"🎨","Görsel Oluşturucu":"🖼️"}
        keys=list(modes.keys()); idx=keys.index(st.session_state.app_mode) if st.session_state.app_mode in keys else 0
        selected=st.radio("Mod:",options=keys,index=idx,format_func=lambda k:f"{modes[k]} {k}",horizontal=True,label_visibility="collapsed",key="mode_radio")
        if selected!=st.session_state.app_mode: st.session_state.app_mode=selected; st.rerun()
        st.markdown("<hr style='margin-top:0.1rem;margin-bottom:0.5rem;'>",unsafe_allow_html=True)
        # Mod İçeriği
        mode=st.session_state.app_mode
        if mode=="Yazılı Sohbet": display_chat_interface_main(main_col)
        elif mode=="Sesli Sohbet (Dosya)": # --- Sesli Sohbet ---
            st.info("Yanıtlanacak ses dosyasını yükleyin.",icon="📢"); a_file=st.file_uploader("Ses:",type=['wav','mp3','ogg','flac','m4a'],label_visibility="collapsed",key="aud_up")
            if a_file:
                st.audio(a_file,format=a_file.type); active_id=st.session_state.get('active_chat_id')
                if not active_id: st.warning("Önce sohbet seçin/başlatın.",icon="⚠️")
                else:
                    txt=None; with st.spinner(f"🔊 '{a_file.name}' işleniyor..."):
                        rec=sr.Recognizer(); try: # BytesIO ile dene
                            with sr.AudioFile(BytesIO(a_file.getvalue())) as src: aud=rec.record(src); txt=rec.recognize_google(aud,language="tr-TR"); st.success(f"**🎙️ Algılanan:**\n> {txt}")
                        except Exception as e: st.error(f"Ses işleme hatası: {e}"); print(f"ERROR: Audio failed: {e}")
                    if txt: u_msg={'role':'user','parts':f"(Ses: {a_file.name}) {txt}"}; st.session_state.all_chats[active_id].append(u_msg); msg_id=f"aud_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter+=1; hist=st.session_state.all_chats[active_id][-20:-1]; with st.spinner("🤖 Yanıt..."): ai_resp,sndr=get_hanogt_response_orchestrator(txt,hist,msg_id,active_id,False); st.markdown(f"#### {sndr} Yanıtı:"); st.markdown(str(ai_resp)); ai_msg={'role':'model','parts':str(ai_resp),'sender_display':sndr}; st.session_state.all_chats[active_id].append(ai_msg); save_all_chats(st.session_state.all_chats); st.success("✅ Yanıt sohbete eklendi!")
        elif mode=="Yaratıcı Stüdyo": # --- Yaratıcı Stüdyo ---
            st.markdown("💡 Fikir verin, AI yaratıcı metin üretsin!"); c_p=st.text_area("Tohum:",key="cr_p",placeholder="Örn: 'Yıldız tozu kahvesi'",height=100); c1,c2=st.columns(2); l_p=c1.selectbox("Uzunluk:",["kısa","orta","uzun"],index=1,key="cr_l"); s_p=c2.selectbox("Stil:",["genel","şiirsel","hikaye"],index=0,key="cr_s")
            if st.button("✨ Üret!",key="cr_g",type="primary",use_container_width=True):
                if c_p and c_p.strip(): active_id=st.session_state.get('active_chat_id','creative_no_chat'); msg_id=f"cr_{st.session_state.message_id_counter}_{int(time.time())}"; st.session_state.message_id_counter+=1; resp,sndr=None,f"{APP_NAME} (Yaratıcı)"
                    if globals().get('gemini_model'): with st.spinner("✨ Gemini ilham arıyor..."): sys_p=f"Yaratıcı asistansın. İstem:'{c_p}'. Stil:'{s_p}', Uzunluk:'{l_p}'."; gem_r=get_gemini_response_cached(sys_p,[],False); resp,sndr=(gem_r,f"{APP_NAME} (Gemini Yaratıcı)") if isinstance(gem_r,str) and not gem_r.startswith(GEMINI_ERROR_PREFIX) else (None,sndr); st.toast("Gemini yaratıcı yanıtı alınamadı.",icon="ℹ️") if not resp else None
                    if not resp: with st.spinner("✨ Hayal gücü..."): resp=creative_response_generator(c_p,l_p,s_p); new_w=advanced_word_generator(c_p.split()[0] if c_p else "k"); resp+=f"\n\n---\n🔮 **Kelimatör:** {new_w}"; sndr=f"{APP_NAME} (Yerel Yaratıcı)"
                    st.markdown(f"#### {sndr} İlhamı:"); st.markdown(resp); log_interaction(c_p,resp,sndr,msg_id,active_id); st.success("✨ Yanıt oluşturuldu!")
                else: st.warning("Lütfen bir metin girin.",icon="✍️")
        elif mode=="Görsel Oluşturucu": # --- Görsel Oluşturucu ---
            st.markdown("🎨 Hayalinizi tarif edin, AI (basitçe) çizsin!"); st.info("ℹ️ Not: Sembolik çizimler.",icon="💡"); i_p=st.text_input("Tarif:",key="img_p",placeholder="Örn: 'Nehir kenarında bir ev'")
            if st.button("🖼️ Oluştur!",key="gen_img",type="primary",use_container_width=True):
                if i_p and i_p.strip():
                    with st.spinner("🖌️ Çiziliyor..."): img=generate_prompt_influenced_image(i_p); st.image(img,caption=f"'{i_p[:60]}' yorumu",use_container_width=True)
                    try: buf=BytesIO(); img.save(buf,format="PNG"); bts=buf.getvalue(); fn_p=re.sub(r'[^\w\s-]','',i_p.lower())[:30].replace(' ','_'); fn=f"hanogt_{fn_p or 'gorsel'}_{int(time.time())}.png"; st.download_button("🖼️ İndir",data=bts,file_name=fn,mime="image/png",use_container_width=True)
                         active_id=st.session_state.get('active_chat_id'); # Sohbete ekle
                         if active_id and active_id in st.session_state.all_chats: u_msg={'role':'user','parts':f"(Görsel: {i_p})"}; ai_msg={'role':'model','parts':"(Görsel oluşturuldu.)",'sender_display':f"{APP_NAME} (Görsel)"}; st.session_state.all_chats[active_id].extend([u_msg,ai_msg]); save_all_chats(st.session_state.all_chats); st.info("İstem sohbete eklendi.",icon="💾")
                    except Exception as e: st.error(f"İndirme hatası: {e}")
                else: st.warning("Lütfen bir tarif girin.",icon="✍️")
        # Footer
        st.markdown("<hr style='margin-top:1rem;margin-bottom:0.5rem;'>",unsafe_allow_html=True); fcols=st.columns(3)
        with fcols[0]: st.caption(f"Kullanıcı: {st.session_state.get('user_name','N/A')}")
        with fcols[1]: st.caption(f"{APP_NAME} v{APP_VERSION} © {CURRENT_YEAR}")
        with fcols[2]: ai_s="Aktif" if globals().get('gemini_model') else "Kapalı"; log_s="Aktif" if globals().get('supabase') else "Kapalı"; st.caption(f"AI:{ai_s} | Log:{log_s}",help=f"Model:{st.session_state.gemini_model_name}")

