# app.py

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
from urllib.parse import urlparse
import google.generativeai as genai # Gemini için eklendi

# --- Sabitler ---
CHAT_HISTORY_FILE = "chat_history.json"
LOGO_PATH = "logo.png"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000

# --- Bilgi Tabanı (Mock) ---
try:
    from knowledge_base import load_knowledge, chatbot_response
except ImportError:
    st.warning("`knowledge_base.py` bulunamadı. Basit yanıtlar kullanılacak (Gemini'dan önce).")
    def load_knowledge():
        return {
            "merhaba": ["Merhaba!", "Selam!", "Hoş geldin!"],
            "nasılsın": ["İyiyim, sorduğun için teşekkürler!", "Harika hissediyorum!", "İşler yolunda."],
            "hanogt kimdir": ["Hanogt AI, öğrenen ve gelişen bir yapay zeka asistanıdır. Artık Gemini gücünü de kullanıyorum!", "Ben Hanogt, size Gemini modeliyle desteklenerek yardımcı olmak için buradayım."]
        }
    def chatbot_response(query, knowledge):
        query_lower = query.lower()
        for key, responses in knowledge.items():
            if key in query_lower: return random.choice(responses)
        return None
KNOWLEDGE_BASE = load_knowledge()

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hanogt AI (Gemini Destekli)", page_icon="✨", layout="wide")

# --- API Anahtarı (Streamlit Secrets ile Güvenli Erişim) ---
st.sidebar.title("🛠️ Ayarlar")

api_key = None
gemini_model = None

# Streamlit Secrets'tan API anahtarını okumayı dene
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Google API Anahtarı Secrets'tan yüklendi.")
else:
    st.sidebar.error("🛑 Google API Anahtarı bulunamadı!")
    st.sidebar.info("Lütfen projenizde `.streamlit/secrets.toml` dosyası oluşturup içine `GOOGLE_API_KEY = 'ANAHTARINIZ'` şeklinde ekleyin.")
    st.error("Uygulamanın çalışması için Google AI API Anahtarı gereklidir. Lütfen `secrets.toml` dosyasını yapılandırın.")
    # Anahtar yoksa uygulamanın devam etmesini engelleyebiliriz veya sınırlı modda çalıştırabiliriz.
    # Şimdilik sadece hata verip Gemini modelini None bırakıyoruz.
    # st.stop() # Uygulamayı tamamen durdurmak için

# API Anahtarı varsa Gemini'ı Yapılandır
if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
        st.sidebar.info("✨ Gemini modeli başarıyla yapılandırıldı.") # Başarı mesajını info'ya çevirdik
    except Exception as e:
        st.sidebar.error(f"API Anahtarı ile Gemini yapılandırılamadı: {e}")
        st.error(f"Gemini modeli yüklenirken bir hata oluştu: {e}")
        gemini_model = None # Başarısız olursa modeli None yap

# --- Logoyu Yükle ve Sidebar'a koy ---
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=100)
else:
    st.sidebar.warning(f"Logo dosyası '{LOGO_PATH}' bulunamadı.")
st.sidebar.title("Hanogt AI Kontrol Paneli")

# --- Yardımcı Fonksiyonlar (Speak, Listen, Scrape, Search Web, History, Creative, Image Gen) ---
# ... (Bu fonksiyonlar önceki kod ile aynı kalır, buraya kopyalanmalıdır) ...
# Metin Okuma Motoru
try: tts_engine = pyttsx3.init()
except Exception as e: st.sidebar.error(f"Metin okuma motoru başlatılamadı: {e}"); tts_engine = None
def speak(text):
    if tts_engine:
        try: tts_engine.say(text); tts_engine.runAndWait()
        except Exception as e: st.error(f"Konuşma sırasında hata: {e}")
    else: st.warning("Metin okuma motoru mevcut değil.")
def listen_to_microphone():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            st.info("🎙️ Dinliyorum... Lütfen konuşun.")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try: audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError: st.warning("Zaman aşımı: Konuşma algılanmadı."); return None
    except OSError as e: st.error(f"Mikrofon bulunamadı veya erişilemedi: {e}"); return None
    except Exception as e: st.error(f"Mikrofon başlatılırken hata: {e}"); return None
    try:
        st.info("İşleniyor..."); text = recognizer.recognize_google(audio, language="tr-TR"); st.success("Metin algılandı."); return text
    except sr.UnknownValueError: st.error("Ne dediğinizi anlayamadım."); return None
    except sr.RequestError as e: st.error(f"Ses tanıma servisine ulaşılamadı; {e}"); return None
    except Exception as e: st.error(f"Ses tanıma sırasında beklenmedik hata: {e}"); return None
def scrape_url_content(url):
    st.info(f"'{url}' adresinden içerik alınmaya çalışılıyor...")
    try:
        parsed_url = urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]): return None
        if parsed_url.scheme not in ['http', 'https']: return None
        headers = {'User-Agent': 'Mozilla/5.0 HanogtAI/2.0'}; response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True); response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower();
        if 'html' not in content_type: return None
        soup = BeautifulSoup(response.content, 'html.parser'); potential_content = []; selectors = ['article', 'main', '.content', '.post-content', '.entry-content', 'body']; content_found = False
        for selector in selectors:
             elements = soup.select(selector)
             if elements:
                  text_content = elements[0].find_all('p')
                  if text_content:
                      potential_content = [p.get_text(strip=True) for p in text_content if p.get_text(strip=True)];
                      if len(" ".join(potential_content)) > 100: content_found = True; break
        if not content_found:
            all_paragraphs = soup.find_all('p'); potential_content = [p.get_text(strip=True) for p in all_paragraphs if p.get_text(strip=True)]
        if not potential_content: return None
        full_text = " ".join(potential_content); cleaned_text = re.sub(r'\s+', ' ', full_text).strip(); final_text = cleaned_text[:SCRAPE_MAX_CHARS]
        if len(cleaned_text) > SCRAPE_MAX_CHARS: final_text += "..."
        st.success(f"URL'den içerik özeti başarıyla alındı."); return final_text
    except requests.exceptions.Timeout: st.error(f"URL okunurken zaman aşımı: {url}"); return None
    except requests.exceptions.RequestException as e: st.error(f"URL okunurken hata: {e}"); return None
    except Exception as e: st.error(f"Sayfa işlenirken beklenmedik hata: {e}"); return None
def search_web(query):
    st.info(f"'{query}' için web'de arama yapılıyor (Gerekirse)..."); summary = None
    try: wikipedia.set_lang("tr"); summary = wikipedia.summary(query, auto_suggest=False); st.success("Wikipedia'dan bilgi bulundu."); return f"**Wikipedia'dan:**\n\n{summary}"
    except wikipedia.exceptions.PageError: pass
    except wikipedia.exceptions.DisambiguationError as e:
        try: summary = wikipedia.summary(e.options[0], auto_suggest=False); st.success(f"Wikipedia'dan '{e.options[0]}' için bilgi bulundu."); return f"**Wikipedia'dan ('{e.options[0]}' için):**\n\n{summary}"
        except Exception: pass
    except Exception as e: st.error(f"Wikipedia araması sırasında hata: {e}")
    ddg_result_text = None; ddg_url = None
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='tr-tr', max_results=1))
            if results:
                snippet = results[0].get('body'); ddg_url = results[0].get('href')
                if snippet: st.success("DuckDuckGo'dan özet bilgi bulundu."); ddg_result_text = f"**Web Özeti (DuckDuckGo):**\n\n{snippet}\n\nKaynak: {ddg_url}"
                else: st.info("DuckDuckGo bir bağlantı buldu ancak özet sağlayamadı. Bağlantı kazınacak.")
    except Exception as e: st.error(f"DuckDuckGo araması sırasında hata: {e}")
    if ddg_url:
        scraped_content = scrape_url_content(ddg_url)
        if scraped_content:
            if len(scraped_content) > (len(ddg_result_text) if ddg_result_text else 0): return f"**Web Sayfasından Alınan İçerik ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
            elif ddg_result_text: return ddg_result_text
            else: return f"Detaylı bilgi için şu adresi ziyaret edebilirsiniz: {ddg_url}"
        elif ddg_result_text: return ddg_result_text
        else: return f"Bu konuda bir web sayfası bulundu ancak içerik otomatik olarak alınamadı. İncelemek isterseniz: {ddg_url}"
    return None
def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e: st.error(f"Sohbet geçmişi kaydedilemedi: {e}")
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f: content = f.read(); return json.loads(content) if content else []
        except Exception as e: st.error(f"Sohbet geçmişi yüklenemedi: {e}"); return []
    else: return []
def creative_response(prompt): # Fallback function if Gemini is not available
    styles = ["Bunu farklı bir açıdan düşünürsek: {}", "Hayal gücümüzü kullanalım: {}", "Belki de olay şöyledir: {}", "Aklıma şöyle bir fikir geldi: {}", "Şöyle bir senaryo canlandı gözümde: {}"]; base_idea = generate_new_idea(prompt); comment = random.choice(styles).format(base_idea); return comment
def generate_new_idea(seed): # Fallback function
    elements = ["kozmik enerji", "zaman döngüleri", "yapay bilinç", "nanobotlar", "ses manzaraları", "dijital ruhlar"]; actions = ["keşfeder", "dönüştürür", "bağlantı kurar", "yeniden şekillendirir", "hızlandırır", "yavaşlatır"]; outcomes = ["evrenin sırlarını", "insanlığın kaderini", "gerçekliğin dokusunu", "unutulmuş anıları", "geleceğin teknolojisini"]; seed_words = seed.lower().split()[:2]; idea = f"{' '.join(seed_words)} {random.choice(actions)} ve {random.choice(elements)} kullanarak {random.choice(outcomes)}."; return idea.capitalize()
def advanced_word_generator(base_word): # Fallback function
    if not base_word or len(base_word) < 3: return "Kelimetor"
    vowels = "aeiouüöı"; consonants = "bcçdfgğhjklmnprsştvyz"; prefix = ["eko", "meta", "neo", "trans", "kripto", "hiper"]; suffix = ["loji", "matik", "nomi", "grafi", "sentez", "versiyon", "izim"]
    if random.random() > 0.5: split_point = random.randint(1, len(base_word) - 1); core = base_word[:split_point] if random.random() > 0.5 else base_word[split_point:]
    else: core = ''.join(random.choice(consonants + vowels) for _ in range(random.randint(3, 5)))
    new_word = core;
    if random.random() > 0.3: new_word = random.choice(prefix) + new_word
    if random.random() > 0.3: new_word += random.choice(suffix)
    return new_word.capitalize()
def generate_prompt_influenced_image(prompt): # Rule-based image generation
    width, height = 512, 512; prompt_lower = prompt.lower()
    keyword_themes = {"güneş": {"bg": [(255, 230, 150), (255, 180, 50)], "shapes": [{"type": "circle", "color": (255, 255, 0), "pos": (0.2, 0.2), "size": 0.15}]},"ay": {"bg": [(20, 20, 80), (50, 50, 120)], "shapes": [{"type": "circle", "color": (240, 240, 240), "pos": (0.8, 0.2), "size": 0.1}]},"gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []},"bulut": {"bg": [(200, 200, 200), (150, 150, 150)], "shapes": [{"type": "ellipse", "color": (255, 255, 255), "pos": (random.uniform(0.1, 0.9), random.uniform(0.1, 0.4)), "size": random.uniform(0.1, 0.3)} for _ in range(3)]},"deniz": {"bg": [(0, 105, 148), (0, 0, 139)], "shapes": []}, "okyanus": {"bg": [(0, 0, 139), (0, 0, 50)], "shapes": []},"ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [{"type": "rectangle", "color": (139, 69, 19), "pos": (0.5, 0.8), "size": (0.05, 0.3)}, {"type": "triangle", "color": (34, 139, 34), "pos": (0.5, 0.6), "size": 0.2}]},"orman": {"bg": [(50, 100, 50), (0, 50, 0)], "shapes": [{"type": "rectangle", "color": (139, 69, 19), "pos": (random.uniform(0.1, 0.9), 0.8), "size": (0.03, 0.25)} for _ in range(5)] + [{"type": "triangle", "color": (34, 139, 34), "pos": (p[1]['pos'][0], 0.65), "size": 0.15} for p in enumerate([{"pos": (random.uniform(0.1, 0.9), 0.8)} for _ in range(5)])]},"gece": {"bg": [(10, 10, 40), (0, 0, 0)], "shapes": [{"type": "circle", "color": (255, 255, 200, 50), "pos": (random.uniform(0, 1), random.uniform(0, 1)), "size": 0.005} for _ in range(50)]},"ateş": {"bg": [(150, 0, 0), (255, 100, 0)], "shapes": [{"type": "triangle", "color": (255, 165, 0), "pos": (random.uniform(0.4, 0.6), random.uniform(0.6, 0.9)), "size": random.uniform(0.05, 0.15)} for _ in range(10)]},"kırmızı": {"bg": [(255, 100, 100), (180, 0, 0)], "shapes": []}, "mavi": {"bg": [(100, 100, 255), (0, 0, 180)], "shapes": []},"yeşil": {"bg": [(100, 255, 100), (0, 180, 0)], "shapes": []}, "sarı": {"bg": [(255, 255, 150), (220, 180, 0)], "shapes": []},}
    bg_color1 = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)); bg_color2 = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)); shapes_to_draw = []; theme_applied = False
    for keyword, theme in keyword_themes.items():
        if keyword in prompt_lower: bg_color1, bg_color2 = theme["bg"]; shapes_to_draw.extend(theme["shapes"]); theme_applied = True; break
    img = Image.new('RGB', (width, height), color=bg_color1); draw = ImageDraw.Draw(img)
    for y in range(height): ratio = y / height; r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio); g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio); b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio); draw.line([(0, y), (width, y)], fill=(r, g, b))
    for shape in shapes_to_draw:
        s_type = shape["type"]; s_color = shape["color"]; s_center_x = int(shape["pos"][0] * width); s_center_y = int(shape["pos"][1] * height)
        if s_type == "circle": s_radius = int(shape["size"] * min(width, height) / 2); draw.ellipse((s_center_x - s_radius, s_center_y - s_radius, s_center_x + s_radius, s_center_y + s_radius), fill=s_color)
        elif s_type == "ellipse": s_radius_x = int(shape["size"] * width / 2*random.uniform(0.8, 1.2)); s_radius_y = int(shape["size"] * height / 2*random.uniform(0.5, 1.0)); draw.ellipse((s_center_x - s_radius_x, s_center_y - s_radius_y, s_center_x + s_radius_x, s_center_y + s_radius_y), fill=s_color)
        elif s_type == "rectangle": s_width = int(shape["size"][0] * width); s_height = int(shape["size"][1] * height); draw.rectangle((s_center_x - s_width // 2, s_center_y - s_height // 2, s_center_x + s_width // 2, s_center_y + s_height // 2), fill=s_color)
        elif s_type == "triangle": s_base = int(shape["size"] * width * 0.8); s_height_tri = int(shape["size"] * height); points = [(s_center_x, s_center_y - s_height_tri // 2), (s_center_x - s_base // 2, s_center_y + s_height_tri // 2), (s_center_x + s_base // 2, s_center_y + s_height_tri // 2)]; draw.polygon(points, fill=s_color)
    if not theme_applied:
        for _ in range(random.randint(3, 8)):
            x1=random.randint(0, width); y1=random.randint(0, height); radius=random.randint(15, 60); shape_fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            if random.random() > 0.5: draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=shape_fill, outline=(255,255,255, 100))
            else: w_rect, h_rect = random.randint(20, 100), random.randint(20, 100); draw.rectangle((x1-w_rect//2, y1-h_rect//2, x1+w_rect//2, y1+h_rect//2), fill=shape_fill, outline=(255,255,255,100))
    try:
        try: font = ImageFont.load_default(size=24)
        except TypeError:
             try: font = ImageFont.truetype("arial.ttf", 24)
             except IOError: font = ImageFont.load_default(); st.warning("Metin için özel font yüklenemedi.")
        bbox = draw.textbbox((0, 0), prompt, font=font, anchor="lt"); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]; text_x = (width - text_width) / 2; text_y = height * 0.9 - text_height; text_y = max(text_y, height * 0.7)
        text_color = (255, 255, 255); shadow_color = (0, 0, 0, 180)
        draw.text((text_x + 1, text_y + 1), prompt, font=font, fill=shadow_color); draw.text((text_x, text_y), prompt, font=font, fill=text_color)
    except Exception as e: st.error(f"Metin yazdırılırken hata: {e}")
    return img

# --- Gemini Yanıt Fonksiyonu ---
def get_gemini_response(prompt, chat_history):
    if not gemini_model: return "Gemini modeli kullanılamıyor (API Anahtarı eksik veya geçersiz)."
    gemini_history = []
    for sender, message in chat_history: role = "user" if sender.startswith("Sen") else "model"; gemini_history.append({'role': role, 'parts': [message]})
    try:
        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(prompt, stream=False)
        # Yanıt ve güvenlik kontrolü
        if not response.parts:
             # Yanıt engellenmiş olabilir
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 st.warning(f"Yanıt güvenlik nedeniyle engellendi: {response.prompt_feedback.block_reason}")
                 return f"Üzgünüm, isteğiniz güvenlik politikaları nedeniyle işlenemedi ({response.prompt_feedback.block_reason})."
             else:
                 st.warning(f"Gemini'dan boş yanıt alındı: {response}")
                 return "Gemini'dan beklenmedik bir yanıt alındı, ancak içerik boş."
        # Normal yanıt
        return "".join(part.text for part in response.parts)
    except Exception as e: st.error(f"Gemini API ile iletişimde hata: {e}"); return f"Gemini'dan yanıt alınamadı. Hata: {e}"

# --- Session State Başlatma ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Yazılı Sohbet"

# --- Ana Sayfa Başlığı ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🚀 Hanogt AI (Gemini Gücüyle) 🚀</h1>", unsafe_allow_html=True)
if not gemini_model: # Sadece model yüklenemediyse ana uyarıyı göster
     st.warning("⚠️ Gemini modeli yüklenemedi. API anahtarınızı `.streamlit/secrets.toml` dosyasında kontrol edin. Uygulama sınırlı modda çalışacak.")
st.markdown("<p style='text-align: center;'>Size nasıl yardımcı olabilirim?</p>", unsafe_allow_html=True)

# --- Mod Seçimi (Sidebar) ---
st.sidebar.subheader("Uygulama Modu")
app_mode = st.sidebar.radio("Bir mod seçin:",["Yazılı Sohbet", "Sesli Sohbet", "Yaratıcı Mod", "Görsel Üretici"],key='app_mode_selector')
st.session_state.app_mode = app_mode

# --- Sohbet Geçmişini Temizleme (Sidebar) ---
if st.sidebar.button("🧹 Sohbet Geçmişini Temizle"):
    st.session_state.chat_history = []
    try:
        if os.path.exists(CHAT_HISTORY_FILE): os.remove(CHAT_HISTORY_FILE)
        st.sidebar.success("Sohbet geçmişi temizlendi!")
    except OSError as e: st.sidebar.error(f"Geçmiş dosyası silinirken hata: {e}")
    st.rerun()

# --- Ana İçerik Alanı ---
header_icons = {"Yazılı Sohbet": "✏️", "Sesli Sohbet": "🎙️", "Yaratıcı Mod": "✨", "Görsel Üretici": "🖼️"}
st.header(f"{header_icons.get(app_mode, '🤔')} {app_mode}")

# --- Uygulama Modları ---

if app_mode == "Yazılı Sohbet" or app_mode == "Sesli Sohbet":
    if app_mode == "Yazılı Sohbet":
        for sender, message in st.session_state.chat_history:
            role = "user" if sender.startswith("Sen") else "assistant"
            with st.chat_message(role): st.markdown(message)
    user_prompt = None
    if app_mode == "Yazılı Sohbet": user_prompt = st.chat_input("Mesajınızı buraya yazın...")
    elif app_mode == "Sesli Sohbet":
        st.info("Aşağıdaki butona basarak konuşmaya başlayabilirsiniz.")
        if st.button("🎙️ Konuşmayı Başlat"):
            with st.spinner("Sizi dinliyorum..."): user_prompt = listen_to_microphone()
            if user_prompt: st.write(f"**Sen:** {user_prompt}")
    if user_prompt:
        sender_prefix = "Sen (Sesli)" if app_mode == "Sesli Sohbet" else "Sen"
        st.session_state.chat_history.append((sender_prefix, user_prompt))
        if app_mode == "Yazılı Sohbet":
             with st.chat_message("user"): st.markdown(user_prompt)
        response = None; ai_sender = "Hanogt AI"
        with st.spinner("🤖 Düşünüyorum..."):
            response = chatbot_response(user_prompt, KNOWLEDGE_BASE)
            if response: ai_sender = "Hanogt AI (Bilgi Tabanı)"
            if not response and gemini_model:
                response = get_gemini_response(user_prompt, st.session_state.chat_history); ai_sender = "Hanogt AI (Gemini)"
            elif not response:
                st.info("Gemini kullanılamıyor veya yanıt vermedi, web'de aranıyor...") # Mesajı güncelledik
                response = search_web(user_prompt)
                if response:
                    if response.startswith("**Wikipedia"): ai_sender = "Hanogt AI (Wikipedia)"
                    elif response.startswith("**Web Özeti"): ai_sender = "Hanogt AI (Web Özeti)"
                    elif response.startswith("**Web Sayfasından"): ai_sender = "Hanogt AI (Sayfa İçeriği)"
                    else: ai_sender = "Hanogt AI (Web Link)"
            if not response or response.strip() == "" or response.startswith("Gemini'dan yanıt alınamadı.") or response.startswith("Üzgünüm, isteğiniz güvenlik politikaları"):
                 # Gemini hatası veya boş yanıt durumunda da varsayılanı ver
                 response = random.choice(["Üzgünüm, bu isteği işleyemedim.","Yanıt bulamadım.","Bir sorun oluştu."])
                 ai_sender = "Hanogt AI"
        st.session_state.chat_history.append((ai_sender, response))
        if app_mode == "Yazılı Sohbet":
            with st.chat_message("assistant"): st.markdown(response)
        elif app_mode == "Sesli Sohbet":
             st.markdown(f"**{ai_sender}:**"); st.markdown(response); speak(response)
        save_chat_history(st.session_state.chat_history)

elif app_mode == "Yaratıcı Mod":
    st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Gemini (varsa) veya yerel yaratıcılığım size yanıt versin!")
    creative_prompt = st.text_input("Yaratıcılık tohumu:", key="creative_input", placeholder="Örn: Zaman makinesi icat eden bir kedinin günlüğü")
    if creative_prompt:
        ai_sender = "Hanogt AI (Yerel Yaratıcı)"; final_response = None
        if gemini_model:
             st.info("Yaratıcı yanıt için Gemini kullanılıyor...")
             with st.spinner("✨ İlham perileri Gemini ile fısıldaşıyor..."):
                  gemini_creative_prompt = f"Aşağıdaki isteme yaratıcı, ilginç ve özgün bir yanıt ver:\n\n{creative_prompt}"
                  final_response = get_gemini_response(gemini_creative_prompt, [])
                  # Gemini'dan geçerli yanıt geldiyse göndericiyi ayarla
                  if final_response and not final_response.startswith("Gemini modeli kullanılamıyor") and not final_response.startswith("Gemini'dan yanıt alınamadı") and not final_response.startswith("Üzgünüm, isteğiniz güvenlik"):
                       ai_sender = "Hanogt AI (Gemini Yaratıcı)"
                  else: # Gemini başarısız olursa veya engellenirse fallback'e geç
                       st.warning("Gemini yanıtı alınamadı, yerel yaratıcılık kullanılıyor.")
                       final_response = creative_response(creative_prompt)
                       new_word = advanced_word_generator(creative_prompt)
                       final_response += f"\n\nAyrıca türettiğim kelime: **{new_word}**"
                       ai_sender = "Hanogt AI (Yerel Yaratıcı - Fallback)"

        else: # Gemini hiç yüklenmediyse
             st.warning("Gemini modeli yüklü değil, yerel yaratıcılık modülü devrede.")
             with st.spinner("✨ Kendi fikirlerimi demliyorum..."):
                 final_response = creative_response(creative_prompt)
                 new_word = advanced_word_generator(creative_prompt)
                 final_response += f"\n\nAyrıca türettiğim kelime: **{new_word}**"
        st.markdown(f"**{ai_sender}:**"); st.markdown(final_response)

elif app_mode == "Görsel Üretici":
    st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!")
    st.info("Not: Bu mod henüz Gemini Vision veya ImageFX gibi API'leri kullanmıyor, kural tabanlı çizim yapar.")
    image_prompt = st.text_input("Ne çizmemi istersiniz?", key="image_input", placeholder="Örn: Karlı bir dağda parlayan güneş")
    if st.button("🎨 Görseli Oluştur"):
        if image_prompt:
            with st.spinner("Fırçalarım hazırlanıyor..."): image = generate_prompt_influenced_image(image_prompt)
            st.image(image, caption=f"Hanogt AI'ın '{image_prompt}' yorumu (Kural Tabanlı)", use_container_width=True)
            buf = BytesIO(); image.save(buf, format="PNG"); byte_im = buf.getvalue()
            st.download_button(label="Görseli İndir (PNG)", data=byte_im, file_name=f"hanogt_ai_rulebased_{image_prompt[:20].replace(' ','_')}.png", mime="image/png")
        else: st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!")

# --- Alt Bilgi ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Hanogt AI (Gemini Destekli) - 2025</p>", unsafe_allow_html=True)

