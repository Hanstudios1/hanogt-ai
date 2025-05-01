# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr # Ses tanıma için
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
import google.generativeai as genai # Gemini için

# --- Sabitler ---
CHAT_HISTORY_FILE = "chat_history.json"
LOGO_PATH = "logo.png"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000
GEMINI_ERROR_PREFIX = "GeminiError:" # Gemini hatalarını ayırt etmek için

# --- Bilgi Tabanı (Mock) ---
try:
    from knowledge_base import load_knowledge, chatbot_response as kb_chatbot_response # İsim çakışmasını önle
except ImportError:
    st.warning("`knowledge_base.py` bulunamadı. Yerel bilgi tabanı yanıtları kullanılamayacak.")
    def load_knowledge(): return {} # Boş döndür
    def kb_chatbot_response(query, knowledge): return None # Çalışmayan fonksiyon
KNOWLEDGE_BASE = load_knowledge()

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hanogt AI v3 (Gemini Öncelikli)", page_icon="🚀", layout="wide")

# --- API Anahtarı (Streamlit Secrets ile Güvenli Erişim) ---
st.sidebar.title("🛠️ Ayarlar")
api_key = None
gemini_model = None
gemini_error_message = None # Yapılandırma hatasını saklamak için

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    gemini_error_message = "🛑 Google API Anahtarı Secrets'ta bulunamadı!"
    st.sidebar.error(gemini_error_message)
    st.sidebar.info("Lütfen `.streamlit/secrets.toml` dosyasını yapılandırın.")

if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Önce 'gemini-pro' deneyelim, daha genel kullanıma uygun olabilir
        gemini_model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
        # Modeli test etmek için küçük bir istek atılabilir (isteğe bağlı, başlangıcı yavaşlatabilir)
        # try:
        #     gemini_model.generate_content("Test", generation_config=genai.types.GenerationConfig(candidate_count=1))
        # except Exception as test_e:
        #      gemini_model = None # Test başarısız olursa modeli None yap
        #      gemini_error_message = f"🛑 Gemini modeli test edilemedi: {test_e}"
        #      st.sidebar.error(gemini_error_message)

    except Exception as e:
        gemini_error_message = f"🛑 Gemini yapılandırma hatası: {e}"
        st.sidebar.error(gemini_error_message)
        gemini_model = None

# Gemini Durum Göstergesi
if gemini_model:
    st.sidebar.success("✅ Gemini Modeli Aktif!")
else:
    st.sidebar.error("🛑 Gemini Modeli Aktif Değil!")
    st.warning("⚠️ Gemini modeli yüklenemediği için uygulama sınırlı modda (bilgi tabanı ve web arama ile) çalışacak.")


# --- Logoyu Yükle ve Sidebar'a koy ---
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=100)
else:
    # Logo bulunamazsa hata yerine sadece uyarı verelim, uygulama çökmesin
    st.sidebar.warning(f"Logo dosyası '{LOGO_PATH}' bulunamadı.")
st.sidebar.title("Hanogt AI Kontrol Paneli")

# --- Metin Okuma Motoru ---
tts_engine = None
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    st.sidebar.warning(f"Metin okuma motoru başlatılamadı: {e}. Sesli yanıtlar çalışmayabilir.")

def speak(text):
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Konuşma sırasında hata: {e}")
    else:
        st.warning("Metin okuma motoru kullanılamıyor.")

# --- Web Arama ve Kazıma Fonksiyonları ---
def scrape_url_content(url):
    # ... (Önceki kod ile aynı, hata yönetimi iyileştirilebilir) ...
    st.info(f"'{url}' adresinden içerik alınmaya çalışılıyor...")
    try:
        parsed_url = urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]): return None
        if parsed_url.scheme not in ['http', 'https']: return None
        headers = {'User-Agent': 'Mozilla/5.0 HanogtAI/3.0'}; response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True); response.raise_for_status()
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
        st.success(f"URL'den içerik özeti başarıyla alındı.")
        return final_text
    except requests.exceptions.Timeout: st.warning(f"URL ({url}) okunurken zaman aşımı."); return None
    except requests.exceptions.RequestException as e: st.warning(f"URL ({url}) okunurken hata: {e}"); return None
    except Exception as e: st.warning(f"Sayfa ({url}) işlenirken hata: {e}"); return None

def search_web(query):
    # ... (Önceki kod ile aynı, Wikipedia ve DDG araması yapar) ...
    st.info(f"'{query}' için web'de arama yapılıyor...")
    summary = None
    try: # Wikipedia
        wikipedia.set_lang("tr"); summary = wikipedia.summary(query, auto_suggest=False); st.success("Wikipedia'dan bilgi bulundu."); return f"**Wikipedia'dan:**\n\n{summary}"
    except wikipedia.exceptions.PageError: pass
    except wikipedia.exceptions.DisambiguationError as e:
        try: summary = wikipedia.summary(e.options[0], auto_suggest=False); st.success(f"Wikipedia'dan '{e.options[0]}' için bilgi bulundu."); return f"**Wikipedia'dan ('{e.options[0]}' için):**\n\n{summary}"
        except Exception: pass
    except Exception as e: st.warning(f"Wikipedia araması sırasında hata: {e}")

    ddg_result_text = None; ddg_url = None
    try: # DuckDuckGo
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='tr-tr', max_results=1))
            if results:
                snippet = results[0].get('body'); ddg_url = results[0].get('href')
                if snippet: st.success("DuckDuckGo'dan özet bilgi bulundu."); ddg_result_text = f"**Web Özeti (DuckDuckGo):**\n\n{snippet}\n\nKaynak: {ddg_url}"
                else: st.info("DDG bağlantı buldu, kazıma denenecek.")
    except Exception as e: st.warning(f"DuckDuckGo araması sırasında hata: {e}")

    if ddg_url: # Scrape
        scraped_content = scrape_url_content(ddg_url)
        if scraped_content:
            # Önceki kodda sadece uzunsa kazımayı döndürüyordu, şimdi hep kazımayı tercih edelim (varsa)
            return f"**Web Sayfasından ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
        elif ddg_result_text: return ddg_result_text # Kazıma başarısızsa DDG özeti
        else: return f"Detaylı bilgi için şu adresi ziyaret edebilirsiniz: {ddg_url}" # İkisi de yoksa link

    if ddg_result_text: return ddg_result_text # Sadece DDG özeti varsa

    return None # Hiçbir şey bulunamadı

# --- Sohbet Geçmişi Fonksiyonları ---
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

# --- Gemini Yanıt Fonksiyonu (İyileştirilmiş Hata Yönetimi) ---
def get_gemini_response(prompt, chat_history):
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."

    # Geçmişi hazırla (son mesaj user olmalı kuralını esnetebiliriz, model genelde anlar)
    gemini_history = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]}
                      for sender, message in chat_history]
    try:
        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(prompt, stream=False)

        # Yanıtın içeriğini ve olası engellemeleri kontrol et
        if not response.parts:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 st.warning(f"Gemini yanıtı güvenlik nedeniyle engellendi: {block_reason}")
                 return f"{GEMINI_ERROR_PREFIX} Güvenlik filtresi: {block_reason}"
             else: # Boş yanıt ama engellenmemiş?
                 st.warning(f"Gemini'dan boş yanıt alındı (Engellenmedi): {response}")
                 return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
        # Başarılı yanıt
        return "".join(part.text for part in response.parts)

    # Belirli API hatalarını yakalamak daha iyi olabilir ama genel Exception şimdilik yeterli
    except Exception as e:
        st.error(f"Gemini API hatası: {e}")
        # Hatanın türüne göre kullanıcıya daha anlamlı mesajlar verilebilir
        error_message = str(e)
        if "API key not valid" in error_message: return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        if "billing account" in error_message.lower(): return f"{GEMINI_ERROR_PREFIX} Faturalandırma sorunu."
        if "API has not been used" in error_message: return f"{GEMINI_ERROR_PREFIX} API projede etkin değil."
        # Diğer genel hatalar
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı."


# --- Yaratıcı Mod ve Görsel Üretici Fonksiyonları ---
# Bunlar önceki kod ile aynı kalabilir veya Yaratıcı Mod da Gemini kullanacak şekilde güncellenebilir.
# Görsel Üretici hala kural tabanlı.
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
    # ... (Önceki kod ile aynı) ...
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

# --- Session State Başlatma ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Yazılı Sohbet"

# --- Ana Sayfa Başlığı ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🚀 Hanogt AI v3 (Gemini Öncelikli) 🚀</h1>", unsafe_allow_html=True)
# Gemini durumu sidebar'da gösterildiği için buradaki uyarı kaldırılabilir veya değiştirilebilir.
# if not gemini_model:
#      st.warning("...")
st.markdown("<p style='text-align: center;'>Size nasıl yardımcı olabilirim?</p>", unsafe_allow_html=True)


# --- Mod Seçimi (Sidebar) ---
st.sidebar.subheader("Uygulama Modu")
app_mode = st.sidebar.radio("Bir mod seçin:",["Yazılı Sohbet", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Mod", "Görsel Üretici"],key='app_mode_selector') # Sesli sohbet adı güncellendi
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
header_icons = {"Yazılı Sohbet": "✏️", "Sesli Sohbet (Dosya Yükle)": "🎙️", "Yaratıcı Mod": "✨", "Görsel Üretici": "🖼️"}
st.header(f"{header_icons.get(app_mode, '🤔')} {app_mode}")

# --- Uygulama Modları ---

# -- YAZILI SOHBET --
if app_mode == "Yazılı Sohbet":
    # Geçmiş mesajları göster
    for sender, message in st.session_state.chat_history:
        role = "user" if sender.startswith("Sen") else "assistant"
        with st.chat_message(role): st.markdown(message)

    # Kullanıcı girdisi
    if prompt := st.chat_input("Mesajınızı buraya yazın..."):
        st.session_state.chat_history.append(("Sen", prompt))
        with st.chat_message("user"): st.markdown(prompt)

        # --- YANIT OLUŞTURMA AKIŞI (GEMINI ÖNCELİKLİ) ---
        response = None
        ai_sender = "Hanogt AI"
        with st.spinner("🤖 Düşünüyorum..."):
            # 1. ÖNCE GEMINI'I DENE (Eğer model yüklendiyse)
            if gemini_model:
                response = get_gemini_response(prompt, st.session_state.chat_history)
                # Gemini'dan geçerli bir yanıt geldiyse veya bilerek engellendiyse
                if response and not response.startswith(GEMINI_ERROR_PREFIX):
                    ai_sender = "Hanogt AI (Gemini)"
                elif response and response.startswith(GEMINI_ERROR_PREFIX):
                    # Gemini hatasını veya engelleme mesajını göster
                    st.warning(f"Gemini yanıtı alınamadı: {response.replace(GEMINI_ERROR_PREFIX, '')}")
                    response = None # Hata oluştuğu için diğer adımlara geç
                else: # Boş yanıt veya beklenmedik durum
                    response = None # Diğer adımlara geç

            # 2. GEMINI BAŞARISIZ OLDUYSA VEYA YOKSA, BİLGİ TABANINI DENE
            if not response:
                kb_resp = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
                if kb_resp:
                    response = kb_resp
                    ai_sender = "Hanogt AI (Bilgi Tabanı)"

            # 3. HALA YANIT YOKSA, WEB'DE ARA
            if not response:
                st.info("Bilgi tabanında bulunamadı, web'de aranıyor...")
                web_resp = search_web(prompt)
                if web_resp:
                    response = web_resp
                    # Kaynağa göre gönderici adını ayarla
                    if response.startswith("**Wikipedia"): ai_sender = "Hanogt AI (Wikipedia)"
                    elif response.startswith("**Web Özeti"): ai_sender = "Hanogt AI (Web Özeti)"
                    elif response.startswith("**Web Sayfasından"): ai_sender = "Hanogt AI (Sayfa İçeriği)"
                    else: ai_sender = "Hanogt AI (Web Link)"

            # 4. HİÇBİR ŞEY BULUNAMAZSA VARSAYILAN YANIT
            if not response:
                response = random.choice([
                    "Üzgünüm, bu konuda size yardımcı olamıyorum.",
                    "Bu isteği anlayamadım veya yanıtlayamadım.",
                    "Farklı bir şekilde sorabilir misiniz?"
                ])
                ai_sender = "Hanogt AI"

        # --- Yanıtı Göster ve Kaydet ---
        st.session_state.chat_history.append((ai_sender, response))
        with st.chat_message("assistant"): st.markdown(response)
        save_chat_history(st.session_state.chat_history)

# -- SESLİ SOHBET (DOSYA YÜKLEME) --
elif app_mode == "Sesli Sohbet (Dosya Yükle)":
    st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası yükleyin (WAV, MP3, OGG vb.).")

    uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file) # Yüklenen sesi dinleme imkanı
        user_prompt = None
        ai_sender = "Hanogt AI"
        response = None

        with st.spinner("Ses dosyası işleniyor ve yazıya dökülüyor..."):
            recognizer = sr.Recognizer()
            try:
                # Dosyayı sr.AudioFile ile aç
                with sr.AudioFile(uploaded_file) as source:
                    audio_data = recognizer.record(source)
                # Yazıya dök
                user_prompt = recognizer.recognize_google(audio_data, language="tr-TR")
                st.success(f"**Algılanan Metin:** {user_prompt}")

            except sr.UnknownValueError:
                st.error("Ses dosyasındaki konuşma anlaşılamadı.")
                user_prompt = None # Hata durumunda prompt'u None yap
            except sr.RequestError as e:
                st.error(f"Ses tanıma servisine ulaşılamadı; {e}")
                user_prompt = None
            except Exception as e:
                # Ses dosyası formatı desteklenmiyor olabilir veya başka bir sorun
                st.error(f"Ses dosyası işlenirken bir hata oluştu: {e}")
                st.info("Lütfen farklı formatta (örn. WAV) bir dosya deneyin.")
                user_prompt = None

        # Eğer yazıya dökme başarılıysa yanıt oluştur
        if user_prompt:
            # Geçmişe algılanan metni ekle
            st.session_state.chat_history.append(("Sen (Ses Dosyası)", user_prompt))

            # --- YANIT OLUŞTURMA AKIŞI (GEMINI ÖNCELİKLİ - Yazılı sohbet ile aynı) ---
            with st.spinner("🤖 Yanıt oluşturuluyor..."):
                 # 1. ÖNCE GEMINI'I DENE
                if gemini_model:
                    response = get_gemini_response(user_prompt, st.session_state.chat_history)
                    if response and not response.startswith(GEMINI_ERROR_PREFIX): ai_sender = "Hanogt AI (Gemini)"
                    elif response and response.startswith(GEMINI_ERROR_PREFIX): st.warning(f"Gemini yanıtı alınamadı: {response.replace(GEMINI_ERROR_PREFIX, '')}"); response = None
                    else: response = None
                # 2. GEMINI BAŞARISIZSA VEYA YOKSA, BİLGİ TABANINI DENE
                if not response:
                    kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
                    if kb_resp: response = kb_resp; ai_sender = "Hanogt AI (Bilgi Tabanı)"
                # 3. HALA YANIT YOKSA, WEB'DE ARA
                if not response:
                    st.info("Bilgi tabanında bulunamadı, web'de aranıyor...")
                    web_resp = search_web(user_prompt)
                    if web_resp:
                        response = web_resp
                        if response.startswith("**Wikipedia"): ai_sender = "Hanogt AI (Wikipedia)"
                        elif response.startswith("**Web Özeti"): ai_sender = "Hanogt AI (Web Özeti)"
                        elif response.startswith("**Web Sayfasından"): ai_sender = "Hanogt AI (Sayfa İçeriği)"
                        else: ai_sender = "Hanogt AI (Web Link)"
                # 4. HİÇBİR ŞEY BULUNAMAZSA VARSAYILAN YANIT
                if not response:
                    response = random.choice(["Sesinizi yazıya döktüm ancak isteğinizi yanıtlayamıyorum.","Bu konuda bir fikrim yok.","Farklı bir şey sorar mısınız?"]); ai_sender = "Hanogt AI"

            # --- Yanıtı Göster, Kaydet ve Oku ---
            st.markdown(f"**{ai_sender}:**")
            st.markdown(response)
            speak(response)
            st.session_state.chat_history.append((ai_sender, response))
            save_chat_history(st.session_state.chat_history)

# -- YARATICI MOD --
elif app_mode == "Yaratıcı Mod":
    st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Gemini (varsa) veya yerel yaratıcılığım size yanıt versin!")
    creative_prompt = st.text_input("Yaratıcılık tohumu:", key="creative_input", placeholder="Örn: Mars'ta yaşayan filozof bir robot")

    if creative_prompt:
        ai_sender = "Hanogt AI (Yerel Yaratıcı)"
        final_response = None

        # Önce Gemini'ı dene
        if gemini_model:
             st.info("Yaratıcı yanıt için Gemini kullanılıyor...")
             with st.spinner("✨ İlham perileri Gemini ile fısıldaşıyor..."):
                  gemini_creative_prompt = f"Aşağıdaki isteme yaratıcı, ilginç ve özgün bir yanıt ver. Hikaye, şiir veya farklı bir formatta olabilir:\n\n\"{creative_prompt}\""
                  # Yaratıcı modda temiz bir başlangıç için boş geçmiş gönder
                  gemini_resp = get_gemini_response(gemini_creative_prompt, [])
                  if gemini_resp and not gemini_resp.startswith(GEMINI_ERROR_PREFIX):
                      final_response = gemini_resp
                      ai_sender = "Hanogt AI (Gemini Yaratıcı)"
                  else: # Gemini başarısız olursa uyar ve fallback'e geç
                      st.warning(f"Gemini yaratıcı yanıtı alınamadı ({gemini_resp.replace(GEMINI_ERROR_PREFIX, '') if gemini_resp else 'Hata'}). Yerel modül kullanılıyor.")
                      final_response = None # Fallback'e geçmek için None yap

        # Gemini yoksa veya başarısız olduysa yerel fallback
        if not final_response:
             st.info("Yerel yaratıcılık modülü kullanılıyor...")
             with st.spinner("✨ Kendi fikirlerimi demliyorum..."):
                 final_response = creative_response(creative_prompt)
                 new_word = advanced_word_generator(creative_prompt)
                 final_response += f"\n\n_(Ayrıca türettiğim kelime: **{new_word}**)_"
                 ai_sender = "Hanogt AI (Yerel Yaratıcı)"


        st.markdown(f"**{ai_sender}:**")
        st.markdown(final_response)

# -- GÖRSEL ÜRETİCİ --
elif app_mode == "Görsel Üretici":
    st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!")
    st.info("Not: Bu mod henüz Gemini Vision veya ImageFX gibi API'leri kullanmıyor, kural tabanlı çizim yapar.")
    image_prompt = st.text_input("Ne çizmemi istersiniz?", key="image_input", placeholder="Örn: Mor bir gün batımında uçan ejderha silüeti")

    if st.button("🎨 Görseli Oluştur"):
        if image_prompt:
            with st.spinner("Fırçalarım hazırlanıyor..."):
                 image = generate_prompt_influenced_image(image_prompt)
            st.image(image, caption=f"Hanogt AI'ın '{image_prompt}' yorumu (Kural Tabanlı)", use_container_width=True)
            buf = BytesIO(); image.save(buf, format="PNG"); byte_im = buf.getvalue()
            st.download_button(label="Görseli İndir (PNG)", data=byte_im, file_name=f"hanogt_ai_rulebased_{image_prompt[:20].replace(' ','_')}.png", mime="image/png")
        else:
            st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!")

# --- Alt Bilgi ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Hanogt AI v3 (Gemini Öncelikli) - 2025</p>", unsafe_allow_html=True)

