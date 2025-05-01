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
import google.generativeai as genai

# --- Sabitler ---
CHAT_HISTORY_FILE = "chat_history.json"
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000
GEMINI_ERROR_PREFIX = "GeminiError:" # Gemini hatalarını ayırt etmek için

# --- Bilgi Tabanı (Mock/Placeholder) ---
try:
    # knowledge_base.py varsa import et, yoksa boş çalışsın
    from knowledge_base import load_knowledge, chatbot_response as kb_chatbot_response
except ImportError:
    st.toast("`knowledge_base.py` bulunamadı.", icon="ℹ️")
    def load_knowledge(): return {}
    def kb_chatbot_response(query, knowledge): return None
KNOWLEDGE_BASE = load_knowledge()

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hanogt AI", page_icon="🤖", layout="wide")

# --- API Anahtarı ve Gemini Yapılandırması ---
# Streamlit Secrets kullanarak API anahtarını güvenli bir şekilde alır.
api_key = None
gemini_model = None
gemini_init_error = None # Başlatma sırasında hata olursa saklamak için

# Secrets'tan API anahtarını oku
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    gemini_init_error = "🛑 Google API Anahtarı Secrets'ta bulunamadı! Lütfen yapılandırın."

# API anahtarı varsa Gemini modelini yapılandır
if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]
        # Güncel modeli kullan (flash genellikle daha hızlıdır)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
    except Exception as e:
        gemini_init_error = f"🛑 Gemini yapılandırma hatası: {e}"
        gemini_model = None

# --- Yardımcı Fonksiyonlar ---

# Metin Okuma (TTS) Motoru Başlatma
tts_engine = None
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    st.toast(f"⚠️ Metin okuma motoru başlatılamadı: {e}. Sesli okuma çalışmayabilir.", icon="🔊")

def speak(text):
    """Verilen metni sesli olarak okur."""
    if tts_engine:
        try:
            # Konuşma hızını ayarlamak isterseniz: tts_engine.setProperty('rate', 150)
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Konuşma sırasında hata: {e}")

# Web Arama ve Kazıma Fonksiyonları
def scrape_url_content(url):
    """Verilen URL'den metin içeriğini kazımayı dener."""
    st.toast(f"🌐 '{urlparse(url).netloc}' adresinden içerik alınıyor...", icon="⏳")
    try:
        parsed_url = urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']: return None
        headers = {'User-Agent': 'Mozilla/5.0 HanogtAI/Final'}; response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True); response.raise_for_status()
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
        return final_text
    except Exception as e: st.toast(f"⚠️ Sayfa ({urlparse(url).netloc}) işlenirken hata: {e}", icon='🌐'); return None

def search_web(query):
    """Web'de arama yapar (Wikipedia > DuckDuckGo > Scrape)."""
    st.toast(f"🔍 '{query}' için web'de aranıyor...", icon="⏳")
    summary = None
    # 1. Wikipedia
    try: wikipedia.set_lang("tr"); summary = wikipedia.summary(query, auto_suggest=False); st.toast("ℹ️ Wikipedia'dan bilgi bulundu.", icon="✅"); return f"**Wikipedia'dan:**\n\n{summary}"
    except Exception: pass
    # 2. DuckDuckGo
    ddg_result_text = None; ddg_url = None
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='tr-tr', max_results=1))
            if results:
                snippet = results[0].get('body'); ddg_url = results[0].get('href')
                if snippet: ddg_result_text = f"**Web Özeti (DuckDuckGo):**\n\n{snippet}\n\nKaynak: {ddg_url}"
    except Exception: pass
    # 3. Scrape
    if ddg_url:
        scraped_content = scrape_url_content(ddg_url)
        if scraped_content: return f"**Web Sayfasından ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
        elif ddg_result_text: return ddg_result_text
        else: return f"Detaylı bilgi için: {ddg_url}"
    if ddg_result_text: return ddg_result_text # Sadece DDG varsa
    st.toast("ℹ️ Web'de doğrudan yanıt bulunamadı.", icon="❌")
    return None

# Sohbet Geçmişi Yönetimi
def load_chat_history():
    """Sohbet geçmişini JSON dosyasından yükler."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f: content = f.read()
            if content and content.strip(): return json.loads(content)
            else: return []
        except Exception as e: st.error(f"Geçmiş dosyası ({CHAT_HISTORY_FILE}) yüklenemedi: {e}"); return []
    else: return []
def save_chat_history(history):
    """Sohbet geçmişini JSON dosyasına kaydeder."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e: st.error(f"Geçmiş kaydedilemedi: {e}")

# Gemini Yanıt Alma Fonksiyonu
def get_gemini_response(prompt, chat_history):
    """Gemini modelinden yanıt alır."""
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    # Geçmişi Gemini formatına uygun hazırla
    gemini_history = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]}
                      for sender, message in chat_history]
    try:
        # Modeli başlat ve mesajı gönder
        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(prompt, stream=False)
        # Yanıtı ve olası engellemeleri kontrol et
        if not response.parts:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 reason = response.prompt_feedback.block_reason
                 st.warning(f"Gemini yanıtı engellendi: {reason}")
                 return f"{GEMINI_ERROR_PREFIX} Güvenlik: {reason}"
             else:
                 st.warning(f"Gemini'dan boş yanıt: {response}")
                 return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
        # Başarılı yanıtı birleştir ve döndür
        return "".join(part.text for part in response.parts)
    except Exception as e:
        # Hata durumunu yönet
        st.error(f"Gemini API hatası: {e}")
        msg = str(e)
        if "API key not valid" in msg: return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        # Diğer bilinen hata mesajları buraya eklenebilir
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı."

# --- YENİ: Merkezi Yanıt Oluşturma Fonksiyonu ---
def get_hanogt_response(user_prompt, chat_history):
    """Kullanıcı istemine göre yanıt oluşturur (Gemini > KB > Web > Default)."""
    response = None
    ai_sender = "Hanogt AI" # Varsayılan gönderici

    # 1. ÖNCE GEMINI'I DENE (Eğer model yüklendiyse)
    if gemini_model:
        response = get_gemini_response(user_prompt, chat_history)
        if response and not response.startswith(GEMINI_ERROR_PREFIX):
            ai_sender = "Hanogt AI (Gemini)"
            return response, ai_sender # Başarılı Gemini yanıtı varsa hemen döndür
        elif response and response.startswith(GEMINI_ERROR_PREFIX):
            # Gemini hatasını göster ama diğer adımlara devam et
            st.toast(f"⚠️ Gemini: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖")
            response = None # Hata veya engelleme durumunda response'u sıfırla
        else: # Boş yanıt vb.
            response = None

    # 2. GEMINI BAŞARISIZ OLDUYSA VEYA YOKSA, BİLGİ TABANINI DENE
    if not response:
        kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
        if kb_resp:
            response = kb_resp
            ai_sender = "Hanogt AI (Bilgi Tabanı)"
            return response, ai_sender # Başarılı KB yanıtı varsa döndür

    # 3. HALA YANIT YOKSA, WEB'DE ARA
    if not response:
        web_resp = search_web(user_prompt)
        if web_resp:
            response = web_resp
            # Kaynağa göre gönderici adını belirle (search_web'den gelen formata göre)
            if "**Wikipedia" in response: ai_sender = "Hanogt AI (Wikipedia)"
            elif "**Web Özeti" in response: ai_sender = "Hanogt AI (Web Özeti)"
            elif "**Web Sayfasından" in response: ai_sender = "Hanogt AI (Sayfa İçeriği)"
            else: ai_sender = "Hanogt AI (Web Link)"
            return response, ai_sender # Başarılı Web yanıtı varsa döndür

    # 4. HİÇBİR ŞEY BULUNAMAZSA VARSAYILAN YANIT
    if not response:
        response = random.choice([
            "Üzgünüm, bu konuda size yardımcı olamıyorum.",
            "Bu isteği anlayamadım veya yanıtlayamadım.",
            "Farklı bir şekilde sorabilir misiniz?"
        ])
        ai_sender = "Hanogt AI"

    return response, ai_sender

# --- Yerel Yaratıcı/Görsel Fonksiyonları (Fallback + Eski Görsel) ---
def creative_response(prompt):
    styles = ["Farklı düşünürsek: {}", "Hayal edelim: {}", "Belki de şöyledir: {}", "Aklıma geldi: {}"]; base_idea = generate_new_idea(prompt); return random.choice(styles).format(base_idea)
def generate_new_idea(seed):
    elements = ["zaman kristalleri", "psişik ağaçlar", "rüya mimarisi", "kuantum köpüğü"]; actions = ["dokur", "çözer", "yansıtır", "inşa eder"]; outcomes = ["kaderin iplerini", "varoluşun kodunu", "bilincin sınırlarını", "kayıp uygarlıkları"]; words = seed.lower().split()[:2]; return f"{' '.join(words)} {random.choice(actions)} ve {random.choice(elements)} kullanarak {random.choice(outcomes)}.".capitalize()
def advanced_word_generator(base_word):
    if not base_word or len(base_word) < 2: return "Kelimatör"
    vowels = "aeiouüöı"; consonants = "bcçdfgğhjklmnprsştvyz"; prefix = ["bio", "krono", "psiko", "tera"]; suffix = ["genez", "sfer", "nomi", "tek"]; core_len = random.randint(2,4); core = ''.join(random.choice(consonants + vowels) for _ in range(core_len)); new_word = core
    if random.random() > 0.4: new_word = random.choice(prefix) + new_word
    if random.random() > 0.4: new_word += random.choice(suffix)
    return new_word.capitalize()
# Kural Tabanlı Görsel Üretici (Önceki versiyondaki gibi)
def generate_prompt_influenced_image(prompt):
    width, height = 512, 512; prompt_lower = prompt.lower()
    keyword_themes = { # Temaları genişletebilirsiniz
        "güneş": {"bg": [(255, 230, 150), (255, 180, 50)], "shapes": [{"type": "circle", "color": (255, 255, 0), "pos": (0.2, 0.2), "size": 0.15}]},
        "ay": {"bg": [(20, 20, 80), (50, 50, 120)], "shapes": [{"type": "circle", "color": (240, 240, 240), "pos": (0.8, 0.2), "size": 0.1}]},
        # ... (Diğer temalar önceki koddan buraya eklenebilir) ...
        "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []},
        "deniz": {"bg": [(0, 105, 148), (0, 0, 139)], "shapes": []},
         "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [{"type": "rectangle", "color": (139, 69, 19), "pos": (0.5, 0.8), "size": (0.05, 0.3)}, {"type": "triangle", "color": (34, 139, 34), "pos": (0.5, 0.6), "size": 0.2}]},
    }
    bg_color1 = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)); bg_color2 = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)); shapes_to_draw = []; theme_applied = False
    for keyword, theme in keyword_themes.items():
        if keyword in prompt_lower: bg_color1, bg_color2 = theme["bg"]; shapes_to_draw.extend(theme["shapes"]); theme_applied = True; break
    img = Image.new('RGB', (width, height), color=bg_color1); draw = ImageDraw.Draw(img)
    # ... (Gradient, Şekil çizme ve Metin yazdırma kodları önceki versiyondan buraya eklenecek) ...
    # Gradient Arka Plan
    for y in range(height): ratio = y / height; r = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio); g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio); b = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio); draw.line([(0, y), (width, y)], fill=(r, g, b))
    # Şekilleri Çiz
    for shape in shapes_to_draw:
        s_type = shape["type"]; s_color = shape["color"]; s_center_x = int(shape["pos"][0] * width); s_center_y = int(shape["pos"][1] * height)
        if s_type == "circle": s_radius = int(shape["size"] * min(width, height) / 2); draw.ellipse((s_center_x - s_radius, s_center_y - s_radius, s_center_x + s_radius, s_center_y + s_radius), fill=s_color)
        elif s_type == "ellipse": s_radius_x = int(shape["size"] * width / 2*random.uniform(0.8, 1.2)); s_radius_y = int(shape["size"] * height / 2*random.uniform(0.5, 1.0)); draw.ellipse((s_center_x - s_radius_x, s_center_y - s_radius_y, s_center_x + s_radius_x, s_center_y + s_radius_y), fill=s_color)
        elif s_type == "rectangle": s_width = int(shape["size"][0] * width); s_height = int(shape["size"][1] * height); draw.rectangle((s_center_x - s_width // 2, s_center_y - s_height // 2, s_center_x + s_width // 2, s_center_y + s_height // 2), fill=s_color)
        elif s_type == "triangle": s_base = int(shape["size"] * width * 0.8); s_height_tri = int(shape["size"] * height); points = [(s_center_x, s_center_y - s_height_tri // 2), (s_center_x - s_base // 2, s_center_y + s_height_tri // 2), (s_center_x + s_base // 2, s_center_y + s_height_tri // 2)]; draw.polygon(points, fill=s_color)
    if not theme_applied: # Rastgele şekiller
        for _ in range(random.randint(3, 8)):
            x1=random.randint(0, width); y1=random.randint(0, height); radius=random.randint(15, 60); shape_fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            if random.random() > 0.5: draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=shape_fill, outline=(255,255,255, 100))
            else: w_rect, h_rect = random.randint(20, 100), random.randint(20, 100); draw.rectangle((x1-w_rect//2, y1-h_rect//2, x1+w_rect//2, y1+h_rect//2), fill=shape_fill, outline=(255,255,255,100))
    # Metni Yazdır
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
# Bu bölümün TÜM fonksiyon tanımlarından SONRA geldiğinden emin olun
if 'chat_history' not in st.session_state: st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Yazılı Sohbet"
if 'user_name' not in st.session_state: st.session_state.user_name = None
if 'user_avatar_bytes' not in st.session_state: st.session_state.user_avatar_bytes = None
# user_name varsa show_main_app'i True yap, yoksa False (ve sonra kontrol et)
st.session_state.show_main_app = bool(st.session_state.user_name)
if 'greeting_message_shown' not in st.session_state: st.session_state.greeting_message_shown = False

# --- Ana Başlık (Roketsiz) ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Hanogt AI</h1>", unsafe_allow_html=True)

# Gemini yüklenemezse uyarı göster
if gemini_init_error: st.error(gemini_init_error)

# --- Kullanıcı Adı Sorgulama ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Tanışalım...")
    name_input = st.text_input("Size nasıl hitap etmeliyim?", key="name_input_key", placeholder="İsminiz...")
    if st.button("Kaydet", key="save_name_button"):
        if name_input.strip(): st.session_state.user_name = name_input.strip(); st.session_state.show_main_app = True; st.session_state.greeting_message_shown = False; st.rerun()
        else: st.error("Lütfen bir isim girin.")

# --- ANA UYGULAMA BÖLÜMÜ ---
elif st.session_state.show_main_app:
    # Tanışma mesajı
    if not st.session_state.greeting_message_shown and st.session_state.user_name:
         st.success(f"Tanıştığıma memnun oldum, {st.session_state.user_name}! Size nasıl yardımcı olabilirim?"); st.session_state.greeting_message_shown = True

    # --- Ayarlar Bölümü ---
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        # Ad Değiştirme
        st.text_input("Adınızı Değiştirin:", value=st.session_state.user_name, key="change_name_input_key", on_change=lambda: setattr(st.session_state, 'user_name', st.session_state.change_name_input_key))
        st.caption(f"Mevcut adınız: {st.session_state.user_name}")
        st.divider()
        # Avatar Yönetimi
        st.write("**Avatar (Profil Resmi):**"); uploaded_avatar = st.file_uploader("Yeni Avatar Yükle (PNG, JPG - Maks 1MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader")
        if uploaded_avatar is not None:
            if uploaded_avatar.size > 1 * 1024 * 1024: st.error("Dosya > 1MB!")
            else: st.session_state.user_avatar_bytes = uploaded_avatar.getvalue(); st.success("Avatar güncellendi!"); st.rerun()
        if st.session_state.user_avatar_bytes:
            st.image(st.session_state.user_avatar_bytes, width=64, caption="Mevcut Avatarınız");
            if st.button("Avatarı Kaldır", key="remove_avatar"): st.session_state.user_avatar_bytes = None; st.rerun()
        else: st.caption("Henüz bir avatar yüklemediniz.")
        st.caption("Not: Avatar sadece bu oturum için geçerlidir.")
        st.divider()
        # Geçmiş Temizleme
        if st.button("🧹 Sohbet Geçmişini Temizle", key="clear_history_main"):
            st.session_state.chat_history = []; save_chat_history([]); st.success("Sohbet geçmişi temizlendi!"); time.sleep(1); st.rerun()

    st.markdown("---")

    # --- Mod Seçim Butonları ---
    st.write("**Uygulama Modu:**")
    modes = ["Yazılı Sohbet", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Mod", "Görsel Üretici"] # Görsel modu eski adıyla
    icons = ["✏️", "🎙️", "✨", "🖼️"]
    cols = st.columns(len(modes))
    current_mode = st.session_state.app_mode
    for i, col in enumerate(cols):
        with col:
            button_type = "primary" if modes[i] == current_mode else "secondary"
            if st.button(f"{icons[i]} {modes[i]}", key=f"mode_btn_{i}", use_container_width=True, type=button_type):
                st.session_state.app_mode = modes[i]; st.rerun()

    app_mode = st.session_state.app_mode
    st.markdown("---")

    # --- MODLARA GÖRE ARAYÜZLER ---

    # -- YAZILI SOHBET --
    if app_mode == "Yazılı Sohbet":
        # Mesajları göster (OKU BUTONU EKLENDİ)
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            is_user = sender.startswith("Sen"); role = "user" if is_user else "assistant"; display_avatar = None
            if is_user and st.session_state.user_avatar_bytes:
                try: display_avatar = Image.open(BytesIO(st.session_state.user_avatar_bytes))
                except Exception: display_avatar = "🧑"
            elif not is_user: display_avatar = "🤖" # AI için avatar

            with st.chat_message(role, avatar=display_avatar):
                 display_name = ""
                 if not is_user and "(" in sender and ")" in sender: source = sender[sender.find("(")+1:sender.find(")")]; display_name = f"({source}) "
                 st.markdown(f"{display_name}{message}") # Mesajı yazdır

                 # <<< YENİ EKLENEN SESLİ OKUMA BUTONU >>>
                 if not is_user and tts_engine: # Sadece AI mesajı ve TTS aktifse
                     if st.button(f"🔊 Oku", key=f"speak_msg_{i}", help="Mesajı sesli oku"): # Help eklendi
                         speak(message)
                 # <<< SESLİ OKUMA BUTONU SONU >>>

        # Yeni mesaj girişi
        if prompt := st.chat_input(f"{st.session_state.user_name} olarak mesaj yazın..."):
            st.session_state.chat_history.append(("Sen", prompt))
            # YANIT OLUŞTURMA (Refaktör Edilmiş Fonksiyonu Kullan)
            with st.spinner("🤖 Düşünüyorum..."):
                response, ai_sender = get_hanogt_response(prompt, st.session_state.chat_history)
            # Yanıtı kaydet ve arayüzü yenile
            st.session_state.chat_history.append((ai_sender, response)); save_chat_history(st.session_state.chat_history); st.rerun()

    # -- SESLİ SOHBET (DOSYA YÜKLEME) --
    elif app_mode == "Sesli Sohbet (Dosya Yükle)":
        st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası yükleyin.")
        uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed")
        if uploaded_file is not None:
            st.audio(uploaded_file); user_prompt = None; ai_sender = "Hanogt AI"; response = None
            with st.spinner("Ses dosyası işleniyor..."):
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(uploaded_file) as source: audio_data = recognizer.record(source)
                    user_prompt = recognizer.recognize_google(audio_data, language="tr-TR"); st.success(f"**Algılanan Metin:** {user_prompt}")
                except Exception as e: st.error(f"Ses dosyası işlenemedi: {e}"); user_prompt = None

            if user_prompt:
                st.session_state.chat_history.append(("Sen (Ses Dosyası)", user_prompt))
                # YANIT OLUŞTURMA (Refaktör Edilmiş Fonksiyonu Kullan)
                with st.spinner("🤖 Yanıt oluşturuluyor..."):
                     response, ai_sender = get_hanogt_response(user_prompt, st.session_state.chat_history)
                # Yanıtı göster, oku ve kaydet
                st.markdown(f"**{ai_sender}:**"); st.markdown(response); speak(response)
                st.session_state.chat_history.append((ai_sender, response)); save_chat_history(st.session_state.chat_history)

    # -- YARATICI MOD --
    elif app_mode == "Yaratıcı Mod":
        st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Gemini (varsa) veya yerel yaratıcılığım size yanıt versin!")
        creative_prompt = st.text_input("Yaratıcılık tohumu:", key="creative_input", placeholder="Örn: Ay'da kamp yapan astronotlar")
        if creative_prompt:
            ai_sender = "Hanogt AI (Yerel Yaratıcı)"; final_response = None
            if gemini_model:
                 with st.spinner("✨ İlham perileri fısıldaşıyor..."):
                      gemini_creative_prompt = f"Aşağıdaki isteme yaratıcı, ilginç ve özgün bir yanıt ver:\n\n\"{creative_prompt}\""
                      # Yaratıcı modda geçmişi göndermeyelim, her seferinde temiz başlasın
                      gemini_resp = get_gemini_response(gemini_creative_prompt, [])
                      if gemini_resp and not gemini_resp.startswith(GEMINI_ERROR_PREFIX): final_response = gemini_resp; ai_sender = "Hanogt AI (Gemini Yaratıcı)"
                      else: st.warning(f"Gemini yaratıcı yanıtı alınamadı. Yerel modül kullanılıyor."); final_response = None
            if not final_response:
                 with st.spinner("✨ Kendi fikirlerimi demliyorum..."):
                     final_response = creative_response(creative_prompt); new_word = advanced_word_generator(creative_prompt); final_response += f"\n\n_(Ayrıca türettiğim kelime: **{new_word}**)_"; ai_sender = "Hanogt AI (Yerel Yaratıcı)"
            st.markdown(f"**{ai_sender}:**"); st.markdown(final_response)

    # -- GÖRSEL ÜRETİCİ (Kural Tabanlı) --
    elif app_mode == "Görsel Üretici":
        st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!")
        st.info("Not: Bu mod, girilen anahtar kelimelere göre basit, kural tabanlı çizimler yapar.")
        image_prompt = st.text_input("Ne çizmemi istersiniz?", key="image_input", placeholder="Örn: Yeşil çimenler üzerinde kırmızı bir top")
        if st.button("🎨 Görseli Oluştur", key="generate_rule_image_btn"):
            if image_prompt:
                with st.spinner("Fırçalarım hazırlanıyor..."):
                     image = generate_prompt_influenced_image(image_prompt) # Kural tabanlı fonksiyon
                st.image(image, caption=f"Hanogt AI'ın '{image_prompt}' yorumu (Kural Tabanlı)", use_container_width=True)
                buf = BytesIO(); image.save(buf, format="PNG"); byte_im = buf.getvalue()
                st.download_button(label="Görseli İndir (PNG)", data=byte_im, file_name=f"hanogt_ai_rulebased_{image_prompt[:20].replace(' ','_')}.png", mime="image/png")
            else: st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!")

# --- Alt Bilgi ---
if st.session_state.show_main_app:
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: small;'>Hanogt AI v3.2 - {st.session_state.get('user_name', 'Misafir')} için çalışıyor - 2025</p>", unsafe_allow_html=True) # Sürüm güncellendi

