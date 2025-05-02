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
from urllib.parse import urlparse
import google.generativeai as genai

# --- Sayfa Yapılandırması (İLK STREAMLIT KOMUTU OLMALI!) ---
st.set_page_config(
    page_title="Hanogt AI",
    page_icon="🤖",
    layout="wide"
)

# --- Sabitler ---
CHAT_HISTORY_FILE = "chat_history.json"
COLLECTED_DATA_FILE = "collected_data.jsonl" # Veri toplama için dosya adı
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000
GEMINI_ERROR_PREFIX = "GeminiError:"

# --- Bilgi Tabanı (Basitleştirilmiş) ---
# Bu blokta Streamlit komutu yok, sadece Python fonksiyonları var
knowledge_base_load_error = None # Hata mesajını saklamak için
def load_knowledge():
    # Örnek basit anahtar kelime eşleşmeleri
    return {
        "merhaba": ["Merhaba!", "Selam!", "Hoş geldin!"],
        "selam": ["Merhaba!", "Selam sana da!"],
        "nasılsın": ["İyiyim, teşekkürler! Siz nasılsınız?", "Harika hissediyorum!", "İşler yolunda."],
        "hanogt kimdir": ["Ben Hanogt AI, size yardımcı olmaya çalışan bir yapay zeka.", "Hanogt AI, sorularınızı yanıtlamak için burada."],
        "teşekkür ederim": ["Rica ederim!", "Ne demek!", "Yardımcı olabildiğime sevindim."],
        "görüşürüz": ["Görüşmek üzere!", "Hoşça kal!", "İyi günler!"]
    }
def kb_chatbot_response(query, knowledge):
    query_lower = query.lower()
    if query_lower in knowledge: return random.choice(knowledge[query_lower])
    possible_responses = []
    for key, responses in knowledge.items():
        if key in query_lower: possible_responses.extend(responses)
    if possible_responses: return random.choice(possible_responses)
    return None
# Bilgi tabanını yükle (try-except olmadan, basitçe çağır)
KNOWLEDGE_BASE = load_knowledge()

# --- API Anahtarı ve Gemini Yapılandırması ---
# Bu blokta da Streamlit komutu yok
api_key = st.secrets.get("GOOGLE_API_KEY")
gemini_model = None
gemini_init_error = None

if not api_key:
    gemini_init_error = "🛑 Google API Anahtarı Secrets'ta bulunamadı! Lütfen yapılandırın."
else:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
    except Exception as e:
        gemini_init_error = f"🛑 Gemini yapılandırma hatası: {e}"
        gemini_model = None

# --- YARDIMCI FONKSİYONLAR ---
# (Fonksiyon tanımları burada başlar)

# Metin Okuma (TTS)
tts_engine = None
tts_init_error = None
try: tts_engine = pyttsx3.init()
except Exception as e: tts_init_error = f"⚠️ Metin okuma başlatılamadı: {e}."

def speak(text):
    # ... (Fonksiyon içeriği aynı) ...
    if not tts_engine: st.warning("Metin okuma motoru aktif değil.", icon="🔊"); return
    try: tts_engine.say(text); tts_engine.runAndWait()
    except Exception as e: st.error(f"Konuşma sırasında hata: {e}")


# Web Arama ve Kazıma
# ... (scrape_url_content ve search_web fonksiyonları aynı) ...
def scrape_url_content(url):
    st.toast(f"🌐 '{urlparse(url).netloc}' alınıyor...", icon="⏳")
    try:
        parsed_url=urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]) or parsed_url.scheme not in ['http', 'https']: return None
        headers={'User-Agent': 'Mozilla/5.0 HanogtAI/Final'}; response=requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True); response.raise_for_status()
        content_type=response.headers.get('content-type', '').lower();
        if 'html' not in content_type: return None
        soup=BeautifulSoup(response.content, 'html.parser'); potential_content=[]; selectors=['article', 'main', '.content', '.post-content', '.entry-content', 'body']; content_found=False
        for selector in selectors:
             elements=soup.select(selector)
             if elements:
                  text_content=elements[0].find_all('p', limit=10)
                  if text_content:
                      potential_content=[p.get_text(strip=True) for p in text_content if p.get_text(strip=True)];
                      if len(" ".join(potential_content)) > 100: content_found=True; break
        if not content_found:
            all_paragraphs=soup.find_all('p', limit=15); potential_content=[p.get_text(strip=True) for p in all_paragraphs if p.get_text(strip=True)]
        if not potential_content: return None
        full_text=" ".join(potential_content); cleaned_text=re.sub(r'\s+', ' ', full_text).strip(); final_text=cleaned_text[:SCRAPE_MAX_CHARS]
        if len(cleaned_text) > SCRAPE_MAX_CHARS: final_text+="..."
        return final_text
    except Exception as e: st.toast(f"⚠️ Sayfa işlenirken hata: {e}", icon='🌐'); return None
def search_web(query):
    st.toast(f"🔍 '{query}' web'de aranıyor...", icon="⏳")
    summary=None
    try: wikipedia.set_lang("tr"); summary=wikipedia.summary(query, sentences=3, auto_suggest=False); st.toast("ℹ️ Wikipedia'dan.", icon="✅"); return f"**Wikipedia'dan:**\n\n{summary}"
    except Exception: pass
    ddg_result_text=None; ddg_url=None
    try:
        with DDGS() as ddgs:
            results=list(ddgs.text(query, region='tr-tr', max_results=1))
            if results:
                snippet=results[0].get('body'); ddg_url=results[0].get('href')
                if snippet: ddg_result_text=f"**Web Özeti (DuckDuckGo):**\n\n{snippet}\n\nKaynak: {ddg_url}"
    except Exception: pass
    if ddg_url:
        scraped_content=scrape_url_content(ddg_url)
        if scraped_content: return f"**Web Sayfasından ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
        elif ddg_result_text: return ddg_result_text
        else: return f"Detaylı bilgi için: {ddg_url}"
    if ddg_result_text: return ddg_result_text
    st.toast("ℹ️ Web'de yanıt bulunamadı.", icon="❌")
    return None


# Sohbet Geçmişi Yönetimi
# <<< BU FONKSİYONLARIN TANIMLARI Session State'den ÖNCE >>>
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f: content = f.read()
            if content and content.strip(): return json.loads(content)
            else: return []
        except Exception as e: st.error(f"Geçmiş dosyası ({CHAT_HISTORY_FILE}) yüklenemedi: {e}"); return []
    else: return []
def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e: st.error(f"Geçmiş kaydedilemedi: {e}")

# Gemini Yanıt Alma
# ... (get_gemini_response fonksiyonu aynı) ...
def get_gemini_response(prompt, chat_history):
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    gemini_history=[{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]} for sender, message in chat_history]
    try:
        chat=gemini_model.start_chat(history=gemini_history); response=chat.send_message(prompt, stream=False)
        if not response.parts:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: reason=response.prompt_feedback.block_reason; st.warning(f"Gemini yanıtı engellendi: {reason}"); return f"{GEMINI_ERROR_PREFIX} Güvenlik: {reason}"
             else: st.warning(f"Gemini'dan boş yanıt: {response}"); return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
        return "".join(part.text for part in response.parts)
    except Exception as e:
        st.error(f"Gemini API hatası: {e}"); msg=str(e)
        if "API key not valid" in msg: return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı."


# Veri Toplama Fonksiyonu
# ... (log_interaction fonksiyonu aynı) ...
def log_interaction(prompt, response, source):
    try:
        log_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "user_prompt": prompt, "ai_response": response, "response_source": source}
        with open(COLLECTED_DATA_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e: st.toast(f"⚠️ Loglama hatası: {e}", icon="📝")


# Merkezi Yanıt Oluşturma Fonksiyonu
# ... (get_hanogt_response fonksiyonu aynı) ...
def get_hanogt_response(user_prompt, chat_history):
    response=None; ai_sender="Hanogt AI"
    if gemini_model: # 1. Gemini
        response=get_gemini_response(user_prompt, chat_history)
        if response and not response.startswith(GEMINI_ERROR_PREFIX): log_interaction(user_prompt, response, "Hanogt AI (Gemini)"); return response, ai_sender
        elif response and response.startswith(GEMINI_ERROR_PREFIX): st.toast(f"⚠️ Gemini: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖"); response=None
        else: response=None
    if not response: # 2. Bilgi Tabanı (Basitleştirilmiş olan)
        kb_resp=kb_chatbot_response(user_prompt, KNOWLEDGE_BASE);
        if kb_resp: response=kb_resp; log_interaction(user_prompt, response, "Hanogt AI (Bilgi Tabanı)"); return response, ai_sender
    if not response: # 3. Web Arama
        web_resp=search_web(user_prompt);
        if web_resp: response=web_resp; log_interaction(user_prompt, response, "Hanogt AI (Web)"); return response, ai_sender
    if not response: # 4. Varsayılan
        response=random.choice(["Yanıt veremiyorum.","Anlayamadım.","Başka soru?"]); ai_sender="Hanogt AI (Varsayılan)"
        log_interaction(user_prompt, response, ai_sender)
    return response, ai_sender


# Yerel Yaratıcı/Görsel Fonksiyonları
# ... (creative_response, generate_new_idea, advanced_word_generator, generate_prompt_influenced_image fonksiyonları aynı) ...
def creative_response(prompt): styles = ["Farklı düşünürsek: {}", "Hayal edelim: {}", "Belki de şöyledir: {}", "Aklıma geldi: {}"]; base_idea = generate_new_idea(prompt); return random.choice(styles).format(base_idea)
def generate_new_idea(seed): elements = ["zaman kristalleri", "psişik ağaçlar", "rüya mimarisi", "kuantum köpüğü"]; actions = ["dokur", "çözer", "yansıtır", "inşa eder"]; outcomes = ["kaderin iplerini", "varoluşun kodunu", "bilincin sınırlarını", "kayıp uygarlıkları"]; words = seed.lower().split()[:2]; return f"{' '.join(words)} {random.choice(actions)} ve {random.choice(elements)} kullanarak {random.choice(outcomes)}.".capitalize()
def advanced_word_generator(base_word):
     if not base_word or len(base_word) < 2: return "Kelimatör"
     vowels="aeiouüöı"; consonants="bcçdfgğhjklmnprsştvyz"; prefix=["bio", "krono", "psiko", "tera"]; suffix=["genez", "sfer", "nomi", "tek"]; core_len=random.randint(2,4); core=''.join(random.choice(consonants + vowels) for _ in range(core_len)); new_word=core
     if random.random() > 0.4: new_word=random.choice(prefix) + new_word
     if random.random() > 0.4: new_word+=random.choice(suffix)
     return new_word.capitalize()
def generate_prompt_influenced_image(prompt):
    # ... (Kural Tabanlı Görsel Kodu) ...
    width, height=512, 512; prompt_lower=prompt.lower()
    keyword_themes = { "güneş": {"bg": [(255, 230, 150), (255, 180, 50)], "shapes": [{"type": "circle", "color": (255, 255, 0), "pos": (0.2, 0.2), "size": 0.15}]}, "ay": {"bg": [(20, 20, 80), (50, 50, 120)], "shapes": [{"type": "circle", "color": (240, 240, 240), "pos": (0.8, 0.2), "size": 0.1}]}, "gökyüzü": {"bg": [(135, 206, 250), (70, 130, 180)], "shapes": []}, "deniz": {"bg": [(0, 105, 148), (0, 0, 139)], "shapes": []}, "ağaç": {"bg": [(180, 220, 180), (140, 190, 140)], "shapes": [{"type": "rectangle", "color": (139, 69, 19), "pos": (0.5, 0.8), "size": (0.05, 0.3)}, {"type": "triangle", "color": (34, 139, 34), "pos": (0.5, 0.6), "size": 0.2}]}, }
    bg_color1=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)); bg_color2=(random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)); shapes_to_draw=[]; theme_applied=False
    for keyword, theme in keyword_themes.items():
        if keyword in prompt_lower: bg_color1, bg_color2=theme["bg"]; shapes_to_draw.extend(theme["shapes"]); theme_applied=True; break
    img=Image.new('RGB', (width, height), color=bg_color1); draw=ImageDraw.Draw(img)
    for y in range(height): ratio=y / height; r=int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio); g=int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio); b=int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio); draw.line([(0, y), (width, y)], fill=(r, g, b))
    for shape in shapes_to_draw:
        s_type=shape["type"]; s_color=shape["color"]; s_center_x=int(shape["pos"][0] * width); s_center_y=int(shape["pos"][1] * height)
        if s_type == "circle": s_radius=int(shape["size"] * min(width, height) / 2); draw.ellipse((s_center_x - s_radius, s_center_y - s_radius, s_center_x + s_radius, s_center_y + s_radius), fill=s_color)
    if not theme_applied:
        for _ in range(random.randint(3, 8)):
            x1=random.randint(0, width); y1=random.randint(0, height); radius=random.randint(15, 60); shape_fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            if random.random() > 0.5: draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=shape_fill)
            else: w_rect, h_rect=random.randint(20, 100), random.randint(20, 100); draw.rectangle((x1-w_rect//2, y1-h_rect//2, x1+w_rect//2, y1+h_rect//2), fill=shape_fill)
    try: font=ImageFont.load_default(size=24)
    except TypeError: font=ImageFont.load_default()
    try:
        bbox=draw.textbbox((0, 0), prompt, font=font, anchor="lt"); text_width=bbox[2] - bbox[0]; text_height=bbox[3] - bbox[1]; text_x=(width - text_width) / 2; text_y=height * 0.9 - text_height; text_y=max(text_y, height * 0.7)
        draw.text((text_x + 1, text_y + 1), prompt, font=font, fill=(0,0,0,180)); draw.text((text_x, text_y), prompt, font=font, fill=(255,255,255))
    except Exception as e: st.error(f"Metin yazdırılırken hata: {e}")
    return img

# --- Session State Başlatma ---
# Bu bölüm TÜM fonksiyon tanımlarından SONRA gelir.
if 'chat_history' not in st.session_state: st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Yazılı Sohbet"
if 'user_name' not in st.session_state: st.session_state.user_name = None
if 'user_avatar_bytes' not in st.session_state: st.session_state.user_avatar_bytes = None
if 'show_main_app' not in st.session_state: st.session_state.show_main_app = bool(st.session_state.user_name)
if 'greeting_message_shown' not in st.session_state: st.session_state.greeting_message_shown = False

# --- UYGULAMA ARAYÜZÜ BAŞLANGICI ---

# --- Ana Başlık ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Hanogt AI</h1>", unsafe_allow_html=True)

# Uygulama başlangıcında oluşan hataları göster (API, TTS vb.)
# Bunları ana UI alanına taşıdık, set_page_config'den sonra olmaları için
if gemini_init_error: st.error(gemini_init_error)
if tts_init_error: st.toast(tts_init_error, icon="🔊") # Toast olarak kalsın
# Bilgi tabanı yükleme hatasını burada gösterelim
if knowledge_base_load_error: st.warning(knowledge_base_load_error, icon="ℹ️")


# --- Kullanıcı Adı Sorgulama ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Tanışalım...")
    name_input = st.text_input("Size nasıl hitap etmeliyim?", key="name_input_key", placeholder="İsminiz...")
    if st.button("Kaydet", key="save_name_button"):
        if name_input.strip(): st.session_state.user_name = name_input.strip(); st.session_state.show_main_app = True; st.session_state.greeting_message_shown = False; st.rerun()
        else: st.error("Lütfen bir isim girin.")

# --- ANA UYGULAMA BÖLÜMÜ ---
elif st.session_state.show_main_app:

    # Karşılama mesajı
    if not st.session_state.greeting_message_shown and st.session_state.user_name:
         st.success(f"Tanıştığıma memnun oldum, {st.session_state.user_name}! Size nasıl yardımcı olabilirim?"); st.session_state.greeting_message_shown = True

    # --- Ayarlar Bölümü ---
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        # ... (Ayarlar içeriği önceki kod ile aynı) ...
        def update_name(): st.session_state.user_name = st.session_state.change_name_input_key; st.toast("Adınız güncellendi!")
        st.text_input("Adınızı Değiştirin:", value=st.session_state.user_name, key="change_name_input_key", on_change=update_name)
        st.caption(f"Mevcut adınız: {st.session_state.user_name}"); st.divider()
        st.write("**Avatar (Profil Resmi):**"); uploaded_avatar = st.file_uploader("Yeni Avatar Yükle (PNG, JPG - Maks 1MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader")
        if uploaded_avatar is not None:
            if uploaded_avatar.size > 1 * 1024 * 1024: st.error("Dosya > 1MB!")
            else: st.session_state.user_avatar_bytes = uploaded_avatar.getvalue(); st.toast("Avatar güncellendi!", icon="🖼️"); st.rerun()
        if st.session_state.user_avatar_bytes:
            st.image(st.session_state.user_avatar_bytes, width=64, caption="Mevcut Avatarınız");
            if st.button("Avatarı Kaldır", key="remove_avatar"): st.session_state.user_avatar_bytes = None; st.toast("Avatar kaldırıldı.", icon="🗑️"); st.rerun()
        else: st.caption("Henüz bir avatar yüklemediniz.")
        st.caption("Not: Avatar sadece bu tarayıcı oturumu için geçerlidir."); st.divider()
        if st.button("🧹 Sohbet Geçmişini Temizle", key="clear_history_main"):
            st.session_state.chat_history = []; save_chat_history([]); st.toast("Sohbet geçmişi temizlendi!", icon="🧹"); time.sleep(1); st.rerun()


    st.markdown("---")

    # --- Mod Seçim Butonları ---
    st.write("**Uygulama Modu:**")
    modes = ["Yazılı Sohbet", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Mod", "Görsel Üretici"]
    icons = ["✏️", "🎙️", "✨", "🖼️"]
    cols = st.columns(len(modes))
    current_mode = st.session_state.app_mode
    new_mode = current_mode
    for i, col in enumerate(cols):
        with col:
            is_active = (modes[i] == current_mode); button_type = "primary" if is_active else "secondary"
            if st.button(f"{icons[i]} {modes[i]}", key=f"mode_btn_{i}", use_container_width=True, type=button_type): new_mode = modes[i]
    if new_mode != current_mode: st.session_state.app_mode = new_mode; st.rerun()
    app_mode = st.session_state.app_mode
    st.markdown("---")

    # --- MODLARA GÖRE ARAYÜZLER ---

    # -- YAZILI SOHBET --
    if app_mode == "Yazılı Sohbet":
        chat_container = st.container()
        with chat_container:
             # ... (Mesaj gösterme ve Oku/Kopyala butonları önceki kod ile aynı) ...
            for i, (sender, message) in enumerate(st.session_state.chat_history):
                is_user = sender.startswith("Sen"); role = "user" if is_user else "assistant"; display_avatar = None
                if is_user and st.session_state.user_avatar_bytes:
                    try: display_avatar = Image.open(BytesIO(st.session_state.user_avatar_bytes))
                    except Exception: display_avatar = "🧑"
                elif not is_user: display_avatar = "🤖"
                with st.chat_message(role, avatar=display_avatar):
                    st.markdown(message) # Sadece mesaj
                    if not is_user: # AI mesajıysa butonları/kodu ekle
                        b_cols = st.columns([0.15, 0.85]) # Buton için dar sütun
                        with b_cols[0]:
                             if tts_engine:
                                if st.button(f"🔊", key=f"speak_msg_{i}", help="Mesajı sesli oku"): speak(message)
                        # Kopyalama alanı direkt mesajın altına
                        st.code(message, language=None)

        if prompt := st.chat_input(f"{st.session_state.user_name} olarak mesaj yazın..."):
            # ... (Yanıt alma ve gösterme önceki kod ile aynı) ...
            st.session_state.chat_history.append(("Sen", prompt))
            with st.spinner("🤖 Düşünüyorum..."): response, ai_sender = get_hanogt_response(prompt, st.session_state.chat_history)
            st.session_state.chat_history.append((ai_sender, response)); save_chat_history(st.session_state.chat_history); st.rerun()

    # -- SESLİ SOHBET (DOSYA YÜKLEME) --
    elif app_mode == "Sesli Sohbet (Dosya Yükle)":
         # ... (Kod önceki ile aynı) ...
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
                with st.spinner("🤖 Yanıt oluşturuluyor..."): response, ai_sender = get_hanogt_response(user_prompt, st.session_state.chat_history)
                st.markdown(f"**{ai_sender}:**"); st.markdown(response); speak(response)
                st.code(response, language=None)
                st.session_state.chat_history.append((ai_sender, response)); save_chat_history(st.session_state.chat_history)


    # -- YARATICI MOD --
    elif app_mode == "Yaratıcı Mod":
         # ... (Kod önceki ile aynı) ...
        st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Gemini (varsa) veya yerel yaratıcılığım size yanıt versin!")
        creative_prompt = st.text_input("Yaratıcılık tohumu:", key="creative_input", placeholder="Örn: Zaman yolculuğu yapan bir tost makinesi")
        if creative_prompt:
            ai_sender = "Hanogt AI"; final_response = None
            if gemini_model:
                 with st.spinner("✨ İlham perileri fısıldaşıyor..."):
                      gemini_creative_prompt = f"Aşağıdaki isteme yaratıcı, ilginç ve özgün bir yanıt ver:\n\n\"{creative_prompt}\""
                      gemini_resp = get_gemini_response(gemini_creative_prompt, [])
                      if gemini_resp and not gemini_resp.startswith(GEMINI_ERROR_PREFIX): final_response = gemini_resp;
                      else: st.warning(f"Gemini yaratıcı yanıtı alınamadı. Yerel modül kullanılıyor."); final_response = None
            if not final_response:
                 with st.spinner("✨ Kendi fikirlerimi demliyorum..."):
                     final_response = creative_response(creative_prompt); new_word = advanced_word_generator(creative_prompt); final_response += f"\n\n_(Ayrıca türettiğim kelime: **{new_word}**)_";
            st.markdown(f"**{ai_sender}:**"); st.markdown(final_response)
            st.code(final_response, language=None)


    # -- GÖRSEL ÜRETİCİ (Kural Tabanlı) --
    elif app_mode == "Görsel Üretici":
         # ... (Kod önceki ile aynı) ...
         st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!")
         st.info("Not: Bu mod, girilen anahtar kelimelere göre basit, kural tabanlı çizimler yapar.")
         image_prompt = st.text_input("Ne çizmemi istersiniz?", key="image_input", placeholder="Örn: Mavi bir nehir kenarında yeşil ağaçlar")
         if st.button("🎨 Görseli Oluştur", key="generate_rule_image_btn"):
            if image_prompt:
                with st.spinner("Fırçalarım hazırlanıyor..."): image = generate_prompt_influenced_image(image_prompt)
                st.image(image, caption=f"Hanogt AI'ın '{image_prompt}' yorumu (Kural Tabanlı)", use_container_width=True)
                buf = BytesIO(); image.save(buf, format="PNG"); byte_im = buf.getvalue()
                st.download_button(label="Görseli İndir (PNG)", data=byte_im, file_name=f"hanogt_ai_rulebased_{image_prompt[:20].replace(' ','_')}.png", mime="image/png")
            else: st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!")


# --- Alt Bilgi ---
if st.session_state.show_main_app:
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: small;'>Hanogt AI v3.6 - {st.session_state.get('user_name', 'Misafir')} için çalışıyor - 2025</p>", unsafe_allow_html=True) # Sürüm güncellendi

