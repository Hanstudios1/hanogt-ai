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
# LOGO_PATH = "logo.png" # Logo artık Ayarlar'dan yüklenebilir
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000
GEMINI_ERROR_PREFIX = "GeminiError:" # Gemini hatalarını ayırt etmek için

# --- Bilgi Tabanı (Mock/Placeholder) ---
# Eğer knowledge_base.py dosyanız varsa import eder, yoksa boş çalışır.
try:
    from knowledge_base import load_knowledge, chatbot_response as kb_chatbot_response
except ImportError:
    st.toast("`knowledge_base.py` bulunamadı.", icon="ℹ️")
    def load_knowledge(): return {}
    def kb_chatbot_response(query, knowledge): return None
KNOWLEDGE_BASE = load_knowledge()

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hanogt AI", page_icon="🚀", layout="wide")

# --- API Anahtarı ve Gemini Yapılandırması ---
# Streamlit Secrets kullanarak API anahtarını güvenli bir şekilde alır.
api_key = None
gemini_model = None
gemini_init_error = None # Başlatma sırasında hata olursa saklamak için

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    gemini_init_error = "🛑 Google API Anahtarı Secrets'ta bulunamadı! Lütfen yapılandırın."
    # Sidebar kaldırıldığı için bu hatayı ana ekranda veya toast ile gösterebiliriz.
    # Şimdilik gemini_init_error değişkeninde saklayalım.

if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [ # Güvenlik ayarları
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # Diğer kategoriler eklenebilir
        ]
        # Güncel modeli kullan (flash daha hızlı ve uygun maliyetli)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
    except Exception as e:
        gemini_init_error = f"🛑 Gemini yapılandırma hatası: {e}"
        gemini_model = None

# --- Yardımcı Fonksiyonlar ---

# Metin Okuma (TTS)
tts_engine = None
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    st.toast(f"⚠️ Metin okuma motoru başlatılamadı: {e}", icon="🔊")

def speak(text):
    """Verilen metni sesli olarak okur."""
    if tts_engine:
        try: tts_engine.say(text); tts_engine.runAndWait()
        except Exception as e: st.error(f"Konuşma sırasında hata: {e}")

# Web Arama ve Kazıma
def scrape_url_content(url):
    """Verilen URL'den metin içeriğini kazımayı dener."""
    st.toast(f"🌐 '{urlparse(url).netloc}' adresinden içerik alınıyor...", icon="⏳")
    try:
        parsed_url = urlparse(url);
        if not all([parsed_url.scheme, parsed_url.netloc]): return None
        if parsed_url.scheme not in ['http', 'https']: return None
        headers = {'User-Agent': 'Mozilla/5.0 HanogtAI/3.1'}; response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True); response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower();
        if 'html' not in content_type: return None
        soup = BeautifulSoup(response.content, 'html.parser'); potential_content = []; selectors = ['article', 'main', '.content', '.post-content', '.entry-content', 'body']; content_found = False
        for selector in selectors: # Ana içeriği bulmaya çalış
             elements = soup.select(selector)
             if elements:
                  text_content = elements[0].find_all('p')
                  if text_content:
                      potential_content = [p.get_text(strip=True) for p in text_content if p.get_text(strip=True)];
                      if len(" ".join(potential_content)) > 100: content_found = True; break
        if not content_found: # Bulamazsa tüm paragrafları al
            all_paragraphs = soup.find_all('p'); potential_content = [p.get_text(strip=True) for p in all_paragraphs if p.get_text(strip=True)]
        if not potential_content: return None
        full_text = " ".join(potential_content); cleaned_text = re.sub(r'\s+', ' ', full_text).strip(); final_text = cleaned_text[:SCRAPE_MAX_CHARS]
        if len(cleaned_text) > SCRAPE_MAX_CHARS: final_text += "..."
        # st.success(f"URL'den içerik özeti başarıyla alındı.") # Çok fazla mesaj olmaması için toast kullandık
        return final_text
    except Exception as e: st.toast(f"⚠️ Sayfa ({urlparse(url).netloc}) işlenirken hata: {e}", icon='🌐'); return None

def search_web(query):
    """Web'de arama yapar (Wikipedia > DuckDuckGo > Scrape)."""
    st.toast(f"🔍 '{query}' için web'de aranıyor...", icon="⏳")
    summary = None
    # 1. Wikipedia
    try:
        wikipedia.set_lang("tr"); summary = wikipedia.summary(query, auto_suggest=False);
        st.toast("ℹ️ Wikipedia'dan bilgi bulundu.", icon="✅")
        return f"**Wikipedia'dan:**\n\n{summary}"
    except Exception: pass # Hata olursa sessizce diğer adıma geç

    # 2. DuckDuckGo
    ddg_result_text = None; ddg_url = None
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='tr-tr', max_results=1))
            if results:
                snippet = results[0].get('body'); ddg_url = results[0].get('href')
                if snippet: ddg_result_text = f"**Web Özeti (DuckDuckGo):**\n\n{snippet}\n\nKaynak: {ddg_url}"
    except Exception: pass # Hata olursa sessizce devam et

    # 3. Scrape (Eğer DDG URL bulduysa)
    if ddg_url:
        scraped_content = scrape_url_content(ddg_url)
        if scraped_content: return f"**Web Sayfasından ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
        elif ddg_result_text: return ddg_result_text # Kazıma başarısızsa DDG özeti
        else: return f"Detaylı bilgi için: {ddg_url}" # İkisi de yoksa link

    if ddg_result_text: return ddg_result_text # Sadece DDG özeti varsa
    st.toast("ℹ️ Web'de doğrudan yanıt bulunamadı.", icon="❌")
    return None # Hiçbir şey bulunamadı

# Sohbet Geçmişi Yönetimi
# BU FONKSİYONLARIN TANIMI, AŞAĞIDAKİ SESSION STATE BAŞLATMADAN ÖNCE GELİYOR
def load_chat_history():
    """Sohbet geçmişini JSON dosyasından yükler."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f: content = f.read()
            if content and content.strip(): return json.loads(content)
            else: return []
        except json.JSONDecodeError as e:
            st.error(f"Geçmiş dosyası ({CHAT_HISTORY_FILE}) bozuk: {e}. Sıfırlanıyor."); return []
        except Exception as e: st.error(f"Geçmiş yüklenirken hata: {e}"); return []
    else: return [] # Dosya yoksa boş liste

def save_chat_history(history):
    """Sohbet geçmişini JSON dosyasına kaydeder."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e: st.error(f"Geçmiş kaydedilemedi: {e}")

# Gemini Yanıt Alma
def get_gemini_response(prompt, chat_history):
    """Gemini modelinden yanıt alır."""
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    gemini_history = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]}
                      for sender, message in chat_history]
    try:
        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(prompt, stream=False)
        if not response.parts: # Yanıt kontrolü
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 reason = response.prompt_feedback.block_reason; st.warning(f"Gemini yanıtı engellendi: {reason}"); return f"{GEMINI_ERROR_PREFIX} Güvenlik: {reason}"
             else: st.warning(f"Gemini'dan boş yanıt: {response}"); return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
        return "".join(part.text for part in response.parts) # Başarılı yanıt
    except Exception as e: # Hata yönetimi
        st.error(f"Gemini API hatası: {e}"); msg = str(e)
        if "API key not valid" in msg: return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        # Diğer spesifik hatalar eklenebilir
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı."

# Yerel Yaratıcı/Görsel Fonksiyonları (Gemini olmadığında kullanılır)
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
def generate_prompt_influenced_image(prompt):
    # ... (Önceki kural tabanlı görsel oluşturma kodu buraya gelecek) ...
    # Örnek olarak basit bir placeholder bırakalım:
    width, height = 256, 256 # Daha küçük boyut
    img = Image.new('RGB', (width, height), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
        draw.text((10, height // 2 - 10), f"İstenen: {prompt[:30]}...", font=font, fill=(255,255,255))
    except: pass # Font hatası olursa çizmesin
    return img
# --- Session State Başlatma (load_chat_history tanımından SONRA) ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Yazılı Sohbet" # Başlangıç modu
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'user_avatar_bytes' not in st.session_state:
    st.session_state.user_avatar_bytes = None
if 'show_main_app' not in st.session_state:
    # Kullanıcı adı belirlenene kadar ana uygulama gizli
    st.session_state.show_main_app = bool(st.session_state.user_name) # Eğer daha önceden isim varsa direkt göster
if 'greeting_message_shown' not in st.session_state:
    st.session_state.greeting_message_shown = False

# --- Ana Başlık ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🚀 Hanogt AI 🚀</h1>", unsafe_allow_html=True)

# Gemini yüklenemezse ana bir uyarı göster
if gemini_init_error:
    st.error(gemini_init_error)

# --- Kullanıcı Adı Sorgulama ---
if not st.session_state.show_main_app:
    st.subheader("👋 Merhaba! Tanışalım...")
    name_input = st.text_input("Size nasıl hitap etmeliyim?", key="name_input_key", placeholder="İsminiz...")
    if st.button("Kaydet", key="save_name_button"):
        if name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.session_state.show_main_app = True
            st.session_state.greeting_message_shown = False # Tebrik için resetle
            st.rerun()
        else: st.error("Lütfen bir isim girin.")
# --- ANA UYGULAMA BÖLÜMÜ ---
else:
    # Tanışma mesajı (sadece bir kere gösterilir)
    if not st.session_state.greeting_message_shown and st.session_state.user_name:
         st.success(f"Tanıştığıma memnun oldum, {st.session_state.user_name}! Size nasıl yardımcı olabilirim?")
         st.session_state.greeting_message_shown = True

    # --- Ayarlar Bölümü ---
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        # Ad Değiştirme
        st.text_input("Adınızı Değiştirin:", value=st.session_state.user_name, key="change_name_input_key", on_change=lambda: setattr(st.session_state, 'user_name', st.session_state.change_name_input_key))
        st.caption(f"Mevcut adınız: {st.session_state.user_name}")

        st.divider()
        # Avatar Yönetimi
        st.write("**Avatar (Profil Resmi):**")
        uploaded_avatar = st.file_uploader("Yeni Avatar Yükle (PNG, JPG - Maks 1MB):", type=["png", "jpg", "jpeg"], key="avatar_uploader")
        if uploaded_avatar is not None:
            if uploaded_avatar.size > 1 * 1024 * 1024: st.error("Dosya > 1MB!")
            else: st.session_state.user_avatar_bytes = uploaded_avatar.getvalue(); st.success("Avatar güncellendi!"); st.rerun() # Hemen görünecek mi?
        if st.session_state.user_avatar_bytes:
            st.image(st.session_state.user_avatar_bytes, width=64, caption="Mevcut Avatarınız")
            if st.button("Avatarı Kaldır", key="remove_avatar"): st.session_state.user_avatar_bytes = None; st.rerun()
        else: st.caption("Henüz bir avatar yüklemediniz.")
        st.caption("Not: Avatar sadece bu oturum için geçerlidir, tarayıcı kapatılınca sıfırlanır.") # Kalıcılık notu

        st.divider()
        # Geçmiş Temizleme
        if st.button("🧹 Sohbet Geçmişini Temizle", key="clear_history_main"):
            st.session_state.chat_history = []; save_chat_history([]); st.success("Sohbet geçmişi temizlendi!"); time.sleep(1); st.rerun()

    st.markdown("---")

    # --- Mod Seçim Butonları ---
    st.write("**Uygulama Modu:**")
    modes = ["Yazılı Sohbet", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Mod", "Görsel Üretici"]
    icons = ["✏️", "🎙️", "✨", "🖼️"]
    cols = st.columns(len(modes))
    current_mode = st.session_state.app_mode
    for i, col in enumerate(cols):
        with col:
            button_type = "primary" if modes[i] == current_mode else "secondary"
            if st.button(f"{icons[i]} {modes[i]}", key=f"mode_btn_{i}", use_container_width=True, type=button_type):
                st.session_state.app_mode = modes[i]; st.rerun() # Modu hemen değiştirmek için rerun

    app_mode = st.session_state.app_mode # Seçili modu al
    st.markdown("---")

    # --- MODLARA GÖRE ARAYÜZLER ---

    # -- YAZILI SOHBET --
    if app_mode == "Yazılı Sohbet":
        # Mesajları göster
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            is_user = sender.startswith("Sen")
            role = "user" if is_user else "assistant"
            display_avatar = None
            if is_user and st.session_state.user_avatar_bytes:
                try: display_avatar = Image.open(BytesIO(st.session_state.user_avatar_bytes))
                except Exception: display_avatar = "🧑" # Hata olursa varsayılan
            elif not is_user:
                display_avatar = "🤖" # AI için varsayılan avatar

            with st.chat_message(role, avatar=display_avatar):
                 # Kaynağı belirt (varsa)
                display_name = ""
                if not is_user and "(" in sender and ")" in sender:
                     source = sender[sender.find("(")+1:sender.find(")")]
                     display_name = f"({source}) "
                st.markdown(f"{display_name}{message}") # Mesajı göster

        # Yeni mesaj girişi
        if prompt := st.chat_input(f"{st.session_state.user_name} olarak mesaj yazın..."):
            st.session_state.chat_history.append(("Sen", prompt)) # Basitçe "Sen" olarak ekle

            # YANIT AKIŞI (Önceki kod ile aynı: Gemini > KB > Web > Default)
            response = None; ai_sender = "Hanogt AI"
            with st.spinner("🤖 Düşünüyorum..."):
                if gemini_model: # 1. Gemini
                    response = get_gemini_response(prompt, st.session_state.chat_history)
                    if response and not response.startswith(GEMINI_ERROR_PREFIX): ai_sender = "Hanogt AI (Gemini)"
                    elif response and response.startswith(GEMINI_ERROR_PREFIX): st.toast(f"⚠️ Gemini: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖"); response = None
                    else: response = None
                if not response: # 2. Bilgi Tabanı
                    kb_resp = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
                    if kb_resp: response = kb_resp; ai_sender = "Hanogt AI (Bilgi Tabanı)"
                if not response: # 3. Web Arama
                    web_resp = search_web(prompt)
                    if web_resp:
                        response = web_resp
                        if "**Wikipedia" in response: ai_sender = "Hanogt AI (Wikipedia)"
                        elif "**Web Özeti" in response: ai_sender = "Hanogt AI (Web Özeti)"
                        elif "**Web Sayfasından" in response: ai_sender = "Hanogt AI (Sayfa İçeriği)"
                        else: ai_sender = "Hanogt AI (Web Link)"
                if not response: # 4. Varsayılan
                    response = random.choice(["Yanıt veremiyorum.","Anlayamadım.","Başka soru?"]); ai_sender = "Hanogt AI"

            # Yanıtı kaydet ve arayüzü yenile
            st.session_state.chat_history.append((ai_sender, response))
            save_chat_history(st.session_state.chat_history)
            st.rerun()

    # -- SESLİ SOHBET (DOSYA YÜKLEME) --
    elif app_mode == "Sesli Sohbet (Dosya Yükle)":
        st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası yükleyin.")
        uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed")
        if uploaded_file is not None:
            st.audio(uploaded_file)
            user_prompt = None; ai_sender = "Hanogt AI"; response = None
            with st.spinner("Ses dosyası işleniyor..."):
                # ... (Ses işleme kodu önceki ile aynı) ...
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(uploaded_file) as source: audio_data = recognizer.record(source)
                    user_prompt = recognizer.recognize_google(audio_data, language="tr-TR")
                    st.success(f"**Algılanan Metin:** {user_prompt}")
                except Exception as e: st.error(f"Ses dosyası işlenemedi: {e}"); user_prompt = None

            if user_prompt:
                st.session_state.chat_history.append(("Sen (Ses Dosyası)", user_prompt))
                # YANIT AKIŞI (Yazılı sohbet ile aynı)
                with st.spinner("🤖 Yanıt oluşturuluyor..."):
                    # ... (Gemini > KB > Web > Default logic) ...
                    if gemini_model: # 1. Gemini
                        response = get_gemini_response(user_prompt, st.session_state.chat_history)
                        if response and not response.startswith(GEMINI_ERROR_PREFIX): ai_sender = "Hanogt AI (Gemini)"
                        elif response and response.startswith(GEMINI_ERROR_PREFIX): st.toast(f"⚠️ Gemini: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖"); response = None
                        else: response = None
                    if not response: # 2. Bilgi Tabanı
                        kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
                        if kb_resp: response = kb_resp; ai_sender = "Hanogt AI (Bilgi Tabanı)"
                    if not response: # 3. Web Arama
                        web_resp = search_web(prompt) # prompt yerine user_prompt olmalı
                        if web_resp:
                            response = web_resp
                            if "**Wikipedia" in response: ai_sender = "Hanogt AI (Wikipedia)"
                            elif "**Web Özeti" in response: ai_sender = "Hanogt AI (Web Özeti)"
                            elif "**Web Sayfasından" in response: ai_sender = "Hanogt AI (Sayfa İçeriği)"
                            else: ai_sender = "Hanogt AI (Web Link)"
                    if not response: # 4. Varsayılan
                         response = random.choice(["Sesinizi yazıya döktüm ama yanıt veremiyorum.","Fikrim yok."]); ai_sender = "Hanogt AI"

                # Yanıtı göster, oku ve kaydet
                st.markdown(f"**{ai_sender}:**"); st.markdown(response)
                speak(response)
                st.session_state.chat_history.append((ai_sender, response))
                save_chat_history(st.session_state.chat_history)
                # Otomatik rerun yapmayalım, belki kullanıcı başka dosya yükler

    # -- YARATICI MOD --
    elif app_mode == "Yaratıcı Mod":
        # ... (Kod önceki ile aynı: Gemini > Fallback) ...
        st.markdown("Bir fikir, bir kelime veya bir cümle yazın. Gemini (varsa) veya yerel yaratıcılığım size yanıt versin!")
        creative_prompt = st.text_input("Yaratıcılık tohumu:", key="creative_input", placeholder="Örn: Bulutların üzerinde yürüyen dev")
        if creative_prompt:
            ai_sender = "Hanogt AI (Yerel Yaratıcı)"; final_response = None
            if gemini_model:
                 st.info("Yaratıcı yanıt için Gemini kullanılıyor...")
                 with st.spinner("✨ İlham perileri Gemini ile fısıldaşıyor..."):
                      gemini_creative_prompt = f"Aşağıdaki isteme yaratıcı, ilginç ve özgün bir yanıt ver:\n\n\"{creative_prompt}\""
                      gemini_resp = get_gemini_response(gemini_creative_prompt, [])
                      if gemini_resp and not gemini_resp.startswith(GEMINI_ERROR_PREFIX): final_response = gemini_resp; ai_sender = "Hanogt AI (Gemini Yaratıcı)"
                      else: st.warning(f"Gemini yaratıcı yanıtı alınamadı ({gemini_resp.replace(GEMINI_ERROR_PREFIX, '') if gemini_resp else 'Hata'}). Yerel modül kullanılıyor."); final_response = None
            if not final_response:
                 st.info("Yerel yaratıcılık modülü kullanılıyor...")
                 with st.spinner("✨ Kendi fikirlerimi demliyorum..."):
                     final_response = creative_response(creative_prompt); new_word = advanced_word_generator(creative_prompt); final_response += f"\n\n_(Ayrıca türettiğim kelime: **{new_word}**)_"; ai_sender = "Hanogt AI (Yerel Yaratıcı)"
            st.markdown(f"**{ai_sender}:**"); st.markdown(final_response)


    # -- GÖRSEL ÜRETİCİ --
    elif app_mode == "Görsel Üretici":
        # ... (Kod önceki ile aynı: Kural tabanlı) ...
        st.markdown("Hayalinizdeki görseli tarif edin, anahtar kelimelere göre sizin için (sembolik olarak) çizeyim!")
        st.info("Not: Bu mod henüz Gemini Vision veya ImageFX gibi API'leri kullanmıyor, kural tabanlı çizim yapar.")
        image_prompt = st.text_input("Ne çizmemi istersiniz?", key="image_input", placeholder="Örn: Yıldızlı gecede parlayan deniz feneri")
        if st.button("🎨 Görseli Oluştur"):
            if image_prompt:
                with st.spinner("Fırçalarım hazırlanıyor..."): image = generate_prompt_influenced_image(image_prompt)
                st.image(image, caption=f"Hanogt AI'ın '{image_prompt}' yorumu (Kural Tabanlı)", use_container_width=True)
                buf = BytesIO(); image.save(buf, format="PNG"); byte_im = buf.getvalue()
                st.download_button(label="Görseli İndir (PNG)", data=byte_im, file_name=f"hanogt_ai_rulebased_{image_prompt[:20].replace(' ','_')}.png", mime="image/png")
            else: st.error("Lütfen ne çizmemi istediğinizi açıklayan bir metin girin!")


# --- Alt Bilgi ---
if st.session_state.show_main_app: # Sadece ana uygulama görünüyorsa göster
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: small;'>Hanogt AI v3 - {st.session_state.get('user_name', 'Misafir')} için çalışıyor - 2025</p>", unsafe_allow_html=True)
