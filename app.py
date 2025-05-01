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
from PIL import Image, ImageDraw, ImageFont # PIL eklendi (önceki kodda vardı)
import time
from io import BytesIO
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import google.generativeai as genai

# --- Sabitler ---
CHAT_HISTORY_FILE = "chat_history.json"
# LOGO_PATH = "logo.png" # Logo artık kullanıcı tarafından yüklenecek veya varsayılan olacak
DEFAULT_ERROR_MESSAGE = "Üzgünüm, bir şeyler ters gitti. Lütfen tekrar deneyin."
REQUEST_TIMEOUT = 10
SCRAPE_MAX_CHARS = 1000
GEMINI_ERROR_PREFIX = "GeminiError:"

# --- Bilgi Tabanı (Mock) ---
try:
    from knowledge_base import load_knowledge, chatbot_response as kb_chatbot_response
except ImportError:
    st.warning("`knowledge_base.py` bulunamadı. Yerel bilgi tabanı yanıtları kullanılamayacak.")
    def load_knowledge(): return {}
    def kb_chatbot_response(query, knowledge): return None
KNOWLEDGE_BASE = load_knowledge()

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hanogt AI", page_icon="🚀", layout="wide")

# --- API Anahtarı ve Gemini Yapılandırması ---
api_key = None
gemini_model = None
gemini_init_error = None

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    gemini_init_error = "🛑 Google API Anahtarı Secrets'ta bulunamadı! Lütfen yapılandırın."

if api_key:
    try:
        genai.configure(api_key=api_key)
        safety_settings = [ # Güvenlik ayarları (isteğe bağlı)
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # Diğer kategoriler eklenebilir veya ayarlar kaldırılabilir
        ]
        # Güncel ve genellikle çalışan bir model kullanalım
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
        # Hızlı bir test çağrısı (isteğe bağlı, açılışı yavaşlatabilir)
        # gemini_model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1))
    except Exception as e:
        gemini_init_error = f"🛑 Gemini yapılandırma hatası: {e}"
        gemini_model = None

# --- Session State Başlatma ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Yazılı Sohbet"
if 'user_name' not in st.session_state:
    st.session_state.user_name = None # Kullanıcı adı başlangıçta boş
if 'user_avatar_bytes' not in st.session_state:
    st.session_state.user_avatar_bytes = None # Kullanıcı avatarı başlangıçta boş
if 'show_main_app' not in st.session_state:
     # Kullanıcı adı belirlenene kadar ana uygulamayı gösterme flag'i
    st.session_state.show_main_app = False
if 'greeting_message_shown' not in st.session_state:
    st.session_state.greeting_message_shown = False # Tanışma mesajı gösterildi mi?

# --- Yardımcı Fonksiyonlar (TTS, Web Arama, Gemini Yanıt vb.) ---
# Metin Okuma
tts_engine = None
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    # Bu hatayı sidebar yerine geçici bir uyarı olarak gösterelim
    st.toast(f"⚠️ Metin okuma motoru başlatılamadı: {e}", icon="🔊")

def speak(text):
    if tts_engine:
        try: tts_engine.say(text); tts_engine.runAndWait()
        except Exception as e: st.error(f"Konuşma sırasında hata: {e}")
    # else: st.toast("🔊 Metin okuma motoru kullanılamıyor.", icon="⚠️") # Çok sık uyarı vermemek için kaldırıldı

# Web Arama & Kazıma (Önceki kod ile aynı)
def scrape_url_content(url):
    # ... (Kod yukarıdaki ile aynı) ...
    try:
        # ... (scraping logic) ...
        return final_text # Başarılı olursa metni döndür
    except Exception as e: st.toast(f"⚠️ Sayfa ({url}) işlenirken hata: {e}", icon='🌐'); return None # Hata durumunda None döndür
def search_web(query):
    # ... (Wikipedia ve DDG kodu yukarıdaki ile aynı) ...
    try: # Wikipedia
        # ... (wikipedia logic) ...
        return summary
    except Exception as e: st.toast(f"⚠️ Wikipedia araması hatası: {e}", icon='🌐'); pass # Hata olursa devam et
    try: # DuckDuckGo & Scrape
        # ... (ddg logic) ...
        if ddg_url:
            scraped_content = scrape_url_content(ddg_url)
            if scraped_content: return f"**Web Sayfasından ({urlparse(ddg_url).netloc}):**\n\n{scraped_content}\n\nKaynak: {ddg_url}"
            elif ddg_result_text: return ddg_result_text
            else: return f"Detaylı bilgi için: {ddg_url}"
        elif ddg_result_text: return ddg_result_text
    except Exception as e: st.toast(f"⚠️ Web araması hatası: {e}", icon='🌐')
    return None

# Sohbet Geçmişi
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

# Gemini Yanıt (Önceki kod ile aynı)
def get_gemini_response(prompt, chat_history):
    if not gemini_model: return f"{GEMINI_ERROR_PREFIX} Model aktif değil."
    gemini_history = [{'role': ("user" if sender.startswith("Sen") else "model"), 'parts': [message]}
                      for sender, message in chat_history]
    try:
        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(prompt, stream=False)
        if not response.parts:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason; st.warning(f"Gemini yanıtı engellendi: {block_reason}"); return f"{GEMINI_ERROR_PREFIX} Güvenlik filtresi: {block_reason}"
             else: st.warning(f"Gemini'dan boş yanıt alındı: {response}"); return f"{GEMINI_ERROR_PREFIX} Boş yanıt."
        return "".join(part.text for part in response.parts)
    except Exception as e:
        st.error(f"Gemini API hatası: {e}"); error_message = str(e)
        if "API key not valid" in error_message: return f"{GEMINI_ERROR_PREFIX} API Anahtarı geçersiz."
        if "billing account" in error_message.lower(): return f"{GEMINI_ERROR_PREFIX} Faturalandırma sorunu."
        if "API has not been used" in error_message: return f"{GEMINI_ERROR_PREFIX} API projede etkin değil."
        return f"{GEMINI_ERROR_PREFIX} API ile iletişim kurulamadı."

# Yerel Yaratıcı/Görsel Fonksiyonları (Fallback için)
def creative_response(prompt): # ... (Kod yukarıdaki ile aynı) ...
    styles = ["Bunu farklı bir açıdan düşünürsek: {}", "Hayal gücümüzü kullanalım: {}", "Belki de olay şöyledir: {}", "Aklıma şöyle bir fikir geldi: {}", "Şöyle bir senaryo canlandı gözümde: {}"]; base_idea = generate_new_idea(prompt); comment = random.choice(styles).format(base_idea); return comment
def generate_new_idea(seed): # ... (Kod yukarıdaki ile aynı) ...
    elements = ["kozmik enerji", "zaman döngüleri", "yapay bilinç", "nanobotlar", "ses manzaraları", "dijital ruhlar"]; actions = ["keşfeder", "dönüştürür", "bağlantı kurar", "yeniden şekillendirir", "hızlandırır", "yavaşlatır"]; outcomes = ["evrenin sırlarını", "insanlığın kaderini", "gerçekliğin dokusunu", "unutulmuş anıları", "geleceğin teknolojisini"]; seed_words = seed.lower().split()[:2]; idea = f"{' '.join(seed_words)} {random.choice(actions)} ve {random.choice(elements)} kullanarak {random.choice(outcomes)}."; return idea.capitalize()
def advanced_word_generator(base_word): # ... (Kod yukarıdaki ile aynı) ...
    if not base_word or len(base_word) < 3: return "Kelimetor"
    vowels = "aeiouüöı"; consonants = "bcçdfgğhjklmnprsştvyz"; prefix = ["eko", "meta", "neo", "trans", "kripto", "hiper"]; suffix = ["loji", "matik", "nomi", "grafi", "sentez", "versiyon", "izim"]
    if random.random() > 0.5: split_point = random.randint(1, len(base_word) - 1); core = base_word[:split_point] if random.random() > 0.5 else base_word[split_point:]
    else: core = ''.join(random.choice(consonants + vowels) for _ in range(random.randint(3, 5)))
    new_word = core;
    if random.random() > 0.3: new_word = random.choice(prefix) + new_word
    if random.random() > 0.3: new_word += random.choice(suffix)
    return new_word.capitalize()
def generate_prompt_influenced_image(prompt): # ... (Kod yukarıdaki ile aynı) ...
    # ... (rule-based image generation) ...
    return img

# --- Ana Başlık ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🚀 Hanogt AI 🚀</h1>", unsafe_allow_html=True)

# --- Kullanıcı Adı Sorgulama ve Ayarlar ---
if not st.session_state.user_name:
    st.session_state.show_main_app = False # Henüz ana uygulamayı gösterme
    st.subheader("👋 Merhaba! Tanışalım...")
    name_input = st.text_input("Size nasıl hitap etmeliyim?", key="name_input_key", placeholder="İsminiz veya takma adınız...")
    if st.button("Kaydet", key="save_name_button"):
        if name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.session_state.show_main_app = True # Artık ana uygulamayı gösterebiliriz
            st.session_state.greeting_message_shown = False # Tebrik mesajını göstermek için resetle
            st.rerun() # Sayfayı yenileyerek ana arayüzü yükle
        else:
            st.error("Lütfen bir isim girin.")
else:
    # Kullanıcı adı belirlendiyse ve tebrik mesajı gösterilmediyse göster
    if not st.session_state.greeting_message_shown:
         st.success(f"Tanıştığıma memnun oldum, {st.session_state.user_name}! Size nasıl yardımcı olabilirim?")
         st.session_state.greeting_message_shown = True # Mesaj gösterildi olarak işaretle
    # Kullanıcı adı varsa ana uygulamayı göster
    st.session_state.show_main_app = True

# --- ANA UYGULAMA (Kullanıcı adı belirlendiyse gösterilir) ---
if st.session_state.show_main_app:

    # --- Ayarlar Bölümü (Expander içinde) ---
    with st.expander("⚙️ Ayarlar & Kişiselleştirme", expanded=False):
        st.write(f"**Mevcut Kullanıcı Adı:** {st.session_state.user_name}")
        new_name = st.text_input("Adınızı Değiştirin:", placeholder="Yeni isim...", key="change_name_input")
        if st.button("Adımı Güncelle", key="update_name_button"):
            if new_name.strip():
                st.session_state.user_name = new_name.strip()
                st.success("Adınız güncellendi!")
                time.sleep(1) # Kısa bekleme
                st.rerun() # Yeni adı hemen göstermek için
            else:
                st.warning("Lütfen geçerli bir isim girin.")

        st.divider()

        st.write("**Avatarınızı (Profil Resmi) Değiştirin:**")
        uploaded_avatar = st.file_uploader("Resim dosyası seçin (PNG, JPG):", type=["png", "jpg", "jpeg"], key="avatar_uploader")
        if uploaded_avatar is not None:
            # Dosya boyutunu kontrol et (isteğe bağlı, örn. 1MB limit)
            if uploaded_avatar.size > 1 * 1024 * 1024:
                 st.error("Dosya boyutu çok büyük (Maksimum 1MB).")
            else:
                # Dosya içeriğini byte olarak oku ve session state'e kaydet
                st.session_state.user_avatar_bytes = uploaded_avatar.getvalue()
                st.success("Avatarınız güncellendi! Mesaj gönderdiğinizde görünecek.")
                # Yüklenen resmi hemen göstermek için (isteğe bağlı)
                st.image(st.session_state.user_avatar_bytes, width=64)

        if st.session_state.user_avatar_bytes:
            st.write("Mevcut Avatar:")
            st.image(st.session_state.user_avatar_bytes, width=64)
            if st.button("Avatarı Kaldır", key="remove_avatar"):
                st.session_state.user_avatar_bytes = None
                st.success("Avatar kaldırıldı.")
                st.rerun()

        st.divider()

        if st.button("🧹 Sohbet Geçmişini Temizle", key="clear_history_main"):
            st.session_state.chat_history = []
            try:
                if os.path.exists(CHAT_HISTORY_FILE): os.remove(CHAT_HISTORY_FILE)
                st.success("Sohbet geçmişi temizlendi!")
                time.sleep(1)
            except OSError as e: st.error(f"Geçmiş dosyası silinirken hata: {e}")
            st.rerun() # Temizliği göstermek için

    st.markdown("---") # Ayarlar ve mod butonları arasına çizgi

    # --- Mod Seçim Butonları ---
    st.write("**Uygulama Modu Seçin:**")
    cols = st.columns(4)
    modes = ["Yazılı Sohbet", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Mod", "Görsel Üretici"]
    icons = ["✏️", "🎙️", "✨", "🖼️"]

    # Mevcut modu session state'den al
    current_mode = st.session_state.app_mode

    for i, col in enumerate(cols):
        with col:
            # Seçili moda göre buton tipini değiştir (daha belirgin)
            button_type = "primary" if modes[i] == current_mode else "secondary"
            if st.button(f"{icons[i]} {modes[i]}", key=f"mode_btn_{i}", use_container_width=True, type=button_type):
                st.session_state.app_mode = modes[i]
                # Mod değiştiğinde sayfayı yenilemek state'i temizleyebilir, dikkatli kullanılmalı.
                # Şimdilik yenileme yapmayalım, bir sonraki etkileşimde mod değişir.
                # st.rerun() # Gerekirse modu hemen aktif etmek için

    app_mode = st.session_state.app_mode # Güncel modu al
    st.markdown("---") # Mod butonları ve sohbet alanı arasına çizgi

    # --- Seçilen Modun Arayüzü ---

    # -- YAZILI SOHBET --
    if app_mode == "Yazılı Sohbet":
        # Geçmiş mesajları göster (Avatar ve İsim ile)
        for sender, message in st.session_state.chat_history:
            is_user = sender.startswith("Sen")
            role = "user" if is_user else "assistant"
            display_avatar = None
            if is_user and st.session_state.user_avatar_bytes:
                try:
                    # Byte verisinden PIL Image oluştur (hata kontrolü önemli)
                    pil_image = Image.open(BytesIO(st.session_state.user_avatar_bytes))
                    display_avatar = pil_image
                except Exception as img_err:
                    st.warning(f"Avatar yüklenirken hata: {img_err}. Varsayılan kullanılıyor.")
                    display_avatar = None #"🧑" # Veya başka bir varsayılan

            with st.chat_message(role, avatar=display_avatar):
                # Mesajın başına ismi ekleyelim (isteğe bağlı)
                display_name = st.session_state.user_name if is_user else "Hanogt AI" # Veya AI gönderici adını kullan
                # Mesajın kaynağını belirtmek için:
                if not is_user and "(" in sender and ")" in sender:
                     source = sender[sender.find("(")+1:sender.find(")")]
                     display_name = f"Hanogt AI ({source})"

                # Kullanıcı adı mesajın başına eklenebilir: st.markdown(f"**{display_name}:**\n{message}")
                # Veya sadece mesaj gösterilir:
                st.markdown(message)


        # Kullanıcı girdisi
        if prompt := st.chat_input(f"{st.session_state.user_name} olarak mesajınızı yazın..."):
            # Geçmişe eklerken "Sen" yerine kullanıcı adını kullanabiliriz (isteğe bağlı)
            # sender_tag = f"Sen ({st.session_state.user_name})"
            sender_tag = "Sen" # Şimdilik basit tutalım
            st.session_state.chat_history.append((sender_tag, prompt))
            # Mesajı hemen gösterme (yukarıdaki döngü zaten gösterecek)

            # --- YANIT OLUŞTURMA AKIŞI (GEMINI ÖNCELİKLİ) ---
            response = None; ai_sender = "Hanogt AI"
            with st.spinner("🤖 Düşünüyorum..."):
                if gemini_model:
                    response = get_gemini_response(prompt, st.session_state.chat_history)
                    if response and not response.startswith(GEMINI_ERROR_PREFIX): ai_sender = "Hanogt AI (Gemini)"
                    elif response and response.startswith(GEMINI_ERROR_PREFIX): st.toast(f"⚠️ Gemini yanıtı alınamadı: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖"); response = None
                    else: response = None
                if not response:
                    kb_resp = kb_chatbot_response(prompt, KNOWLEDGE_BASE)
                    if kb_resp: response = kb_resp; ai_sender = "Hanogt AI (Bilgi Tabanı)"
                if not response:
                    st.toast("🌐 Web'de aranıyor...", icon="🔍")
                    web_resp = search_web(prompt)
                    if web_resp:
                        response = web_resp
                        if "**Wikipedia" in response: ai_sender = "Hanogt AI (Wikipedia)"
                        elif "**Web Özeti" in response: ai_sender = "Hanogt AI (Web Özeti)"
                        elif "**Web Sayfasından" in response: ai_sender = "Hanogt AI (Sayfa İçeriği)"
                        else: ai_sender = "Hanogt AI (Web Link)"
                if not response: response = random.choice(["Yanıt veremiyorum.","Anlayamadım.","Başka soru?"]); ai_sender = "Hanogt AI"

            # --- Yanıtı Kaydet ---
            st.session_state.chat_history.append((ai_sender, response))
            save_chat_history(st.session_state.chat_history)
            st.rerun() # Yeni mesajları göstermek için sayfayı yenile

    # -- SESLİ SOHBET (DOSYA YÜKLEME) --
    elif app_mode == "Sesli Sohbet (Dosya Yükle)":
        st.info("Lütfen yanıtlamamı istediğiniz konuşmayı içeren bir ses dosyası yükleyin.")
        uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility="collapsed")

        if uploaded_file is not None:
            st.audio(uploaded_file)
            user_prompt = None; ai_sender = "Hanogt AI"; response = None

            with st.spinner("Ses dosyası işleniyor..."):
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(uploaded_file) as source: audio_data = recognizer.record(source)
                    user_prompt = recognizer.recognize_google(audio_data, language="tr-TR")
                    st.success(f"**Algılanan Metin:** {user_prompt}")
                except Exception as e: st.error(f"Ses dosyası işlenemedi: {e}"); user_prompt = None

            if user_prompt:
                # sender_tag = f"Sen ({st.session_state.user_name} - Ses)"
                sender_tag = "Sen (Ses Dosyası)"
                st.session_state.chat_history.append((sender_tag, user_prompt))

                # --- YANIT OLUŞTURMA AKIŞI (GEMINI ÖNCELİKLİ) ---
                with st.spinner("🤖 Yanıt oluşturuluyor..."):
                    if gemini_model:
                        response = get_gemini_response(user_prompt, st.session_state.chat_history)
                        if response and not response.startswith(GEMINI_ERROR_PREFIX): ai_sender = "Hanogt AI (Gemini)"
                        elif response and response.startswith(GEMINI_ERROR_PREFIX): st.toast(f"⚠️ Gemini yanıtı alınamadı: {response.replace(GEMINI_ERROR_PREFIX, '')}", icon="🤖"); response = None
                        else: response = None
                    if not response:
                        kb_resp = kb_chatbot_response(user_prompt, KNOWLEDGE_BASE)
                        if kb_resp: response = kb_resp; ai_sender = "Hanogt AI (Bilgi Tabanı)"
                    if not response:
                        st.toast("🌐 Web'de aranıyor...", icon="🔍")
                        web_resp = search_web(user_prompt)
                        if web_resp:
                            response = web_resp
                            if "**Wikipedia" in response: ai_sender = "Hanogt AI (Wikipedia)"
                            elif "**Web Özeti" in response: ai_sender = "Hanogt AI (Web Özeti)"
                            elif "**Web Sayfasından" in response: ai_sender = "Hanogt AI (Sayfa İçeriği)"
                            else: ai_sender = "Hanogt AI (Web Link)"
                    if not response: response = random.choice(["Sesinizi yazıya döktüm ama yanıt veremiyorum.","Fikrim yok."]); ai_sender = "Hanogt AI"

                # --- Yanıtı Göster, Kaydet ve Oku ---
                st.markdown(f"**{ai_sender}:**"); st.markdown(response)
                speak(response)
                st.session_state.chat_history.append((ai_sender, response))
                save_chat_history(st.session_state.chat_history)
                # Sesli yanıttan sonra otomatik rerun yapmayalım, kullanıcı tekrar dosya yükleyebilir.

    # -- YARATICI MOD --
    elif app_mode == "Yaratıcı Mod":
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

# --- Alt Bilgi (Kullanıcı adı belirlenmediyse görünmez) ---
if st.session_state.show_main_app:
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: small;'>Hanogt AI v3 - {st.session_state.user_name} için çalışıyor - 2025</p>", unsafe_allow_html=True)

