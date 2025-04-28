# knowledge_base.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modeli yükle
# Burada dikkat: Eğer internet yoksa, modeli yerel indirip kullanmalısın!
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Bilgi tabanını yükle
def load_knowledge():
    knowledge = {
        "Merhaba": "Merhaba! Size nasıl yardımcı olabilirim?",
        "Nasılsın": "Ben bir yapay zekayım, her zaman iyiyim! Size nasıl yardımcı olabilirim?",
        "Hava durumu": "Şu anda bulunduğunuz lokasyonun hava durumu bilgilerine ulaşamıyorum, ama genel bilgiler verebilirim.",
        "Sen kimsin": "Ben Hanogt AI, sizin için buradayım!",
        "Ne yapabilirsin": "Size sorularınızda yardımcı olabilirim, bilgiler sunabilirim ve yaratıcı fikirler üretebilirim."
    }
    return knowledge

# Kullanıcıdan gelen mesaja cevap veren fonksiyon
def chatbot_response(user_input, knowledge):
    # Bilgi tabanındaki anahtar kelimeleri encode et
    keys = list(knowledge.keys())
    key_embeddings = model.encode(keys)

    # Kullanıcının sorusunu encode et
    user_embedding = model.encode(user_input)

    # Benzerlik hesapla
    similarities = cosine_similarity([user_embedding], key_embeddings)[0]

    # En yakın anahtar kelimeyi bul
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]

    # Eşik değer (kalite için)
    threshold = 0.45

    if best_similarity >= threshold:
        best_key = keys[best_idx]
        return knowledge[best_key]
    else:
        return None