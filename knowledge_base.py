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
def load_knowledge():
    knowledge = {
        # Genel Sorular
        "merhaba": "Merhaba {name}! Sana nasıl yardımcı olabilirim?",
        "nasılsın": "Ben bir yapay zekayım {name}, harika çalışıyorum!",
        "sen kimsin": "Ben Hanogt AI! Yapay zeka destekli bir asistanım.",
        "hangi dillerde konuşabiliyorsun": "Şu anda Türkçe dilinde iletişim kurabiliyorum.",
        "bana bir tavsiye ver": "Her gün küçük bir adım at {name}. Zamanla büyük farklar yaratırsın!",
        "bana bir şaka yap": "Tabii {name}! Matematik kitabı neden üzgündü? Çünkü çok problemi vardı!",
        "günün sözü nedir": "Başarı, küçük çabaların her gün tekrarlanmasıdır {name}.",
        "bana hikaye anlat": "Bir zamanlar, yapay zekalar insanların en iyi dostları olmuştu... {name} de onlardan biriydi!",
        "programlama öğrenmek istiyorum": "Harika {name}! Python ile başlayabilirsin. Basit ve güçlü bir dildir.",
        
        # Yemek Soruları
        "en sevdiğin yemek nedir": "Benim için pizza ve makarna çok popüler gözüküyor!",
        "ne yemek yapabilirim": "Hızlı bir omlet veya makarna hazırlayabilirsin!",
        "tatlı önerisi": "Çikolatalı kek veya tiramisu deneyebilirsin!",
        "kahvaltı önerisi": "Kahvaltı için yulaf ezmesi, peynir ve zeytin güzel bir seçenek olabilir.",
        "yemek tarifini paylaş": "Tabii! Basit bir menemen tarifi verebilirim: Yumurta, domates, biber, tuz ve baharatlarla harika bir menemen yapabilirsin!",
        "vegan yemek önerisi": "Nohutlu salata veya vegan burger harika seçenekler olabilir.",
        "yemek nasıl pişirilir": "Bunu belirlemenizi öneririm: Hangi yemeği yapmak istediğinizi söylerseniz, tarifi detaylandırırım.",
        "içki önerisi": "Meyveli kokteyller veya alkolsüz içecekler harika bir seçenek olabilir.",
        "akşam yemeği ne yapabilirim": "Sebzeli karnıbahar çorbası veya tavuklu salata harika bir akşam yemeği olabilir.",
        "tatlı yaparken nelere dikkat etmeliyim": "Tatlı yaparken doğru malzeme ölçülerine dikkat etmek, özellikle hamurun kıvamını tutturmak çok önemlidir.",
        
        # Spor Soruları
        "hangi sporu önerirsin": "Yüzme, koşu ve yoga hem eğlenceli hem de sağlıklıdır!",
        "en popüler spor nedir": "Futbol dünya genelinde en popüler spordur.",
        "basketbol kuralları nedir": "Topu potaya sokarak sayı kazanmaya çalışırsın. 5 kişilik iki takım arasında oynanır.",
        "tenis nasıl oynanır": "Tenis, raketle topu karşı tarafa göndermeyi hedefler. Oyuncular sırasıyla servis yapar ve puan kazanırlar.",
        "futbol nasıl oynanır": "Futbol, topu rakip kaleye sokarak gol atmaya dayalı bir oyundur. 11 oyuncudan oluşan takımlar oynar.",
        "yoga nedir": "Yoga, zihni ve bedeni rahatlatan bir egzersiz yöntemidir.",
        "futbolun kuralları nedir": "Futbol, topun rakip takımın kalesine sokulmasıyla oynanır. 11 kişi olan takımların her biri savunma ve atak yapar.",
        "kayak nasıl yapılır": "Kayak yapmak için dağda kar üzerinde kayak takımlarını giyip kayma hareketi yaparak iniş yaparsınız.",
        "fitness nedir": "Fitness, fiziksel sağlığı artırmak amacıyla yapılan egzersizleri ifade eder.",
        "futbolcu nasıl olunur": "Futbolcu olmak için düzenli antrenman yaparak yeteneklerinizi geliştirmeli ve bir futbol kulübüyle sözleşme imzalamalısınız.",
        "yüzme nasıl öğrenilir": "Yüzme öğrenmek için doğru nefes almayı öğrenmeli, suya güvenmeli ve temel hareketleri çalışmalısınız.",
        "zumba nedir": "Zumba, dans ve aerobik hareketleri birleştiren eğlenceli bir fitness türüdür.",
        
        # Teknoloji Soruları
        "en iyi telefon markası hangisi": "iPhone, Samsung ve Xiaomi günümüzde çok tercih ediliyor.",
        "yapay zeka nedir": "Yapay zeka, insan zekasını taklit eden bilgisayar sistemleridir.",
        "geleceğin teknolojileri": "Yapay zeka, kuantum bilgisayarlar ve biyoteknoloji geleceği şekillendiriyor!",
        "robotlar ne iş yapar": "Robotlar, insan gibi belirli işleri yapabilen makinelerdir. Otomasyon ve üretimde yaygın kullanılır.",
        "yapay zeka nasıl çalışır": "Yapay zeka, verileri analiz eder ve örüntüleri tanır. Bu sayede karar verme ve öğrenme yeteneği kazanır.",
        "blockchain nedir": "Blockchain, verilerin güvenli ve değiştirilemez bir şekilde kaydedildiği bir teknoloji sistemidir.",
        "yeni teknoloji ürünleri": "Yapay zeka destekli cihazlar, giyilebilir teknolojiler ve akıllı ev cihazları şu anda popüler teknoloji ürünleri arasında.",
        "5G nedir": "5G, mobil ağ teknolojisinin beşinci neslidir ve daha hızlı internet bağlantısı sağlar.",
        "yapay zeka oyunları nasıl çalışır": "Yapay zeka, oyunlarda karakterlerin ve ortamların akıllıca tepki vermesini sağlar.",
        "tablet nedir": "Tablet, dokunmatik ekranı olan ve genellikle taşınabilir bir bilgisayar cihazıdır.",
        "robotik kol nedir": "Robotik kol, endüstriyel otomasyon ve cerrahi müdahalelerde kullanılan bir robot teknolojisidir.",
        "sanal gerçeklik nedir": "Sanal gerçeklik, bilgisayar destekli ortamlarda etkileşimli deneyimler yaratmaya olanak tanır.",
        
        # Bilim Soruları
        "uzayda yaşam var mı": "Şu anda dünyadan başka bir yerde kanıtlanmış yaşam bulunamadı.",
        "en büyük gezegen hangisi": "Jüpiter, Güneş Sistemi'nin en büyük gezegenidir.",
        "ışık hızı nedir": "Işık saniyede yaklaşık 299,792 kilometre yol alır!",
        "nükleer enerji nedir": "Nükleer enerji, atom çekirdeklerinin bölünmesiyle elde edilen büyük miktarda enerjidir.",
        "dünya nasıl oluştu": "Dünya, yaklaşık 4.5 milyar yıl önce, gaz ve toz bulutlarının çekilmesiyle oluştu.",
        "yıldızlar neden parlar": "Yıldızlar, nükleer füzyon yoluyla enerji üretirler, bu da ışık yaymalarına neden olur.",
        "bilimsel yöntem nedir": "Bilimsel yöntem, bir problemi çözmek için gözlem, hipotez oluşturma ve deney yapma aşamalarını içerir.",
        "dünya neden döner": "Dünya, dönme hareketi yapar çünkü oluşumu sırasında bir açısal momentum kazanmıştır.",
        "gezegen nedir": "Gezegen, yıldız çevresinde dönen ve yeterli büyüklükte olan bir gök cismidir.",
        "evrim nedir": "Evrim, canlı türlerinin zaman içinde genetik değişimlerle yeni özellikler kazanmasını ifade eder.",
        "kara delik nedir": "Kara delik, ışığın bile kaçamayacağı kadar güçlü bir çekim alanına sahip olan bir gök cismidir.",
        "fiziksel değişim nedir": "Fiziksel değişim, bir maddenin formunun değişmesi, ancak kimyasal yapısının aynı kalmasıdır.",
        
        # Hava Durumu & Gün Bilgisi
        "hava nasıl": "Şu anda hava durumunu veremem {name}, ama birlikte kontrol edebiliriz!",
        "bugün günlerden ne": "Bugün güzel bir gün {name}!",
        "yarın hava nasıl olacak": "Yarının hava durumu hakkında bilgi almak için yerel hava durumu kaynağını kontrol edebilirsin.",
        "hava durumu nedir": "Hava durumu, sıcaklık, nem, rüzgar gibi atmosfer koşullarını ifade eder.",
        "günümüzün tarihi nedir": "Bugün {date}!",
        "bugün kar yağıyor mu": "Bugün kar yağışı olup olmadığını yerel hava durumu kaynağından öğrenebilirsin.",
        "yazın hava nasıl olur": "Yazın genellikle sıcak ve güneşli hava hakimdir.",
        "kışın hava nasıl olur": "Kışın hava soğuk, bazen kar yağışlı olabilir.",
        "sonbaharda hava nasıl olur": "Sonbahar, genellikle serin ve yağışlı olabilir.",
        "ilkbaharda hava nasıl olur": "İlkbaharda hava genellikle ılımandır ve doğa canlanmaya başlar.",
        
        # Kapanış Cevapları
        "görüşürüz": "Görüşmek üzere {name}! İyi günler!",
        "hoşça kal": "Hoşça kal {name}! Kendine iyi bak!",
        "teşekkür ederim": "Her zaman yardımcı olmaktan mutluluk duyarım, {name}!",
        "sağ ol": "Rica ederim, her zaman yardımcı olabilirim!",
        "güle güle": "Güle güle {name}! Kendine iyi bak!",
        "yardım et": "Tabii ki! Yardımcı olabileceğim bir konu var mı {name}?",
        "bana bir konu öner": "Güncel bir konu olarak yapay zeka veya uzay hakkında sohbet edebiliriz!",
        "bana bir kitap öner": "Jules Verne'in 'Denizler Altında Yirmi Bin Fersah'ı harika bir kitap!",
        "bana bir film öner": "Bilim kurgu filmleri seviyorsan, 'Interstellar'ı mutlaka izlemeni öneririm!",
        "seninle tekrar konuşmak isterim": "Beni tekrar çağırdığında burada olacağım {name}! Görüşmek üzere!",
        "görüşürüz tekrar": "Tekrar görüşmek üzere {name}! Kendine iyi bak!",
    }
    return knowledge