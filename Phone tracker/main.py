import phonenumbers
from phonenumbers import geocoder, carrier
from opencage.geocoder import OpenCageGeocode
import folium
import webbrowser
import requests
from myphone import number  # Pastikan myphone.py ada variabel: number = "+62822xxxxxxx"

def analyze_phone_number_realistic(phone_number, api_key):
    """
    Analisis realistis nomor telepon dengan penjelasan keterbatasan
    """
    print("="*60)
    print("ANALISIS NOMOR TELEPON - TUJUAN EDUKASI")
    print("="*60)
    
    try:
        # Parse nomor
        parsed_number = phonenumbers.parse(phone_number)
        
        if not phonenumbers.is_valid_number(parsed_number):
            print("[ERROR] Nomor telepon tidak valid!")
            return None
            
        country_code = parsed_number.country_code
        national_number = str(parsed_number.national_number)
        
        print(f"[INFO] Nomor yang dianalisis: {phone_number}")
        print(f"[INFO] Country Code: +{country_code}")
        print(f"[INFO] National Number: {national_number}")
        print()
        
        # Informasi dasar
        location = geocoder.description_for_number(parsed_number, "en")
        provider = carrier.name_for_number(parsed_number, "en")
        
        print("[INFORMASI DASAR]")
        print(f"• Lokasi (dari library): {location}")
        print(f"• Provider: {provider}")
        print()
        
        # Analisis khusus Indonesia
        if country_code == 62:
            return analyze_indonesia_number(national_number, api_key)
        else:
            print("[INFO] Kode ini khusus untuk nomor Indonesia")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error parsing: {e}")
        return None

def analyze_indonesia_number(national_number, api_key):
    """
    Analisis khusus untuk nomor Indonesia dengan data yang lebih akurat
    """
    print("[ANALISIS NOMOR INDONESIA]")
    
    # Database prefix operator Indonesia yang lebih lengkap
    operator_data = {
        # Telkomsel
        '811': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '812': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '813': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '821': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '822': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '823': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '851': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '852': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        '853': {'operator': 'Telkomsel', 'type': 'GSM', 'region': 'Nasional'},
        
        # Indosat
        '814': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '815': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '816': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '855': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '856': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '857': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        '858': {'operator': 'Indosat', 'type': 'GSM', 'region': 'Nasional'},
        
        # XL Axiata
        '817': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        '818': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        '819': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        '859': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        '877': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        '878': {'operator': 'XL', 'type': 'GSM', 'region': 'Nasional'},
        
        # Tri (3)
        '895': {'operator': 'Tri', 'type': 'GSM', 'region': 'Nasional'},
        '896': {'operator': 'Tri', 'type': 'GSM', 'region': 'Nasional'},
        '897': {'operator': 'Tri', 'type': 'GSM', 'region': 'Nasional'},
        '898': {'operator': 'Tri', 'type': 'GSM', 'region': 'Nasional'},
        '899': {'operator': 'Tri', 'type': 'GSM', 'region': 'Nasional'},
        
        # Smartfren
        '881': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '882': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '883': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '884': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '885': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '886': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '887': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '888': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
        '889': {'operator': 'Smartfren', 'type': 'CDMA', 'region': 'Nasional'},
    }
    
    # Cari informasi operator
    operator_info = None
    for prefix in ['881', '882', '883', '811', '812', '813']:
        if national_number.startswith(prefix):
            operator_info = operator_data.get(prefix)
            break
    
    if not operator_info:
        # Coba dengan 3 digit pertama
        first_three = national_number[:3]
        operator_info = operator_data.get(first_three)
    
    if operator_info:
        print(f"• Operator: {operator_info['operator']}")
        print(f"• Teknologi: {operator_info['type']}")
        print(f"• Jangkauan: {operator_info['region']}")
    else:
        print("• Operator: Tidak teridentifikasi")
    
    print()
    print("[KETERBATASAN PELACAKAN]")
    print("• Nomor mobile TIDAK bisa dilacak ke alamat spesifik")
    print("• Hanya bisa diketahui negara dan kadang provinsi")
    print("• Lokasi real-time memerlukan akses ke BTS operator")
    print("• Hasil geocoding sering tidak akurat untuk mobile")
    print()
    
    # Buat peta realistis untuk Indonesia
    return create_indonesia_coverage_map(operator_info)

def create_indonesia_coverage_map(operator_info):
    """
    Buat peta coverage area Indonesia yang realistis
    """
    print("[MEMBUAT PETA COVERAGE INDONESIA]")
    
    # Koordinat pusat Indonesia
    indonesia_center = [-2.5489, 118.0149]
    
    # Koordinat kota-kota besar Indonesia
    major_cities = {
        'Jakarta': [-6.2088, 106.8456],
        'Surabaya': [-7.2575, 112.7521],
        'Bandung': [-6.9175, 107.6191],
        'Medan': [3.5952, 98.6722],
        'Semarang': [-6.9932, 110.4203],
        'Makassar': [-5.1477, 119.4327],
        'Palembang': [-2.9761, 104.7754],
        'Tangerang': [-6.1783, 106.6319],
        'Depok': [-6.4025, 106.7942],
        'Bekasi': [-6.2383, 106.9756],
        'Yogyakarta': [-7.7956, 110.3695],  # Lokasi Anda yang sebenarnya
        'Bantul': [-7.8879, 110.3281],      # Kabupaten Bantul
    }
    
    # Buat peta
    my_map = folium.Map(
        location=indonesia_center,
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Tambahkan marker untuk kota-kota besar
    for city, coords in major_cities.items():
        color = 'red' if city in ['Yogyakarta', 'Bantul'] else 'blue'
        icon_name = 'home' if city in ['Yogyakarta', 'Bantul'] else 'building'
        
        popup_text = f"""
        <b>{city}</b><br>
        Koordinat: {coords[0]:.4f}, {coords[1]:.4f}<br>
        {'<b>LOKASI ANDA</b>' if city in ['Yogyakarta', 'Bantul'] else 'Kota Besar Indonesia'}
        """
        
        folium.Marker(
            coords,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=city,
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
        ).add_to(my_map)
    
    # Tambahkan circle untuk Yogyakarta area
    folium.Circle(
        major_cities['Yogyakarta'],
        radius=50000,  # 50km radius
        popup="Area Yogyakarta - Kemungkinan coverage area",
        color='green',
        fill=True,
        fillColor='green',
        fillOpacity=0.2,
        weight=2
    ).add_to(my_map)
    
    # Tambahkan informasi disclaimer
    disclaimer_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 300px; height: auto; 
                background-color: rgba(255,255,255,0.9); border:2px solid red; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;
                ">
    <h4 style="color: red; margin: 0;">⚠️ DISCLAIMER</h4>
    <p><b>Pelacakan nomor mobile memiliki keterbatasan:</b></p>
    <ul>
        <li>Tidak bisa menunjukkan lokasi real-time</li>
        <li>Hanya estimasi coverage area operator</li>
        <li>Akurasi tergantung database publik</li>
        <li>Untuk edukasi, bukan surveillance</li>
    </ul>
    <p><b>Operator:</b> {operator_info['operator'] if operator_info else 'Unknown'}</p>
    </div>
    '''
    
    my_map.get_root().html.add_child(folium.Element(disclaimer_html))
    
    return my_map

def create_educational_explanation():
    """
    Penjelasan edukatif tentang pelacakan nomor telepon
    """
    print("="*60)
    print("PENJELASAN EDUKATIF: MENGAPA HASIL TIDAK AKURAT")
    print("="*60)
    
    explanations = [
        "1. KETERBATASAN TEKNOLOGI:",
        "   • Nomor mobile tidak terikat pada lokasi fisik tertentu",
        "   • Database publik hanya menyimpan info operator & region",
        "   • Geocoding service tidak memiliki data real-time BTS",
        "",
        "2. MENGAPA HASIL DI TENGAH LAUT:",
        "   • OpenCage/geocoding service menggunakan data tidak lengkap",
        "   • Default fallback ke koordinat center Indonesia",
        "   • Algoritma geocoding salah interpretasi query",
        "",
        "3. APA YANG SEBENARNYA BISA DILACAK:",
        "   • Negara asal nomor (dari country code)",
        "   • Operator/provider (dari prefix)",
        "   • Teknologi (GSM/CDMA)",
        "   • Region/provinsi (terbatas)",
        "",
        "4. PELACAKAN AKURAT MEMERLUKAN:",
        "   • Akses ke database operator (tidak publik)",
        "   • Data real-time dari BTS/cell tower",
        "   • Kerjasama dengan penegak hukum",
        "   • Triangulasi sinyal (butuh hardware khusus)",
        "",
        "5. ALTERNATIF YANG LEBIH AKURAT:",
        "   • GPS tracking (dengan izin)",
        "   • Find My Device / Google Find My Device",
        "   • Family sharing location",
        "   • Apps dengan location sharing",
    ]
    
    for line in explanations:
        print(line)
    
    print("="*60)

# Main execution
def main():
    # API Key OpenCage
    api_key = 'd9b612d5de1c4e23a8d4d81e8e9f3b26'
    
    # Penjelasan edukatif terlebih dahulu
    create_educational_explanation()
    
    print(f"\n[INFO] Menganalisis nomor: {number}")
    
    # Analisis nomor
    result_map = analyze_phone_number_realistic(number, api_key)
    
    if result_map:
        # Simpan peta
        map_file = "realistic_phone_coverage.html"
        result_map.save(map_file)
        print(f"[INFO] Peta coverage disimpan ke: {map_file}")
        
        # Buka di browser
        webbrowser.open(map_file)
        
        print("\n[KESIMPULAN]")
        print("• Peta menunjukkan coverage area Indonesia secara umum")
        print("• Marker merah menunjukkan lokasi Yogyakarta & Bantul")
        print("• Circle hijau adalah estimasi area coverage")
        print("• Ini bukan lokasi real-time, hanya estimasi regional")
    
    print("\n[PESAN EDUKASI]")
    print("Pelacakan nomor telepon untuk tujuan jahat adalah ilegal!")
    print("Gunakan pengetahuan ini hanya untuk edukasi dan keamanan.")

if __name__ == "__main__":
    main()